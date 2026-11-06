import copy
import importlib.util
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from weight_edit_lab.coding.environment import CodingConfig, CodingEditEnv
from tests.test_editing_benchmark import MemoryStudent


class DirectAcceptanceTests(unittest.TestCase):
    def test_partial_case_regression_rejects_even_when_full_functions_improve(self):
        student = MemoryStudent()
        env = CodingEditEnv(student, CodingConfig(horizon=1))
        env.reset()
        from weight_edit_lab.coding.actions import starter_catalog
        env.step(starter_catalog(24)[1]["action"])
        env.current["coding"]["case_accuracy"] = .4
        self.assertGreater(env.current["coding"]["accuracy"], env.baseline["coding"]["accuracy"])
        outcome = env.finalize(protect_case_accuracy=True)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(student.state, 0)
        self.assertEqual(env.optimizer_steps, 8)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "optional coding runtime")
class TensorProgramTests(unittest.TestCase):
    def setUp(self):
        import torch
        from weight_edit_lab.coding.backend import Student
        self.torch = torch
        self.student = Student.__new__(Student)
        s = self.student
        s.device, s.dtype, s.originals, s.num_layers = torch.device("cpu"), torch.float32, {}, 4
        s.manifest = {"model_id": "tiny", "revision": "fixed"}
        s.model = torch.nn.Module()
        s.model.model = torch.nn.Module()
        s.model.model.layers = torch.nn.ModuleList()
        for i in range(4):
            block = torch.nn.Module()
            block.mlp, block.self_attn = torch.nn.Module(), torch.nn.Module()
            for parent, name in [(block.mlp,"down_proj"), *[(block.self_attn,k) for k in ("v_proj","o_proj","q_proj")]]:
                linear = torch.nn.Linear(4,4,bias=False)
                with torch.no_grad(): linear.weight.fill_(1+i/10)
                setattr(parent, name, linear)
            s.model.model.layers.append(block)
        s.model.requires_grad_(False)
        self.name = "model.layers.0.mlp.down_proj"

    def test_authored_projection_changes_only_selected_columns_without_gradients(self):
        s, t = self.student, self.torch
        before = s.model.get_submodule(self.name).weight.clone()
        code = f"W=weight('{self.name}')\nu=unit(array([1,0,0,0]))\nC=W[:,[1,3]]\nD=C-0.05*outer(u,transpose(C)@u)\nwrite('{self.name}',columns(W,[1,3],D))"
        with patch.object(t.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")), patch.object(t.Tensor, "backward", side_effect=AssertionError("gradient forbidden")):
            result = s.edit({"kind":"tensor_program","code":code}, [], [])
        expected = before.clone();expected[0,[1,3]] -= .05
        self.assertTrue(t.equal(s.model.get_submodule(self.name).weight, expected))
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["tensors"][0]["changed_entries"], 2)
        s.restore()
        self.assertTrue(t.equal(s.model.get_submodule(self.name).weight, before))

    def test_inspection_reads_real_values_and_cannot_mutate(self):
        s = self.student
        original = s.checkpoint_id()
        p = s.tensor_program(f"W=weight('{self.name}')\nemit('values',W[:2,:2])\nemit('top',topk(norm(W,0),2))")
        self.assertEqual(p.output["values"], [[1.,1.],[1.,1.]])
        with self.assertRaises(ValueError): s.tensor_program(f"write('{self.name}',weight('{self.name}'))")
        self.assertEqual(s.checkpoint_id(), original)

    def test_forbidden_programs_and_resource_abuse_do_not_mutate(self):
        s = self.student
        sources = ["import os", "while True: pass", "x=[i for i in range(5)]", "x=open('/etc/passwd')",
            "x=weight('lm_head')", f"x=weight('{self.name}').numpy()", "x='a'*1000000000",
            f"x=weight('{self.name}')/0", "x=__import__('os')", "exec('1+1')"]
        before = s.checkpoint_id()
        for source in sources:
            with self.assertRaises((ValueError, SyntaxError, ZeroDivisionError), msg=source): s.tensor_program(source, writable=True)
            self.assertEqual(s.checkpoint_id(), before)
        from weight_edit_lab.coding.tensor_program import TensorProgram
        kernel = TensorProgram(lambda _: self.torch.ones(3000))
        with self.assertRaises(ValueError): kernel.run("x=weight('x')\ny=outer(x,x)")

    def test_patch_validation_is_atomic_and_never_silently_clips(self):
        s, t = self.student, self.torch
        other = "model.layers.1.mlp.down_proj"
        before = s.model.get_submodule(self.name).weight.clone()
        with self.assertRaises(ValueError):
            s.apply_tensor_values({self.name: before*1.01, other: t.full((4,4), float('nan'))})
        with self.assertRaises(ValueError): s.apply_tensor_values({self.name: before*2})
        s.apply_tensor_values({self.name: before*1.01}, validate_only=True)
        self.assertTrue(t.equal(s.model.get_submodule(self.name).weight, before))
        self.assertFalse(s.originals)

    def test_controls_match_requested_norm_and_move_location(self):
        from weight_edit_lab.coding.direct_controls import matched_controls
        s, t = self.student, self.torch
        candidate = {self.name: s.model.get_submodule(self.name).weight.clone()}
        candidate[self.name][0,0] += .1
        controls, info = matched_controls(s, candidate, 12)
        actual = candidate[self.name]-s.model.get_submodule(self.name).weight
        for label, values in controls.items():
            self.assertEqual(len(values), 1)
            name, value = next(iter(values.items()))
            delta = value-s.model.get_submodule(name).weight
            self.assertAlmostEqual(delta.norm().item(), actual.norm().item(), places=6)
            self.assertEqual(int(t.count_nonzero(delta)), 1)
        self.assertNotEqual(info["wrong_layer_mapping"][self.name], self.name)
        self.assertTrue(t.equal(s.model.get_submodule(self.name).weight, t.ones(4,4)))

    def test_stale_probe_is_not_silently_reused_after_edit(self):
        s, t = self.student, self.torch
        s.probe_cache = {"probe-1": {"checkpoint": s.checkpoint_id(), "values": {self.name: {"input": t.ones(4)}}}}
        self.assertEqual(s.tensor_program(f"emit('a',activation('probe-1','{self.name}','input'))").output['a'], [1.]*4)
        s.apply_tensor_values({self.name: t.ones(4,4)*1.01})
        with self.assertRaises(ValueError): s.tensor_program(f"emit('a',activation('probe-1','{self.name}','input'))")

    def test_real_exchange_binds_requests_and_returns_authored_code(self):
        from weight_edit_lab.coding.direct_editor import DirectFileEditor
        from weight_edit_lab.runner import atomic_json
        with tempfile.TemporaryDirectory() as directory:
            bridge = DirectFileEditor(Path(directory)/'exchange', self.student, timeout=3)
            errors=[]
            def controller():
                try:
                    for i in range(2):
                        path=bridge.directory/f'request-{i}.json'
                        until=time.monotonic()+3
                        while not path.exists() and time.monotonic()<until: time.sleep(.01)
                        req=json.loads(path.read_text())
                        self.assertNotIn('final',req['observation'])
                        if i==0:
                            response={'request_id':req['request_id'],'kind':'inspect','code':f"emit('value',weight('{self.name}')[0,0])"}
                        else:
                            self.assertEqual(req['previous_result']['output']['value'],1.)
                            response=copy.deepcopy(req['responses']['patch'])
                            response.update(request_id=req['request_id'],code=f"write('{self.name}',weight('{self.name}')*1.01)")
                        atomic_json(bridge.directory/f'response-{i}.json',response)
                except BaseException as exc: errors.append(exc)
            thread=threading.Thread(target=controller);thread.start()
            decision=bridge({'support':[], 'development':{}, 'final':{'secret':True}})
            thread.join(4)
            self.assertFalse(errors)
            self.assertEqual(decision['action']['kind'],'tensor_program')
            self.assertEqual(bridge.inspections,1)
            self.assertEqual(bridge.patch_attempts,1)
            self.assertFalse(self.student.originals)

    def test_trial_preserves_raw_candidates_and_closes_author_before_final_tests(self):
        from examples import direct_trial
        from weight_edit_lab.coding.backend import Student
        from tests.test_editing_benchmark import evaluation
        s = self.student
        s.batch_size, s.max_new_tokens = 4, 512
        lifecycle = []
        def evaluate(problems):
            if problems[0].name.startswith('hard-test'):
                self.assertIn('author closed', lifecycle)
            # A deliberately harmful candidate tests that its raw effects survive rollback.
            changed = s.checkpoint_id() != start_id
            result = evaluation([not changed]*4, 'x')
            for row, problem in zip(result['rows'], problems):
                row.update(name=problem.name, instruction=problem.instruction)
            return result
        s.evaluate = evaluate
        s.inspect = lambda: []
        start_id = s.checkpoint_id()
        real_edit = s.edit
        def edit(action, support, rehearsal, seed=0):
            if action['kind'] == 'lora':
                result = s.apply_tensor_values({self.name: s.model.get_submodule(self.name).weight*1.02})
                result['steps'] = 8
                return result
            return real_edit(action, support, rehearsal, seed)
        s.edit = edit
        code = f"write('{self.name}',weight('{self.name}')*1.01)"
        class Editor:
            trace = []
            def __init__(self,*args,**kwargs): pass
            def __call__(self,observation): return {'action':{'kind':'tensor_program','code':code}}
            def costs(self): return {}
            def close(self,status): lifecycle.append('author closed')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model.json';model.write_text(json.dumps(s.manifest))
            out=root/'trial'
            with patch.object(direct_trial,'Student',return_value=s), patch.object(direct_trial,'DirectFileEditor',Editor), patch('sys.argv',['direct_trial','--model',str(model),'--out',str(out),'--device','cpu']):
                direct_trial.main()
            journal=json.loads((out/'live.json').read_text())
            self.assertEqual(journal['status'],'completed')
            events=journal['events']
            self.assertLess(max(i for i,e in enumerate(events) if e['kind']=='completed'),min(i for i,e in enumerate(events) if e['kind']=='terminal_started'))
            record=json.loads((out/'base-astra_direct-terminal.json').read_text())
            self.assertFalse(record['finalization']['accepted'])
            self.assertEqual(record['candidate_final']['coding']['accuracy'],0.)
            self.assertEqual(record['final']['coding']['accuracy'],1.)
            self.assertEqual(len([e for e in events if e['kind']=='terminal_completed']),5)

    def test_invalid_probe_cost_is_a_request_not_a_forward_pass(self):
        from weight_edit_lab.coding.direct_editor import DirectFileEditor, LANGUAGE
        from weight_edit_lab.runner import atomic_json
        self.assertIn('one or two distinct modules', LANGUAGE)
        with tempfile.TemporaryDirectory() as directory:
            bridge=DirectFileEditor(Path(directory)/'exchange',self.student,timeout=3)
            errors=[]
            def controller():
                try:
                    for i in range(2):
                        path=bridge.directory/f'request-{i}.json';until=time.monotonic()+3
                        while not path.exists() and time.monotonic()<until:time.sleep(.01)
                        req=json.loads(path.read_text())
                        if i==0:
                            response={'request_id':req['request_id'],'kind':'probe','prompt':'test',
                                      'modules':[f'model.layers.{j}.mlp.down_proj' for j in range(3)]}
                        else:
                            self.assertEqual(req['remaining']['probe_requests'],3)
                            self.assertEqual(req['previous_result']['error'],'ValueError')
                            response={'request_id':req['request_id'],'kind':'stop','rationale':'invalid probe test'}
                        atomic_json(bridge.directory/f'response-{i}.json',response)
                except BaseException as exc: errors.append(exc)
            thread=threading.Thread(target=controller);thread.start();bridge({});thread.join(4)
            self.assertFalse(errors)
            self.assertEqual(bridge.costs()['probe_requests'],1)
            self.assertEqual(bridge.costs()['probe_forward_passes'],0)


if __name__ == '__main__':
    unittest.main()
