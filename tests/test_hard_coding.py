import itertools
import importlib.util
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from weight_edit_lab.coding.calibration import calibration_groups, failure_category
from weight_edit_lab.coding.environment import CodingConfig, CodingEditEnv, episode_problems
from weight_edit_lab.coding.hard_tasks import (TRAIN_FAMILIES, EVAL_FAMILIES, ANCHORS,
    make_problem, anchor_problem, specification, oracle)
from weight_edit_lab.coding.interpreter import Interpreter, grade, extract_prefilled_function
from tests.test_coding import FakeStudent


class HardTaskTests(unittest.TestCase):
    def test_reference_solutions_agree_with_independent_oracles(self):
        for family in TRAIN_FAMILIES+EVAL_FAMILIES:
            for split in ("support","score","test"):
                for seed in (0,1):
                    p = make_problem(family,split,seed)
                    self.assertEqual(len({tuple(xs) for xs,_ in p.cases}),64)
                    self.assertTrue(grade(p.solution,p.cases,p.instruction_budget)["correct"],p.name)
        for family in ANCHORS:
            p=anchor_problem(family,"rehearsal",0)
            self.assertTrue(grade(p.solution,p.cases,p.instruction_budget)["correct"],p.name)

    def test_boundary_fixtures_and_exhaustive_delete_dp(self):
        fixtures=[("subarray_target",[2,-2,2],2,3),
                  ("longest_bounded_sum",[5,-10,5],0,3),
                  ("window_mode",[2,1,2,1],2,[1,1,1]),
                  ("clamp_last",[9,1,3,-8,-2],2,[1,2,-2]),
                  ("next_margin",[3,4,8,1],2,[2,1,0,0]),
                  ("disjoint_pairs",[2,2,2,2,2],4,2),
                  ("circular_run",[4,4,0,4],2,3),
                  ("coin_change",[0,-2,3,5],7,-1),
                  ("lis_gap",[1,5,2,8],3,3),
                  ("balanced_signs",[-1,0,1,-2,0,3],3,6),
                  ("clipped_water",[4,0,4],3,3)]
        for family,xs,p,expected in fixtures:
            self.assertEqual(oracle(family,xs,p),expected,family)
        for penalty in (2,5):
            _,body=specification("max_sum_delete",penalty)
            candidate=Interpreter("def solve(xs):\n"+body,budget=200000)
            for n in range(6):
                for xs in itertools.product((-3,0,3),repeat=n):
                    self.assertEqual(candidate.run(xs),oracle("max_sum_delete",list(xs),penalty))

    def test_family_holdout_and_terminal_isolation(self):
        train = CodingConfig(task_suite="algorithmic")
        evaluation = CodingConfig(task_suite="algorithmic",family_partition="evaluation")
        for seed in range(8):
            support=episode_problems(train,"support",seed)
            score=episode_problems(train,"score",seed)
            test=episode_problems(train,"test",seed)
            held=episode_problems(evaluation,"support",seed)
            self.assertEqual([p.family for p in support],[p.family for p in score])
            self.assertEqual([p.family for p in support],[p.family for p in test])
            self.assertFalse({p.instruction for p in support}&{p.instruction for p in score})
            self.assertFalse({p.family for p in support}&{p.family for p in held})
        student=FakeStudent()
        env=CodingEditEnv(student,train)
        env.reset(seed=0)
        self.assertFalse(any("hard-test-" in name or "retention_test" in name for name in student.seen))
        with self.assertRaises(RuntimeError):env.final_evaluation()
        env.close_episode()
        env.final_evaluation()
        self.assertTrue(any("hard-test-" in name for name in student.seen))

    def test_editor_receives_only_the_evaluated_function(self):
        student=FakeStudent()
        evaluate=student.evaluate
        def with_suffix(ps):
            r=evaluate(ps)
            for row in r["rows"]:
                row["code"]="def solve(xs): return 0\nprint('unrelated example')"
                row["evaluated_code"]="def solve(xs): return 0"
            return r
        student.evaluate=with_suffix
        obs,_=CodingEditEnv(student,CodingConfig(task_suite="algorithmic")).reset(seed=0)
        self.assertTrue(all("print" not in p["student_code"] for p in obs["support"]))

    def test_calibration_never_queries_reserved_families(self):
        with patch("weight_edit_lab.coding.calibration.make_problem",wraps=make_problem) as generate:
            groups=calibration_groups((0,1))
        self.assertEqual(len(groups["bridge"]),8)
        self.assertEqual(len(groups["stress"]),16)
        self.assertTrue(all(call.args[0] in TRAIN_FAMILIES and call.args[1]=="score" for call in generate.call_args_list))

    def test_constant_shortcuts_fail_every_hard_family(self):
        for family in TRAIN_FAMILIES+EVAL_FAMILIES:
            p=make_problem(family,"score",0)
            for code in ("return 0","return []","return len(xs)"):
                self.assertFalse(grade("def solve(xs):\n    "+code,p.cases,p.instruction_budget)["correct"],family)


class InterpreterExtensionTests(unittest.TestCase):
    def test_extracts_completed_function_without_running_or_reading_suffix(self):
        source='def solve(xs):\n    """Line one\nno indentation inside docstring\n    """\n    return sum(xs)\n'
        for suffix in ("print([1,2,3", "```\nExplanation", "open('/nonexistent', 'w')"):
            extracted=extract_prefilled_function(source+suffix)
            self.assertEqual(extracted,source)
            self.assertTrue(grade(extracted,(([2,3],5),))["correct"])
        broken="def solve(xs):\n    return (\n"
        self.assertEqual(extract_prefilled_function(broken),broken)
        self.assertFalse(grade(broken,(([],0),))["correct"])

    def test_python_container_and_name_semantics(self):
        self.assertFalse(grade("def solve(xs):\n    return range(3)",(([],[0,1,2]),))["correct"])
        mutation="def solve(xs):\n    d={1:2,3:4}\n    for k in d.keys():\n        d[5]=6\n    return 0"
        self.assertEqual(grade(mutation,(([],0),))["error"],"RuntimeError")
        self.assertEqual(grade("def solve(xs):\n    return undefined",(([],0),))["error"],"NameError")
        self.assertEqual(grade("def solve(xs):\n    sum=0\n    return sum(xs)",(([],0),))["error"],"TypeError")

    def test_counts_tuple_keys_key_functions_and_stack(self):
        code="""def solve(xs):
    counts = {x: xs.count(x) for x in set(xs)}
    keys = sorted(counts, key=lambda x: (-counts[x], x))
    stack = []
    for x in keys:
        stack.append(x)
    if stack:
        stack.pop()
    data = {(1, 2): len(stack)}
    return data.get((1, 2), 0)
"""
        self.assertTrue(grade(code,(([2,1,2,3],2),([],0)))["correct"])
        self.assertTrue(grade("def solve(xs):\n    return min(xs, default=0)",(([],0),([3,-1],-1)))["correct"])
        self.assertTrue(grade("def solve(xs):\n    return list(dict.fromkeys(xs))",(([2,1,2],[2,1]),))["correct"])
        self.assertTrue(grade("def solve(xs):\n    best = float('inf')\n    return min(best, 10 ** 9)",(([],10**9),))["correct"])

    def test_new_paths_cannot_escape_or_allocate_without_bounds(self):
        programs=["return sorted(xs, key=lambda x: open('/etc/passwd'))",
            "return 10 ** 1000000000", "a={}\n    a[0]=a\n    return a",
            "return dict.fromkeys(range(1000000000))", "a=[]\n    a.extend([a])\n    return a",
            "return sorted(xs, key=lambda x: x.__class__)","return float('nan')"]
        for code in programs:
            self.assertFalse(grade("def solve(xs):\n    "+code,(([1],1),))["correct"],code)

    def test_failures_are_distinguished_from_low_coding_accuracy(self):
        for source,category in (("def solve(xs):\n    return xs[99]","runtime_error"),
            ("def solve(xs):\n    return xs.copy()","unsupported_subset"),
            ("def solve(xs):\n    return 5","wrong_answer"),
            ("def solve(xs):\n    while True:\n        pass","resource_limit"),
            ("def solve(","syntax")):
            row={"grade":grade(source,(([],0),)),"truncated":False}
            self.assertEqual(failure_category(row),category)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),"optional coding runtime")
class CalibrationCacheTests(unittest.TestCase):
    def test_cached_code_regrades_without_model_and_rejects_changed_protocol(self):
        from weight_edit_lab.coding.calibration import calibrate
        class Model:
            device="cpu"
            dtype="float32"
            def __init__(self,*args,**kwargs):pass
            def evaluate(self,ps):
                return {"rows":[{"name":p.name,"instruction":p.instruction,"code":p.solution,
                    "grade":grade(p.solution,p.cases,p.instruction_budget),"tokens":40,"truncated":False} for p in ps]}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest={"model_id":"test/model","revision":"pinned"}
            with patch("weight_edit_lab.coding.backend.Student",Model), patch("builtins.print"):
                original=calibrate(manifest,root/"first",seeds=(0,))
            with patch("weight_edit_lab.coding.backend.Student",side_effect=AssertionError("must not load a model")), patch("builtins.print"):
                replay=calibrate(manifest,root/"second",seeds=(0,),reuse_generations=root/"first")
                self.assertEqual(original["cohorts"],replay["cohorts"])
                self.assertTrue(replay["reused_generations"])
                with self.assertRaisesRegex(ValueError,"protocol mismatch"):
                    calibrate(manifest,root/"bad",seeds=(1,),reuse_generations=root/"first")


if __name__ == "__main__":unittest.main()
