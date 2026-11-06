"""Local, pinned causal LM and bounded numerical edits (optional torch extra)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .actions import MODULES, validate_action
from .interpreter import grade, extract_prefilled_function


class LowRankLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, seed: int):
        super().__init__()
        self.base = base
        self.scale = 2.0
        # A CPU generator keeps initialization independent of global/MPS RNG.
        rng = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.randn(rank, base.in_features, generator=rng, dtype=torch.float32) * .01
        self.a = nn.Parameter(a.to(base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        base.requires_grad_(False)

    def forward(self, x):
        result = self.base(x)
        update = nn.functional.linear(nn.functional.linear(x.float(), self.a), self.b)
        return result + (update * self.scale).to(result.dtype)

    def delta(self):
        return (self.b @ self.a) * self.scale


def replace_module(model, name, value):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent), leaf, value)


def tensor_digest(tensor):
    # numpy has no bfloat16 dtype; hashing the raw uint8 view preserves bits.
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Student:
    def __init__(self, manifest: dict, device="auto", max_new_tokens=192, batch_size=4):
        if not 16 <= max_new_tokens <= 512 or not 1 <= batch_size <= 8:
            raise ValueError("invalid generation limits")
        if not manifest.get("revision") or not Path(manifest.get("path", "")).is_dir():
            raise ValueError("a downloaded, revision-pinned local manifest is required")
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        # Float16 was numerically invalid on the pilot MPS batched prompts.
        self.dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.manifest = dict(manifest)
        self.tokenizer = AutoTokenizer.from_pretrained(manifest["path"], local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(manifest["path"], local_files_only=True,
            dtype=self.dtype, attn_implementation="eager").to(self.device).eval()
        self.model.requires_grad_(False)
        self.num_layers = self.model.config.num_hidden_layers
        # Fail closed on architectures without the advertised module interface.
        for module in MODULES:
            if not isinstance(self.model.get_submodule(f"model.layers.{self.num_layers-1}.{module}"), nn.Linear):
                raise ValueError("unsupported model architecture")
        self.max_new_tokens, self.batch_size = max_new_tokens, batch_size
        self.originals = {}
        self.generation = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id, use_cache=True)

    def prompt(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True)

    def training_batch(self, problems):
        rows, labels = [], []
        for problem in problems:
            if not problem.solution.startswith(problem.completion_prefix):
                raise ValueError("training solution does not match its fixed function prefix")
            prefix = self.tokenizer.encode(self.prompt(problem.prompt)+problem.completion_prefix, add_special_tokens=False)
            completion = self.tokenizer.encode(problem.solution[len(problem.completion_prefix):], add_special_tokens=False) + [self.tokenizer.eos_token_id]
            if len(prefix) + len(completion) > 512:
                raise ValueError("training sequence exceeds budget")
            rows.append(prefix + completion)
            labels.append([-100] * len(prefix) + completion)
        width = max(map(len, rows))
        ids = [row + [self.tokenizer.pad_token_id] * (width-len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width-len(row)) for row in rows]
        labels = [row + [-100] * (width-len(row)) for row in labels]
        return {k: torch.tensor(v, device=self.device) for k, v in
                (("input_ids", ids), ("attention_mask", mask), ("labels", labels))}

    def evaluate(self, problems):
        self.model.eval()
        started = time.monotonic()
        rows = []
        with torch.inference_mode():
            for start in range(0, len(problems), self.batch_size):
                group = problems[start:start+self.batch_size]
                inputs = self.tokenizer([self.prompt(p.prompt)+p.completion_prefix for p in group], padding=True,
                    return_tensors="pt").to(self.device)
                # Catch broken numeric backends before treating outputs as skill.
                first = self.model(**inputs, use_cache=False).logits[:, -1, :]
                if not bool(torch.isfinite(first).all()):
                    raise RuntimeError("nonfinite inference logits; this is not a valid coding evaluation")
                output = self.model.generate(**inputs, generation_config=self.generation)
                for problem, ids in zip(group, output[:, inputs["input_ids"].shape[1]:]):
                    tokens = ids.tolist()
                    eos = self.tokenizer.eos_token_id
                    used = tokens.index(eos) + 1 if eos in tokens else len(tokens)
                    code = problem.completion_prefix + self.tokenizer.decode(tokens[:used], skip_special_tokens=True)
                    evaluated_code = extract_prefilled_function(code) if problem.completion_prefix else code
                    rows.append({"name": problem.name, "instruction": problem.instruction, "code": code,
                        "grade": grade(evaluated_code, problem.cases, budget=problem.instruction_budget), "tokens": used,
                        "evaluated_code":evaluated_code,
                        "truncated": eos not in tokens and used == self.max_new_tokens})
        return {"accuracy": sum(r["grade"]["correct"] for r in rows)/len(rows),
            "case_accuracy": sum(r["grade"]["passed"] for r in rows)/sum(r["grade"]["total"] for r in rows),
            "rows": rows, "seconds": time.monotonic()-started}

    def inspect(self):
        result = []
        for layer in sorted({self.num_layers//2-1, self.num_layers//2, self.num_layers-2, self.num_layers-1}):
            for module in ("mlp.down_proj", "self_attn.o_proj"):
                name = f"model.layers.{layer}.{module}"
                weight = self.model.get_submodule(name).weight.detach().float()
                result.append({"module": name, "shape": list(weight.shape),
                    "rms": weight.square().mean().sqrt().item(), "norm": weight.norm().item()})
        return result

    def restore(self):
        with torch.no_grad():
            for name, original in self.originals.items():
                self.model.get_submodule(name).weight.copy_(original.to(self.device))
        self.originals.clear()

    def snapshot(self):
        """Copy touched tensors; the immutable pinned base supplies all others."""
        return {name: self.model.get_submodule(name).weight.detach().cpu().clone()
                for name in self.originals}

    def checkpoint_id(self):
        from ..tasks import digest
        changed = {name: tensor_digest(tensor) for name, tensor in self.snapshot().items()
                   if not torch.equal(tensor, self.originals[name])}
        return digest({"model_id": self.manifest["model_id"], "revision": self.manifest["revision"],
                       "dtype": str(self.dtype), "changed_tensors": changed})

    def restore_snapshot(self, tensors):
        self._validate_tensors(tensors)
        self.restore()
        self._remember(tensors)
        with torch.no_grad():
            for name, tensor in tensors.items():
                self.model.get_submodule(name).weight.copy_(tensor.to(self.device))

    def _remember(self, names):
        for name in names:
            if name not in self.originals:
                self.originals[name] = self.model.get_submodule(name).weight.detach().cpu().clone()

    def edit(self, action, support, rehearsal, seed=0):
        action = validate_action(action, self.num_layers)
        started = time.monotonic()
        kind = action["kind"]
        if kind == "tensor_program":
            program = self.tensor_program(action["code"], writable=True)
            result = self.apply_tensor_values(program.writes)
            result.update(program_output=program.output, program_work=program.work,
                          program_sha256=hashlib.sha256(action["code"].encode()).hexdigest())
            return result
        if kind in ("noop", "stop", "rollback"):
            if kind == "rollback":
                self.restore()
            return {"kind": kind, "steps": 0, "delta_norm": 0., "seconds": time.monotonic()-started, "tensors": []}
        layers = action["layers"] if kind == "lora" else [action["layer"]]
        names = [f"model.layers.{layer}.{action['module']}" for layer in layers]
        if kind == "scale_columns" and max(action["columns"]) >= self.model.get_submodule(names[0]).in_features:
            raise ValueError("column out of range")
        self._remember(names)
        before = {name: self.model.get_submodule(name).weight.detach().clone() for name in names}
        wrappers, losses = {}, []
        try:
            if kind == "lora":
                for i, name in enumerate(names):
                    wrappers[name] = LowRankLinear(self.model.get_submodule(name), action["rank"], seed+i)
                    replace_module(self.model, name, wrappers[name])
                parameters = [p for w in wrappers.values() for p in (w.a, w.b)]
                optimizer = torch.optim.AdamW(parameters, lr=action["lr"], weight_decay=0.)
                batch = self.training_batch(support if action["data"] == "support" else rehearsal)
                for _ in range(action["steps"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.model(**batch, use_cache=False).loss
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("nonfinite student training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(loss.item())
                with torch.no_grad():
                    for wrapper in wrappers.values():
                        wrapper.base.weight.add_(wrapper.delta().to(wrapper.base.weight.dtype))
            else:
                with torch.no_grad():
                    weight = self.model.get_submodule(names[0]).weight
                    weight[:, action["columns"]] *= action["factor"]
        except BaseException:
            for name, wrapper in wrappers.items():
                replace_module(self.model, name, wrapper.base)
            with torch.no_grad():
                for name, tensor in before.items():
                    self.model.get_submodule(name).weight.copy_(tensor)
            raise
        finally:
            for name, wrapper in wrappers.items():
                replace_module(self.model, name, wrapper.base)
            self.model.requires_grad_(False)
            self.model.eval()
        records = []
        with torch.no_grad():
            for name in names:
                weight = self.model.get_submodule(name).weight
                original = self.originals[name].to(self.device).float()
                delta = weight.float() - original
                limit = .1 * original.norm()
                norm = delta.norm()
                if not bool(torch.isfinite(norm)):
                    for n in names:
                        self.model.get_submodule(n).weight.copy_(before[n])
                    raise RuntimeError("nonfinite edit")
                clipped = bool(norm > limit)
                if clipped:
                    weight.copy_((original + delta * (limit/norm)).to(weight.dtype))
                actual = weight.float() - before[name].float()
                # Small auditable samples of numerical changes; these do not
                # identify which entries caused a behavioral effect.
                indices = actual.abs().flatten().topk(min(4, actual.numel())).indices.cpu().tolist()
                samples = []
                for index in indices:
                    row, column = divmod(index, weight.shape[1])
                    samples.append({"row": row, "column": column,
                        "before": before[name][row, column].item(), "after": weight[row, column].item(),
                        "delta": actual[row, column].item()})
                records.append({"name": name, "shape": list(weight.shape), "delta_norm": actual.norm().item(),
                    "total_relative_norm": ((weight.float()-original).norm()/original.norm()).item(),
                    "clipped": clipped, "before_sha256": tensor_digest(before[name]), "after_sha256": tensor_digest(weight),
                    "largest_entry_changes": samples})
        return {"kind": kind, "steps": len(losses), "training_loss": losses,
            "delta_norm": sum(r["delta_norm"]**2 for r in records)**.5,
            "seconds": time.monotonic()-started, "tensors": records}

    def tensor_catalog(self):
        return [{"name": f"model.layers.{i}.{m}",
                 "shape": list(self.model.get_submodule(f"model.layers.{i}.{m}").weight.shape)}
                for i in range(self.num_layers) for m in MODULES]

    def tensor_program(self, code, writable=False):
        from .tensor_program import TensorProgram
        allowed = {x["name"] for x in self.tensor_catalog()}
        def read(name):
            if name not in allowed:
                raise ValueError("tensor is outside the weight interface")
            return self.model.get_submodule(name).weight
        def activation(probe_id, name, side):
            entry = getattr(self, "probe_cache", {}).get(probe_id)
            if entry is None or entry["checkpoint"] != self.checkpoint_id():
                raise ValueError("activation probe is missing or belongs to different weights")
            if side not in ("input", "output") or name not in entry["values"]:
                raise ValueError("activation was not recorded")
            return entry["values"][name][side]
        return TensorProgram(read, writable=writable, read_activation=activation).run(code)

    def apply_tensor_values(self, values, validate_only=False):
        """Atomically apply authored values; reject excessive edits, never optimize them."""
        if not 1 <= len(values) <= 4:
            raise ValueError("a direct patch must write one to four tensors")
        allowed = {x["name"] for x in self.tensor_catalog()}
        prepared, before = {}, {}
        started = time.monotonic()
        for name, value in values.items():
            if name not in allowed or not isinstance(value, torch.Tensor):
                raise ValueError("invalid direct tensor")
            weight = self.model.get_submodule(name).weight
            if value.shape != weight.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("invalid direct shape or values")
            before[name] = weight.detach().cpu().clone()
            prepared[name] = value.detach().cpu().to(self.dtype)
            original = self.originals.get(name, before[name]).float()
            relative = (prepared[name].float()-original).norm()/original.norm().clamp_min(1e-12)
            if not bool(torch.isfinite(prepared[name]).all()) or not bool(relative <= .1):
                raise ValueError("direct patch exceeds the 10% pretrained-tensor norm limit")
        if validate_only:
            return {"valid": True, "tensors": len(prepared)}
        self._remember(values)
        records = []
        try:
            with torch.no_grad():
                for name, value in prepared.items():
                    weight = self.model.get_submodule(name).weight
                    weight.copy_(value.to(self.device))
                    delta = value.float()-before[name].float()
                    original = self.originals[name].float()
                    samples = []
                    for index in delta.abs().flatten().topk(min(4, delta.numel())).indices.tolist():
                        row, column = divmod(index, delta.shape[1])
                        samples.append({"row": row, "column": column,
                            "before": before[name][row,column].item(), "after": value[row,column].item(),
                            "delta": delta[row,column].item()})
                    records.append({"name": name, "shape": list(value.shape), "delta_norm": delta.norm().item(),
                        "total_relative_norm": ((value.float()-original).norm()/original.norm().clamp_min(1e-12)).item(),
                        "changed_entries": int(torch.count_nonzero(delta)), "clipped": False,
                        "before_sha256": tensor_digest(before[name]), "after_sha256": tensor_digest(value),
                        "largest_entry_changes": samples})
        except BaseException:
            with torch.no_grad():
                for name, value in before.items():
                    self.model.get_submodule(name).weight.copy_(value.to(self.device))
            raise
        return {"kind": "tensor_program", "steps": 0, "training_loss": [],
                "delta_norm": sum(x["delta_norm"]**2 for x in records)**.5,
                "seconds": time.monotonic()-started, "tensors": records}

    def probe_activations(self, probe_id, prompt, modules):
        """One forward pass; expose the last prompt token's selected activations."""
        allowed = {x["name"] for x in self.tensor_catalog()}
        if not isinstance(prompt, str) or not 1 <= len(prompt) <= 4000:
            raise ValueError("probe text must contain 1..4000 characters")
        if (not isinstance(modules, list) or not 1 <= len(modules) <= 2
                or any(not isinstance(x, str) or x not in allowed for x in modules)
                or len(set(modules)) != len(modules)):
            raise ValueError("select one or two different supported modules")
        inputs = self.tokenizer(self.prompt(prompt), return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError("probe exceeds 512 tokens")
        captured, hooks = {}, []
        def capture(name):
            def hook(module, arguments, output):
                captured[name] = {"input": arguments[0][0,-1].detach().cpu().float().clone(),
                                  "output": output[0,-1].detach().cpu().float().clone()}
            return hook
        try:
            for name in modules:
                hooks.append(self.model.get_submodule(name).register_forward_hook(capture(name)))
            with torch.inference_mode():
                self.diagnostic_forward_passes = getattr(self, "diagnostic_forward_passes", 0)+1
                logits = self.model(**inputs, use_cache=False).logits[0,-1].float().cpu()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("nonfinite probe logits")
        finally:
            for hook in hooks:
                hook.remove()
        if not hasattr(self, "probe_cache"):
            self.probe_cache = {}
        if len(self.probe_cache) >= 8:
            raise ValueError("activation cache budget exceeded")
        self.probe_cache[probe_id] = {"checkpoint": self.checkpoint_id(), "values": captured}
        report = {}
        for name, values in captured.items():
            report[name] = {}
            for side, vector in values.items():
                indices = vector.abs().topk(min(16, vector.numel())).indices.tolist()
                report[name][side] = {"shape": list(vector.shape), "norm": vector.norm().item(),
                    "largest": [{"index": i, "value": vector[i].item()} for i in indices]}
        top = logits.topk(8).indices.tolist()
        return {"probe_id": probe_id, "checkpoint": self.checkpoint_id(), "tokens": inputs["input_ids"].shape[1],
                "position": "last token of chat-formatted prompt; no generated answer", "activations": report,
                "next_tokens": [{"token": self.tokenizer.decode([i]), "logit": logits[i].item()} for i in top]}

    def save_patch(self, path: Path):
        from safetensors.torch import save_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {name: self.model.get_submodule(name).weight.detach().cpu().contiguous()
                   for name in self.originals}
        # Save complete touched tensors so bfloat16 reloading is bit exact.
        save_file(tensors, str(path), metadata={"model_id": self.manifest["model_id"],
                    "revision": self.manifest["revision"], "format": "weight-edit-patch-v1"})
        return {name: tensor_digest(t) for name, t in tensors.items()}

    def load_patch(self, path: Path):
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if (metadata.get("format") != "weight-edit-patch-v1" or metadata.get("revision") != self.manifest["revision"]
                    or metadata.get("model_id") != self.manifest["model_id"]):
                raise ValueError("patch model/revision mismatch")
            tensors = {name: f.get_tensor(name) for name in f.keys()}
        self._validate_tensors(tensors)
        self._remember(tensors)
        with torch.no_grad():
            for name, tensor in tensors.items():
                self.model.get_submodule(name).weight.copy_(tensor.to(self.device))
        return {name: tensor_digest(t) for name, t in tensors.items()}

    def _validate_tensors(self, tensors):
        # Validate every tensor before changing anything.
        allowed = {f"model.layers.{i}.{m}" for i in range(self.num_layers) for m in MODULES}
        for name, tensor in tensors.items():
            if name not in allowed or tensor.shape != self.model.get_submodule(name).weight.shape:
                raise ValueError("invalid patch tensor")
            if tensor.dtype != self.dtype or not bool(torch.isfinite(tensor).all()):
                raise ValueError("invalid patch dtype or values")
