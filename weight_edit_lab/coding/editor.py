"""An actual LM adapter trained with group-relative, clipped categorical policy gradients.

This is a finite-action contextual bandit, not free-form sequence GRPO. Candidate
actions are executed on fresh student weights; deterministic results can be cached.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..tasks import digest
from .backend import LowRankLinear, Student, replace_module, tensor_digest


def editor_prompt(observation, catalog):
    # This whitelist is intentional. No terminal evaluation fields, reward
    # tables, dev reference solutions, or hidden tests are editor context.
    public = {key: observation[key] for key in
              ("support", "development", "weight_statistics", "last_edit", "remaining_actions")}
    public.update({key: observation[key] for key in
                   ("budget", "baseline_development", "action_history") if key in observation})
    return ("You edit a small coding model's weights. Choose an action that improves coding while "
            "preserving existing skills. Support examples may train the student. Development scores "
            "are feedback. Tensor statistics do not establish a causal mechanism.\n"
            + json.dumps(public, separators=(",", ":")) + "\nActions:\n"
            + json.dumps(catalog, separators=(",", ":"))
            + "\nRespond with exactly one action label: " + ", ".join(a["label"] for a in catalog) + ".")


class EditorPolicy:
    def __init__(self, manifest, catalog, device="auto", rank=4, seed=0, exploration=.1):
        if not 1 <= rank <= 8 or not 0 <= exploration < 1:
            raise ValueError("invalid editor rank/exploration")
        self.backend = Student(manifest, device=device)
        self.catalog, self.exploration = catalog, exploration
        self.model = self.backend.model
        self.labels = [a["label"] for a in catalog]
        encoded = [self.backend.tokenizer.encode(label, add_special_tokens=False) for label in self.labels]
        if any(len(ids) != 1 for ids in encoded) or len({ids[0] for ids in encoded}) != len(encoded):
            raise ValueError("catalog labels must each encode as one distinct token")
        self.token_ids = torch.tensor([ids[0] for ids in encoded], device=self.backend.device)
        self.wrappers = {}
        for layer in range(self.backend.num_layers-2, self.backend.num_layers):
            for suffix in ("self_attn.q_proj", "self_attn.v_proj"):
                name = f"model.layers.{layer}.{suffix}"
                wrapper = LowRankLinear(self.model.get_submodule(name), rank, seed+len(self.wrappers))
                replace_module(self.model, name, wrapper)
                self.wrappers[name] = wrapper
        self.parameters = [p for w in self.wrappers.values() for p in (w.a, w.b)]
        self.rank = rank

    def probabilities(self, observation):
        text = self.backend.prompt(editor_prompt(observation, self.catalog))
        inputs = self.backend.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.backend.device)
        if inputs["input_ids"].shape[1] > 4096:
            raise ValueError("editor context exceeds 4096-token pilot budget")
        hidden = self.model.model(**inputs, use_cache=False).last_hidden_state[:, -1, :]
        # Conditional probabilities over the declared single-token action set.
        # Projecting only these rows avoids allocating all vocabulary logits.
        weight = self.model.lm_head.weight.index_select(0, self.token_ids)
        logits = torch.nn.functional.linear(hidden, weight).float().squeeze(0)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("nonfinite editor logits")
        p = torch.softmax(logits, dim=-1)
        return (1-self.exploration)*p + self.exploration/len(self.labels)

    def choose(self, observation):
        with torch.no_grad():
            probabilities = self.probabilities(observation).cpu().tolist()
        selected = max(range(len(probabilities)), key=probabilities.__getitem__)
        return self.catalog[selected], probabilities

    def fingerprints(self):
        return {f"{name}.{letter}": tensor_digest(getattr(w, letter))
                for name, w in self.wrappers.items() for letter in ("a", "b")}

    def save(self, path: Path):
        from safetensors.torch import save_file
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f"{name}.{letter}": getattr(w, letter).detach().cpu().contiguous()
                   for name, w in self.wrappers.items() for letter in ("a", "b")}
        metadata = {"format": "categorical-editor-v1", "model_id": self.backend.manifest["model_id"],
            "revision": self.backend.manifest["revision"], "catalog_sha256": digest(self.catalog),
            "rank": str(self.rank), "exploration": str(self.exploration)}
        save_file(tensors, str(path), metadata=metadata)

    def load(self, path: Path):
        from safetensors import safe_open
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            expected = {"format": "categorical-editor-v1", "model_id": self.backend.manifest["model_id"],
                "revision": self.backend.manifest["revision"], "catalog_sha256": digest(self.catalog),
                "rank": str(self.rank), "exploration": str(self.exploration)}
            if metadata != expected or set(f.keys()) != set(self.fingerprints()):
                raise ValueError("editor adapter metadata/tensor mismatch")
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for name, w in self.wrappers.items():
            for letter in ("a", "b"):
                target, source = getattr(w, letter), tensors[f"{name}.{letter}"]
                if target.shape != source.shape or not bool(torch.isfinite(source).all()):
                    raise ValueError("invalid editor adapter tensor")
        with torch.no_grad():
            for name, w in self.wrappers.items():
                for letter in ("a", "b"):
                    getattr(w, letter).copy_(tensors[f"{name}.{letter}"].to(self.backend.device))


def train_editor(policy, contexts, reward_fn, updates=8, group_size=8, seed=0, lr=.0003,
                 clip=.2, beta=.02, epochs=2, progress=None):
    if not contexts or not 1 <= updates <= 1000 or not 2 <= group_size <= 32 or not 1 <= epochs <= 8:
        raise ValueError("invalid training budget")
    if not 0 < lr <= .005 or not 0 < clip < 1 or beta < 0:
        raise ValueError("invalid RL hyperparameters")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(policy.parameters, lr=lr, weight_decay=0.)
    with torch.no_grad():
        reference = [policy.probabilities(obs).detach() for obs in contexts]
    before = policy.fingerprints()
    history = []
    for update in range(updates):
        index = update % len(contexts)
        observation = contexts[index]
        with torch.no_grad():
            old = policy.probabilities(observation).detach()
            selected = torch.multinomial(old.cpu(), group_size, replacement=True, generator=rng).tolist()
        rewards = [float(reward_fn(index, action_index)) for action_index in selected]
        reward = torch.tensor(rewards, device=old.device, dtype=torch.float32)
        if not bool(torch.isfinite(reward).all()):
            raise ValueError("nonfinite rewards")
        advantages = (reward-reward.mean()) / (reward.std(unbiased=False)+1e-8)
        indices = torch.tensor(selected, device=old.device)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            p = policy.probabilities(observation)
            ratio = p[indices]/old[indices]
            surrogate = torch.minimum(ratio*advantages, ratio.clamp(1-clip, 1+clip)*advantages).mean()
            kl = (p*(p.log()-reference[index].log())).sum()
            loss = -surrogate + beta*kl
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite policy loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters, 1., error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            after = policy.probabilities(observation).detach().cpu().tolist()
        row = {"update": update, "context_index": index, "actions": selected, "rewards": rewards,
            "mean_reward": reward.mean().item(), "advantage_std": advantages.std(unbiased=False).item(),
            "loss": loss.item(), "kl_to_reference": kl.item(), "probabilities_before": old.cpu().tolist(),
            "probabilities_after": after}
        history.append(row)
        if progress:
            progress(row)
    after = policy.fingerprints()
    with torch.no_grad():
        final = [policy.probabilities(obs).detach().cpu().tolist() for obs in contexts]
    return {"algorithm": "finite-action group-relative clipped policy gradient", "updates": updates,
        "group_size": group_size, "epochs_per_group": epochs, "lr": lr, "clip": clip, "beta": beta,
        "seed": seed, "reference_probabilities": [r.cpu().tolist() for r in reference],
        "final_probabilities": final, "history": history, "weights_before": before, "weights_after": after,
        "changed_adapter_tensors": sum(before[k] != after[k] for k in before)}
