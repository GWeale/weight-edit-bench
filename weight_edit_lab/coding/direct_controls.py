"""Predeclared controls for a frozen authored patch; never shown to its author."""

import torch


def matched_controls(student, candidate, seed):
    """Call with the student at the ORIGINAL starting state, after the author ends.

    Match requested delta magnitudes before dtype rounding. Actual applied norms
    and nonzero counts must also be reported; bfloat16 can change either.
    """
    rng = torch.Generator(device="cpu").manual_seed(seed)
    random_values, wrong_values, mapping, norms = {}, {}, {}, {}
    for name, edited in sorted(candidate.items()):
        original = student.model.get_submodule(name).weight.detach().cpu().float()
        delta = edited.float()-original
        if not bool(delta.any()):
            continue
        signs = torch.randint(0, 2, delta.shape, generator=rng).float()*2-1
        random_values[name] = original + delta.abs()*signs
        layer = int(name.split(".")[2])
        target = None
        for offset in range(student.num_layers):
            other = (layer+student.num_layers//2+offset) % student.num_layers
            proposed = name.replace(f"layers.{layer}.", f"layers.{other}.")
            if proposed not in candidate and proposed not in wrong_values:
                weight = student.model.get_submodule(proposed).weight.detach().cpu().float()
                if weight.shape == delta.shape:
                    target = proposed
                    wrong_values[target] = weight+delta
                    break
        if target is None:
            raise ValueError("no unused same-shaped layer for the location control")
        mapping[name] = target
        norms[name] = delta.norm().item()
    return {"random_sign": random_values, "wrong_layer": wrong_values}, {
        "seed": seed, "wrong_layer_mapping": mapping, "source_delta_norms": norms,
        "matching": "same requested delta L2 and changed-entry support before dtype rounding; random signs at original locations versus original deltas in other layers"}
