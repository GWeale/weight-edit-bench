"""A deliberately finite, auditable action boundary for the first coding pilot."""

import math


MODULES = ("mlp.down_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.q_proj")


def validate_action(action: dict, num_layers: int) -> dict:
    if not isinstance(action, dict):
        raise ValueError("action must be an object")
    kind = action.get("kind")
    if kind in ("noop", "stop", "rollback"):
        if set(action) != {"kind"}:
            raise ValueError("unexpected action fields")
        return dict(action)
    if kind == "tensor_program":
        if set(action) != {"kind", "code"} or not isinstance(action["code"], str) or not 1 <= len(action["code"]) <= 12000:
            raise ValueError("tensor_program requires 1..12000 characters of code")
        return dict(action)
    if kind == "lora":
        if set(action) != {"kind", "layers", "module", "rank", "steps", "lr", "data"}:
            raise ValueError("lora requires layers/module/rank/steps/lr/data")
        layers = action["layers"]
        if (not isinstance(layers, list) or not 1 <= len(layers) <= 4
                or any(type(x) is not int or not 0 <= x < num_layers for x in layers)
                or len(set(layers)) != len(layers)):
            raise ValueError("choose one to four distinct valid layers")
        if type(action["rank"]) is not int or not 1 <= action["rank"] <= 8:
            raise ValueError("rank must be in 1..8")
        if type(action["steps"]) is not int or not 1 <= action["steps"] <= 24:
            raise ValueError("steps must be in 1..24")
        lr = action["lr"]
        if type(lr) not in (float, int) or not math.isfinite(lr) or not 1e-5 <= lr <= .005:
            raise ValueError("learning rate must be in 0.00001..0.005")
        if action["data"] not in ("support", "rehearsal"):
            raise ValueError("only support and fixed rehearsal data may train the student")
    elif kind == "scale_columns":
        if set(action) != {"kind", "layer", "module", "columns", "factor"}:
            raise ValueError("scale_columns requires layer/module/columns/factor")
        if type(action["layer"]) is not int or not 0 <= action["layer"] < num_layers:
            raise ValueError("invalid layer")
        columns = action["columns"]
        if (not isinstance(columns, list) or not 1 <= len(columns) <= 32
                or any(type(x) is not int or x < 0 for x in columns) or len(set(columns)) != len(columns)):
            raise ValueError("choose one to 32 distinct nonnegative columns")
        factor = action["factor"]
        if type(factor) not in (float, int) or not math.isfinite(factor) or not .8 <= factor <= 1.2:
            raise ValueError("factor must be in 0.8..1.2")
    else:
        raise ValueError("unknown action kind")
    if action["module"] not in MODULES:
        raise ValueError("module is outside the edit interface")
    import copy
    return copy.deepcopy(action)


def starter_catalog(num_layers: int, with_recovery: bool = False) -> list[dict]:
    if num_layers < 4:
        raise ValueError("starter catalog needs at least four layers")
    def lora(layers, module, steps, lr):
        return dict(kind="lora", layers=layers, module=module, steps=steps, lr=lr, rank=4, data="support")
    catalog = [
        {"label": "A", "name": "no_edit", "action": {"kind": "noop"}},
        {"label": "B", "name": "late_mlp_small", "action": lora(list(range(num_layers-2, num_layers)), "mlp.down_proj", 8, .001)},
        {"label": "C", "name": "late_mlp_larger", "action": lora(list(range(num_layers-2, num_layers)), "mlp.down_proj", 16, .003)},
        {"label": "D", "name": "attention_output", "action": lora(list(range(num_layers-4, num_layers)), "self_attn.o_proj", 8, .001)},
        {"label": "E", "name": "earlier_mlp_control", "action": lora([num_layers//2-1, num_layers//2], "mlp.down_proj", 8, .001)},
        {"label": "F", "name": "direct_scale_control", "action": dict(kind="scale_columns", layer=num_layers-1,
             module="mlp.down_proj", columns=list(range(16)), factor=.95)},
    ]
    if with_recovery:
        import copy
        recovery = dict(catalog[1]["action"], steps=4, data="rehearsal")
        catalog.extend([
            {"label": "G", "name": "edit_then_recover", "sequence": [copy.deepcopy(catalog[2]["action"]), recovery]},
            {"label": "H", "name": "recovery_only", "action": copy.deepcopy(recovery)},
        ])
    return catalog


def recipe_actions(candidate):
    import copy
    return copy.deepcopy(candidate["sequence"] if "sequence" in candidate else [candidate["action"]])
