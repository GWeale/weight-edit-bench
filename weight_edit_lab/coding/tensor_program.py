"""An interpreted tensor-expression language: no Python exec, imports or fitting.

Programs may assign variables, slice tensors, do bounded algebra, emit small
results and stage complete replacement tensors. The caller owns model mutation.
"""

import ast
import math
import operator

import torch


MAX_ELEMENTS = 8_000_000
MAX_WORK = 160_000_000


class TensorProgram:
    def __init__(self, read_weight, *, writable=False, read_activation=None):
        self.read_weight, self.read_activation = read_weight, read_activation
        self.writable = writable
        self.variables, self.weights, self.writes, self.output = {}, {}, {}, {}
        self.work = 0

    def charge(self, amount):
        self.work += int(amount)
        if self.work > MAX_WORK:
            raise ValueError("tensor computation budget exceeded")

    def checked(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() > MAX_ELEMENTS or value.ndim > 2:
                raise ValueError("tensor size limit exceeded")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite tensor result")
            self.charge(value.numel())
        elif type(value) in (int, float):
            if not math.isfinite(value) or abs(value) > 1e12:
                raise ValueError("numeric result exceeds limit")
        return value

    def run(self, source):
        if not isinstance(source, str) or not 1 <= len(source) <= 12000:
            raise ValueError("program must contain 1..12000 characters")
        tree = ast.parse(source, mode="exec")
        if len(tree.body) > 64 or sum(1 for _ in ast.walk(tree)) > 1500:
            raise ValueError("program syntax budget exceeded")
        with torch.no_grad():
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                    name = statement.targets[0].id
                    if name.startswith("_"):
                        raise ValueError("private names are unavailable")
                    self.variables[name] = self.expression(statement.value)
                elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    self.expression(statement.value)
                else:
                    raise ValueError("only variable assignments and function calls are allowed")
                retained = sum(x.numel() for x in self.variables.values() if isinstance(x, torch.Tensor))
                if retained > 32_000_000:
                    raise ValueError("retained tensor budget exceeded")
        return self

    def expression(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
            return self.checked(node.value)
        if isinstance(node, ast.Name) and node.id in self.variables:
            return self.variables[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > 2048:
                raise ValueError("literal list too long")
            values = [self.expression(x) for x in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Slice):
            values = [self.expression(x) if x is not None else None for x in (node.lower, node.upper, node.step)]
            if any(x is not None and type(x) is not int for x in values):
                raise ValueError("integer slice required")
            return slice(*values)
        if isinstance(node, ast.Subscript):
            tensor = self.expression(node.value)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("only tensors may be indexed")
            return self.checked(tensor[self.expression(node.slice)])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.numeric(self.expression(node.operand))
            return self.checked(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.MatMult):
            a, b = self.numeric(self.expression(node.left)), self.numeric(self.expression(node.right))
            if isinstance(node.op, ast.MatMult):
                if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not a.ndim or not b.ndim:
                    raise ValueError("matrix multiplication requires tensors")
                inner = a.shape[-1]
                if inner != b.shape[0]:
                    raise ValueError("matrix dimensions differ")
                rows = a.shape[0] if a.ndim == 2 else 1
                columns = b.shape[1] if b.ndim == 2 else 1
                self.charge(rows * inner * columns)
                if rows * columns > MAX_ELEMENTS:
                    raise ValueError("matrix result too large")
            else:
                shape = torch.broadcast_shapes(a.shape if isinstance(a, torch.Tensor) else (), b.shape if isinstance(b, torch.Tensor) else ())
                if math.prod(shape) > MAX_ELEMENTS:
                    raise ValueError("broadcast result too large")
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Div: operator.truediv, ast.MatMult: operator.matmul}
            return self.checked(operations[type(node.op)](a, b))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [self.expression(x) for x in node.args]
            return self.checked(self.call(node.func.id, args))
        raise ValueError(f"unsupported tensor expression: {type(node).__name__}")

    @staticmethod
    def numeric(value):
        if type(value) not in (int, float) and not isinstance(value, torch.Tensor):
            raise ValueError("arithmetic requires numbers or tensors")
        return value

    def call(self, name, args):
        if name == "weight" and len(args) == 1 and isinstance(args[0], str):
            key = args[0]
            if key not in self.weights:
                if len(self.weights) >= 4:
                    raise ValueError("at most four tensors may be read per program")
                self.weights[key] = self.checked(self.read_weight(key).detach().cpu().float().clone())
            return self.weights[key]
        if name == "activation" and len(args) == 3 and self.read_activation is not None:
            return self.read_activation(*args).detach().cpu().float().clone()
        if name == "write" and len(args) == 2:
            if not self.writable:
                raise ValueError("inspection cannot write weights")
            key, tensor = args
            original = self.call("weight", [key])
            if key in self.writes or not isinstance(tensor, torch.Tensor) or tensor.shape != original.shape:
                raise ValueError("write requires one complete replacement per tensor, with identical shape")
            self.writes[key] = tensor.detach().float().clone()
            return 0
        if name == "emit" and len(args) == 2:
            key, value = args
            if not isinstance(key, str) or not 1 <= len(key) <= 80 or key in self.output or len(self.output) >= 12:
                raise ValueError("emit requires a new short label")
            self.output[key] = self.serialize(value)
            import json
            if len(json.dumps(self.output, allow_nan=False)) > 40000:
                raise ValueError("inspection output too large")
            return 0
        if name == "array" and len(args) == 1:
            return torch.tensor(args[0], dtype=torch.float32)
        if name == "shape" and len(args) == 1 and isinstance(args[0], torch.Tensor):
            return list(args[0].shape)
        if name == "transpose" and len(args) == 1 and isinstance(args[0], torch.Tensor):
            return args[0].T
        if name == "outer" and len(args) == 2 and all(isinstance(x, torch.Tensor) and x.ndim == 1 for x in args):
            if args[0].numel()*args[1].numel() > MAX_ELEMENTS:
                raise ValueError("outer product too large")
            return torch.outer(*args)
        if name == "topk" and len(args) == 2:
            tensor, count = args
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or type(count) is not int or not 1 <= count <= 128:
                raise ValueError("topk requires a vector and count in 1..128")
            return torch.topk(tensor, min(count, tensor.numel())).indices
        if name in ("sum", "mean", "std", "norm") and len(args) in (1, 2) and isinstance(args[0], torch.Tensor):
            tensor = args[0]
            axis = args[1] if len(args) == 2 else None
            if axis is not None and (type(axis) is not int or not -tensor.ndim <= axis < tensor.ndim):
                raise ValueError("invalid reduction axis")
            if name == "norm":
                return torch.linalg.vector_norm(tensor, dim=axis)
            if name == "std":
                return torch.std(tensor, dim=axis, correction=0)
            return getattr(torch, name)(tensor, dim=axis)
        if name in ("abs", "sqrt") and len(args) == 1:
            value = torch.as_tensor(self.numeric(args[0]))
            return torch.abs(value) if name == "abs" else torch.sqrt(value)
        if name == "unit" and len(args) == 1 and isinstance(args[0], torch.Tensor):
            return args[0] / torch.linalg.vector_norm(args[0]).clamp_min(1e-12)
        if name == "columns" and len(args) == 3:
            tensor, indices, values = args
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or not isinstance(values, torch.Tensor):
                raise ValueError("columns requires a matrix, indices and replacement columns")
            indices = torch.as_tensor(indices)
            if indices.dtype != torch.int64 or indices.ndim != 1 or not 1 <= indices.numel() <= 128:
                raise ValueError("choose 1..128 integer columns")
            if len(set(indices.tolist())) != indices.numel() or bool((indices < 0).any()) or bool((indices >= tensor.shape[1]).any()):
                raise ValueError("columns must be distinct and in range")
            if tuple(values.shape) != (tensor.shape[0], indices.numel()):
                raise ValueError("replacement column shape differs")
            result = tensor.clone()
            result[:, indices] = values
            return result
        raise ValueError(f"unknown function or invalid arguments: {name}")

    def serialize(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() > 1024:
                raise ValueError("emit at most 1024 values per label; slice or summarize first")
            return value.tolist()
        if type(value) in (int, float, str):
            return value
        if isinstance(value, (tuple, list)) and len(value) <= 1024:
            return [self.serialize(x) for x in value]
        raise ValueError("unsupported emitted value")
