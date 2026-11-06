"""A bounded interpreter for a pure Python subset. Never executes Python source with exec/eval."""

import ast
import math
import operator
import re
import io
import tokenize

DATA_CONTAINERS = (list, tuple, dict, set, range, type({}.keys()), type({}.values()), type({}.items()))


class RejectedProgram(ValueError):
    pass


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class LoopSignal(Exception):
    def __init__(self, kind):
        self.kind = kind


def extract_function(text: str) -> str:
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    source = fences[0] if fences else text
    return source.strip()


def extract_prefilled_function(source: str) -> str:
    """Keep the completed, fixed-signature function; never run trailing examples.

    Tokenization handles dedented lines inside docstrings correctly. A capped
    example/Markdown suffix cannot invalidate a complete preceding function.
    Incomplete function bodies are left intact and fail normal syntax checks.
    """
    if not source.startswith("def solve(xs):\n"):
        raise ValueError("missing fixed function signature")
    depth = 0
    entered = False
    lines = source.splitlines(keepends=True)
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.INDENT:
                depth += 1
                entered = True
            elif token.type == tokenize.DEDENT:
                depth -= 1
                if entered and depth == 0:
                    return "".join(lines[:token.start[0]-1]).rstrip() + "\n"
    except (tokenize.TokenError,IndentationError):
        pass
    return source


class Interpreter:
    def __init__(self, source: str, budget: int = 20000):
        if len(source) > 12000:
            raise RejectedProgram("program too large")
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError) as exc:
            raise RejectedProgram("invalid syntax") from exc
        if sum(1 for _ in ast.walk(tree)) > 2000:
            raise RejectedProgram("program too complex")
        body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                                           and isinstance(n.value.value, str))]
        if len(body) != 1 or not isinstance(body[0], ast.FunctionDef) or body[0].name != "solve":
            raise RejectedProgram("exactly one solve function required")
        fn = body[0]
        if (fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs
                or fn.args.defaults or fn.args.posonlyargs or len(fn.args.args) != 1):
            raise RejectedProgram("unsupported function signature")
        self.body, self.argument, self.budget = fn.body, fn.args.args[0].arg, budget

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise RejectedProgram("instruction budget exceeded")

    def bounded(self, v):
        # Bound the whole value, including aliases/cycles introduced by append or
        # index assignment. Builtin comparisons cannot bypass the work budget.
        def visit(value, ancestors, depth):
            self.tick()
            if depth > 8:
                raise RejectedProgram("nesting limit exceeded")
            if type(value) is int and abs(value) > 10**12:
                raise RejectedProgram("numeric limit exceeded")
            # Infinity is a useful DP sentinel, not an allocation-size integer.
            if type(value) is float and (math.isnan(value) or (math.isfinite(value) and abs(value) > 10**12)):
                raise RejectedProgram("numeric limit exceeded")
            if isinstance(value, DATA_CONTAINERS+(str,)) and len(value) > 512:
                raise RejectedProgram("container limit exceeded")
            if isinstance(value, DATA_CONTAINERS):
                if id(value) in ancestors:
                    raise RejectedProgram("cyclic values forbidden")
                for item in value:
                    visit(item, ancestors | {id(value)}, depth + 1)
                    if isinstance(value, dict):
                        visit(value[item], ancestors | {id(value)}, depth + 1)
            elif type(value) not in (int, float, bool, str, type(None)):
                raise RejectedProgram("unsupported value")
        visit(v, set(), 0)
        return v

    def run(self, xs):
        self.remaining = self.budget
        env = {self.argument: self.bounded(list(xs))}
        try:
            self.block(self.body, env)
        except Returned as result:
            return self.bounded(result.value)
        return None

    def assign(self, node, value, env):
        self.tick()
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise RejectedProgram("private names forbidden")
            env[node.id] = self.bounded(value)
        elif isinstance(node, (ast.Tuple, ast.List)) and isinstance(value, (list, tuple)) and len(node.elts) == len(value):
            for target, item in zip(node.elts, value):
                self.assign(target, item, env)
        elif isinstance(node, ast.Subscript):
            container = self.expr(node.value, env)
            index = self.expr(node.slice, env)
            if not isinstance(container, (list, dict)) or (isinstance(container, list) and type(index) is not int):
                raise RejectedProgram("only list/dict index assignment supported")
            container[index] = self.bounded(value)
            self.bounded(container)
        else:
            raise RejectedProgram("unsupported assignment")

    def block(self, statements, env):
        for node in statements:
            self.tick()
            if isinstance(node, ast.Return):
                raise Returned(self.expr(node.value, env) if node.value else None)
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, env)
                for target in node.targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AugAssign):
                value = self.binary(node.op, self.expr(node.target, env), self.expr(node.value, env))
                self.assign(node.target, value, env)
            elif isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, (ast.For, ast.While)):
                iterable = iter(self.expr(node.iter, env)) if isinstance(node, ast.For) else None
                broken = False
                while True:
                    self.tick()
                    if iterable is not None:
                        try:
                            item = next(iterable)
                        except StopIteration:
                            break
                        self.assign(node.target, item, env)
                    elif not self.expr(node.test, env):
                        break
                    try:
                        self.block(node.body, env)
                    except LoopSignal as signal:
                        if signal.kind == "break":
                            broken = True
                            break
                if not broken:
                    self.block(node.orelse, env)
            elif isinstance(node, (ast.Break, ast.Continue)):
                raise LoopSignal("break" if isinstance(node, ast.Break) else "continue")
            elif not isinstance(node, ast.Pass):
                raise RejectedProgram(f"unsupported statement: {type(node).__name__}")

    def binary(self, op, a, b):
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                      ast.Pow: operator.pow}
        if type(op) not in operations:
            raise RejectedProgram("unsupported operator")
        if isinstance(op, ast.Mod) and isinstance(a, str):
            raise RejectedProgram("string formatting forbidden")
        if isinstance(op, ast.Pow) and (type(a) not in (int,float) or type(b) is not int or abs(b) > 12):
            raise RejectedProgram("power exceeds numeric budget")
        if isinstance(op, ast.Mult):
            for container, multiplier in ((a, b), (b, a)):
                if isinstance(container, (list, tuple, str)) and (type(multiplier) is not int or len(container)*max(0,multiplier) > 512):
                    raise RejectedProgram("repetition too large")
        return self.bounded(operations[type(op)](a, b))

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, bool, str, type(None)):
                raise RejectedProgram("unsupported constant")
            return self.bounded(node.value)
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise RejectedProgram("private names forbidden")
            if node.id not in env:
                raise NameError(node.id)
            return self.bounded(env[node.id])
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            constructor = {ast.List:list, ast.Tuple:tuple, ast.Set:set}[type(node)]
            return self.bounded(constructor(self.expr(n, env) for n in node.elts))
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise RejectedProgram("dictionary expansion unsupported")
            return self.bounded({self.expr(k,env):self.expr(v,env) for k,v in zip(node.keys,node.values)})
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expr(node.left, env), self.expr(node.right, env))
        if isinstance(node, ast.UnaryOp):
            funcs = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
            if type(node.op) not in funcs:
                raise RejectedProgram("unsupported unary operator")
            return self.bounded(funcs[type(node.op)](self.expr(node.operand, env)))
        if isinstance(node, ast.BoolOp):
            result = self.expr(node.values[0], env)
            for value in node.values[1:]:
                if (isinstance(node.op, ast.And) and not result) or (isinstance(node.op, ast.Or) and result):
                    break
                result = self.expr(value, env)
            return result
        if isinstance(node, ast.Compare):
            funcs = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
                     ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a,b:a in b,
                     ast.NotIn: lambda a,b:a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}
            left = self.expr(node.left, env)
            for op, other in zip(node.ops, node.comparators):
                right = self.expr(other, env)
                if type(op) not in funcs or not funcs[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Subscript):
            value = self.expr(node.value, env)
            if not isinstance(value, (list, tuple, str, dict, range)):
                raise RejectedProgram("unsupported subscript")
            return self.bounded(value[self.expr(node.slice, env)])
        if isinstance(node, ast.Slice):
            return slice(*(self.expr(n, env) if n else None for n in (node.lower, node.upper, node.step)))
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)):
            if len(node.generators) != 1 or node.generators[0].is_async:
                raise RejectedProgram("only one comprehension loop supported")
            gen = node.generators[0]
            items = {} if isinstance(node,ast.DictComp) else set() if isinstance(node,ast.SetComp) else []
            local = dict(env)
            for value in self.expr(gen.iter, env):
                self.assign(gen.target, value, local)
                if all(self.expr(condition, local) for condition in gen.ifs):
                    if isinstance(node,ast.DictComp):
                        items[self.expr(node.key,local)] = self.expr(node.value,local)
                    elif isinstance(node,ast.SetComp):
                        items.add(self.expr(node.elt,local))
                    else:
                        items.append(self.expr(node.elt, local))
                    self.bounded(items)
            return items
        if isinstance(node, ast.Call):
            args = [self.expr(a, env) for a in node.args]
            kwargs = {}
            for keyword in node.keywords:
                if keyword.arg not in ("key", "reverse", "default") or keyword.arg in kwargs:
                    raise RejectedProgram("unsupported keyword argument")
                kwargs[keyword.arg] = self.key_function(keyword.value,env) if keyword.arg == "key" else self.expr(keyword.value,env)
            if isinstance(node.func, ast.Attribute):
                if (isinstance(node.func.value,ast.Name) and node.func.value.id == "dict"
                        and "dict" not in env and node.func.attr == "fromkeys"):
                    if kwargs:
                        raise RejectedProgram("unsupported method keyword")
                    return self.bounded(dict.fromkeys(*args))
                target = self.expr(node.func.value, env)
                allowed = {list: ("append", "pop", "count", "index", "sort", "reverse", "extend", "insert"),
                           dict: ("get", "items", "keys", "values", "pop", "setdefault"),
                           set: ("add", "discard", "remove")}
                if node.func.attr not in allowed.get(type(target), ()):
                    raise RejectedProgram("method not permitted")
                if kwargs and (node.func.attr != "sort" or set(kwargs)-{"key","reverse"}):
                    raise RejectedProgram("unsupported method keyword")
                value = getattr(target,node.func.attr)(*args,**kwargs)
                self.bounded(target)
                return self.bounded(value)
            if not isinstance(node.func, ast.Name):
                raise RejectedProgram("unsupported call")
            name = node.func.id
            if name in env:
                raise TypeError("local value is not callable")
            builtins = {"sum": sum, "len": len, "min": min, "max": max, "abs": abs,
                        "sorted": sorted, "list": list, "any": any, "all": all, "int": int,
                        "float": float, "dict": dict, "set": set, "tuple": tuple}
            allowed_keywords = {"min": {"key","default"}, "max": {"key","default"}, "sorted": {"key","reverse"}}
            if set(kwargs) - allowed_keywords.get(name,set()):
                raise RejectedProgram("unsupported builtin keyword")
            if name == "range":
                value = range(*args)
                if len(value) > 512:
                    raise RejectedProgram("range too large")
                return value
            if name in ("enumerate", "zip", "reversed"):
                fn = {"enumerate": enumerate, "zip": zip, "reversed": reversed}[name]
                return self.bounded(list(fn(*args)))
            if name in builtins:
                return self.bounded(builtins[name](*args,**kwargs))
            if name in ("open","eval","exec","__import__","globals","locals","compile") or name.startswith("__"):
                raise RejectedProgram("call not permitted")
            raise NameError(name)
        raise RejectedProgram(f"unsupported expression: {type(node).__name__}")

    def key_function(self, node, env):
        """Only bounded expression lambdas and whitelisted data lookups are keys."""
        if isinstance(node,ast.Constant) and node.value is None:
            return None
        if isinstance(node,ast.Lambda):
            a = node.args
            if len(a.args) != 1 or a.posonlyargs or a.kwonlyargs or a.defaults or a.vararg or a.kwarg:
                raise RejectedProgram("key lambda must take exactly one argument")
            return lambda value: self.bounded(self.expr(node.body, dict(env, **{a.args[0].arg:value})))
        if isinstance(node,ast.Name) and node.id == "abs":
            return lambda value: self.bounded(abs(value))
        if isinstance(node,ast.Attribute):
            target = self.expr(node.value,env)
            if (type(target) is dict and node.attr == "get") or (type(target) is list and node.attr == "count"):
                return lambda value: self.bounded(getattr(target,node.attr)(value))
        raise RejectedProgram("unsupported key function")


def grade(source: str, cases: tuple, budget: int = 20000) -> dict:
    try:
        interpreter = Interpreter(extract_function(source),budget=budget)
        passed = 0
        for xs, expected in cases:
            actual = interpreter.run(xs)
            passed += int(type(actual) is type(expected) and actual == expected)
        return {"passed": passed, "total": len(cases), "correct": passed == len(cases), "error": None}
    except (ValueError, ArithmeticError, TypeError, IndexError, KeyError, RuntimeError, NameError, LoopSignal) as exc:
        return {"passed": 0, "total": len(cases), "correct": False, "error": type(exc).__name__, "error_detail": str(exc)[:200]}
