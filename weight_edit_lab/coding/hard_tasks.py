"""Algorithmic development tasks and separately reserved editor-evaluation families.

Expected outputs come from trusted Python oracles, never model-generated code.
Canonical training solutions use different algorithms where practical and are
checked against the oracles. These are synthetic tasks, not contamination-proof.
"""

from collections import Counter

from ..tasks import rng_for
from .tasks import Problem


PROTOCOL = "algorithmic-coding-v2"
BRIDGE_FAMILIES = ("window_sums", "first_prefix", "positive_runs", "filter_unique")
STRESS_FAMILIES = ("subarray_target", "longest_bounded_sum", "max_sum_delete",
                  "window_mode", "clamp_last", "next_margin", "disjoint_pairs", "circular_run")
TRAIN_FAMILIES = BRIDGE_FAMILIES + STRESS_FAMILIES
EVAL_FAMILIES = ("coin_change", "lis_gap", "balanced_signs", "clipped_water")
ANCHORS = ("sum", "reverse", "minimum", "positive_count", "prefix_sum", "absolute", "sorted", "unique")
SPLITS = ("support", "score", "test", "rehearsal", "retention_score", "retention_test")


def inputs_for(split, seed, family, parameter, count=64):
    rng = rng_for(PROTOCOL, "cases", split, seed, family)
    p = parameter
    raw = [[], [0], [1], [-1], [p], [p-1], [p+1], [0]*16, [p]*16, [-p]*16,
        [1,-1]*8, [-1,0,1]*5, list(range(-8,9)), list(range(8,-9,-1)),
        [p,-p,0,p], [p+1,-1], [-20,5,5,-20,5], [3,1,3,1,2,2],
        [p,0,p], [1,2,1,2,1], [0,p,0,0,p,0], [1,p-1,1,p-1], [p-1,p,p+1], [p-1,p,p+1]*5]
    seen, result = set(), []
    while len(result) < count:
        xs = raw.pop(0) if raw else [rng.randint(-12,12) for _ in range(rng.randint(1,16))]
        key = tuple(xs)
        if key not in seen:
            seen.add(key)
            result.append(xs)
    return result


def oracle(family, xs, p):
    n = len(xs)
    slices = lambda: (xs[i:j] for i in range(n) for j in range(i+1,n+1))
    if family == "window_sums":
        return [sum(xs[i:i+p]) for i in range(n-p+1)]
    if family == "first_prefix":
        return next((i for i in range(n) if sum(xs[:i+1]) >= p),-1)
    if family == "positive_runs":
        from itertools import groupby
        return [len(list(group)) for flag,group in groupby(x>p for x in xs) if flag]
    if family == "filter_unique":
        return list(dict.fromkeys(x for x in xs if abs(x)>=p))
    if family == "subarray_target":
        return sum(sum(part) == p for part in slices())
    if family == "longest_bounded_sum":
        return max((len(part) for part in slices() if sum(part) <= p), default=0)
    if family == "max_sum_delete":
        candidates = [sum(part) for part in slices()]
        candidates += [sum(part)-x-p for part in slices() if len(part) > 1 for x in part]
        return max(candidates, default=0)
    if family == "window_mode":
        result = []
        for start in range(n-p+1):
            counts = Counter(xs[start:start+p])
            result.append(min(counts, key=lambda x: (-counts[x], x)))
        return result
    if family == "clamp_last":
        values = [max(-p,min(p,x)) for x in xs]
        return [x for i,x in enumerate(values) if x not in values[i+1:]]
    if family == "next_margin":
        return [next((j-i for j in range(i+1,n) if xs[j] >= x+p),0) for i,x in enumerate(xs)]
    if family == "disjoint_pairs":
        counts = Counter(xs)
        return sum(min(counts[x],counts[p-x]) for x in counts if x < p-x) + (counts[p//2]//2 if p%2 == 0 else 0)
    if family == "circular_run":
        best = 0
        for start in range(n):
            length = 0
            while length < n and xs[(start+length)%n] > p:
                length += 1
            best = max(best,length)
        return best
    if family == "coin_change":
        # Breadth-first reachable totals, independent of the solution's DP.
        coins = {x for x in xs if 0 < x <= p}
        frontier, visited = {0}, {0}
        for depth in range(1,p+1):
            frontier = {v+c for v in frontier for c in coins if v+c <= p} - visited
            if p in frontier:
                return depth
            visited |= frontier
        return -1
    if family == "lis_gap":
        # Exhaustively enumerate valid subsequences; inputs are small.
        tails = [()]
        for x in xs:
            tails += [seq+(x,) for seq in tails if not seq or x-seq[-1] >= p]
        return max(map(len,tails))
    if family == "balanced_signs":
        return max((len(part) for part in slices()
                    if sum(x<0 for x in part) == sum(x==0 for x in part) == sum(x>0 for x in part)), default=0)
    if family == "clipped_water":
        heights = [max(0,min(p,x)) for x in xs]
        # Count enclosed cells one horizontal level at a time.
        total = 0
        for level in range(1,p+1):
            walls = [i for i,h in enumerate(heights) if h >= level]
            if walls:
                total += sum(heights[i] < level for i in range(walls[0],walls[-1]))
        return total
    raise ValueError("unknown family")


def specification(family, p):
    if family == "window_sums":
        return (f"Return the sum of each contiguous window of exactly {p} elements, in window order. Return [] if fewer than {p} elements are available.", f"""    result = []
    for i in range(len(xs) - {p} + 1):
        result.append(sum(xs[i:i + {p}]))
    return result""")
    if family == "first_prefix":
        return (f"Return the smallest zero-based index i for which the sum of xs[0] through xs[i] is at least {p}. Elements may be negative. Return -1 if no such index exists.", f"""    total = 0
    for i, x in enumerate(xs):
        total += x
        if total >= {p}:
            return i
    return -1""")
    if family == "positive_runs":
        return (f"Return the lengths of maximal consecutive runs of values strictly greater than {p}. List run lengths in input order. Skip values at most {p}; do not emit zero-length runs.", f"""    result = []
    count = 0
    for x in xs:
        if x > {p}:
            count += 1
        elif count:
            result.append(count)
            count = 0
    if count:
        result.append(count)
    return result""")
    if family == "filter_unique":
        return (f"Keep only values whose absolute value is at least {p}, then remove duplicates while preserving first-occurrence order. Return the original signed values, not their absolute values.", f"""    result = []
    for x in xs:
        if abs(x) >= {p} and x not in result:
            result.append(x)
    return result""")
    if family == "subarray_target":
        return (f"Return a single integer: the number of nonempty contiguous subarrays whose sum equals {p}. Values may be negative. Each distinct start/end index pair counts as one subarray.", f"""    counts = {{0: 1}}
    total = 0
    answer = 0
    for x in xs:
        total += x
        answer += counts.get(total - {p}, 0)
        counts[total] = counts.get(total, 0) + 1
    return answer""")
    if family == "longest_bounded_sum":
        return (f"Return the maximum length of a contiguous subarray with sum at most {p}. Values may be negative. Return 0 if no nonempty subarray qualifies.", f"""    best = 0
    for start in range(len(xs)):
        total = 0
        for end in range(start, len(xs)):
            total += xs[end]
            if total <= {p}:
                best = max(best, end - start + 1)
    return best""")
    if family == "max_sum_delete":
        return (f"Choose a nonempty contiguous segment. You may delete at most one element from it, paying a penalty of {p} if you delete. At least one element must remain. Return the largest remaining sum minus any penalty. Return 0 only for an empty input; negative answers are allowed.", f"""    if not xs:
        return 0
    keep = xs[0]
    removed = -1000000000
    best = keep
    for x in xs[1:]:
        removed = max(keep - {p}, removed + x)
        keep = max(x, keep + x)
        best = max(best, keep, removed)
    return best""")
    if family == "window_mode":
        return (f"For every contiguous window of exactly {p} elements, return its most frequent value. Break frequency ties by choosing the smallest value. Return the answers in window order, or [] if the input is shorter than {p}.", f"""    result = []
    for start in range(len(xs) - {p} + 1):
        window = xs[start:start + {p}]
        best = min(window)
        for x in window:
            if window.count(x) > window.count(best) or (window.count(x) == window.count(best) and x < best):
                best = x
        result.append(best)
    return result""")
    if family == "clamp_last":
        return (f"First clamp every value to the inclusive interval [-{p}, {p}]. Then keep only the last occurrence of each clamped value. Return those values ordered by the positions of their last occurrences in the original list.", f"""    seen = []
    result = []
    for raw in reversed(xs):
        value = max(-{p}, min({p}, raw))
        if value not in seen:
            seen.append(value)
            result.append(value)
    return result[::-1]""")
    if family == "next_margin":
        return (f"For each index i, find the smallest j greater than i such that xs[j] is at least xs[i] + {p}. Return a list of distances j-i, using 0 when no such j exists.", f"""    result = []
    for i in range(len(xs)):
        distance = 0
        for j in range(i + 1, len(xs)):
            if xs[j] >= xs[i] + {p}:
                distance = j - i
                break
        result.append(distance)
    return result""")
    if family == "disjoint_pairs":
        return (f"Return the largest number of disjoint pairs of indices whose values sum to {p}. Each index may be used at most once. Equal values at different indices are distinct elements.", f"""    values = sorted(xs)
    left = 0
    right = len(values) - 1
    answer = 0
    while left < right:
        total = values[left] + values[right]
        if total == {p}:
            answer += 1
            left += 1
            right -= 1
        elif total < {p}:
            left += 1
        else:
            right -= 1
    return answer""")
    if family == "circular_run":
        return (f"Treat the list as circular. Return the longest run of consecutive elements strictly greater than {p}, allowing a run to cross the end once. No position can be counted twice, so the result is at most len(xs). Return 0 for an empty input.", f"""    best = 0
    run = 0
    for x in xs + xs:
        if x > {p}:
            run += 1
            best = max(best, min(run, len(xs)))
        else:
            run = 0
    return best""")
    if family == "coin_change":
        return (f"Use the strictly positive values in xs as coin denominations, with unlimited copies of each. Ignore zero and negative values. Return the minimum number of coins needed to make exactly {p}, or -1 if impossible.", f"""    dp = [{p+1}] * {p+1}
    dp[0] = 0
    for total in range(1, {p+1}):
        for coin in xs:
            if 0 < coin <= total:
                dp[total] = min(dp[total], dp[total - coin] + 1)
    return dp[{p}] if dp[{p}] <= {p} else -1""")
    if family == "lis_gap":
        return (f"Return the longest subsequence length such that each selected value is at least {p} greater than the preceding selected value. Preserve input order; elements need not be adjacent. Return 0 for an empty list.", f"""    dp = [1] * len(xs)
    for i in range(len(xs)):
        for j in range(i):
            if xs[i] - xs[j] >= {p}:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if dp else 0""")
    if family == "balanced_signs":
        return (f"First subtract {p} from every value. Return the maximum length of a contiguous subarray with equal counts of negative, zero, and positive transformed values. Return 0 if none exists.", f"""    first = {{(0, 0): -1}}
    neg = 0
    zero = 0
    pos = 0
    best = 0
    for i, x in enumerate(xs):
        if x < {p}:
            neg += 1
        elif x == {p}:
            zero += 1
        else:
            pos += 1
        key = (neg - zero, pos - zero)
        if key in first:
            best = max(best, i - first[key])
        else:
            first[key] = i
    return best""")
    if family == "clipped_water":
        return (f"Interpret each value as a unit-width bar height after clamping it to [0, {p}]. Return the total units of rainwater trapped between bars. Empty input traps 0.", f"""    heights = [max(0, min({p}, x)) for x in xs]
    answer = 0
    for i in range(1, len(heights) - 1):
        level = min(max(heights[:i]), max(heights[i+1:]))
        answer += max(0, level - heights[i])
    return answer""")
    raise ValueError("unknown family")


def make_problem(family, split, seed):
    if family not in TRAIN_FAMILIES + EVAL_FAMILIES or split not in ("support", "score", "test"):
        raise ValueError("invalid algorithmic task")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be nonnegative")
    p = 2 + seed % 3 + 3 * ("support", "score", "test").index(split)
    instruction, code = specification(family, p)
    inputs = inputs_for(split, seed, family, p)
    cases = tuple((xs, oracle(family, [x-p for x in xs] if family == "balanced_signs" else xs, p)) for xs in inputs)
    return Problem(f"hard-{split}-{seed}-{family}", instruction,
        "def solve(xs):\n" + code + "\n", cases, family, 200000, "def solve(xs):\n")


def anchor_problem(family, split, seed):
    specs = {
        "sum": ("Return the sum of all elements.", "    return sum(xs)", lambda xs: sum(xs)),
        "reverse": ("Return the elements in reverse order.", "    return xs[::-1]", lambda xs: list(reversed(xs))),
        "minimum": ("Return the smallest element, or 0 if the list is empty.", "    return min(xs) if xs else 0", lambda xs: min(xs,default=0)),
        "positive_count": ("Return the number of strictly positive elements.", "    return sum(1 for x in xs if x > 0)", lambda xs: len([x for x in xs if x>0])),
        "prefix_sum": ("Return a list of running sums: element i is the sum of input elements 0 through i.", "    result = []\n    total = 0\n    for x in xs:\n        total += x\n        result.append(total)\n    return result", lambda xs: [sum(xs[:i+1]) for i in range(len(xs))]),
        "absolute": ("Return the absolute value of each element, preserving order.", "    return [abs(x) for x in xs]", lambda xs: [-x if x<0 else x for x in xs]),
        "sorted": ("Return all elements sorted in ascending order, preserving duplicates.", "    return sorted(xs)", lambda xs: sorted(xs)),
        "unique": ("Remove duplicate values, keeping only the first occurrence of each value in input order.", "    result = []\n    for x in xs:\n        if x not in result:\n            result.append(x)\n    return result", lambda xs: list(dict.fromkeys(xs))),
    }
    instruction, code, fn = specs[family]
    cases = tuple((xs, fn(xs)) for xs in inputs_for(split,seed,family,3))
    return Problem(f"hard-{split}-{seed}-anchor_{family}",instruction,"def solve(xs):\n"+code+"\n",cases,"anchor_"+family,200000,"def solve(xs):\n")


def hard_problems(split, seed, count=4, partition="train", curriculum="mixed"):
    if (split not in SPLITS or partition not in ("train", "evaluation") or type(seed) is not int or seed < 0
            or curriculum not in ("bridge","mixed","stress")):
        raise ValueError("invalid hard task split/partition/seed")
    anchors = split in ("rehearsal", "retention_score", "retention_test")
    pool = ANCHORS if anchors else TRAIN_FAMILIES if partition == "train" else EVAL_FAMILIES
    if not anchors and partition == "train" and curriculum != "mixed":
        pool = BRIDGE_FAMILIES if curriculum == "bridge" else STRESS_FAMILIES
    if type(count) is not int or not 1 <= count <= len(pool):
        raise ValueError("invalid hard task count")
    # Same task-family selection for an episode's support/dev/test partitions.
    rng = rng_for(PROTOCOL,"families",seed,"anchors" if anchors else partition)
    if not anchors and partition == "train" and curriculum == "mixed":
        bridge_count = min(len(BRIDGE_FAMILIES),(count+1)//2)
        families = rng.sample(list(BRIDGE_FAMILIES),bridge_count) + rng.sample(list(STRESS_FAMILIES),count-bridge_count)
    else:
        families = rng.sample(list(pool),count)
    constructor = anchor_problem if anchors else make_problem
    return [constructor(f,split,seed) for f in families]
