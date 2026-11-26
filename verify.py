"""Check an anchored kissing code with integer arithmetic."""

import argparse
import json
from itertools import combinations
from pathlib import Path


def dot(a: list[int], b: list[int]) -> int:
    return sum(x * y for x, y in zip(a, b))


def directions(data: list, dimension: int) -> list[list[int]]:
    if not isinstance(data, list):
        raise ValueError("directions must be a list")
    for row in data:
        if (not isinstance(row, list) or len(row) != dimension
                or any(type(x) is not int for x in row) or not any(row)):
            raise ValueError(f"expected a nonzero integer direction in dimension {dimension}")
    return data


def at_most(a: list[int], b: list[int], reciprocal: int) -> bool:
    inner = dot(a, b)
    return inner <= 0 or (reciprocal * inner) ** 2 <= dot(a, a) * dot(b, b)


def anchor_accepts(a: list[int], v: list[int]) -> bool:
    """Check dot(normalize(a), (sqrt(3)/2*normalize(v), 1/2)) <= 1/2."""
    n = dot(a, a)
    s = dot(v, v)
    t = a[-1]
    inner = dot(a[:4], v)
    if inner <= 0:
        return True
    rational = s * (n + t * t) - 3 * inner * inner
    radical = 2 * s * t
    if radical >= 0:
        return rational >= 0 and rational * rational >= radical * radical * n
    return rational >= 0 or rational * rational <= radical * radical * n


def verify(anchor: list, completion: list) -> dict:
    anchor = directions(anchor, 4)
    completion = directions(completion, 5)
    if len(anchor) != 14 or not all(at_most(a, b, 3) for a, b in combinations(anchor, 2)):
        raise ValueError("the fourteen anchor directions do not form a valid code")
    pole = [0, 0, 0, 0, 1]
    violations = {
        "pole": sum(not at_most(a, pole, 2) for a in completion),
        "anchor": sum(not anchor_accepts(a, v) for a in completion for v in anchor),
        "completion": sum(not at_most(a, b, 2) for a, b in combinations(completion, 2)),
    }
    size = 15 + len(completion)
    valid = not any(violations.values())
    return {"size": size, "valid": valid, "violations": violations,
            "refutes_40": valid and size >= 41}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("completion", type=Path)
    parser.add_argument("--anchor", type=Path)
    args = parser.parse_args()
    data = json.loads(args.completion.read_text(encoding="utf-8"))
    if args.anchor or "anchor" not in data:
        path = args.anchor or Path(__file__).with_name("anchor.json")
        anchor = json.loads(path.read_text(encoding="utf-8"))["directions"]
    else:
        anchor = data["anchor"]
    print(json.dumps(verify(anchor, data["directions"]), sort_keys=True))


if __name__ == "__main__":
    main()
