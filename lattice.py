"""Exact maximum clique of integer directions compatible with the fixed anchor."""

import argparse
import json
from itertools import combinations, product
from math import gcd
from pathlib import Path

from verify import at_most, anchor_accepts, verify


def rays(radius, anchor):
    pole = [0, 0, 0, 0, 1]
    vertices = []
    for ray in product(range(-radius, radius + 1), repeat=5):
        if gcd(*ray) != 1:
            continue
        ray = list(ray)
        if at_most(ray, pole, 2) and all(anchor_accepts(ray, v) for v in anchor):
            vertices.append(ray)
    return vertices


def graph(vertices):
    adjacent = [0] * len(vertices)
    for i, j in combinations(range(len(vertices)), 2):
        if at_most(vertices[i], vertices[j], 2):
            adjacent[i] |= 1 << j
            adjacent[j] |= 1 << i
    return adjacent


def maximum_clique(neighbours, target=None):
    best = ()
    visits = 0

    def visit(remaining, chosen):
        nonlocal best, visits
        visits += 1
        order, bounds = [], []
        unused = remaining
        colour = 0
        while unused:
            colour += 1
            independent = unused
            while independent:
                bit = independent & -independent
                vertex = bit.bit_length() - 1
                order.append(vertex)
                bounds.append(colour)
                unused ^= bit
                independent &= ~(bit | neighbours[vertex])
        for vertex, bound in zip(reversed(order), reversed(bounds)):
            cutoff = len(best) + 1 if target is None else target
            if len(chosen) + bound < cutoff:
                return
            extension = chosen + (vertex,)
            children = remaining & neighbours[vertex]
            if children:
                visit(children, extension)
            elif len(extension) > len(best):
                best = extension
            if target is not None and len(best) >= target:
                return
            remaining ^= 1 << vertex

    visit((1 << len(neighbours)) - 1, ())
    return best, visits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("radius", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", type=int,
                        help="search for this many completion points, skipping smaller cliques")
    args = parser.parse_args()
    if args.radius < 1:
        parser.error("radius must be positive")
    if args.target is not None and args.target < 1:
        parser.error("target must be positive")
    anchor = json.loads(Path(__file__).with_name("anchor.json").read_text(encoding="utf-8"))["directions"]
    points = rays(args.radius, anchor)
    compatibility = graph(points)
    print(json.dumps({"radius": args.radius, "rays": len(points),
                      "edges": sum(x.bit_count() for x in compatibility) // 2}), flush=True)
    clique, visits = maximum_clique(compatibility, args.target)
    selected = [points[i] for i in clique]
    checked = verify(anchor, selected)
    if not checked["valid"]:
        raise RuntimeError(f"the extracted clique failed verification: {checked}")
    report = {"radius": args.radius, "rays": len(points), "search_nodes": visits}
    if args.target is None:
        report["maximum_completion"] = len(clique)
    else:
        report.update({"target": args.target, "target_found": len(clique) >= args.target})
    if clique:
        report.update({"exact": checked, "directions": selected})
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
