"""Vary the fourteen contacts while searching for an exact completion."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from search import normalize
from verify import directions, verify


def search(size, initial, seed, attempts, output):
    rng = np.random.default_rng(seed)
    a = 14
    n = size - 1
    b = 5 * n
    i, j = np.triu_indices(n, k=1)
    scales = np.ones((n, 5))
    scales[:a] = (np.sqrt(3) / 2,) * 4 + (0,)
    shift = np.zeros((n, 5))
    shift[:a, 4] = 0.5
    fixed = scales == 0
    rows = np.arange(len(i))
    columns = np.arange(5)
    target = np.zeros(b + 1)
    target[-1] = 1
    cap_rows = np.zeros((n - a, b + 1))
    cap_rows[np.arange(n - a), 5 * np.arange(a, n) + 4] = -1
    cap_rows[:, -1] = 1
    norm_rows = np.arange(n)[:, None]
    norm_columns = norm_rows * 5 + columns
    bound = [(0, 0) if f else (-1.05, 1.05) for f in fixed.ravel()] + [(-1, 1)]

    def values(raw):
        points = raw * scales + shift
        return points, np.r_[np.einsum("ij,ij->i", points[i], points[j]), points[a:, 4]]

    def project(raw):
        raw = raw.copy()
        raw[fixed] = 0
        return normalize(raw)

    def soft(x, beta):
        raw = x.reshape(n, 5).copy()
        raw[fixed] = 0
        length = np.linalg.norm(raw, axis=1)[:, None]
        unit = raw / length
        points, scores = values(unit)
        z = logsumexp(beta * scores)
        weights = np.exp(beta * scores - z)
        w = weights[:len(i)]
        grad = np.zeros_like(points)
        for k in columns:
            grad[:, k] = (np.bincount(i, w * points[j, k], minlength=n)
                          + np.bincount(j, w * points[i, k], minlength=n))
        grad[a:, 4] += weights[len(i):]
        grad *= scales
        grad -= (grad * unit).sum(axis=1)[:, None] * unit
        return z / beta, (grad / length).ravel()

    def slack(x):
        return x[-1] - values(x[:-1].reshape(n, 5))[1]

    def slack_jac(x):
        points, _ = values(x[:-1].reshape(n, 5))
        gradient = np.zeros((len(i), b + 1))
        gradient[rows[:, None], 5 * i[:, None] + columns] = -points[j] * scales[i]
        gradient[rows[:, None], 5 * j[:, None] + columns] = -points[i] * scales[j]
        gradient[:, -1] = 1
        return np.vstack((gradient, cap_rows))

    def norm_jac(x):
        gradient = np.zeros((n, b + 1))
        gradient[norm_rows, norm_columns] = 2 * x[:-1].reshape(n, 5)
        return gradient

    best = None
    for k in range(attempts):
        unit = initial if k == 0 else project(initial + rng.uniform(0.02, 0.25)
                                               * rng.normal(size=(n, 5)))
        for beta in (100, 400):
            result = minimize(soft, unit.ravel(), args=(beta,), jac=True, method="L-BFGS-B",
                              options={"maxiter": 400, "ftol": 1e-12})
            unit = project(result.x.reshape(n, 5))
        entry = np.r_[unit.ravel(), max(values(unit)[1])]
        result = minimize(lambda x: x[-1], entry, jac=lambda _: target, method="SLSQP",
                          bounds=bound, constraints=[
                              {"type": "ineq", "fun": slack, "jac": slack_jac},
                              {"type": "eq", "fun": lambda x: (
                                  np.sum(x[:-1].reshape(n, 5) ** 2, axis=1) - 1),
                               "jac": norm_jac}],
                          options={"maxiter": 150, "ftol": 1e-10})
        unit = project(result.x[:-1].reshape(n, 5))
        score = float(max(values(unit)[1]))
        integers = np.rint(unit * (10**10)).astype(np.int64)
        anchor = integers[:a, :4].tolist()
        completion = integers[a:].tolist()
        try:
            checked = verify(anchor, completion)
        except ValueError:
            checked = {"valid": False, "reason": "fourteen-contact layer exceeds its bound"}
        report = {"size": size, "seed": seed, "restart": k,
                  "max_dot_excluding_fixed_contacts": score, "exact": checked}
        print(json.dumps(report), flush=True)
        if checked["valid"] or best is None or score < best:
            best = score
            output.write_text(json.dumps({**report, "anchor": anchor, "directions": completion},
                                         indent=2) + "\n", encoding="utf-8")
        if checked["valid"]:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--restarts", type=int, default=16)
    args = parser.parse_args()
    data = json.loads(args.initial.read_text(encoding="utf-8"))
    a = data.get("anchor")
    if a is None:
        a = json.loads(Path(__file__).with_name("anchor.json").read_text(encoding="utf-8"))["directions"]
    a = directions(a, 4)
    b = directions(data["directions"], 5)
    if len(a) != 14 or len(b) < 2 or args.restarts < 1:
        parser.error("supply fourteen contact directions, two or more free points, and a positive restart count")
    initial = np.vstack((np.c_[normalize(np.array(a)), np.zeros(14)], normalize(np.array(b))))
    search(15 + len(b), initial, args.seed, args.restarts, args.output)


if __name__ == "__main__":
    main()
