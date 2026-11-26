"""Search spherical completions of the published fourteen-contact anchor."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from verify import verify


def normalize(points):
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def run(anchor, count, seed, restarts, output, southern=False, initial=None):
    rng = np.random.default_rng(seed)
    points = np.c_[np.sqrt(3) * normalize(np.array(anchor, dtype=float)) / 2,
                   np.full(14, 0.5)]
    fixed = np.vstack((np.array([0, 0, 0, 0, 1]), points))
    i, j = np.triu_indices(count, k=1)
    n = len(i)
    best = np.inf

    def scores(free):
        return np.r_[np.sum(free[i] * free[j], axis=1), (free @ fixed.T).ravel()]

    def smooth(flat, beta):
        raw = flat.reshape(-1, 5)
        length = np.linalg.norm(raw, axis=1)[:, None]
        free = raw / length
        weighted = scores(free) * beta
        logsum = logsumexp(weighted)
        mass = np.exp(weighted - logsum)
        grad = mass[n:].reshape(free.shape[0], -1) @ fixed
        for axis in range(5):
            grad[:, axis] += (np.bincount(i, mass[:n] * free[j, axis], minlength=count)
                              + np.bincount(j, mass[:n] * free[i, axis], minlength=count))
        grad -= np.sum(grad * free, axis=1)[:, None] * free
        return logsum / beta, (grad / length).ravel()

    def polish(free):
        # Use exact constraints on the real-valued model, to floating-point
        # solver tolerances. Only verify.py establishes actual feasibility.
        width = 5 * count + 1
        rows = np.arange(n)
        offsets = np.arange(5)
        linear = np.zeros((count * len(fixed), width))
        for k in range(count):
            linear[k * len(fixed):(k + 1) * len(fixed), 5 * k:5 * (k + 1)] = -fixed
        linear[:, -1] = 1
        grad = np.zeros(width)
        grad[-1] = 1

        def cone(x):
            return x[-1] - scores(x[:-1].reshape(-1, 5))

        def cone_grad(x):
            p = x[:-1].reshape(-1, 5)
            jac = np.zeros((n, width))
            jac[rows[:, None], 5 * i[:, None] + offsets] = -p[j]
            jac[rows[:, None], 5 * j[:, None] + offsets] = -p[i]
            jac[:, -1] = 1
            return np.vstack((jac, linear))

        def norm_grad(x):
            jac = np.zeros((count, width))
            k = np.arange(count)
            jac[k[:, None], 5 * k[:, None] + offsets] = 2 * x[:-1].reshape(-1, 5)
            return jac

        result = minimize(lambda x: x[-1], np.r_[free.ravel(), max(scores(free))],
                          method="SLSQP", jac=lambda _: grad,
                          constraints=[{"type": "ineq", "fun": cone, "jac": cone_grad},
                                       {"type": "eq", "fun": lambda x: (
                                           (x[:-1].reshape(-1, 5) ** 2).sum(axis=1) - 1),
                                        "jac": norm_grad}],
                          options={"maxiter": 150, "ftol": 1e-10})
        return normalize(result.x[:-1].reshape(-1, 5))

    for index in range(restarts):
        raw = rng.normal(size=(count, 5))
        if southern:
            raw[:, 4] = -abs(raw[:, 4])
        if initial is not None:
            if index == 0:
                raw = initial.copy()
            else:
                raw = initial + rng.uniform(0.015, 0.16) * raw
        free = normalize(raw)
        for beta in (25, 100, 400):
            result = minimize(smooth, free.ravel(), args=(beta,), method="L-BFGS-B",
                              jac=True, options={"maxiter": 375, "ftol": 1e-12})
            free = normalize(result.x.reshape(-1, 5))
        free = polish(free)
        maximum = max(float(max(scores(free))), 0.5)
        witness = (free * 10**10).round().astype(np.int64).tolist()
        checked = verify(anchor, witness)
        print(json.dumps({"seed": seed, "restart": index, "size": 15 + count,
                          "max_dot": maximum, "exact": checked}), flush=True)
        if checked["valid"] or maximum < best:
            best = maximum
            output.write_text(json.dumps({"seed": seed, "restart": index,
                                          "max_dot": best, "directions": witness},
                                         indent=2) + "\n", encoding="utf-8")
        if checked["valid"]:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("size", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--southern", action="store_true")
    parser.add_argument("--initial", type=Path)
    args = parser.parse_args()
    if args.size < 17 or args.restarts < 1:
        parser.error("supply at least two free points and a positive restart count")
    anchor = json.loads(Path(__file__).with_name("anchor.json").read_text(encoding="utf-8"))["directions"]
    verify(anchor, [])
    initial = None
    if args.initial:
        raw = json.loads(args.initial.read_text(encoding="utf-8"))["directions"]
        if len(raw) != args.size - 15:
            parser.error("the initial file must have size minus fifteen directions")
        initial = normalize(np.array(raw, dtype=float))
    run(anchor, args.size - 15, args.seed, args.restarts, args.output, args.southern, initial)


if __name__ == "__main__":
    main()
