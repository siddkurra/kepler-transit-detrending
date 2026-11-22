"""paired McNemar comparisons on shared injections, from results/runs.jsonl.
usage: python paired.py exp05 kepler17 "name A" "name B" """
import sys

from common import HERE  # noqa: F401
from evaluate import compare, runs_df

if __name__ == "__main__":
    exp, star, a, b = sys.argv[1:5]
    n10, n01, p, n = compare(runs_df(exp), star, a, b)
    print(f"{star}: only A {n10}, only B {n01}, n={n}, McNemar p={p:.4g}")
