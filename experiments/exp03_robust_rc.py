"""exp03: robust (self-masking) rc. hold the trend when flux drops > k*sigma below it.
hypothesis: keeps more depth at high fc (so better kept/cdpp -> better recall) and,
unlike the bls 2-pass mask, doesnt need pass 1 to find the planet, so it shouldnt
inflate nulls the same way."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import PLOTS, STARS, TUNE_SEED, save
from evaluate import run, summarize


def main_b():
    """exp03b: same idea, but decide what's a dip from a zero-lag residual (2-way pass,
    lower-tail sigma clip, refill, refit) instead of the causal hold."""
    cfgs = []
    for m in ({"method": "rc"}, {"method": "butter", "order": 2}):
        for fc in (2.0, 3.0, 5.0):
            cfgs.append({**m, "fc": fc, "two_way": True})
            cfgs.append({**m, "fc": fc, "two_way": True, "clip": 3.0})
    t1 = time.time()
    df = run("exp03b", STARS, cfgs, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp03b_clip")
    cols = ["star", "name", "cdpp", "kept", "asym", "null_max", "thr", "recall", "recall_lo",
            "recall_hi", "r150", "r300", "r600"]
    print(s[cols].to_string(index=False))
    print(f"{time.time() - t1:.0f} s")


def main():
    cfgs = [{"method": "rc", "fc": 3.0, "two_way": True},
            {"method": "butter", "order": 2, "fc": 3.0, "two_way": True},
            {"method": "rc", "fc": 3.0, "two_way": True, "masked": True}]
    for fc in (1.0, 2.0, 3.0, 5.0):
        for k in (2.5, 3.5):
            cfgs.append({"method": "robust_rc", "fc": fc, "k": k, "two_way": True})
    t1 = time.time()
    df = run("exp03", STARS, cfgs, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp03_robust_rc")
    cols = ["star", "name", "cdpp", "kept", "asym", "null_max", "thr", "recall", "recall_lo",
            "recall_hi", "r150", "r300", "r600"]
    print(s[cols].to_string(index=False))
    print(f"{time.time() - t1:.0f} s")
    plot(s)


def plot(s):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, st in enumerate(STARS):
        g = s[s.star == st]
        for _, r in g.iterrows():
            c = "C3" if r["name"].startswith("robust") else "C0"
            ax[j].errorbar(r.kept * 100, r.recall, yerr=[[r.recall - r.recall_lo], [r.recall_hi - r.recall]],
                           fmt="o", color=c, alpha=0.8, capsize=2)
            ax[j].annotate(r["name"].replace("_2way", "").replace("robust_rc", "R"), (r.kept * 100, r.recall),
                           fontsize=5, xytext=(3, 2), textcoords="offset points")
        ax[j].set(title=f"{st}  (red = robust rc)", xlabel="real planet depth kept (%)", ylim=(0, 1.05))
        ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("recall (null-calibrated)")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/exp03_robust_rc.png", dpi=130)


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        plot(pd.read_csv("../results/exp03_robust_rc.csv"))
    elif "--b" in sys.argv:
        main_b()
    else:
        main()
