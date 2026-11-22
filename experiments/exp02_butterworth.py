"""exp02: butterworth order x cutoff x direction.
prediction (RESEARCH.md): order 1 can't fit spots (0.17/d) to 10% while keeping the
transit (10.4/d) at 99%; order 2 can for fc in ~0.5-3.9/d. so order>=2 should beat rc."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import numpy as np
from scipy import signal

from common import PLOTS, STARS, TUNE_SEED, save
from evaluate import run, summarize
from filters import butter_sos

DT = 29.4 / 60 / 24
F_SPOT, F_TR = 0.166, 10.4


def predicted():
    rows = []
    for order in (1, 2, 4):
        for fc in (0.3, 0.5, 1.0, 2.0, 3.0):
            _, h = signal.sosfreqz(butter_sos(fc, DT, order), worN=[F_SPOT, F_TR], fs=1 / DT)
            rows.append((order, fc, abs(h[0]), abs(h[1])))
    return rows


def main():
    print("order  fc   |H(spot)|  |H(transit)|   (want <0.1, >0.99)")
    for o, fc, hs, ht in predicted():
        ok = "  <- fits" if hs < 0.1 and ht > 0.99 else ""
        print(f"{o:5d} {fc:4.1f} {hs:10.3f} {ht:12.4f}{ok}")

    cfgs = [{"method": "rc", "fc": 1.0, "two_way": True}]
    for order in (1, 2, 4):
        for fc in (0.3, 0.5, 1.0, 2.0, 3.0):
            for tw in (False, True):
                cfgs.append({"method": "butter", "order": order, "fc": fc, "two_way": tw})
    t1 = time.time()
    df = run("exp02", STARS, cfgs, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp02_butterworth")
    print(s.drop(columns=["edge"]).to_string(index=False))
    print(f"{time.time() - t1:.0f} s")
    plot(s)


def plot(s):
    fig, ax = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for j, st in enumerate(STARS):
        for tw, ls in ((True, "-"), (False, "--")):
            for order, c in ((1, "C0"), (2, "C1"), (4, "C2")):
                g = s[(s.star == st) & s.name.str.startswith("butter" + ("_2way" if tw else " "))
                      & s.name.str.contains(f"order={order}")]
                fc = g.name.str.extract(r"fc=([\d.]+)")[0].astype(float)
                lab = f"N={order} {'2-way' if tw else '1-way'}"
                ax[0, j].plot(fc, g.recall, ls, color=c, marker="o", label=lab)
                ax[1, j].plot(fc, 100 * g.kept, ls, color=c, marker="o", label=lab)
        ref = s[(s.star == st) & (s.name == "rc_2way fc=1.0")]
        ax[0, j].axhline(ref.recall.iloc[0], color="k", ls=":", label="rc 2-way fc=1 (class best)")
        ax[1, j].axhline(100 * ref.kept.iloc[0], color="k", ls=":")
        ax[0, j].set(title=st, xscale="log", ylim=(0, 1.02))
        ax[1, j].set(xlabel="fc (1/day)")
    ax[0, 0].set_ylabel("recall (null-calibrated)")
    ax[1, 0].set_ylabel("real planet depth kept (%)")
    ax[0, 0].legend(fontsize=6)
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.set_xticks([0.3, 0.5, 1, 2, 3], ["0.3", "0.5", "1", "2", "3"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/exp02_butterworth.png", dpi=130)


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        plot(pd.read_csv("../results/exp02_butterworth.csv"))
    else:
        main()
