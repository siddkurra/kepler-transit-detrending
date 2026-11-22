"""exp08: final held-out evaluation (seed 2, never touched during tuning).
configs fixed in LOG.md before running. 15 injections per depth (60 per star),
30 scrambled nulls per (star, config). paired mcnemar vs the class baseline and vs
the published-best detrender, holm-corrected."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from common import PLOTS, STARS, TEST_SEED, save
from evaluate import compare, run, runs_df, summarize
from metrics import wilson

CFGS = {
    "raw": {"method": "none"},
    "class: rc 2-way fc=1": {"method": "rc", "fc": 1.0, "two_way": True},
    "class: rc 2-way mask fc=1": {"method": "rc", "fc": 1.0, "two_way": True, "masked": True},
    "biweight 0.5 d (rule of thumb)": {"method": "biweight", "window": 0.5},
    "biweight 0.2 d (tuned)": {"method": "biweight", "window": 0.2},
    "butter2 2-way fc=2": {"method": "butter", "order": 2, "two_way": True, "fc": 2.0},
    "butter2 2-way fc=auto (final)": {"method": "butter", "order": 2, "two_way": True, "fc": "auto:snr"},
}
FINAL = "butter2 2-way fc=auto (final)"
REFS = ["class: rc 2-way fc=1", "biweight 0.5 d (rule of thumb)", "biweight 0.2 d (tuned)"]


def holm(ps):
    order = np.argsort(ps)
    out = np.empty(len(ps))
    run_max = 0
    for rank, i in enumerate(order):
        run_max = max(run_max, (len(ps) - rank) * ps[i])
        out[i] = min(1, run_max)
    return out


def main():
    from track import cfg_name
    label = {cfg_name(c): k for k, c in CFGS.items()}
    t1 = time.time()
    df = run("exp08", STARS, list(CFGS.values()), inj_seed=TEST_SEED, n_per_depth=15, n_null=30)
    print(f"{time.time() - t1:.0f} s")
    report(df, label)


def report(df, label):
    s = summarize(df)
    s.insert(1, "config", s.name.map(label))
    save(s, "exp08_heldout")
    print(s[["star", "config", "cdpp", "kept", "null_max", "thr", "recall", "recall_lo", "recall_hi",
             "r150", "r300", "r600", "r1200"]].to_string(index=False))
    inv = {v: k for k, v in label.items()}
    rows = []
    for st in STARS:
        for ref in REFS:
            n10, n01, p, n = compare(df, st, inv[FINAL], inv[ref])
            rows.append(dict(star=st, vs=ref, only_final=n10, only_ref=n01, n=n, p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p.values)
    save(t, "exp08_paired")
    print(t.to_string(index=False))
    plot(df, label)


def plot(df, label):
    inj = df[df.kind == "inj"].copy()
    inj["config"] = inj.name.map(label)
    s = summarize(df)
    thr = {(r.star, r["name"]): r.thr for _, r in s.iterrows()}
    inj["hit"] = [(m == 1) and (sd > thr[(st, nm)]) for m, sd, st, nm in zip(inj.match, inj.sde, inj.star, inj.name)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for j, st in enumerate(STARS):
        for c, (k, cfg) in zip(plt.cm.tab10.colors, CFGS.items()):
            g = inj[(inj.star == st) & (inj.config == k)]
            dep, rec, lo, hi = [], [], [], []
            for d, gg in g.groupby("depth"):
                a, b = wilson(gg.hit.sum(), len(gg))
                dep.append(d * 1e6); rec.append(gg.hit.mean()); lo.append(a); hi.append(b)
            dep, rec = np.array(dep), np.array(rec)
            lw = 2.5 if k == FINAL else 1
            ax[j].plot(dep, rec, "o-", color=c, lw=lw, label=k)
            ax[j].fill_between(dep, lo, hi, color=c, alpha=0.08)
        ax[j].set(xscale="log", title=st, xlabel="injected depth (ppm)", ylim=(-0.02, 1.02))
        ax[j].set_xticks([150, 300, 600, 1200], ["150", "300", "600", "1200"])
        ax[j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("recall (held-out seed, null-calibrated)")
    ax[2].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/exp08_heldout.png", dpi=130)


def plot_simple(df):
    """readme version: 3 lines, plain labels"""
    from track import cfg_name
    s = summarize(df)
    thr = {(r.star, r["name"]): r.thr for _, r in s.iterrows()}
    inj = df[df.kind == "inj"].copy()
    inj["hit"] = [(m == 1) and (sd > thr[(st, nm)]) for m, sd, st, nm in zip(inj.match, inj.sde, inj.star, inj.name)]
    lines = [("class: rc 2-way fc=1", "our class project (RC filter)", "#9a9a9a", 1.5),
             ("biweight 0.5 d (rule of thumb)", "wotan biweight (standard tool)", "#5b8fd1", 1.5),
             (FINAL, "this repo (Butterworth, auto cutoff)", "#d1495b", 3)]
    titles = {"kepler17": "Kepler-17 (spotted)", "kepler63": "Kepler-63 (very spotted)", "hatp7": "HAT-P-7 (quiet)"}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for j, st in enumerate(STARS):
        for key, lab, col, lw in lines:
            g = inj[(inj.star == st) & (inj.name == cfg_name(CFGS[key]))]
            r = g.groupby("depth").hit.mean()
            ax[j].plot(r.index * 1e6, r.values, "o-", color=col, lw=lw, ms=4 + lw, label=lab)
        ax[j].set(xscale="log", title=titles[st], xlabel="transit depth (ppm)", ylim=(-0.03, 1.03))
        ax[j].set_xticks([150, 300, 600, 1200], ["150", "300", "600", "1200"])
        ax[j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[j].grid(alpha=0.25)
        for side in ("top", "right"):
            ax[j].spines[side].set_visible(False)
    ax[0].set_ylabel("fraction of fake planets found")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(f"{PLOTS}/heldout_simple.png", dpi=150)


if __name__ == "__main__":
    import sys
    if "--simple" in sys.argv:
        plot_simple(runs_df("exp08"))
    elif "--report" in sys.argv:
        from track import cfg_name
        report(runs_df("exp08"), {cfg_name(c): k for k, c in CFGS.items()})
    else:
        main()

