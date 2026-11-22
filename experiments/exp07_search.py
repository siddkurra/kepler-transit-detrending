"""exp07: same detrended light curves, different search.
bls (box, current) vs mes_white (matched filter fold, no whitening) vs mes (whitened
matched filter, kepler tps style) vs tls (limb-darkened template, hippke & heller 2019).
filter fixed at butter2 2-way fc=2 (middle of the exp06 plateau)."""
import time

from common import STARS, TUNE_SEED, save
from evaluate import compare, run, summarize

B = {"method": "butter", "order": 2, "two_way": True, "fc": 2.0}
SEARCHES = ("bls", "mes_white", "mes", "tls")


def main():
    cfgs = [{**B, "search": s} for s in SEARCHES]
    t1 = time.time()
    df = run("exp07", STARS, cfgs, shape=False, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp07_search")
    print(s[["star", "name", "null_max", "thr", "recall", "recall_lo", "recall_hi",
             "r150", "r300", "r600"]].to_string(index=False))
    print(f"{time.time() - t1:.0f} s")
    base = "butter_2way fc=2.0 order=2 search=bls"
    for st in STARS:
        for sname in SEARCHES[1:]:
            n10, n01, p, n = compare(df, st, f"butter_2way fc=2.0 order=2 search={sname}", base)
            print(f"{st:<9} {sname:<10} vs bls: only {sname} {n10}, only bls {n01}, p={p:.3f}")




def main_b():
    """exp07b: whitening with the psd floored at the white level, paired vs exp07's bls."""
    from evaluate import runs_df
    cfgs = [{**B, "search": "mes"}]
    df = run("exp07b", STARS, cfgs, shape=False, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp07b_mes_floor")
    print(s[["star", "name", "null_max", "thr", "recall", "recall_lo", "recall_hi",
             "r150", "r300", "r600"]].to_string(index=False))
    old, new = runs_df("exp07"), runs_df("exp07b")
    for st in STARS:
        for ref in ("bls", "mes"):
            n10, n01, p, n = compare(new, st, "butter_2way fc=2.0 order=2 search=mes",
                                     f"butter_2way fc=2.0 order=2 search={ref}", df_b=old)
            print(f"{st:<9} mes(floored) vs exp07 {ref:<4}: only floored {n10}, only {ref} {n01}, p={p:.3f}")


def main_c():
    """exp07c: can whitening replace the high-pass? mes on undetrended flux (tps style,
    whitening is the only noise handling) vs the butter + bls pipeline."""
    from evaluate import runs_df
    cfgs = [{"method": "none", "search": "mes"}, {"method": "none", "search": "bls"}]
    df = run("exp07c", STARS, cfgs, shape=False, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp07c_whiten_only")
    print(s[["star", "name", "null_max", "thr", "recall", "recall_lo", "recall_hi",
             "r150", "r300", "r600"]].to_string(index=False))
    old, new = runs_df("exp07"), runs_df("exp07c")
    for st in STARS:
        n10, n01, p, n = compare(new, st, "none search=mes", "butter_2way fc=2.0 order=2 search=bls",
                                 df_b=old)
        print(f"{st:<9} whiten-only mes vs butter+bls: only whiten {n10}, only butter {n01}, p={p:.4f}")


if __name__ == "__main__":
    import sys
    main_b() if "--b" in sys.argv else main_c() if "--c" in sys.argv else main()
