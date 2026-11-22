"""exp06: pick fc from the data, never seeing injections.
snr: maximize box_retention(fc)/cdpp(fc)  (exp02 says recall tracks kept/cdpp)
acf: cofiam-style, minimize |autocorrelation| of residuals at the transit time scale
compare vs the hand-tuned fixed fc from exp02."""
import time

from common import STARS, TUNE_SEED, save
from evaluate import compare, run, summarize

B = {"method": "butter", "order": 2, "two_way": True}


def main():
    cfgs = [{**B, "fc": fc} for fc in (1.5, 2.0, 3.0)] + [{**B, "fc": "auto:snr"}, {**B, "fc": "auto:acf"}]
    t1 = time.time()
    df = run("exp06", STARS, cfgs, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp06_autocut")
    print(s[["star", "name", "cdpp", "kept", "null_max", "thr", "recall", "recall_lo", "recall_hi",
             "r150", "r300", "r600"]].to_string(index=False))
    print(f"{time.time() - t1:.0f} s")
    for st in STARS:
        for b in ("butter_2way fc=3.0 order=2", "butter_2way fc=auto:acf order=2"):
            n10, n01, p, n = compare(df, st, "butter_2way fc=auto:snr order=2", b)
            print(f"{st:<9} auto:snr vs {b:<32} only snr {n10}  only other {n01}  p={p:.3f}")


if __name__ == "__main__":
    main()
