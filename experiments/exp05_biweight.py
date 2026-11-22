"""exp05: benchmark vs wotan sliding biweight (hippke+2019 best general detrender,
rule of thumb window ~3x transit duration ~ 0.3-0.5 d here).
question: does the linear butter2 2-way search filter hold up against it?"""
import time

from common import STARS, TUNE_SEED, save
from evaluate import run, summarize


def main():
    cfgs = [{"method": "butter", "order": 2, "two_way": True, "fc": fc} for fc in (2.0, 3.0)]
    cfgs += [{"method": "biweight", "window": w} for w in (0.2, 0.3, 0.5, 0.75, 1.0)]
    t1 = time.time()
    df = run("exp05", STARS, cfgs, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp05_biweight")
    print(s[["star", "name", "cdpp", "kept", "asym", "null_max", "thr", "recall", "recall_lo",
             "recall_hi", "r150", "r300", "r600"]].to_string(index=False))
    print(f"{time.time() - t1:.0f} s")


if __name__ == "__main__":
    main()
