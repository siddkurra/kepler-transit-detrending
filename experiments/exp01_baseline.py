"""exp01: port the class baseline onto the new harness.
1) regression: kepler-17 shape numbers should match the class project
2) pilot: injection + scrambled-null on all 3 stars to check calibration/timing"""
import time

from common import STARS, TUNE_SEED, save
from evaluate import run, summarize


def main():
    CUTS = [0.1, 0.3, 1.0, 5.0]
    cfgs = [{"method": "none"}]
    for fc in CUTS:
        cfgs += [{"method": "rc", "fc": fc}, {"method": "rc", "fc": fc, "two_way": True}]

    t1 = time.time()
    df = run("exp01a", ["kepler17"], cfgs)
    print(summarize(df)[["name", "rms", "kept", "asym", "edge", "cdpp"]].to_string(index=False))

    pilot = [{"method": "none"}, {"method": "rc", "fc": 0.3}, {"method": "rc", "fc": 1.0, "two_way": True},
             {"method": "rc", "fc": 1.0, "two_way": True, "masked": True}]
    df = run("exp01b", STARS, pilot, inj_seed=TUNE_SEED, n_per_depth=8, n_null=20)
    s = summarize(df)
    save(s, "exp01_pilot")
    print(s.drop(columns=["edge"]).to_string(index=False))
    print(f"{time.time() - t1:.0f} s")


if __name__ == "__main__":
    main()
