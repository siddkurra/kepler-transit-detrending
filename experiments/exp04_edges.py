"""exp04: gap-edge handling. odd-reflection padding of ~3 time constants vs start-from-rest.
expect: big win at low fc (time constant ~ edge window), ~nothing at fc=3."""
import time

import numpy as np
import pandas as pd

from common import STARS, TUNE_SEED, save
from evaluate import run, star, summarize
from filters import detrend
from metrics import _edges


def edge_ratio(key, cfg):
    s = star(key)
    o = detrend(s.t, s.flux, **cfg)
    oot = np.abs(s.phase()) > 1.5 * s.dur
    e = _edges(s.t)
    return np.nanstd(o[e & oot]) / np.nanstd(o[~e & oot])


def main():
    rows = []
    for key in STARS:
        for m in ({"method": "rc", "two_way": True}, {"method": "butter", "order": 2, "two_way": True},
                  {"method": "butter", "order": 2}):
            for fc in (0.3, 1.0, 3.0):
                r = {"star": key, "filter": m["method"] + ("_2way" if m.get("two_way") else ""), "fc": fc}
                for pad in (False, True):
                    r["pad" if pad else "nopad"] = edge_ratio(key, {**m, "fc": fc, "pad": pad})
                rows.append(r)
    er = pd.DataFrame(rows)
    save(er, "exp04_edge_ratio")
    print("edge rms / interior rms (1 = no edge penalty)")
    print(er.to_string(index=False))

    t1 = time.time()
    cfgs = [{"method": "butter", "order": 2, "two_way": True, "fc": fc, "pad": p}
            for fc in (0.5, 3.0) for p in (False, True)]
    df = run("exp04", STARS, cfgs, inj_seed=TUNE_SEED, n_per_depth=6, n_null=15)
    s = summarize(df)
    save(s, "exp04_edges")
    print(s[["star", "name", "cdpp", "edge", "null_max", "thr", "recall", "recall_lo", "recall_hi",
             "r150", "r300"]].to_string(index=False))
    print(f"{time.time() - t1:.0f} s")


if __name__ == "__main__":
    main()
