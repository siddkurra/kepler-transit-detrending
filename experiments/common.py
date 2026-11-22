import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:.3g}")

STARS = ["kepler17", "kepler63", "hatp7"]
TUNE_SEED, TEST_SEED = 1, 2
OUT = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots")


def save(df, name):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
