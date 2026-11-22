import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs.jsonl")


def cfg_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def cfg_name(cfg):
    c = dict(cfg)
    s = c.pop("method")
    if c.pop("two_way", False):
        s += "_2way"
    if c.pop("masked", False):
        s += "_mask"
    for k, v in sorted(c.items()):
        s += f" {k}={v}"
    return s


def log(exp, kind, star, cfg, metrics, **extra):
    os.makedirs(os.path.dirname(RUNS), exist_ok=True)
    rec = dict(exp=exp,
               kind=kind, star=star, cfg=cfg, cfg_id=cfg_id(cfg), name=cfg_name(cfg),
               **extra, **metrics)
    with open(RUNS, "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


def load(exp=None, kind=None):
    import pandas as pd
    if not os.path.exists(RUNS):
        return pd.DataFrame()
    df = pd.read_json(RUNS, lines=True, precise_float=True)
    if exp is not None:
        df = df[df.exp == exp]
    if kind is not None:
        df = df[df.kind == kind]
    return df
