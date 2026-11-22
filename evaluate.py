import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import data
import inject
import search
import track
from filters import detrend
from metrics import transit_shape, wilson

warnings.filterwarnings("ignore")
WORKERS = max(1, (os.cpu_count() or 2) - 1)
_STARS = {}


def star(key):
    if key not in _STARS:
        _STARS[key] = data.load(key)
    return _STARS[key]


def apply(t, f, cfg):
    cfg = dict(cfg)
    masked = cfg.pop("masked", False)
    if isinstance(cfg.get("fc"), str):
        from autocut import choose_fc
        how = cfg.pop("fc").split(":")[1]
        cfg["fc"], _ = choose_fc(t, f, cfg, how=how)
    out = detrend(t, f, **cfg)
    if not masked:
        return out
    h = search.bls(t, out)
    ph = ((t - h.t0 + h.P / 2) % h.P) - h.P / 2
    return detrend(t, f, mask=np.abs(ph) < 0.75 * h.dur, **cfg)


def _shape(key, cfg):
    cfg, _ = _split(cfg)
    s = star(key)
    t1 = time.time()
    out = apply(s.t, s.flux, cfg)
    m = transit_shape(s, out)
    m["secs"] = time.time() - t1
    return m


def _split(cfg):
    cfg = dict(cfg)
    return cfg, cfg.pop("search", "bls")


def _recover(key, cfg, arg):
    i, pl = arg
    cfg, how = _split(cfg)
    t, f = inject.strip_known(star(key))
    h = search.find(t, apply(t, inject.inject(t, f, pl), cfg), how)
    return dict(sde=h.sde, P_found=h.P, match=search.matches(h.P, pl.P), inj=i, **inject.as_dict(pl))


def _null(key, cfg, seed):
    cfg, how = _split(cfg)
    t, f = inject.scramble(*inject.strip_known(star(key)), seed)
    h = search.find(t, apply(t, f, cfg), how)
    return dict(sde=h.sde, P_found=h.P, seed=seed)


def run(exp, stars, cfgs, shape=True, inj_seed=None, n_per_depth=0, n_null=0):
    tasks = []
    for k in stars:
        for c in cfgs:
            if shape:
                tasks.append(("shape", k, c, None))
            for i, pl in enumerate(inject.draw(inj_seed, n_per_depth) if n_per_depth else []):
                tasks.append(("inj", k, c, (i, pl)))
            for sd in range(n_null):
                tasks.append(("null", k, c, 1000 + sd))
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_dispatch, kind, k, c, a) for kind, k, c, a in tasks]
        res = [f.result() for f in futs]
    for (kind, k, c, a), m in zip(tasks, res):
        track.log(exp, kind, k, c, m, inj_seed=inj_seed if kind == "inj" else None)
    return pd.DataFrame([dict(kind=kind, star=k, name=track.cfg_name(c), cfg_id=track.cfg_id(c), **m)
                         for (kind, k, c, a), m in zip(tasks, res)])


def _dispatch(kind, k, c, a):
    if kind == "shape":
        return _shape(k, c)
    if kind == "inj":
        return _recover(k, c, a)
    return _null(k, c, a)


def summarize(df, sde_floor=7.0):
    """per star x filter: shape metrics, null-calibrated threshold, recovery by depth."""
    rows = []
    for (s, name), g in df.groupby(["star", "name"], sort=False):
        r = dict(star=s, name=name)
        sh = g[g.kind == "shape"]
        if len(sh):
            r.update({k: sh[k].iloc[0] for k in ["rms", "cdpp", "kept", "asym", "edge"]})
        nl = g[g.kind == "null"]
        thr = max(sde_floor, nl.sde.max()) if len(nl) else sde_floor
        r["null_max"] = nl.sde.max() if len(nl) else np.nan
        r["thr"] = thr
        inj = g[g.kind == "inj"]
        if len(inj):
            hit = (inj.match == 1) & (inj.sde > thr)
            r["recall"] = hit.mean()
            lo, hi = wilson(hit.sum(), len(hit))
            r["recall_lo"], r["recall_hi"] = lo, hi
            for dep, gg in inj.groupby("depth"):
                hh = (gg.match == 1) & (gg.sde > thr)
                r[f"r{int(round(dep * 1e6))}"] = hh.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def hits(df, star_key, name, sde_floor=7.0):
    """per-injection hit vector for one (star, filter), keyed by planet so configs line up."""
    g = df[(df.star == star_key) & (df.name == name)]
    nl = g[g.kind == "null"]
    thr = max(sde_floor, nl.sde.max()) if len(nl) else sde_floor
    inj = g[g.kind == "inj"].copy()
    # key on the planet itself so runs from different experiments line up
    inj["key"] = list(zip(inj.P.round(6), inj.t0.round(6), inj.depth.round(9)))
    return inj.set_index("key").apply(lambda r: (r.match == 1) and (r.sde > thr), axis=1)


def paired(ha, hb):
    from metrics import mcnemar
    common = ha.index.intersection(hb.index)
    if len(common) == 0 or len(common) < min(len(ha), len(hb)):
        raise ValueError(f"pairing failed: {len(ha)} vs {len(hb)} injections, {len(common)} shared")
    return mcnemar(ha[common].values, hb[common].values) + (len(common),)


def compare(df, star_key, a, b, df_b=None):
    return paired(hits(df, star_key, a), hits(df if df_b is None else df_b, star_key, b))


def runs_df(exp):
    """rebuild the run-level frame for an experiment from runs.jsonl"""
    return track.load(exp)
