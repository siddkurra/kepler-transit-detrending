from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Planet:
    P: float
    t0: float
    depth: float
    dur: float
    b: float


def strip_known(star):
    keep = np.abs(star.phase()) > 1.5 * star.dur
    return star.t[keep], star.flux[keep]


def duration(P, b):
    # central transit across a sun-like star: ~13 h at 1 yr, scales P^(1/3)
    return 13 / 24 * (P / 365.25) ** (1 / 3) * np.sqrt(1 - b**2)


def trapezoid(t, pl, ingress=0.15):
    ph = np.abs(((t - pl.t0 + pl.P / 2) % pl.P) - pl.P / 2)
    half = pl.dur / 2
    tin = ingress * pl.dur
    s = np.clip((half - ph) / tin, 0, 1)
    return 1 - pl.depth * s


def inject(t, f, pl):
    return f * trapezoid(t, pl)


def draw(seed, n, depths=(150e-6, 300e-6, 600e-6, 1200e-6), pmin=1.0, pmax=10.0):
    rng = np.random.default_rng(seed)
    out = []
    for dep in depths:
        for _ in range(n):
            P = np.exp(rng.uniform(np.log(pmin), np.log(pmax)))
            b = rng.uniform(0, 0.8)
            out.append(Planet(P, 131.5 + rng.uniform(0, P), dep, duration(P, b), b))
    return out


def scramble(t, f, seed, chunk=6.0):
    """null light curve: chop into ~6 d chunks, shuffle, restitch, flip upside down.
    keeps the noise/variability statistics, kills anything periodic, and real dips
    become bumps so they can't be 'found'."""
    rng = np.random.default_rng(seed)
    edges = np.arange(t[0], t[-1] + chunk, chunk)
    pieces = [np.where((t >= a) & (t < a + chunk))[0] for a in edges]
    pieces = [p for p in pieces if len(p)]
    order = rng.permutation(len(pieces))
    dt = np.median(np.diff(t))
    tt, ff, cur = [], [], t[0]
    for k in order:
        p = pieces[k]
        rel = t[p] - t[p[0]]
        tt.append(cur + rel)
        ff.append(f[p])
        cur = cur + rel[-1] + dt
    return np.concatenate(tt), 2 - np.concatenate(ff)


def as_dict(pl):
    return asdict(pl)
