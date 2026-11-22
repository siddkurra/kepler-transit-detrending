import os
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

# literature periods only used as a sanity window for the BLS search
TARGETS = {
    "kepler17": dict(name="Kepler-17", P=(1.40, 1.60), kind="active G, 2% hot jupiter, ~12 d rotation"),
    "kepler63": dict(name="Kepler-63", P=(9.30, 9.60), kind="very active young G, ~5.4 d rotation"),
    "hatp7": dict(name="HAT-P-7", P=(2.15, 2.25), kind="quiet F star, 0.7% hot jupiter"),
}
QUARTERS = (1, 2, 3, 4)


@dataclass
class Star:
    key: str
    t: np.ndarray
    flux: np.ndarray
    P: float
    t0: float
    dur: float
    depth: float

    def phase(self, t=None):
        t = self.t if t is None else t
        return ((t - self.t0 + self.P / 2) % self.P) - self.P / 2


def _download(name):
    import lightkurve as lk
    res = lk.search_lightcurve(name, mission="Kepler", author="Kepler", exptime=1800)
    T, F = [], []
    for q in QUARTERS:
        lc = res[res.mission == f"Kepler Quarter {q:02d}"].download(flux_column="sap_flux")
        lc = lc.remove_nans()
        raw = lc.flux.value
        raw = np.asarray(getattr(raw, "unmasked", raw), dtype=float)
        f = raw / np.median(raw)
        # noise from point-to-point differences, not from the median-filter residual:
        # on a smooth bright star the window is monotonic, residual is exactly 0
        # for most points and the MAD collapses to 0 (ate 80% of kepler-63)
        sig = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2)
        ok = f - median_filter(f, size=25, mode="nearest") < 5 * sig
        T.append(np.asarray(lc.time.value, float)[ok])
        F.append(f[ok])
    return np.concatenate(T), np.concatenate(F)


def _ephemeris(t, f, prange):
    import lightkurve as lk
    from astropy.timeseries import BoxLeastSquares
    flat = lk.LightCurve(time=t, flux=f).flatten(window_length=101)
    bls = BoxLeastSquares(flat.time.value, flat.flux.value)
    pg = bls.power(np.linspace(*prange, 5000), np.arange(0.04, 0.25, 0.005))
    i = np.argmax(pg.power)
    return pg.period[i], pg.transit_time[i], pg.duration[i]


def _depth(t, f, P, t0, D):
    ph = ((t - t0 + P / 2) % P) - P / 2
    ep = np.round((t - t0) / P)
    corr = np.full(len(f), np.nan)
    w_out = max(0.35, 2.5 * D)
    for e in np.unique(ep):
        w = (ep == e) & (np.abs(ph) < w_out)
        o = w & (np.abs(ph) > 0.75 * D)
        if o.sum() < 10 or not np.any(w & (np.abs(ph) < D / 4)):
            continue
        corr[w] = f[w] / np.polyval(np.polyfit(ph[o], f[o], 2), ph[w])
    inn = np.abs(ph) < D / 4
    out = (np.abs(ph) > 0.75 * D) & (np.abs(ph) < w_out)
    return np.nanmedian(corr[out]) - np.nanmean(corr[inn])


def load(key):
    path = os.path.join(HERE, "data", f"{key}.npz")
    if not os.path.exists(path):
        cfg = TARGETS[key]
        t, f = _download(cfg["name"])
        P, t0, D = _ephemeris(t, f, cfg["P"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, t=t, flux=f, P=P, t0=t0, dur=D, depth=_depth(t, f, P, t0, D))
    z = np.load(path)
    return Star(key, z["t"], z["flux"], float(z["P"]), float(z["t0"]), float(z["dur"]),
                float(z["depth"]))


if __name__ == "__main__":
    for k in TARGETS:
        s = load(k)
        print(f"{k:<9} n={len(s.t):5d}  P={s.P:.5f} d  dur={s.dur*24:.2f} h  "
              f"depth={s.depth*1e6:6.0f} ppm  rms={np.std(s.flux)*1e6:6.0f} ppm")
