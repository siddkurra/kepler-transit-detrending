import numpy as np

from filters import detrend, split_gaps

FC_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0)
DURS = (0.08, 0.12, 0.16)


def box_retention(cfg, fc, dur, dt):
    # what fraction of a box transit's depth survives this exact filter (no data involved)
    t = np.arange(0, 20, dt)
    box = np.where(np.abs(t - 10) < dur / 2, -1.0, 0.0)
    out = detrend(t, 1 + 1e-3 * box, fc=fc, **cfg) - 1
    inside = np.abs(t - 10) < dur / 4
    return -out[inside].mean() / 1e-3


def cdpp(t, f, dur):
    dt = np.median(np.diff(t))
    n = max(1, int(round(dur / dt)))
    vals = []
    for idx in split_gaps(t)[0]:
        x = f[idx]
        x = x[np.isfinite(x)]
        if len(x) > 3 * n:
            vals.append(np.convolve(x - np.median(x), np.ones(n) / n, mode="valid"))
    v = np.concatenate(vals)
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def acf_at(f, lag):
    x = f[np.isfinite(f)] - np.nanmean(f)
    return np.sum(x[:-lag] * x[lag:]) / np.sum(x * x)


def choose_fc(t, f, cfg, how="snr", grid=FC_GRID, durs=DURS):
    dt = np.median(np.diff(t))
    scores = []
    for fc in grid:
        out = detrend(t, f, fc=fc, **cfg)
        if how == "snr":
            scores.append(np.mean([box_retention(cfg, fc, d, dt) / cdpp(t, out, d) for d in durs]))
        elif how == "acf":
            lag = max(1, int(round(np.median(durs) / dt)))
            scores.append(-abs(acf_at(out, lag)))
        else:
            raise ValueError(how)
    return grid[int(np.argmax(scores))], np.array(scores)
