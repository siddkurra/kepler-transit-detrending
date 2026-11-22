## Setup

Three Kepler stars, quarters 1-4, SAP flux: Kepler-17 (spotted, 2% planet), Kepler-63 (very spotted, ~5 day rotation, 0.4% planet) and HAT-P-7 (quiet, 0.65% planet). Using three is so I don't tune to one light curve.

For each filter, I cut out the real planet, inject trapezoid transits at 150/300/600/1200 ppm (periods 1-10 d, durations from P for a sun-like star), detrend, and run BLS. It only counts if the top peak is at the injected period. The threshold is the larger of SDE 7 and the max SDE over scrambled nulls: the same light curve chopped into 6 day chunks, shuffled and flipped upside down, so there's nothing real to find. That keeps every filter at about the same false alarm rate. Injection seed 1 was for tuning, seed 2 only for the final run.

## exp01: baseline on the new setup

First made sure it reproduces the class numbers. It does: rc at fc=1 keeps 76.9% of the depth vs 76.5% before, and the difference is 70 extra points from a bug fix. The bug: the outlier clip used the MAD of (flux - running median), and on Kepler-63 the flux is smooth enough that the running median often *is* the point, so the MAD went to 0 and it threw out 80% of the data. Switched to the MAD of first differences.

The pilot run showed masking makes false alarms worse. Pass 1 finds some noise dip, the mask protects it, and pass 2 finds it again with a higher SDE (null max went from ~6 to 8-10). With the calibrated threshold, masking finds exactly as many planets as no masking. The class best gets ~41% on the two spotted stars and almost nothing under 600 ppm.

## exp02: butterworth

The prediction from RESEARCH.md was that 1st order can't knock the spots to 10% while keeping the transit band at 99%, and 2nd order can for fc ~0.5-3.9/day. I swept order 1/2/4, fc 0.3-3, one-way and two-way.

On the spotted stars, recall went from ~0.46 (class) to 0.88 on Kepler-17 and 1.00 on Kepler-63, at fc 2-3, two-way. But order only mattered for shape. At fc=3 even 1st order two-way tied 4th order. Recall was *negatively* correlated with depth kept (r about -0.8) and tracked depth kept / CDPP instead. So "keep the transit intact," which is what we optimized in class, is the wrong objective for finding planets. It only matters for measuring one you've already found.

One-way 4th order actually flips the transit (negative depth) from ringing, so I dropped one-way butterworth.

## exp03: robust rc

Idea: make the filter ignore dips by itself. If the flux falls more than k·σ below the trend, hold the trend. It worked great on HAT-P-7 (recall 1.00, lowest nulls of anything) and completely failed on the spotted stars (0-0.17). The reason is lag. A first order low-pass trails a slope by slope × RC, which is ~1000 ppm on Kepler-63's spots, way over k·σ, so it held 44% of the time. You can't detect dips against a reference that's lagging.

Second try: decide what's a dip from a zero-lag (two-way) residual, clip the low points, fill them, refit. That kept the real planets at ~100% depth, but didn't change recall for butterworth and hurt it for rc (0.92 → 0.63 on Kepler-63). Shallow transits are below the per-point noise, so none of their points get clipped. So there are really two detrends: an aggressive one for searching, and a masked/clipped one for characterizing what you found.

## exp04: gap edges

Expected filter start-up transients after gaps, so I added odd-reflection padding. It fixes a synthetic ramp but not the real data. On HAT-P-7 the noise is normal for the first 6 hours after a gap, then jumps ~9× between 6 and 12 hours in. That's something in the data (maybe thermal settling after downlinks, not checked), not the filter. At fc ~3 there's no edge penalty anyway, so padding stays optional.

## exp05: wotan biweight

Windows 0.2-1 d vs butterworth two-way at fc 2-3, paired on the same injections. Butterworth found more on Kepler-17 (11-0 vs the 0.5 d window, p=0.001; 7-0 vs 0.2 d, p=0.016) and Kepler-63 (8-0 vs 0.5 d, p=0.008), and never missed one the biweight got. The biweight does keep the real planets at 100% depth, same pattern as exp03. Its rule-of-thumb window (3× the transit duration) was too long for spotted stars, and its best window was the shortest one I tried.

## exp06: picking the cutoff automatically

Since recall tracks depth kept / CDPP, pick fc to maximize that: compute depth kept by pushing a box through the filter, and measure CDPP on the data. No injections involved. It picks fc 1.5-2 and matches the hand-tuned results on all three stars. Recall is flat from fc 1.5 to 3 anyway. The CoFiAM-style autocorrelation rule was jumpier, and on HAT-P-7 it picked fc=7, which wipes out the real planet.

## exp07: the search side

Same detrended data, different search: BLS, a whitened matched filter (TPS-style: Welch PSD, whiten, correlate with a box, fold), and TLS. My first whitening made things worse (Kepler-17 0.88 → 0.71). I was whitening a high-passed series, so dividing by its tiny PSD below fc undid the high-pass. Flooring the PSD at the white-noise level fixed it, and then all three searches tied BLS within a planet.

Whitening alone, with no high-pass, matched the full pipeline on HAT-P-7 but lost ~20 planets on each spotted star. A single global PSD doesn't fit spots that evolve over time. TPS uses wavelets, probably for this reason.

Also found that paired comparisons were silently matching 0 injections across experiments (imprecise float parsing, then mismatched keys). Fixed it, and it now raises an error instead of returning nothing. Earlier comparisons weren't affected.

## exp08: held-out run

Written before running seed 2. Configs: raw, the class rc (plain and masked), biweight 0.5 and 0.2 d, butterworth fc=2, and butterworth with the automatic cutoff. 60 injections and 30 nulls per star per config. Prediction: the final config beats the class filter on both spotted stars, beats the biweight on Kepler-17, and ties on HAT-P-7.

All of that held. Final pipeline vs class filter: Kepler-17 0.85 vs 0.40 (27-0), Kepler-63 1.00 vs 0.37 (38-0), HAT-P-7 tie (0.97). Vs the 0.5 d biweight it was 31-0 and 20-0. Vs the tuned 0.2 d biweight it won on Kepler-17 (16-0) and wasn't significant on Kepler-63 (4-0). All of these survive a Holm correction except that last one. Tuning vs held-out recall barely moved (0.88 → 0.85, 1.00 → 1.00, 0.92 → 0.97), so it doesn't look overfit.

What's still weak: only three stars and one year of data, simple trapezoid injections, long transits would get eaten at these cutoffs, and Kepler-63's null tail is heavier with the final filter (max SDE 11). That last one I don't understand yet.
