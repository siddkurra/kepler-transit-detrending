# Reading notes

What I read before starting the experiments, on how people actually detrend light curves. Nobody uses a plain RC high-pass for this, but most methods still estimate a slow trend and divide it out. They mainly differ in how they avoid eating the transit.

Kepler's own pipeline splits it in two. The PDC step regresses out instrument trends shared across the CCD and deliberately leaves stellar variability in, which is why I use SAP flux: PDCSAP for Kepler-17 still has all the spots. The planet search (TPS) doesn't really detrend at all. It estimates the noise at each time scale with wavelets, whitens, and runs a matched filter. That's the textbook detector for a pulse in colored noise, and my high-pass is basically a hand-picked version of the whitening step.

Savitzky-Golay (what lightkurve's flatten uses) and splines fit smooth curves in a window. They're linear phase, so they don't skew the transit the way a causal RC does. Wotan's biweight came out best in the wotan paper's comparison on known planets. It's a robust estimator, so in-transit points get down-weighted without needing a mask. They suggest a window of ~3× the transit duration.

Also looked at CoFiAM from the exomoon search, which fits cosines and picks how many by minimizing the residual autocorrelation, i.e. a cutoff chosen from the data. GPs model the spots as a quasi-periodic process. They're probably the most principled option but slow, and I didn't try them. And TLS swaps BLS's box for a limb-darkened shape.

So the masking I'd added was already standard. What the better methods had that I didn't: robustness, and a cutoff picked from the data.

One calculation I wanted to check: knocking the spots (~0.17/day) down to 10% while keeping the transit band (~10/day) at 99% needs the two frequencies to be at least (99/0.0203)^(1/2N) apart for an Nth order Butterworth. For N=1 that's 70, and I only have 63, which is the RC problem in one number. For N=2 it's 8.4, so a cutoff between ~0.5 and 3.9/day should work. A box transit isn't really "at" 10/day though, since it has a lot of low-frequency content. How this played out is in LOG.md.

## Stuff I read

- Kepler PDC: https://arxiv.org/abs/1203.1382 and https://arxiv.org/abs/1203.1383
- Kepler TPS / stellar noise: https://ui.adsabs.harvard.edu/abs/2002ApJ...575..493J
- K2 splines: https://arxiv.org/abs/1408.3853
- wotan: https://github.com/hippke/wotan (paper: https://arxiv.org/abs/1906.00966)
- CoFiAM / exomoon search: https://arxiv.org/abs/1301.1853
- K2SC (GPs): https://arxiv.org/abs/1603.09167
- celerite: https://arxiv.org/abs/1703.09710
- TLS: https://github.com/hippke/tls (paper: https://arxiv.org/abs/1901.02015)
