# Validation methodologies for basis function expansion (BFE) potentials: a literature review

Scope: refereed papers in MNRAS, ApJ (incl. ApJL/ApJS), A&A, AJ, and comparable-tier venues that
present a concrete methodology for testing how well a basis function expansion (BFE) / self-consistent
field (SCF) representation reproduces the true gravitational potential, density, or dynamics of an
N-body or cosmological system. Organized by the specific aspect of the approximation each methodology
targets, since different papers stress-test different failure modes (spatial truncation, temporal
undersampling, coefficient noise, force accuracy, downstream orbit/stream fidelity).

## Summary table

| # | Reference | Validation aspect | Method | Headline result |
|---|-----------|-------------------|--------|------------------|
| 1 | Hernquist & Ostriker 1992, ApJ, 386, 375 | Spatial truncation / basis choice (foundational) | Compares SCF potential reconstructed from a finite (n,l,m) truncation against the input analytic/N-body density; basis chosen so the lowest-order term matches the target profile | Establishes that a basis whose zeroth-order term already resembles the target profile converges with far fewer terms than a generic (e.g. spherical Bessel) basis |
| 2 | Lowing, Jenkins, Eke & Frenk 2011, MNRAS, 416, 2697 | Spatial order **and** temporal cadence | Fits Hernquist–Ostriker BFE to a cosmological halo at each of ~400 snapshots (~25 Myr apart), linearly interpolates coefficients between snapshots, integrates test-particle/subhalo orbits and compares to orbits in the raw N-body potential | 12 radial × 6 angular terms reproduce the potential to a few percent; linear interpolation across ~25 Myr snapshots gives orbits that track the true (higher-cadence) potential well, but degrades for orbits with short pericentric periods |
| 3 | Lilley, Sanders, Evans & Erkal 2018, MNRAS, 476, 2092 | Basis-set choice / convergence rate | Derives a new biorthogonal potential-density basis from the Hankel transform of Laguerre polynomials, tuned to the NFW asymptotic falloff; compares the power-law decay rate of expansion coefficients (C_n00) against the classic Hernquist–Ostriker basis for the same target halo | New basis coefficients decay as n^-2 vs. n^-3 for Hernquist–Ostriker on an NFW-like target, i.e. fewer terms needed for a basis matched to the target's asymptotic slope |
| 4 | Lilley, Sanders & Evans 2018, MNRAS, 478, 1281 | Basis-set choice / generality | Extends the biorthogonal family to a two-parameter double-power-law basis and tests reconstruction accuracy across a range of inner/outer density slopes | Demonstrates which basis-family parameters best match a given halo's inner/outer slope, directly informing basis choice for non-Hernquist/NFW targets |
| 5 | Wang, Athanassoula & Mao 2020, A&A, 639, A38 | Basis geometry (spherical harmonic vs. cylindrical) | Directly compares the Hernquist–Ostriker spherical-harmonic expansion against the Vasiliev & Athanassoula cylindrical (CylSP) expansion for a family of generalized Dehnen density profiles of varying flattening | CylSP is more accurate than the spherical-harmonic expansion for nearly all inner slopes and flattenings tested — basis geometry should match the target's intrinsic symmetry (relevant if your satellite or host departs from spherical) |
| 6 | Vasiliev 2019, MNRAS, 482, 1525 (the AGAMA paper) | Spatial/radial-grid resolution; smooth vs. discrete potential | Builds multipole and "CylSpline" potential approximations directly from N-body snapshots and from analytic density laws; compares the resulting smooth potential to both the exact analytic potential and to the raw (noisy) potential implied by the N-body particles themselves | With roughly 15–30 radial grid points the smoothed expansion is *more* accurate than evaluating the potential directly from the discrete N-body particles, because the expansion suppresses Poisson (shot) noise; accuracy degrades sharply below ~15 points |
| 7 | Sanders, Lilley, Vasiliev, Evans & Erkal 2020, MNRAS, 499, 4793 ("Models of distorted and evolving dark matter haloes") | Spatial order **and** temporal cadence, jointly, for a cosmological zoom-in halo | Fits BFE (both Hernquist–Ostriker-type and spline-radial bases) to every snapshot of a Milky Way-like zoom-in simulation, explicitly varies both expansion order and snapshot spacing, and measures the resulting error in reconstructed satellite/test-particle orbits (position, energy, angular momentum) relative to orbits integrated directly in the simulation | ≥15 radial and ≥6 angular terms needed; a conventional Hernquist–Ostriker expansion is hard to beat despite testing alternative bases; time evolution over 5 Gyr introduces ~15% uncertainty in recovered satellite orbital parameters, comparable to observational/potential-model uncertainties — this paper is the closest direct precedent for your cadence experiment |
| 8 | Garavito-Camargo et al. 2019, ApJ, 884, 51 ("Hunting for the Dark Matter Wake...") | Downstream astrophysical fidelity (density/kinematic wake structure) | Compares the dark-matter wake structure (density and velocity perturbations) induced by an LMC-mass satellite as measured directly in a live N-body simulation vs. as reconstructed from a BFE representation of the same simulation | BFE reproduces both the local (near-satellite) and global (halo-wide) wake morphology seen in the raw simulation, validating BFE as a tool for extracting perturbation structure, not just bulk potential shape |
| 9 | Garavito-Camargo et al. 2021, ApJ, 919, 109 | Temporal cadence for a *live, responding* host (coupled MW+LMC wake) | Fits time-dependent BFE coefficients to a live, self-consistently responding MW halo perturbed by an infalling LMC-mass satellite across the simulation's snapshot cadence, and evaluates how well the interpolated, time-varying BFE reproduces the halo's asymmetric, evolving structure at arbitrary intermediate times | Demonstrates BFE can track a genuinely time-evolving (not just orbiting-test-particle) host response, directly relevant if you want to eventually let the host itself respond rather than stay static |
| 10 | Petersen, Weinberg & Katz 2022, MNRAS, 510, 6201 (the EXP code paper) | Force accuracy vs. direct summation; coefficient noise / adaptive basis selection | Runs matched N-body integrations using (a) BFE-derived forces (adaptive, empirically-optimized basis) and (b) direct-summation/tree forces on the same initial conditions; compares per-particle force errors and energy conservation; also describes signal-to-noise-based truncation of noisy high-order coefficients | Median force error of 0.52% for the BFE vs. 1.02% for direct summation on the same test problem — BFE forces can be *more* accurate than direct summation because the expansion suppresses discreteness noise, provided the basis is adapted to the mass distribution |
| 11 | Arora, Sanderson, Regan, Garavito-Camargo, Bregou, Panithanpaisal, Wetzel, Cunningham, Loebman, Dropulic & Shipp 2024, ApJ, 977, 23 ("Efficient and accurate force replay...") | Temporal cadence, directly, in a cosmological-baryonic (FIRE-2) simulation; force- and orbit-level error budgets | Fits low-order BFE (spherical harmonics for the halo, azimuthal harmonics for the disk) to each snapshot of a FIRE-2 Milky Way-mass galaxy at ~25 Myr cadence over 4 Gyr, linearly interpolates forces between snapshots, and separately repeats the experiment at 200 Myr cadence; measures force-reconstruction error against the simulation's true forces and orbit-integration error (position, energy, angular momentum) over multiple orbital periods | At 25 Myr cadence, 95% of particles (outside self-gravitating subhalos) have force errors ≤4%; most orbits have position errors ≤10% for 2–3 orbital periods; after 4 Gyr, 43%/70% of orbits retain energy/angular-momentum errors within 10%; central/high-frequency orbits show the largest errors; integrals of motion remain robust even at 200 Myr cadence even though absolute positions do not — this is the paper methodologically closest to what you're planning, including its explicit framing around satellite disruption |
| 12 | Petersen, Roule, Fouvry, Pichon & Tep 2024, MNRAS, 530, 4378 (LinearResponse.jl) | Cross-validation against independent theory (not N-body) | Predicts the linear dynamical response of self-gravitating spheres/discs (Plummer model, isochrone, Mestel disc) analytically via linear response theory in the same BFE basis used by EXP, then compares to N-body measurements made with the EXP code | Analytic linear-response predictions and EXP's BFE-based N-body measurements agree well, providing a validation route for BFE-based dynamics that is independent of "compare to a higher-resolution/finer-cadence N-body run" |

## Discussion by validation aspect

**Spatial truncation order and basis-set choice** (rows 1, 3, 4, 5, 6, 7). The oldest and most
standard check: fix the target density/potential, vary the number of radial and angular terms
retained, and measure the residual between the truncated BFE and the true field. Hernquist & Ostriker
(1992) established the principle that matching the lowest-order basis function to the target's overall
shape lets you truncate at low order; Lilley et al. (2018a,b) and Wang et al. (2020) show this matters
quantitatively — an NFW-like halo needs a different basis (or coordinate system) than a Hernquist
profile to converge at the same order, and Vasiliev (2019) shows that spatial/radial resolution choices
trade off against noise suppression, not just accuracy. None of this addresses your specific concern
(temporal cadence), but it sets the "spatial error floor" against which any temporal error should be
compared, since the two error sources are usually reported together in the papers below.

**Temporal cadence and coefficient interpolation** (rows 2, 7, 9, 11) — the category most directly
relevant to your question. All four papers fit BFE coefficients snapshot-by-snapshot and interpolate
(typically linearly) between them, then quantify how the resulting time-dependent potential degrades
orbit integration relative to the "true" (fine time-resolution or continuously-computed) potential.
Lowing et al. (2011) established the ~25 Myr cadence as adequate for halo-scale orbits in their
cosmological box; Sanders et al. (2020) is the first to jointly vary spatial order *and* cadence and
attribute error budget between them; Arora et al. (2024) is the most quantitative and closest analogue
to what you're planning — it explicitly tests two cadences (25 Myr vs. 200 Myr), separates force-level
from orbit-level error, and flags that orbits spending more time near the galactic center (i.e., short
dynamical times, exactly your worry) show the largest degradation. Garavito-Camargo et al. (2021) is
the only one of these that lets the *host* itself respond (rather than treating the host as fixed and
only the satellite/test-particles as evolving), which is a natural next step after your static-analytic-host
test.

**Force/energy accuracy vs. direct summation** (rows 6, 10, 11). These papers benchmark BFE-derived
forces against a "ground truth" computed by direct summation or a tree code on the same particle
distribution, independent of any time-dependence. Petersen, Weinberg & Katz (2022) is notable for
showing BFE forces can *beat* direct summation because the expansion filters Poisson noise — a useful
point if you're ever tempted to treat direct N-body forces as automatically "more true" than the BFE.

**Coefficient noise and adaptive basis selection** (row 10, discussed within Petersen et al. 2022).
Distinguishing physical coefficients from those dominated by finite-N discreteness noise is a
prerequisite for trusting high-order terms; EXP's adaptive/empirical basis construction and
signal-to-noise-based coefficient selection is the most complete refereed treatment in your target
journal list.

**Downstream astrophysical / orbit fidelity** (rows 8, 9, 11 again). Beyond generic force/orbit-position
error metrics, Garavito-Camargo et al. (2019, 2021) validate that BFE reproduces physically meaningful
derived structures (the dark-matter wake), not just pointwise force accuracy — a useful template if you
want your validation to report on stream morphology rather than only energy/angular-momentum
conservation.

**Independent cross-validation against theory** (row 12). LinearResponse.jl is a genuinely different
validation strategy: instead of comparing BFE-based dynamics to a finer-resolution/finer-cadence
N-body run (as in rows 2, 7, 9, 11), it compares to an analytic linear-response prediction computed in
the same basis. Not directly applicable to your cadence question, but worth knowing about as an
alternative validation axis if N-body ground truth is ever unavailable or too expensive.

## Most relevant precedents for your specific test

If you want directly comparable prior work before designing your own cadence experiment, prioritize:

- **Arora et al. 2024 (ApJ 977, 23)** — closest in spirit and method to your planned test; explicitly
  compares two snapshot cadences, separates force- from orbit-level error, and frames the motivation
  around satellite disruption.
- **Sanders et al. 2020 (MNRAS 499, 4793)** — the most careful joint treatment of spatial order and
  temporal cadence, with orbit-error quantification for Milky Way satellites.
- **Lowing et al. 2011 (MNRAS 416, 2697)** — the original demonstration that ~25 Myr cadence + linear
  interpolation is adequate for halo-scale (not necessarily central/short-period) orbits.

## Full references

1. Hernquist, L., & Ostriker, J. P. 1992, ApJ, 386, 375, "A Self-consistent Field Method for Galactic Dynamics"
2. Lowing, B., Jenkins, A., Eke, V., & Frenk, C. 2011, MNRAS, 416, 2697, "A halo expansion technique for approximating simulated dark matter haloes"
3. Lilley, E. J., Sanders, J. L., Evans, N. W., & Erkal, D. 2018, MNRAS, 476, 2092, "Galaxy halo expansions: a new biorthogonal family of potential-density pairs"
4. Lilley, E. J., Sanders, J. L., & Evans, N. W. 2018, MNRAS, 478, 1281, "A two-parameter family of double-power-law biorthonormal potential-density expansions"
5. Wang, Y., Athanassoula, E., & Mao, S. 2020, A&A, 639, A38, "Basis function expansions for galactic dynamics: Spherical versus cylindrical coordinates"
6. Vasiliev, E. 2019, MNRAS, 482, 1525, "AGAMA: action-based galaxy modelling architecture"
7. Sanders, J. L., Lilley, E. J., Vasiliev, E., Evans, N. W., & Erkal, D. 2020, MNRAS, 499, 4793, "Models of distorted and evolving dark matter haloes"
8. Garavito-Camargo, N., Besla, G., Laporte, C. F. P., et al. 2019, ApJ, 884, 51, "Hunting for the Dark Matter Wake Induced by the Large Magellanic Cloud"
9. Garavito-Camargo, N., Besla, G., Laporte, C. F. P., et al. 2021, ApJ, 919, 109, "Quantifying the Impact of the Large Magellanic Cloud on the Structure of the Milky Way's Dark Matter Halo Using Basis Function Expansions"
10. Petersen, M. S., Weinberg, M. D., & Katz, N. 2022, MNRAS, 510, 6201, "EXP: N-body integration using basis function expansions"
11. Arora, A., Sanderson, R. E., Regan, C., Garavito-Camargo, N., Bregou, E., Panithanpaisal, N., Wetzel, A., Cunningham, E. C., Loebman, S. R., Dropulic, A., & Shipp, N. 2024, ApJ, 977, 23, "Efficient and Accurate Force Replay in Cosmological-baryonic Simulations"
12. Petersen, M. S., Roule, M., Fouvry, J.-B., Pichon, C., & Tep, K. 2024, MNRAS, 530, 4378, "Predicting the linear response of self-gravitating stellar spheres and discs with LinearResponse.jl"

## Notes on scope and confidence

- All 12 references were confirmed via their journal landing pages (Oxford Academic for MNRAS, IOPscience
  for ApJ, A&A's own site) or NASA ADS during this search, not from memory alone, and all fall within
  MNRAS/ApJ/A&A as requested.
- I deliberately excluded a widely-referenced but harder-to-pin-down Weinberg paper on coefficient
  signal-to-noise regularization/truncation, since I could not verify its exact title/venue with
  confidence in this search session; the same coefficient-selection methodology is, however, documented
  and properly citable via Petersen, Weinberg & Katz (2022) (row 10), which is fully verified.
- I did not include preprints without a confirmed refereed venue (e.g., very recent 2025/2026 arXiv-only
  papers on BFE applications) to stay strictly within your journal-quality requirement; let me know if
  you want those added with an explicit "preprint, not yet refereed" f