r"""Mechanism test for the estimator A-vs-B divergence (panel D), with synthetic
rates of KNOWN spatial profile. Pure ground-truth; no cache, no model.

Claim under test
----------------
Estimator A (rate_variance_components, all-samples ANOVA) weights the FEM
variance by the eye density p(e). Estimator B (pipeline_one_minus_alpha, the
close-pair below_threshold intercept) conditions on Delta_e<thr, which weights
eye positions by p(e)^2 -- concentrated toward the fixation center. So A and B
integrate FEM over DIFFERENT eye distributions. Prediction: on the SAME rates,
the sign of (B - A) for 1-alpha is set by WHERE the rate's eye-sensitivity (its
FEM variance) lives relative to the center:

    * FEM concentrated centrally  -> p^2 weighting emphasizes it -> B > A.
    * FEM growing with eccentricity-> p^2 weighting de-emphasizes it -> B < A.
    * FEM spatially flat (linear) -> mild B < A (center has less eye spread).

If this holds, the opposite Allen(+)/Logan(-) median (B-A) seen on real cells is
the expected consequence of different spatial eye-sensitivity profiles, not a
bug -- and the apples-to-apples panel must use the SAME estimator on both sides.
"""
import numpy as np
from VisionCore.covariance import rate_variance_components, pipeline_one_minus_alpha

SIGMA_EYE = 0.15          # fixational spread (deg), ~realistic
THRESHOLD = 0.05
MIN_TPP = 10


def fem_profiles(e):
    """Map eye (..., 2) to a per-profile FEM drive (..., n_profiles).

    Each profile is mean-centred-ish by construction so PSTH stays separable;
    amplitudes are tuned so the TRUE (A) 1-alpha lands mid-range and comparable.
    """
    x, y = e[..., 0], e[..., 1]
    r2 = x ** 2 + y ** 2
    eccentric = 9.0 * r2                                  # grows with eccentricity
    central = 2.2 * np.exp(-r2 / (2 * (0.6 * SIGMA_EYE) ** 2))  # peaked at center
    linear = 3.2 * x                                      # spatially flat sensitivity
    return np.stack([eccentric, central, linear], axis=-1)


PROFILE_NAMES = ["eccentric (FEM grows with |e|)",
                 "central   (FEM peaked at center)",
                 "linear    (FEM spatially flat)"]


def run(n_trials=160, n_phases=90, sigma_psth=1.0, seed=0):
    rng = np.random.default_rng(seed)
    eye = rng.normal(0.0, SIGMA_EYE, size=(n_trials, n_phases, 2))
    fem = fem_profiles(eye)                               # (trials, phases, 3)
    n_cells = fem.shape[-1]
    psth = rng.normal(0.0, sigma_psth, size=(n_phases, n_cells))
    rate = psth[None, :, :] + fem                         # (trials, phases, cells)

    A = np.array([
        rate_variance_components(rate[:, :, c], min_trials_per_phase=MIN_TPP)
        ["one_minus_alpha"] for c in range(n_cells)
    ])
    Bout = pipeline_one_minus_alpha(rate, eye, threshold=THRESHOLD,
                                    min_trials_per_phase=MIN_TPP, device="cpu")
    B = Bout["one_minus_alpha"]
    return A, B


def main():
    print(f"Synthetic mechanism test (sigma_eye={SIGMA_EYE} deg, thr={THRESHOLD} deg)")
    print("Averaged over 8 seeds; A = all-samples ANOVA, B = close-pair pipeline.\n")
    n_cells = 3
    As, Bs = [], []
    for s in range(8):
        A, B = run(seed=s)
        As.append(A); Bs.append(B)
    A = np.nanmean(As, axis=0); B = np.nanmean(Bs, axis=0)
    print(f"{'profile':36s} {'A (true)':>9s} {'B (pipe)':>9s} {'B - A':>8s}")
    for c in range(n_cells):
        print(f"{PROFILE_NAMES[c]:36s} {A[c]:9.3f} {B[c]:9.3f} {B[c]-A[c]:+8.3f}")
    print("\nPrediction: central -> B>A (+), eccentric -> B<A (-), linear -> mild -.")


if __name__ == "__main__":
    main()
