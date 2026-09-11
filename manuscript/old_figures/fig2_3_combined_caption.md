Figure 2: Fixational eye movements dominate single-neuron and shared
variability, and contribute a compact population covariance component aligned
with the stimulus-driven subspace.

(A) Compact eye-matching example. Top: horizontal eye position during two
representative trials of the same frozen-stimulus sequence. Middle: absolute
eye-position difference, |Delta e(t)|. Bottom: per-bin firing rate of an
example unit on the same two trials. Gray windows mark epochs with different
eye-trajectory separations; spike-rate agreement changes with the eye-match
state, illustrating the conditioning used to estimate eye-dependent rate
variance.

(B) Distribution of 1 - alpha across neurons, stacked by subject at the 8 ms
counting window. Monkey A: n = 1010 neurons, median 0.786 [IQR 0.670, 0.861].
Monkey L: n = 257, median 0.684 [IQR 0.541, 0.792]. Both distributions are
skewed toward 1, indicating that most trial-to-trial rate variance is
attributable to fixational eye movements.

(C) Population Fano factor vs. counting window, by subject (Monkey A: blue;
Monkey L: green). Dashed lines/open markers show raw responses; solid
lines/filled markers show FEM-corrected responses. Error bars are
session-clustered bootstrap 95% CIs; stars below each window give the
per-subject corrected-vs.-uncorrected significance. Across windows and
subjects, FEM correction reduces the population Fano factor toward or below
the Poisson reference (gray dotted line), with larger reductions at longer
counting windows.

(D) Compact covariance decomposition for an example Monkey A session at the
8 ms counting window. Top: the conventional decomposition
Sigma_total = Sigma_PSTH + Sigma_int^uncorr attributes all covariance not
captured by the PSTH to an uncorrected internal component. Bottom: that
uncorrected component is split into an eye-movement-dependent covariance
Sigma_FEM and a residual corrected covariance Sigma_int^corr. The
off-diagonal structure in Sigma_int^uncorr migrates largely into Sigma_FEM,
leaving the residual covariance much closer to diagonal.

(E) Mean Fisher-z noise correlation vs. counting window, per subject. Dashed
lines/open markers show uncorrected responses; solid lines/filled markers show
FEM-corrected responses. Error bars are session-clustered bootstrap 95% CIs,
with subjects dodged in x for clarity. Uncorrected shared variability grows
with counting window, whereas FEM-corrected correlations remain near zero at
short windows and only begin to emerge at the longest window.

(F) Cumulative fraction of own variance carried by the leading eigenvalues of
Sigma_PSTH (solid) and Sigma_FEM (dashed), with each session normalized to its
own total variance. Thin traces are sessions; bold lines are per-subject
medians. FEM spectra saturate within roughly three dimensions in both
subjects, while PSTH spectra are more distributed, especially in Monkey A.

(G) Participation ratio of Sigma_FEM vs. Sigma_PSTH, one point per session.
The legend gives an exact one-sided sign test for PSTH PR > FEM PR per
subject. Monkey A: all 11/11 sessions fall above the diagonal, p = 5e-4;
PR_FEM median 2.59 [IQR 2.38, 3.36] vs. PR_PSTH 4.93 [4.10, 5.41]. Monkey L:
9/13 sessions fall above the diagonal, n.s.; PR_FEM 3.14 [1.98, 3.34] vs.
PR_PSTH 2.86 [2.39, 3.70]. FEM variance therefore occupies roughly two to
three population modes, consistent with the translational degrees of freedom
of gaze.

(H) Subspace alignment between Sigma_PSTH and Sigma_FEM. X is the fraction of
PSTH variance lying in the leading FEM subspace; Y is the fraction of FEM
variance lying in the leading PSTH subspace. Per-subject shuffle clouds
(eye traces permuted across trials, same PSTH subspace) and their means (x)
sit near chance levels; observed sessions sit well above the shuffle cloud,
with red outlines marking sessions jointly significant at p < 0.01 for both X
and Y. Observed means: Monkey A X = 0.61, Y = 0.64 (null 0.31, 0.25); Monkey L
X = 0.77, Y = 0.75 (null 0.43, 0.48). FEM-driven modulations are largely
confined to the dimensions used to encode the stimulus, supporting an
information-limiting correlation geometry rather than orthogonal shared noise.
