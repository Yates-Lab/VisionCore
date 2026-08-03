# ssi_figure_v4 legend and methods draft

This draft is written to match the July 29 manuscript tone and terminology.
It assumes the main Methods already contain the experimental preparation, eye
tracking, stimulus presentation, spike sorting, covariance decomposition, and
digital twin model architecture/training sections.

## Figure Legend Draft

Figure 4: Single-spike information and contour-relative FEMs. (A) Schematic of
the model-based single-spike information (SSI) analysis. A trained digital twin
was used to predict the spatial activation map of each model unit as a natural
image was translated by a measured FEM trajectory. SSI quantifies how far the
resulting activation map departs from a spatially uniform response, in
bits/spike. In the example shown, the FEM-jittered movie produced a more
spatially structured response map than the counterfactually stabilized movie
(0.14 vs. 0.10 bits/spike). (B) SSI change relative to a cell-matched
stabilized baseline as a function of total FEM path length for units split by
preferred spatial frequency. Low-SF units (blue; 71 units, 7100 unit-image
pairs) gained progressively with path length, whereas high-SF units (orange;
29 units, 2900 unit-image pairs) showed little additional benefit at long path
lengths. Open markers indicate drift-only traces and filled markers indicate
traces containing microsaccades. (C) Local image structure was estimated around
each gaze position. A Sobel structure-tensor analysis of a gaze-centered
natural-image patch defined the local contour axis and an orientation coherence
score. Example patches illustrate the coherence bins used in later panels.
(D) The effect of path length depended on whether unit tuning was aligned with
the local contour. Among contour-aligned unit-image pairs, low-SF units
(57 units, 977 pairs) retained a positive path-length dependence, whereas
high-SF aligned units (22 units, 356 pairs) declined with longer paths,
especially for microsaccade-containing traces. (E) For high-SF aligned units,
SSI depended on the contour-relative direction of the eye movement. The x-axis
shows component RMS excursion of drift-only trajectories projected either
normal to the local contour (across, solid) or parallel to it (along, dashed).
Large across-contour excursions produced a stronger loss of SSI than
along-contour excursions; in the last displayed bin, the across-minus-along
contrast was -5.1 percentage points (image bootstrap p = 0.0004). The far tail
of the RMS distribution (>3.8 arcmin) is omitted from the display. (F) Real FEM
position spread was anisotropic around local contours. For each reviewed
BackImage fixation window, eye positions were projected onto axes at different
angles relative to the local contour, and RMS spread was averaged within local
edge-coherence bins. As coherence increased, spread became larger parallel to
the contour and smaller in the orthogonal direction. Dashed horizontal lines
show the orientation-scrambled reference. (G) The observed contour-relative
orientation of real FEMs was compared with a random-rotation null. Positive
values indicate that the real pairing between eye trajectory and local image
axis predicted higher model SSI than randomly rotating the same trajectories
relative to the same local contours. The advantage increased with coherence
for aligned high-SF units and was largest in the highest coherence bin
(0.155 percentage points, 95% CI [0.044, 0.265]). All high-SF and low-SF
populations showed weaker or less selective effects. Open markers indicate
points whose 95% CI includes zero. (H) The relationship between local edge
coherence and edge-following behavior depended on the scale at which local
image structure was measured. For each patch radius, the slope of the
edge-following alignment index versus local coherence was fit over windows
with coherence > 0.3. The slope rose from small patches, peaked near the
foveal scale, and remained positive over larger patches, indicating that FEM
anisotropy is most strongly coupled to local contour structure measured on
approximately degree-scale image neighborhoods. Error bars denote 95% CIs.

## Methods Draft

### Single-spike information analysis

We quantified how retinal image motion altered the spatial specificity of
model responses using the trained digital twin described above. For each unit,
image, FEM trajectory, and time bin, the model produced a two-dimensional
activation map over spatial position. Single-spike information (SSI) was
computed from each nonnegative activation map as the divergence of that map
from a uniform spatial response. Let \(r_t(x)\) denote the predicted response
at position \(x\) in time bin \(t\), and let
\(\bar r_t = \langle r_t(x) \rangle_x\). We computed

\[
SSI_t = \left\langle
\frac{r_t(x)}{\bar r_t}
\log_2 \frac{r_t(x)}{\bar r_t}
\right\rangle_x .
\]

This quantity is reported in bits/spike. It is insensitive to an overall
multiplicative change in firing rate and instead measures how selectively the
activity identifies spatial position. A broad activation map has low SSI,
whereas a localized activation map has high SSI.

Population SSI was computed by pooling over selected units, images, and movie
rows using the model's expected spike count as the weight. For a set of movie
rows \(m\) and units \(u\), we computed

\[
SSI_{\mathrm{pop}} =
\frac{\sum_{m,u} \hat n_{m,u} SSI_{m,u}}
{\sum_{m,u} \hat n_{m,u}},
\]

where \(\hat n_{m,u}\) is the predicted expected spike count. Thus, the
population value is the amount of spatial information carried per predicted
spike by the selected population. All plotted SSI effects are expressed as a
percent change relative to a matched stabilized baseline,

\[
100 \times
\frac{SSI_{\mathrm{FEM}} - SSI_{\mathrm{stable}}}
{SSI_{\mathrm{stable}}}.
\]

### Natural-image FEM movie bank

The SSI analyses used a BackImage movie bank generated from real image patches
and measured fixational eye movements. We selected 100 BackImage image windows
with valid local image measurements and local orientation coherence >= 0.20.
These were crossed with 1000 real FEM snippets sampled from the reviewed
BackImage fixation windows, yielding 100,000 image-by-trajectory movies. The
trace bank contained both drift-only snippets and snippets containing
microsaccades; 200 of the selected traces contained microsaccades. Traces were
sampled over the empirical range of path lengths and were restricted to path
lengths <= 350 arcmin. Movies were evaluated for 40 samples at 120 Hz using the
RR100 model population view.

For each selected image we also generated a counterfactually stabilized movie.
In this condition the same BackImage patch was held fixed for the same number
of time bins with zero retinal displacement. The stabilized responses were
scored with the same model, spatial map, and SSI procedure as the FEM-jittered
movies. For every plotted movement bin, the stabilized baseline was matched to
the same image composition and the same selected cell or unit-image population.
This cell-matched baseline prevents differences in image identity or unit
selection from being interpreted as movement effects.

### Local image structure and unit-image selections

Local image structure was measured in gaze-centered BackImage patches. Unless
otherwise noted, the patch radius was 1 degree. Patches were excluded if less
than 98% of the patch fell inside the image or if more than 5% of the patch was
background. The local contour axis was estimated from the Sobel structure
tensor. If \(g_x\) and \(g_y\) are horizontal and vertical image gradients, we
computed \(J_{xx}=\langle g_x^2\rangle\), \(J_{yy}=\langle g_y^2\rangle\), and
\(J_{xy}=\langle g_x g_y\rangle\) within the patch. The orientation coherence
was

\[
\frac{\sqrt{(J_{xx}-J_{yy})^2 + 4J_{xy}^2}}{J_{xx}+J_{yy}},
\]

and the local edge axis was defined as the axis orthogonal to the dominant
gradient axis. Image-derived axes were reported in gaze coordinates, with
positive x rightward and positive y upward.

Units were divided by the spatial-frequency metric from the model tuning
analysis. Low-SF units had `sf_split_metric < 0.5` cycles/degree, and high-SF
units had `sf_split_metric >= 0.5` cycles/degree. For contour-aligned
analyses, we further required a valid preferred orientation and orientation
selectivity index >= 0.05. Unit-image pairs were classified by the acute
axis-angle difference between the unit's preferred orientation and the local
image contour axis. Aligned pairs had a difference <= 15 degrees. Orthogonal
pairs, used in supporting diagnostics, had a difference >= 67.5 degrees, and
oblique pairs fell between these cutoffs. Because the contour axis varies from
image to image, the aligned population is naturally defined as a set of
unit-image pairs rather than as a fixed set of units alone.

### Path-length and contour-relative model dose curves

For the path-length analyses, total FEM path length was computed as the sum of
Euclidean sample-to-sample eye displacements, converted to arcmin. Traces were
separated into drift-only and microsaccade-containing contexts. Drift-only
traces were divided into eight equal-count path bins; microsaccade-containing
traces were divided into five equal-count path bins. A zero-motion stabilized
point was plotted at path length zero. Panels B and D show the cell-baselined
SSI change in each path bin for low- and high-SF populations, with and without
the unit-contour alignment restriction.

For contour-relative analyses, each two-dimensional FEM trajectory was
projected onto two axes defined by the local image: the contour-parallel axis
and the contour-normal axis. For a trajectory \(e_t=(x_t,y_t)\), contour axis
\(u\), and normal axis \(v\), we computed both accumulated component path and
position-spread metrics. Component path was the sum of the absolute projected
sample-to-sample displacements,

\[
P_u = 60 \sum_t |(e_{t+1}-e_t) \cdot u|,
\qquad
P_v = 60 \sum_t |(e_{t+1}-e_t) \cdot v|.
\]

Component RMS excursion was the standard deviation of centered projected eye
position,

\[
R_u = 60 \sqrt{\left\langle ((e_t-\bar e)\cdot u)^2 \right\rangle_t},
\qquad
R_v = 60 \sqrt{\left\langle ((e_t-\bar e)\cdot v)^2 \right\rangle_t}.
\]

The main contour-relative SSI panel uses component RMS excursion because the
behavior-model bridge showed that real behavior was better described by
position spread than by accumulated path. The RMS dose curves used drift-only
movies and the aligned high-SF unit-image population. Bins were constructed
from pooled contour-normal and contour-parallel RMS values so that the two
components were compared at matched dose ranges. Body bins were defined by
pooled quantiles, with an additional tail bin; the far tail above 3.8 arcmin
was omitted from the displayed panel. The reported across-minus-along contrast
was computed in the last displayed bin by bootstrapping over images.

### Statistical uncertainty for model SSI curves

For each movement bin and selected population, the point estimate was computed
from expected-spike-weighted sums over all contributing unit-image-trajectory
rows. Uncertainty was estimated by resampling images with replacement. For
each bootstrap sample, the numerator and denominator of the moving SSI and the
cell-matched stabilized SSI were recomputed from the resampled image totals,
and the percent change was recomputed from those ratios. Unless otherwise
specified, model SSI confidence intervals used 10,000 image bootstrap
resamples. P-values for displayed across-versus-along contrasts were computed
from the bootstrap distribution of the paired residual difference between the
two contour-relative components.

### Real FEM anisotropy around local contours

To ask whether the animal's natural eye movements were aligned with local
image structure, we analyzed reviewed BackImage fixation windows from 30
recording sessions. The primary window set contained 11,749 windows after
requiring valid gaze samples, valid local image features, and the patch
contamination criteria described above. Windows came from the mid- and
late-fixation phases of the BackImage condition. For each window, the local
contour axis was measured at the mean gaze position, and the eye-position
cloud within the window was centered and projected onto axes spanning all
relative angles from parallel to orthogonal to the contour.

Position spread along each relative axis was quantified as RMS projected eye
position in arcmin. Profiles were grouped into four local edge-coherence bins:
0-0.2, 0.2-0.5, 0.5-0.8, and 0.8-1. Source bins were merged by weighting RMS
values by the number of contributing windows. The orientation-scrambled
reference was computed by replacing the relationship between the measured eye
positions and the local contour axis with randomized relative orientations,
while preserving the empirical eye-position clouds and coherence labels. This
control estimates the spread expected if local image orientation carried no
information about the direction of the measured FEM cloud.

### Behavior-model bridge and random-rotation control

The behavior-model bridge asked whether the contour-relative structure of real
FEMs was beneficial for the same model SSI curves measured in the movie bank.
For each reviewed BackImage fixation window, we extracted the central
40-sample snippet, corresponding to 0.325 s, and projected the centered eye
positions onto the local contour-parallel and contour-normal axes. We then
converted the observed component RMS values into predicted SSI changes by
piecewise-linear interpolation through the model dose curves. Predictions
outside the model curve range were marked invalid and excluded from the
corresponding summaries.

To construct the random-rotation null, each behavior snippet was rotated by an
independent angle drawn uniformly from [0, pi), while keeping the same local
image coherence bin and the same eye-position cloud. This preserves the
movement amplitude and temporal structure of each real trace but breaks the
specific alignment between that trace and the local image contour. We generated
256 random rotations per behavior window. The plotted match advantage is the
observed prediction minus the mean random-rotation prediction, in percentage
points of SSI change relative to the model cell baseline. The displayed panel
uses the component-mean marginal prediction for RMS excursion, defined as the
average of the contour-normal and contour-parallel one-dimensional marginal
predictions. This is not a full two-dimensional SSI surface; it is a compact
summary of how the observed contour-relative distribution samples the two
model dose axes. Confidence intervals were computed by bootstrap resampling of
paired window predictions, and open markers indicate bins in which the 95%
confidence interval included zero.

### Patch-radius sensitivity of local contour measurements

Finally, we tested whether the relationship between local image coherence and
edge-following behavior depended on the spatial scale used to define local
image structure. We recomputed Sobel structure-tensor edge axes and coherence
values for gaze-centered patches with radii from 0.25 to 3.0 degrees. To make
this sweep efficient, full-image gradient fields were computed once per trial,
and gradient products were averaged over each gaze-centered patch with
integral images. The same image-contamination criteria were applied at every
radius.

For each window and radius, we estimated the principal axis of the
eye-position covariance and compared it with the local edge axis. The
edge-following alignment index was

\[
\cos(2\Delta\theta),
\]

where \(\Delta\theta\) is the circular axis-angle difference between the drift
axis and the local edge axis. Values near 1 indicate motion parallel to the
local contour, values near -1 indicate motion normal to the contour, and
values near 0 indicate no consistent axis relationship. For each patch radius,
we fit a window-level ordinary least-squares regression of this alignment
index against local orientation coherence, restricted to windows with
coherence > 0.3. Panel H plots the fitted slope as a function of patch radius.
Confidence intervals are the 95% intervals from the regression standard error
using the appropriate Student-t critical value.
