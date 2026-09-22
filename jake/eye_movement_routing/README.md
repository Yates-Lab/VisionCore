# Eye-movement routing exploration

This analysis asks whether the same natural-image/eye-trajectory movies alter
spatial information differently in two populations defined by their tuning,
and whether those differences appear in a model localization decoder. It uses
the manuscript's selected rank-one checkpoint and corrected Figure 4 replay.
It writes a separate exploratory artifact; it does not change Figure 4.

Run from the repository root:

```sh
conda run --no-capture-output -n yatesfv python jake/eye_movement_routing/analyze.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n yatesfv python jake/eye_movement_routing/decode.py
conda run --no-capture-output -n yatesfv python jake/eye_movement_routing/compose.py
conda run --no-capture-output -n yatesfv python jake/eye_movement_routing/audit.py
```

Outputs are under `outputs/eye_movement_routing_20260914/`. The second command
reuses completed spatial-map replays and verifies their unit and checkpoint
identities. `--decode-only` reuses all maps without loading the model. PNG,
PDF, and SVG versions accompany the numerical CSV/NPZ/JSON artifacts.

## Population comparison

- Use the 145 strictly validated model tuning fits, with their original
  checkpoint unit identities. The lower and upper thirds of preferred TF/SF
  define 48 units each. The middle 49 units are excluded. These are two tuning
  groups, not discovered biological cell classes. Temporal tuning is inferred
  from the model, not independently measured in the recorded neurons.
- Partition the 40 image patches into 20 selection and 20 evaluation patches
  using seed 20260914. No response outcome enters the population definition.
- On selection patches, average each unit's motion-condition passband power.
  Order the 200 trajectories by the difference between the two populations'
  mean log powers (higher-speed group minus lower-speed group). Put the same
  40 trajectories into each quintile for both populations.
- Evaluate rate changes and the original Figure 4 spatial SSI on evaluation
  patches, relative to the same patch stabilized. Rate is averaged over images
  and trajectories before calculating each unit's percentage change; SSI is
  pooled by expected spike count first. Display the median unit effect.
- Confidence bands resample image patches and trajectories within quintiles,
  keeping unit groups fixed and paired across groups. They measure stimulus
  sampling uncertainty, not independent replication across neurons.
- A separate control pairs trajectories within path-length deciles, contrasting
  the outer thirds of spectral balance with at most 15% path-length mismatch.
  The control reports the group-by-movement interaction, including null or
  opposite results. The main quintiles are not matched for movement size.

The group rule and ordering were explored in this task. Evaluation images did
not determine trajectory ordering, but this is exploratory evidence rather
than an untouched confirmatory test. Repeated image content across patches
would further limit the image-resampling interpretation.

## Spatial localization decoder

Use 10 prespecified evaluation patches and four evenly spaced trajectories
from each of the five quintiles. Replay all 200 pairs plus one stabilized
movie per image through the selected model. Check every replay's mean firing
rate against the existing Figure 4 cache before accepting it.

At each of a central 7×7 grid of exact translated-readout positions, retain
predicted counts in all 60 native 240-Hz bins over 250 ms. Grid spacing is four input pixels (approximately
6.4 arcmin). These templates represent a surrogate population's responses as
the known image is translated relative to its receptive fields. No spatial
interpolation is used. The readouts are translated model features, not a
simultaneously recorded population with that retinotopic arrangement.

Draw independent Poisson counts for each unit and time bin. With a uniform position prior,
the ideal observer computes each candidate's log likelihood, summed over units
and time bins, as `sum(count * log(expected_count) - expected_count)`. The factorial term cancels.
Report localization accuracy, error in grid steps, and mean log posterior of
the true position plus `log2(49)` (Monte Carlo mutual information in bits per
250-ms observation). The decoder knows the image and trajectory; it is an
available-information calculation, not a fixed downstream circuit.

The initial count-only decoder collapsed all 250 ms for each unit. It showed
losses in matched-spike localization for the larger, faster-biased movements.
This differs from the moment-by-moment spatial selectivity quantified by SSI,
so the final assay additionally retains spike timing. Count-only results are
preserved in `count_only_decoding_*` as a distinct readout control. This
extension was motivated by that observed discrepancy and is exploratory.
Use `--readout count_only` to reproduce the collapsed-count assay (it writes
the ordinary decoding filenames, so preserve the desired other outputs first).

Repeat after scaling each population/condition's templates to ten expected
total spikes averaged over candidate locations. This removes a global spike
count advantage while retaining spatial structure and relative unit rates.
It does not force the same spike count at every location. Runtime checks cover
identical templates (chance/zero information), separated templates, and
invariance to global gain under the matched-count analysis.

This decoder assumes independent Poisson variability. It does not establish
decoding performance from real spikes, learned image reconstruction, or an
attentional mechanism. A population-selective information effect would
motivate testing whether attention-linked changes in eye trajectories produce
such reweighting in recorded population responses.
