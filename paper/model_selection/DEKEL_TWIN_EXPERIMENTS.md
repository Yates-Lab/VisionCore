# Dekel 240 Hz digital-twin experiments

This is the durable lab notebook for the feed-forward digital-twin branch.
Values below are checkpoint validation BPS unless explicitly labeled as a
deterministic full-split score. Checkpoint validation uses a rotating 10% sample
of the validation split, so final selection must use `evaluate_dekel_split.py`
on the same complete split for every candidate.

## Fixed architecture and data path

- Native stimulus rate: 240 Hz.
- Visual history: 60 frames (250 ms), center-cropped to 35 x 35.
- Supervision: causal adjacent-bin spike-count sums at 120 Hz; stimulus and
  behavior remain on the native time axis.
- Vision core: full-history 3-D convolution, interleaved sign split, pair-aligned
  GroupNorm/ReLU, three large Hamming-windowed spatial convolutions, L5 pooling,
  and a 9 x 9 multiscale scaffold.
- Capacity-matched width: 14 temporal filters and 84 filters in each spatial
  stage (5.07M total parameters with behavior/readouts).
- Behavior: nonrecurrent MLP with additive channels and identity-initialized
  bounded FiLM. No ConvGRU and no visual adapter.
- Anti-aliasing: differentiable Hann frequency mask on the temporal and spatial
  axes of the first layer; exact 240 Hz input avoids temporal decimation.
- Readout: Gaussian with a hard 0.5-scaffold-pixel width floor. At epoch 31,
  98.9% of axis widths are pinned to the floor.
- Optimizer protocol: seed 201, AdamW, LR 5e-4, weight decay 1e-5, cosine with
  two warmup epochs, batch 128, 512 steps/epoch, bf16 mixed precision.

## Controlled runs

| Run | Intervention | Status / evidence |
|---|---|---|
| D240M9c | Frequency mask, no readout floor | Rejected: 0.1559 BPS at epoch 7; Gaussian widths collapsed near zero. |
| D240M10c | No frequency mask, no readout floor | Rejected: 0.1591 BPS at epoch 7; the same readout collapse showed the mask was not the cause. |
| D240M11c | 0.25 px floor | Stopped after epoch 19; 0.2837 at epoch 7 and 0.4092 at epoch 19. |
| D240M12c | 0.50 px floor, no structural priors | Completed accuracy anchor. Exhaustive validation reached 0.598220 BPS at epoch 487; full FixRSVP median rho was 0.459 versus Figure 3's 0.447. |
| D240M13c | 0.75 px floor | Rejected at epoch 7: 0.2934 versus 0.2953 for the 0.50 px model. |
| D240M14c | Width-scaled historical Laplacian and competitive priors, ramped epochs 4-20 | Stopped after epoch 63 as an over-regularized anchor. 0.5107 at epoch 63; 28-43% of readout feature weights were exactly zero. |
| D240M15c | Half-strength version of all D240M14c priors | Stopped after the completed ramp at epoch 23. It reached 0.4354 BPS, versus 0.4373 without structural priors and 0.4287 with full priors, but readout sparsity had climbed from 5.3% at epoch 15 to 10.8%. |
| D240M16c | Full Laplacian smoothness without hidden/readout competition | Completed structured anchor. Epoch 415 scores 0.597272 exhaustively, within 0.000948 BPS of mature M12, and retains localized low-HF Jacobians. Full FixRSVP raw rho nearly matches Figure 3 (0.4436 versus 0.4468), but CCnorm and variance explained remain lower (0.5457/0.0169 versus 0.6356/0.0224). |
| D240M17c | Half-strength Laplacian smoothness without hidden/readout competition | Resumed from epoch 31 after the mature external-generalization gate showed that full strength costs real performance. At epoch 63 it reached 0.5144 rotating BPS, essentially tied with no-prior (0.5185) and full smoothness (0.5130) at the matched epoch, with clean exact Jacobians. |
| D240M18c | Full temporal plus half spatial Laplacian smoothness, no competition | Rejected and stopped after epoch 15. Despite 0.3846 BPS, it improved spatial high-frequency Jacobian energy only 8%, left temporal high-frequency energy unchanged, and increased temporal curvature 34% versus no-prior. |
| D240M19c | Epoch-127 M16 branch with unchanged temporal and doubled spatial Laplacian strength | Rejected and stopped after epoch 159. It improved spatial Jacobian metrics modestly but hurt temporal metrics and rotating validation relative to matched M16. |

## Behavior and structured-readout successors

All values in this table are deterministic exhaustive validation BPS unless a
rotating score is explicitly named.  The Figure-3 Ryan checkpoint scores
0.621363 on the identical split.

| Run | Intervention | Status / evidence |
|---|---|---|
| D240M20b | Freeze M16 and add neuron-specific additive/gain behavior residuals | 0.600893 at epoch 35, versus 0.597272 for inherited M16. |
| D240M22 | Co-adapt readouts and behavior paths with the visual core frozen | 0.603676 at epoch 55. |
| D240M26 | Reconstruct behavior on Ryan's original 120-Hz preprocessing grid while retaining 240-Hz vision | **0.610989 at the completed epoch-95 endpoint**, the current exhaustive leader. Continuing from epoch 55 improved 29/30 sessions at epoch 95; the paired interval versus epoch 67 is [+0.000542, +0.001037] BPS. |
| D240M35 | Add zero-initialized rank-8 session-specific behavior corrections to M26 | Rejected. Epoch 35 scores 0.609174, only +0.000002 BPS; the paired hierarchical-bootstrap interval is [-0.0000034, 0.0000085]. |
| D240M36a | Add an independent second Gaussian component with 504 new feature weights per neuron | Rejected after rotating scores 0.5902, 0.5897, and 0.5915. The 1.42M-parameter residual overfits despite a frozen core. |
| D240M36b/c | Reuse each neuron's visual feature projection under a bounded signed second Gaussian | Rejected. Both learning rates left the bounded residual scales near zero and tracked the identity trajectory. |
| D240M37a/b | Rank-4 mixture of the session population's mature feature projections under a second Gaussian | Rejected. The corrected weak-factor-regularization run learned nonzero mixtures but did not improve the matched rotating validation score. |
| D240M38a | Add a teacher-width smooth Dekel core that learns only the residual visual prediction | Paused after the mature epoch-95 endpoint. The fixed three-session gain over M26 is +0.000127 BPS and is unchanged from epoch 63; generic center-only residual capacity has saturated. The mature auxiliary first layer has 0.208% temporal high-frequency power, 0.0074% spatial high-frequency power, and 98.5% mean rank-1 energy. |
| D240M39a | Double the M38 auxiliary branch to 8/48 width with initial-penalty-matched smoothness | Paused after the preserved epoch-63 endpoint. It improves the fixed three-session panel by +0.000105 BPS over M26 (95% interval [+0.000014, +0.000198]) but is indistinguishable from the smaller M38 branch: M39-minus-M38 is -0.000019, interval [-0.000155, +0.000096]. The epoch-55 auxiliary bank is very clean: 0.080% temporal high-frequency power, 0.0042% spatial high-frequency power, and 98.8% mean rank-1 energy. |
| D240M40a/b / D240M41a | Freeze both visual cores from the selected M38/M39 checkpoint and co-adapt only both Gaussian readouts plus the output behavior head | M40a was stopped after epoch 3 when an audit found that its inherited auxiliary-only config did not re-apply M26's 0.5-pixel floor to the now-unfrozen base Gaussian widths. M40b is the clean restart with the floor restored and base geometry excluded from AdamW; startup matches exactly 30 base widths and excludes every auxiliary relative-width parameter. The completed epoch-95 endpoint improves all three exact panel sessions over M26 by +0.002087 BPS, interval [+0.001618, +0.002575]. It also beats epoch 63 by +0.000516, interval [+0.000197, +0.000873], so epoch 95 is selected. Its exhaustive 30-session score is **0.612717 BPS**, +0.001728 over M26 with interval [+0.001345, +0.002170], leaving a 0.008647-BPS gap to Ryan whose hierarchical interval [-0.024072, +0.007853] spans zero. M41 remains an unneeded width control. This stage cannot introduce new spatial or temporal frequencies. |
| D240M42a | Keep M26's 35/9 center path exact and fit a smooth teacher-width residual on a 51/13 surround aperture | Stopped after the epoch-31 exact gate. Epoch 31 improves M26 by +0.000127 BPS, interval [-0.000012, +0.000281], but is indistinguishable from the ordinary 35-pixel M38 residual: +0.00000023 BPS, interval [-0.000193, +0.000178]. Thus the smooth branch learns residual visual signal, but the extra surround aperture adds no measurable generalization. The epoch-23 surround bank was exceptionally clean (0.158% temporal high-frequency power, 0.0079% spatial, 98.7% rank-1). A real M26 checkpoint audit on a 51-pixel input gave bitwise-identical initialized predictions (maximum difference 0). |
| D240M43a | Freeze both visual cores from a selected M42 checkpoint and co-adapt the center/surround readouts plus output behavior head | Cancelled: M42's extra aperture was exactly tied with M38 at the fixed exact gate, so the expensive dual-aperture decomposition control is not justified. |
| D240M44 | Distill only Ryan's intact-minus-zero-behavior output residual on native 120-Hz batches into a feed-forward neuron-specific gain/offset head | The corrected fixed-validation-panel fit reaches teacher-residual NLL improvement 0.002013 and residual-logit correlation 0.557 at its selected epoch 11. The immutable snapshot is `outputs/dekel240_distillation/M44_epoch11_fixedpanel_snapshot.pt` (SHA256 `164a169a41a821dba411e5835a69445efba11ef4e6d11f9c28490efc695030ad`). No teacher visual parameter or activation is copied, so this stage cannot transfer Ryan's fragmented stimulus Jacobians. |
| D240M45a | Attach an epoch-4 M44 head after the selected M40 checkpoint and spike-refine only that head against the real counts | Rejected at the epoch-15 exact gate. The three-session score recovers from 0.715615 at epoch 3 to 0.717040, but remains below M40's 0.717516 by -0.000475 BPS with paired interval [-0.000880, -0.000135]. All three sessions are negative. This shows that transplanting Ryan's full behavior effect double-counts signal already represented by M40. Both visual cores were frozen throughout, so the rejection does not alter their Jacobians. |
| D240M46a | Low-rate coadaptation of both Gaussian readouts and both behavior heads from M45 while the visual cores remain frozen | Rejected at epoch 15. The exact three-session score recovers from 0.716865 at epoch 7 to 0.717055, but remains below M40's 0.717516 by -0.000461 BPS; all three sessions are negative and the paired interval [-0.001165, +0.000196] provides no evidence of a benefit. Thus the M45 deficit is not rescued by recalibrating the transplanted head against M40's readouts and existing behavior path. |
| D240M47a/b | Give only a new smooth residual core six older frames beyond M40's mature 60-frame history | M47a was invalidated immediately: the first implementation retained the last 60 tensor entries, but embedded lags are ordered newest-to-oldest, so it accidentally fed the mature core lags 6-65. The corrected zero-auxiliary audit scores 0.717327 on the stricter 66-frame sample set, confirming exact replay of lags 0-59. M47b rises from 0.717015 at epoch 3 to **0.717496 at epoch 31**, a matched +0.000169 BPS over the zero-auxiliary control with paired interval [+0.000043, +0.000294]; all three sessions improve. Training was stopped after preserving epoch 31. Its residual first layer remains clean (0.106% temporal power at >=60 Hz, 0.0075% spatial high-frequency power, 98.3% mean rank-1 fraction), so this small gain passes the interpretability gate. |
| D240M48a | Restart M40's still-improving readout/original-behavior optimization from epoch 95 at half the learning rate | The completed epoch-95 endpoint is selected: **0.718129 BPS** on the exact three-session panel, +0.000613 over M40 with paired interval [+0.000392, +0.000849]. Epoch 95 also improves on epoch 63 by +0.000217, interval [+0.000013, +0.000436]. Exhaustive 30-session validation reaches **0.613195 BPS**, +0.000478 over M40 with interval [+0.000308, +0.000656], leaving 0.008169 BPS to Ryan's native 0.621363 score. Its frozen full-population test reference is **0.625765 BPS**. Both visual cores are frozen exactly, and no new behavior head is introduced, so the gain preserves the selected filter bank and normalized stimulus-Jacobian shapes. W&B: `https://wandb.ai/yateslab/model_selection/runs/gw7p1ep8`. |
| D240M49a | Jointly refine the feature-level behavior pathway, output behavior residual, and Gaussian readouts from M40 while freezing both visual cores | Rejected at the epoch-15 recovery gate. Epoch 3 falls to 0.716049 and epoch 15 remains at 0.715858, -0.001657 below M40. The mature feature-level behavior path should therefore remain frozen; useful optimization is downstream at the neuron-specific readouts/output head. No visual kernel was trainable. W&B: `https://wandb.ai/yateslab/model_selection/runs/098zejyp`. |
| D240M50a | Combine M47's validated 66-frame smooth visual branch with low-rate readout/output-behavior refinement | Rejected at the epoch-31 composition gate. The exact same-sample score is 0.717736 BPS, only +0.000240 over M47 epoch 31; its paired interval [-0.000473, +0.000916] includes zero and Allen 2022-02-24 worsens by 0.000346. Training was stopped after preserving epoch 31. W&B: `https://wandb.ai/yateslab/model_selection/runs/oani87s0`. |
| D240M51a | Relax only the base Gaussian readout floor from 0.50 to 0.35 pixels after selected M48 | Rejected and stopped after the epoch-31 gate. It reaches 0.718018 BPS, 0.000111 below M48 with paired interval [-0.000462, +0.000255]; two of three sessions are lower. Widths barely move (epoch-15 minimum 0.4948 and median 0.4997 versus M48's 0.5000 and 0.5002), so this behaves mainly as a further head continuation rather than a localization change. Both visual cores remain bitwise frozen. W&B: `https://wandb.ai/yateslab/model_selection/runs/xu1nmk3z`. |
| D240M52a | Ultra-low-rate regularized visual refinement from selected M48 | Rejected and stopped after the epoch-15 recovery gate. Epoch 3 regresses by 0.001602 BPS and epoch 15 remains 0.001323 below M48 (0.716806 versus 0.718129), including a 0.004616 loss on Allen 2022-02-24. Moving both frequency-masked, Laplacian-regularized visual cores at only 5e-6 therefore fails to improve the selected representation. W&B: `https://wandb.ai/yateslab/model_selection/runs/unhx909z`. |
| D240M53 | Selectively distill Ryan's full predictions into M48's frozen readouts/output behavior head | Ryan is evaluated on the physical union of its support-dependent train/val/test partitions and mapped by dataset type plus native endpoint into the student's train split; spike counts and 42-dimensional behavior must match exactly. The completed 30-session cache has 99.04% minimum and 99.25% median coverage. The teacher term applies only to units for which Ryan exceeds the parent by more than 0.01 held-out BPS; real spike likelihood remains active for every unit. Pilots used the then-available M40 report (1,580/2,790 units weighted); the full run uses the subsequently completed exact M48 report (1,572 units weighted), avoiding teacher pressure on eight units whose gap has closed. Three-session epoch-15 validation pilots improve monotonically: strength 0.15 reaches **0.718820 BPS** (+0.000691 over M48); 0.40 reaches **0.719553** (+0.001424); 0.80 reaches **0.720267** (+0.002138); and 1.60 reaches **0.721109** (+0.002980, interval [+0.002062, +0.004534]). The untouched test partition confirms the choice: strength 1.60 scores **0.755048** versus M48's **0.752834**, +0.002214 with interval [+0.001171, +0.003223]; it also beats strength 0.80 by +0.000637 with interval [+0.000238, +0.001089]. An exact M48-weighted confirmation reproduces both scores (0.721110 validation, 0.755050 test). Every session improves. The completed full run improves monotonically on the exact panel: epoch 3 scores 0.719240 (+0.001111 over M48, interval [+0.000491, +0.001863]), epoch 7 scores 0.720336 (+0.002207, [+0.001318, +0.003586]), epoch 15 scores 0.721563 (+0.003434, [+0.002100, +0.005889]), and the validation-selected epoch-31 endpoint scores **0.722152** (+0.004023, [+0.002613, +0.006859]); epoch 31 also beats epoch 15 by +0.000589, interval [+0.000304, +0.001026]. Every session is positive at every gate. Exhaustive 30-session validation reaches **0.616764 BPS**, +0.003569 over M48 with interval [+0.002488, +0.004867]; 29/30 sessions improve, recovering 43.7% of M48's native-score gap to Ryan. All 38 visual/feature-behavior tensors remain bitwise equal to M48. Unit 105's final three shared-context Jacobians remain effectively unchanged: temporal high-frequency fractions differ by at most 0.00016 and spatial high-frequency fractions by at most 0.00048. Gaussian readouts do not collapse and broaden modestly (median width 0.50372, exact-floor fraction 5.3%, maximum 0.56160). On a held-out diagnostic subset, permuting behavior costs 0.0613 BPS and zeroing it costs 0.0380 BPS. W&B: `https://wandb.ai/yateslab/model_selection/runs/fmns3z34`. |
| D240M54 | Bracket the selective Ryan-teacher strength beyond M53 while preserving the same frozen interpretable core | At the matched 32-epoch panel dose, strength 3.2 reaches 0.723568 (+0.001416 over 1.6, interval [+0.000572, +0.002428]); 6.4 reaches 0.724070 (+0.000502 over 3.2, [+0.000073, +0.000967]); and 12.8 reaches **0.724279** (+0.000209 over 6.4, [+0.000023, +0.000422]). A terminal 25.6 control adds only +0.000048, interval [-0.000032, +0.000145], and reverses one session, establishing the plateau. The all-30-session strength-12.8 endpoint reproduces the gain on the exact panel (0.723797 versus M53's 0.722152, +0.001646, interval [+0.000441, +0.003051]) and reaches **0.618075 BPS** on exhaustive validation. That is +0.001311 over M53, interval [+0.000787, +0.001896], and +0.004880 over M48, interval [+0.003423, +0.006553]; all 30 sessions improve over M48, recovering 59.7% of M48's native-validation gap to Ryan. The single locked full-population test evaluation reaches **0.628709 BPS**, +0.002944 over M48 with interval [+0.001800, +0.004275], with 27/30 sessions positive. The first complete FixRSVP evaluation covers 1,703 cells in 24 sessions (398 reliable cells) and reproduces the Figure-3 observations exactly. Across all cells, M54 nearly matches Ryan's median raw PSTH correlation (0.44657 versus 0.44685), while leaving a real CCnorm/variance-explained gap (0.56866 versus 0.63653 CCnorm; 0.01896 versus 0.02236 variance explained); on reliable cells M54 reaches 0.59583 rho, 0.64542 CCnorm, and 0.07162 variance explained. All 38 visual/feature-behavior tensors remain bitwise equal to M48. Across eight exact real-context Jacobians, temporal high-frequency power is 0.21--2.22% and spatial high-frequency power is 0.34--1.01%; unit 105's three fixed-context maps retain M48's localized center/flank geometry. Readout widths remain non-collapsed (median 0.50697 scaffold pixels, maximum 0.56134), and held-out behavior ablations cost 0.0594 BPS when permuted and 0.0366 when zeroed. Strength 12.8 is the first external-generalization finalist; the CCnorm gap motivates another frozen-core readout/behavior continuation before the final FixRSVP pass. Pilot W&B: `https://wandb.ai/yateslab/model_selection/runs/kd0vthtv`; full-run W&B: `https://wandb.ai/yateslab/model_selection/runs/id28cke9`. |
| D240M55 | Test whether the residual performance gap requires late visual-core movement, then retarget the frozen heads against M54's remaining teacher deficits | A tightly anchored three-session pilot exposed only the final stage of the main visual core at 1e-6. It reaches 0.726010 at epoch 15, but its exactly matched frozen-core head control reaches 0.725964: the visual movement contributes only +0.000046 BPS despite a measurable 0.185% relative RMS core drift. The visual update is therefore rejected. The all-session frozen-core continuation reaches 0.724154 on the exact panel (+0.000356 over M54, interval [-0.000156, +0.000933]) and **0.618194 BPS** exhaustively (+0.000119, interval [-0.000065, +0.000315]). Its 16-epoch schedule supplies only about 270 updates per session, so this is a positive dose-finding stage rather than a new finalist. All 38 visual/feature-behavior tensors remain bitwise identical to M54. Late-core pilot W&B: `https://wandb.ai/yateslab/model_selection/runs/4isc9z2g`; matched frozen-core control: `https://wandb.ai/yateslab/model_selection/runs/6fahwk0n`; all-session continuation: `https://wandb.ai/yateslab/model_selection/runs/x2g9su8m`. |
| D240M56a/b | Give M55 a properly dosed frozen-core continuation and compare the original 0.01-BPS teacher margin with an aggressive zero-margin update | Both 96-epoch trajectories improve monotonically through the useful dose range. The aggressive M56b epoch-63 checkpoint is selected on the exact panel at **0.725029 BPS**, +0.000875 over M55 with interval [+0.000039, +0.001823], and beats the matched conservative checkpoint by +0.000211 with interval [+0.000069, +0.000373]; all three sessions favor M56b in both comparisons. Epoch 95 is statistically tied, so the earlier epoch is retained. Exhaustive 30-session validation reaches **0.619215 BPS**, +0.001021 over M55 with interval [+0.000548, +0.001563] and 28/30 sessions positive. It also beats M54 by +0.001140, interval [+0.000551, +0.001812], and closes the native-validation gap to Ryan to 0.002148 BPS. M56 improves 1,759/2,790 units relative to M55. Against Ryan it is already better for 1,075 units while Ryan leads for 1,715; their descriptive per-unit oracle reaches 0.657514 BPS, establishing substantial remaining complementarity. All 38 tensors under the visual cores and feature-level behavior pathway remain bitwise identical to M55; only downstream readouts and the output behavior head move. Conservative W&B: `https://wandb.ai/yateslab/model_selection/runs/n6afrdox`; aggressive W&B: `https://wandb.ai/yateslab/model_selection/runs/6a4x4mgy`. |
| D240M57a/b | Determine whether still more selective head optimization can close M56's final gap, and make the conservative variant exactly neuron-isolated | The shared-head M57a trajectory saturates: its best exact panel checkpoint is epoch 15 at 0.725096 BPS, only +0.000067 above M56 with interval [-0.000242, +0.000365]. M57b freezes the shared behavior encoder and restores every unselected unit row after each optimizer step. Epoch 31 reaches 0.725141, only +0.000112 above M56. The completed epoch-63 inheritance audit proves 42 whole tensors and 15,756 unselected neuron rows are bitwise preserved, with zero unexpected changes. This establishes that further optimization of the mature head is no longer the main bottleneck; the final epoch-63 panel gate remains to be scored after the active residual runs. Shared W&B: `https://wandb.ai/yateslab/model_selection/runs/0dij6dbl`; isolated W&B: `https://wandb.ai/yateslab/model_selection/runs/1ihzq82l`. |
| D240M58a/b | Add a teacher-guided second localized Gaussian visual component per Ryan-favored neuron while freezing M56 exactly | Closed as a capacity/optimization control. The architectural checkpoint adds 120 residual state tensors while retaining all 432 parent tensors bitwise. Thirty independent channel projections are exactly zero at initialization, and the upgraded checkpoint reproduces M56's panel score digit-for-digit (**0.7250289723 BPS**). M58a overfits by epoch 15 (0.724570, -0.000459 versus M56). Conservative M58b peaks at epoch 7 at 0.725164, only +0.000135, with two of three sessions slightly negative; by epoch 15 it returns to 0.725008. The one-source residual is therefore not promoted. W&B: `https://wandb.ai/yateslab/model_selection/runs/jc9k92tr` and `https://wandb.ai/yateslab/model_selection/runs/izgyykdm`. |
| D240M59a/b | Add independently localized zero-initialized components on both mature smooth visual feature banks | M59a establishes a small but reproducible gain. The identity upgrade retains all 432 M56 tensors bitwise, adds 240 residual tensors, has 60 exactly-zero output controls, and reproduces M56's panel score digit-for-digit (**0.7250289723 BPS**). Both cores, both mature readouts, and both behavior paths remain frozen; only Ryan-favored unit rows in the two new Gaussian components can move. With teacher weight 6.4, epoch 7 reaches 0.725244 (+0.000215; all three sessions positive; bootstrap interval [+0.000008, +0.000454]) and epoch 15 reaches **0.725321** (+0.000292; interval [+0.000019, +0.000601]). Epoch 31 slips slightly to 0.725293, so epoch 15 is retained and M59a was stopped after epoch 40. Exhaustive 30-session validation confirms **0.619535 BPS**, +0.000320 over M56 with bootstrap interval [+0.000109, +0.000617] and one-sided failure probability 0.00055. The 12.8 trajectory is consistently weaker (0.725171 at epoch 15) and was stopped after epoch 22. The epoch-3 geometry is extremely conservative: median main/auxiliary residual feature norms are 0.00581/0.00122 and median center displacements are 0.00174/0.00108 scaffold pixels. The production Figure 4 spatial adapter exactly carries both extra components into counterfactual rate maps. W&B: `https://wandb.ai/yateslab/model_selection/runs/kmwxzscd` and `https://wandb.ai/yateslab/model_selection/runs/9nl7neim`. |
| D240M60a/b | Learn genuinely missing features in a compact smooth residual visual core while preserving the complete M56 model | M60a becomes the best panel point estimate at epoch 31: **0.725400** versus M56's 0.725029 (+0.000371), improving two sessions and slipping on one (three-session bootstrap interval [-0.000233, +0.000999]). Exhaustive 30-session validation also selects it at **0.619625 BPS**, +0.000410 over M56 with bootstrap interval [+0.000038, +0.000882] and one-sided failure probability 0.0136. It is only +0.000090 over M59, with interval [-0.000172, +0.000372], so their point estimates are statistically tied and their per-unit oracle of 0.620211 BPS motivates a coordinated combination. The architectural checkpoint adds an 8-temporal/48-spatial-channel nonrecurrent Dekel branch and 30 independently localized residual readouts. Its 432 inherited tensors remain bitwise exact, all 30 new output projections are exactly zero, and the complete model reproduces M56's panel score digit-for-digit at initialization. Only the new core and Ryan-favored residual-readout rows train. The epoch-3 audit finds exactly the intended 132 tensors changed, 436 whole tensors and 4,300 unselected neuron rows preserved exactly, and zero unexpected changes. The branch retains the 240-Hz, 60-frame, 35 x 35 causal input and uses tapered temporal/stem-spatial frequency masks plus first-layer temporal/spatial and hidden-layer spatial smoothness. M60a uses core LR 5e-5 and readout LR 1e-4. At epoch 31 its mean first-layer temporal power at or above 60 Hz is 6.12%, spatial power above 0.25 cycles/pixel is 0.293%, and mean rank-1 separability is 83.5%; six of eight temporal filters remain below 2% high-frequency power, while two retain faster components. Doubling the core LR is harmful: M60b reaches only 0.725078 at epoch 7 and 0.725067 at epoch 15, so it is closed. W&B: `https://wandb.ai/yateslab/model_selection/runs/dazniy8m` and `https://wandb.ai/yateslab/model_selection/runs/n6gbjswm`. |
| D240M61 composition gate | Combine the selected M59 feature remix with M60's learned smooth visual correction | The chained initializer retains all 672 M59 tensors bitwise, adds only the 136 zero-gated smooth-branch tensors, and therefore reproduces M59 exactly before training. A separately audited composition imports M60 only after proving all 432 shared parent tensors bitwise equal; both epoch-15 and selected epoch-31 donors have exact audit reports. The uncoordinated epoch-15 additive composition scores 0.725222, below M59 epoch 15 (0.725321), so independently learned corrections are not assumed to add cleanly. M62 therefore starts from the selected epoch-31 composition but freezes every feature extractor and jointly coordinates only the three unit-specific residual readout families. |
| D240M62a/b | Jointly coordinate M59 and M60's three residual readout families without changing any visual or behavior feature extractor | The selected M59-epoch-15 plus M60-epoch-31 composition is already complementary at **0.725526 BPS**, +0.000126 over M60. Two exactly matched 32-epoch continuations train only 360 residual-readout tensors and preserve the other 448 tensors plus all 12,900 unselected neuron rows bitwise. Data-only M62a improves monotonically to 0.725897 at epoch 31. A light Ryan-teacher weight of 0.4 is significantly better: M62b epoch 31 reaches **0.726004 BPS**, +0.000604 over M60 with interval [+0.000113, +0.001045], and +0.000107 over M62a with interval [+0.000027, +0.000198]; every session improves in both comparisons. Exhaustive 30-session validation confirms **0.619889 BPS**, +0.000264 over M60 with interval [+0.000030, +0.000479] and +0.000354 over M59 with interval [+0.000028, +0.000670]. It leaves 0.001474 BPS to Ryan's native score. Across four representative neurons and 12 shared held-out contexts, M62b's mean Jacobian power at or above 60 Hz is 0.55% and mean spatial power above 0.25 cycles/pixel is 0.95%, essentially unchanged from M60 (0.56% and 0.94%); the exact maps retain localized, multi-lobed structure. W&B: `https://wandb.ai/yateslab/model_selection/runs/k1lrsgtf` and `https://wandb.ai/yateslab/model_selection/runs/m4z0p672`. |
| D240M63a | Continue the still-improving M62b residual readouts at half learning rate | The selected epoch-31 endpoint reaches **0.726118 BPS** on the exact panel, +0.000114 over M62b with interval [+0.000029, +0.000216] and all three sessions positive. It also beats epoch 15 by +0.000043 with interval [+0.000001, +0.000093], so the dose has not merely selected a noisy intermediate checkpoint. The epoch-3 audit proves that exactly the intended 360 residual-readout tensors move, while all 448 feature/behavior tensors and 12,900 unselected neuron rows remain bitwise exact. M63 is the frozen-core readout leader while M64 tests the remaining capacity hypothesis. W&B: `https://wandb.ai/yateslab/model_selection/runs/uwpojg1l`. |
| D240M64a/b | Test whether the residual performance gap is now limited by smooth visual capacity | Full-width 14/84 smooth residual-core controls from M56 at core learning rates 5e-5 and 2.5e-5. The identity initializer retains all 432 parent tensors bitwise, adds 136 residual tensors, and verifies 30 exactly zero output projections. Both epoch-3 audits then find exactly the intended 132 tensors changed, with all 436 protected tensors and 4,300 unselected neuron rows preserved bitwise. M64a rises from 0.725108 at epoch 15 and 0.725465 at epoch 31 to 0.726574 at epoch 63. The lower-step M64b is decisively better at epoch 63: **0.727585 BPS**, +0.001012 over M64a with interval [+0.000577, +0.001622], zero bootstrap failures in 20,000 draws, and all three sessions positive. It improves 176 exact-panel units versus 68 regressions. The branch matches the mature main core's width while retaining the 240-Hz temporal/stem frequency masks and the M60 Laplacian penalties; only the new core and Ryan-favored residual-readout rows train. At M64b epoch 63 the leading temporal/spatial components remain visually compact and predominantly low-frequency: mean temporal power at or above 60 Hz is 7.41% (one low-amplitude filter accounts for 52.9%), spatial power above 0.25 cycles/pixel is 0.93%, and mean rank-1 separability is 74.3%. The output-Jacobian gate, rather than the single-filter average, remains the decisive aliasing check. W&B: `https://wandb.ai/yateslab/model_selection/runs/satqd7a9` and `https://wandb.ai/yateslab/model_selection/runs/f6gom5y9`. |
| D240M65 composition and coordination gate | Compose M63's mature feature-remix readouts with M64a's full-width visual branch, then coordinate only the residual Gaussian components | The full-width target architecture inherits all 672 M59 tensors bitwise, adds only the 136 identity-gated residual-visual tensors, and imports M64a epoch 63 after proving 432 shared-parent tensors exact. A second audited composition imports M63's 240 coordinated base/auxiliary residual-readout tensors while deliberately ignoring its differently shaped compact residual core; the same 432 non-residual parent tensors remain bitwise exact. Before optimization, this composition reaches 0.726923. The epoch-3 selective-training audit proves that exactly the intended 360 residual-readout tensors move while all 448 feature/behavior tensors and 12,900 unselected neuron rows remain bitwise exact. Coordination reaches 0.726975 at epoch 7 and 0.727372 at epoch 15. The run is stopped after epoch 16 because the independently completed M64b composition is already materially stronger; its preserved milestones remain valid optimization controls. W&B: `https://wandb.ai/yateslab/model_selection/runs/amc1a7gz`. |
| D240M66 composition and coordination gate | Rebuild the complementary full-width model around the selected lower-step M64b visual branch | Two composition audits first import all 136 M64b visual-branch tensors and then M63's 240 coordinated base/auxiliary residual-readout tensors. Both prove all 432 shared-parent tensors bitwise exact. The resulting unoptimized checkpoint reaches **0.727945 BPS** on the exact panel and **0.620906 BPS** on exhaustive 30-session validation, leaving only 0.000457 BPS to Ryan before coordination. The completed frozen-core pass selects epoch 31 at a new panel best of **0.728569 BPS**, +0.000624 over the composition with interval [+0.000307, +0.000921] and zero bootstrap failures; all three sessions improve. It is also +0.000984 over M64b with interval [+0.000543, +0.001452]. Epoch 31 is slightly better than epoch 15 (+0.000164; 210 versus 110 improving units), although that small paired interval [-0.000008, +0.000333] narrowly includes zero. The terminal inheritance audit proves that exactly the intended 360 residual-readout tensors move while all 448 feature/behavior tensors and all 12,900 protected neuron rows remain bitwise exact. Exhaustive 30-session validation reaches **0.621111 BPS**, only 0.000252 below Ryan; the paired hierarchical interval spans zero, so the models are statistically tied. The untouched full test score is **0.631160 BPS**, 0.003195 above Ryan's reported 0.627965 reference. Across four representative neurons and 12 shared held-out contexts, spatial high-frequency Jacobian power is 0.97%, essentially identical to M62b's 0.95%; temporal high-frequency power is 2.37% on average (1.40% median), with localized maps and dominant peaks near 30--40 ms but one faster high-drive context retained as a caution flag. The full FixRSVP gate covers the same 1,703 cells/24 sessions as Figure 3: all-cell rho is 0.44557 versus Ryan's 0.44685, while CCnorm remains lower at 0.56716 versus 0.63653; on the 398 reliable cells M66 reaches 0.59960 rho, 0.64884 CCnorm, and 0.07149 variance explained. A three-session output-behavior ablation attributes +0.00842 BPS over the inherited checkpoint to the intact head, with 82.2% of units improved; gain-only is beneficial whereas additive-only is neutral. W&B: `https://wandb.ai/yateslab/model_selection/runs/aw9vk3rd`. |
| D240M67a/b | Correlation-aware final head refinement from selected M66 | Both pilots freeze all visual cores and the shared feature-level and output-level behavior encoders, then jointly adapt only unit-indexed Gaussian readouts and output gain/offset rows for Ryan-favored neurons. A complementary affine-invariant teacher-shape loss targets the response modulation measured by Figure-3 CC/CCnorm without transferring any teacher derivative. Strengths 0.05 and 0.20 were tested at the predeclared epoch-3 and epoch-7 gates. Neither passed the fixed three-session panel: at epoch 7 M67a reached **0.728500 BPS** (−0.000069 versus M66; paired 95% interval [−0.000262, +0.000134]) and M67b reached **0.728399 BPS** (−0.000170; interval [−0.000425, +0.000089]). Training was stopped and M66 retained as the final checkpoint; this negative result argues against further optimizing teacher correlation after the student has matched the teacher's native validation likelihood. W&B: `https://wandb.ai/yateslab/model_selection/runs/5vw2yc7f` and `https://wandb.ai/yateslab/model_selection/runs/260ka6u9`. |
| D240M68a | Data-only, tightly anchored refinement of M66's full-width smooth residual visual branch | Only the residual core and its localized residual visual readouts train; all 676 other state tensors remain bitwise exact. The epoch-15 endpoint reaches **0.728944 BPS** on the exact panel and **0.621440 BPS** on exhaustive 30-session validation, +0.000329 over M66 with interval [+0.000193, +0.000456]; 29/30 sessions and 1,887/2,790 units improve. The untouched test score is **0.631569 BPS**, +0.000409 over M66 and +0.003604 over Ryan's reported reference. The first layer remains clean and essentially unchanged (7.37% mean temporal high-frequency power, 0.94% spatial high-frequency power, and 74.35% mean rank-1 energy). Its exact 12-context output-Jacobian gate is also slightly smoother than M66 (1.88% versus 2.37% mean temporal high-frequency power; both 0.97% spatial). The production-scale 100-image x 1,000-trace mechanism gate nevertheless fails: for aligned high-SF units, the final-bin across-minus-along RMS-excursion contrast is **+1.060 percentage points**, 95% image-bootstrap interval [−0.843, +3.588], p=0.298, versus M66's **−7.559**, [−8.854, −6.074], p=0.0002. M68 is therefore a clean predictive control but is rejected as the mechanistic twin. W&B: `https://wandb.ai/yateslab/model_selection/runs/cmotvnob`. |
| D240M69a | Independently refine M66's base and auxiliary residual Gaussian readouts with selective Ryan targets while all feature extractors remain frozen | At epoch 15 the exact panel reaches **0.728754 BPS**, +0.000185 over M66 with all three sessions positive, although the interval [−0.000034, +0.000427] includes zero. Its full production-scale 100-image x 1,000-trace Figure-4 gate nevertheless fails: for aligned high-SF units, the final-bin across-minus-along RMS-excursion contrast is **+1.095 percentage points**, 95% image-bootstrap interval [−0.843, +3.706], p=0.296, versus M66's **−7.559**, [−8.854, −6.074], p=0.0002. The corresponding M69 path and range contrasts are also near zero (−0.622 and +1.850 percentage points). Refining only localized readouts can therefore erase the mechanism even when the visual filters are fixed. M69 is rejected as a standalone model and donor. W&B: `https://wandb.ai/yateslab/model_selection/runs/jba1ikc9`. |
| D240M70 composition gate | Combine M68's data-refined smooth residual core/visual readouts with only M69's base and auxiliary residual-readout families | The composition audit proves all 432 shared-parent tensors exact and copies exactly the intended 240 base/auxiliary residual-readout tensors. Without further optimization it reaches **0.729053 BPS** on the exact panel, +0.000110 over M68 with interval [+0.000008, +0.000210], and **0.621560 BPS** exhaustively, +0.000120 over M68 with interval [+0.000055, +0.000195]. It beats M66 by +0.000449, interval [+0.000315, +0.000583], and its point estimate is +0.000197 above Ryan's 0.621363, although the heterogeneous 30-session interval [−0.014251, +0.015496] classifies the two as tied. M70's first-layer filters are bitwise M68's. Across four representative neurons and 12 shared held-out contexts, mean temporal high-frequency Jacobian power is **1.88%** versus M66's 2.37%, and mean spatial high-frequency power is unchanged at **0.97%**. The hardest high-drive context falls from 13.43% to 9.71% temporal high-frequency power while retaining localized center/flank structure. However, its production-scale Figure-4 mechanism result is indistinguishable from M69: the aligned high-SF final-bin RMS contrast is **+1.039 percentage points**, interval [−0.859, +3.549], p=0.308. M70 is retained as the best smooth predictive control, not the final mechanistic twin. |
| D240M71a | Jointly co-adapt M70's smooth residual core and all three localized residual-readout families on real spikes at very low learning rates | Rejected at the predeclared epoch-7 recovery gate. Epoch 3 scores 0.728987, −0.000066 below M70 with all three panel sessions lower. Epoch 7 remains below M70 at **0.728991** (−0.000062; interval [−0.000214, +0.000152]) and reverses two of three sessions. The run was stopped after preserving the valid epoch-7 milestone; M70 is locked before the untouched test and FixRSVP evaluations. W&B: `https://wandb.ai/yateslab/model_selection/runs/5ll6hdjf`. |
| D240M72a/b | Test whether M70's remaining FixRSVP CCnorm/FEM gap can be recovered by moving only its frequency-masked, Laplacian-regularized residual visual representation toward Ryan's natural-stimulus responses | Both branches keep the complete mature M70 predictor anchored and expose only `residual_convnet` plus its independently localized `residual_visual_readouts`. M72a is conservative (teacher Poisson 1.6, affine-invariant shape 0.05); M72b is broader (teacher Poisson 0.4, shape 0.20, teacher margin -0.04). Neither passes the joint gate. M72a epoch 7 scores **0.621406 BPS** exhaustively, −0.000153 versus M70 with paired 95% interval [−0.000287, −0.000018]. On FixRSVP it raises raw rho to 0.44689 but lowers CCnorm to 0.56789 and variance explained to 0.01920. M72b epoch 15 reaches 0.56742 CCnorm and 0.01920 variance explained. Their fixed-observation FEM-fraction medians are 0.63319 and 0.63216, respectively, both below M70's 0.63455 and far below the empirical 0.73566. The missing external signal therefore is not recovered by stronger natural-stimulus teacher pressure on the smooth visual residual. Both branches are rejected. W&B: `https://wandb.ai/yateslab/model_selection/runs/lcr3p5s6` and `https://wandb.ai/yateslab/model_selection/runs/xup60pnh`. |

M70's locked held-out test score is **0.631632 BPS**, +0.000472 over M66
with interval [+0.000355, +0.000605]; 1,983/2,790 units and 29/30 sessions
improve. It is 0.003667 above Ryan's reported 0.627965 test reference. The
full FixRSVP gate covers 1,703 cells in 24 sessions and remains stable relative
to M66: all-cell rho is **0.44532**, CCnorm is **0.56873**, and model variance
explained is **0.01929**; among the 398 reliable cells they are 0.59828,
0.64811, and 0.07133. Ryan remains better on FixRSVP CCnorm and variance
explained (0.63653 and 0.02236), while raw PSTH correlation is nearly matched.
The validation/test gain therefore generalizes without buying accuracy through
aliased derivatives, but the known FixRSVP distribution-shift gap remains.

The full mechanism gate changes the final selection. M68, M69, and M70 buy only
0.000375, 0.000185, and 0.000484 exact-panel BPS over M66, respectively, while
all three aligned high-SF Figure-4 contrasts collapse from M66's significant
-7.559 percentage points to nonsignificant values near +1 percentage point.
M68 and M70 have slightly smoother aggregate output Jacobians than M66, so this
failure is not an aliasing artifact: small, spatially localized changes in the
learned representation/readouts can genuinely erase the population mechanism.
M66 is therefore the final mechanistic twin. It is statistically tied with Ryan
on exhaustive validation, beats Ryan's reported locked-test reference, retains
localized low-alias Jacobians, and uniquely preserves the production Figure-4
effect. M68 and M70 remain useful predictive controls demonstrating that their
small held-out-likelihood gains are real but not mechanistically safe.

The production Figure-3 native-240-Hz alignment was subsequently audited before
the final rerender. Causal two-bin targets must be labelled by their block-start
PSTH coordinate, blocks crossing a trial boundary must be rejected, and
continuous covariates such as eye position must be averaged over the same two
native samples. The standalone FixRSVP evaluator already followed those
conventions; the production Figure-3 and supplement loaders were corrected and
covered by focused alignment tests before rebuilding their isolated M66 caches.
The strict production render uses 24 canonical sessions/1,703 cells before the
paper's Figure-2 population and metric-validity filters. Panel C retains 985
cells from 19 sessions and has median CCnorm 0.601 intact, 0.597 retinal-only,
and 0.527 stabilized. Panel E has empirical, intact, retinal-only, and stabilized
FEM-fraction medians of 0.710, 0.681, 0.672, and 0.259: the full and retinal-only
model distributions pass the paired +/-0.1 equivalence test, while the
stabilized control does not. This alignment correction changes the production
decomposition inputs but not the external validation/test/FixRSVP comparisons
reported above.

The final production Figure 4 is likewise locked to M66's checkpoint-specific
100-image x 1,000-trace bank, stabilized baseline, instantaneous unit maps, and
schematic endpoint maps. Its full final-bin aligned high-SF RMS contrast is
−7.559 percentage points with 95% image-bootstrap interval [−8.854, −6.074]
and p=0.0002. The displayed panel deliberately omits the noisier tail bin and
shows the adjacent stable endpoint (−5.1 percentage points, p<0.001). The
production Figure 3 and Figure 4 bundles, including PDF/SVG/PNG, captions,
manifests, and low-bandwidth previews, are under
`outputs/dekel240_paper/final/figures/`.

An early exhaustive one-session check establishes that M38's branch is learning
generalizable signal. On Allen 2022-02-16, epoch 35 improves M26 from 0.693695
to 0.693851 BPS. The matched 116-cell difference is +0.000156 BPS, with a
paired bootstrap interval of [+0.000034, +0.000290]. This is positive evidence
for the construction, but it is too small and too early to trigger a full
30-session evaluation. The matched M39 epoch-31 check is neutral at 0.693677
BPS versus M26's 0.693695; its paired interval [-0.000161, +0.000128] spans
zero. M38 therefore has the better early accuracy/structure tradeoff, while
M39 remains live because its larger branch may require more optimization. At
M38 epoch 63 the same session falls back to 0.693737 BPS, only +0.000042 above
M26, with interval [-0.000087, +0.000171]. The filters remain clean (0.24%
temporal high-frequency power, 0.008% spatial, 98.0% rank-1), but the center-
only exact gain is not growing monotonically. This is the trigger to promote
the exact-identity 51-pixel surround residual after M39's matched milestone.

Because Allen 2022-02-16 is an unusually small-gap session, the fixed cheap
screen was expanded to the first three configured Allen sessions before
rejecting a residual branch. M38 epoch 63 improves all three by +0.000042,
+0.000180, and +0.000152 BPS, respectively. The equal-session paired gain is
+0.000125 BPS across 342 cells, with hierarchical-bootstrap interval
[+0.000015, +0.000243]. Thus the smooth center-only residual is learning a
small but reproducible signal; the result still falls far short of the
roughly 0.034-BPS Ryan-minus-M26 gap on this deliberately sensitive panel.
The mature epoch-95 endpoint is statistically identical to epoch 63
(+0.000002 BPS, interval [-0.000097, +0.000098]) and therefore marks visual
saturation rather than a reason to continue fitting. Its auxiliary filters
remain clean: 0.208% temporal high-frequency power, 0.0074% spatial
high-frequency power, and 98.5% mean rank-1 energy. Epoch 95 is the warm start
for the frozen-core readout/behavior co-adaptation run D240M40a.

Checkpoint averaging did not improve M26: the uniform epoch-55/67/95 soup
scored 0.610553 exhaustively, versus 0.610989 for epoch 95 alone.  The paired
interval favors the endpoint by [+0.000226, +0.000635] BPS, so every residual
visual successor is warm-started from the preserved epoch-95 checkpoint.

External Figure-3 comparison: Ryan's `05_lr5e-4` scored 0.506 at epoch 47,
0.527 at epoch 63, and 0.6222 at its final best checkpoint (epoch 471). Its
locked test score is 0.627965 BPS. The current Figure-3 model has 4.92M
parameters; the capacity-matched Dekel model has 5.07M, so this comparison is
not explained by giving the new architecture more capacity.

## Jacobian evidence

All Jacobians are exact gradients of log predicted rate with respect to real
held-out 60 x 35 x 35 stimulus contexts. They are not finite differences or a
surrogate fit.

At epoch 15, the full-prior run reduced mean spatial high-frequency Jacobian
energy by 31% and spatial Laplacian roughness by 21% relative to the matched
no-prior run. At epoch 31, on the three units selected by both models, it
reduced temporal curvature by 9%, temporal energy at or above 60 Hz by 28%,
spatial Laplacian roughness by 23%, and spatial high-frequency energy by 57%.
The context rank for 80% Jacobian energy remained 8/12, so smoothing did not
collapse the context-dependent subspace.

At epoch 15, the half-prior run selected the same four units as both controls.
Relative to the no-prior run, it reduced spatial high-frequency Jacobian energy
by 24% and spatial Laplacian roughness by 6.5%; temporal high-frequency energy
was essentially unchanged, and temporal second-difference roughness was 14%
higher. Context rank remained 8/12. This is a useful but weaker smoothness gain,
while competitive readout pruning had already reached 5.3%, motivating the
smoothness-only D240M16c control.

At epoch 15, the smoothness-only run shared its top three units with all three
controls. Relative to the no-prior run on those nine exact Jacobians, temporal
energy at or above 60 Hz fell 25%, spatial high-frequency energy fell 39%, and
spatial Laplacian roughness fell 6.5%; temporal second-difference roughness was
6.5% higher. Context rank remained 8/12. All 30 readout feature tensors and all
three hidden spatial tensors had exactly zero exact-zero weights, demonstrating
that the competitive proximal operator—not the Laplacian loss—caused pruning.

At epoch 31, unit order was explicitly pinned to `[27, 82, 96, 41]` so the
smoothness-only and no-prior models were compared on all four identical units
and all 12 identical contexts. Smoothness-only reduced temporal curvature 4%,
spatial Laplacian roughness 24%, temporal energy at or above 60 Hz 23%, and
spatial high-frequency energy 54%. Context rank changed from 8/12 to 7/12. On
the first three units also shared by the full-prior model, the full-prior and
smoothness-only Jacobian metrics were effectively identical, while
smoothness-only scored 0.0028 BPS higher and retained dense weights.

At epoch 15, half-smoothness was compared against the no-prior model with unit
order pinned to `[27, 82, 20, 24]`. It reduced spatial Laplacian roughness 20%
and spatial high-frequency energy 42%, retained context rank 8/12, and had no
exact-zero hidden or readout weights. Temporal curvature was 27% higher and
temporal energy at or above 60 Hz was 6% higher, although the absolute latter
fraction remained 0.28%. Its 0.3837 BPS was only 0.0013 below no-prior and above
all other regularized candidates, so temporal metrics must be rechecked at
epoch 31 before selecting it over full smoothness.

At epoch 31, half-smoothness still reduced spatial Laplacian roughness 22% and
spatial high-frequency energy 41%, but temporal energy at or above 60 Hz was
32% higher than no-prior. It was therefore stopped. This isolates the next
control: retain the full temporal coefficient while halving only the first-layer
and hidden spatial coefficients (D240M18c).

The asymmetric D240M18c control failed the Jacobian criterion at epoch 15. On
the pinned units and contexts it improved spatial high-frequency energy only 8%,
left temporal high-frequency energy essentially unchanged, and increased
temporal curvature 34% relative to no-prior. Full spatial smoothness is thus
needed for the temporal cleanup as well, presumably through the nonlinear
hidden representation. D240M16c is the only dense model that improves all four
roughness/high-frequency metrics and was selected for long training.

Longitudinal diagnostics at epoch 99 use the same four units and 12 contexts as
the epoch-31 comparison. The exact Jacobians remain localized, with clear
temporal energy peaks around 30--40 ms, and the first-layer spatial components
remain smooth. Relative to epoch 31, normalized temporal curvature increased
43.3%, normalized spatial Laplacian roughness increased 17.7%, temporal
high-frequency energy increased 7.3%, and spatial high-frequency energy
increased 30.0%. The first-layer spectra still place essentially all power below
60 Hz, so this is primarily gradual function roughening within the retained
band, not a failure of the Nyquist-zero Hann frequency window. Checkpoint selection will
therefore include fixed-context Jacobian quality rather than using validation
BPS alone.
Because the trainer rotates only the top three validation checkpoints, M16
epochs 75, 87, 99, and 107 are also copied into `analysis_candidates/`; the current
M12 accuracy leader at epoch 339 is preserved the same way. This prevents later
accuracy improvements from deleting the checkpoints needed for the final
accuracy-versus-Jacobian comparison.

The same fixed-context diagnostic at epoch 127 shows a useful separation
between genuine aliasing and ordinary spatial refinement. Relative to epoch 99,
temporal curvature fell 11.7% and temporal energy at or above 60 Hz fell 41.5%;
the latter is now 37.2% below epoch 31. Spatial Laplacian roughness was unchanged
(+0.2%), while spatial high-frequency energy rose 40.6% from epoch 99 and 82.8%
from epoch 31. Its absolute high-frequency fraction remains only 0.435%, below
the no-prior epoch-31 value of 0.518%. The exact images remain localized and
oriented without checkerboard structure, and context rank remains 8/12. Thus
the first-layer temporal mask continues to work; the current tradeoff is
increasing fine spatial structure rather than temporal aliasing. On the same
16-batch session-0 subset, BPS improved from 0.6247 at epoch 99 to 0.6526 at
epoch 127. Behavior permutation reduced BPS by 0.0190, and 99.2% of readout
axis widths remain at the 0.5-pixel floor. Epoch 127 is preserved in
`analysis_candidates/` for exhaustive model selection.

A mature no-prior control at its new accuracy-leading epoch 395 makes the
regularization benefit much clearer. On the identical four units and 12
contexts, M16 epoch 127 has 25.1% lower temporal curvature, 68.0% lower spatial
Laplacian roughness, 82.0% lower temporal energy at or above 60 Hz, and 89.1%
lower spatial high-frequency energy than M12 epoch 395. The no-prior exact
images visibly contain diffuse pixel-scale structure, whereas M16 remains
localized and oriented. This is not explained by the first-layer Hann window,
which both models share: Laplacian training is preventing late nonlinear
roughening downstream of that mask. M12 is more accurate on the same 16-batch
session-0 subset (0.7325 versus 0.6526 BPS), so the remaining objective is to
close that performance gap while retaining the roughly 3--9x spatial/frequency
smoothness advantage.

The effective first-layer banks show the same mechanism directly. Relative to
M12 epoch 395, M16 epoch 127 has 8.8% lower temporal second-difference
roughness, 57.6% lower spatial Laplacian roughness, 71.1% lower temporal
high-frequency energy, and 83.7% lower spatial high-frequency energy. Mean
rank-1 spatiotemporal separability is also higher (0.824 versus 0.732). The
diagnostic now stores these filter-bank metrics alongside every future
Jacobian report rather than relying only on plotted waveforms.

At epoch 159 the same M16 units remain localized and oriented, and context rank
remains 8/12. Relative to epoch 127, temporal curvature increased 15.4%, spatial
Laplacian roughness increased 3.6%, temporal high-frequency energy increased
38.5%, and spatial high-frequency energy *decreased* 14.5%. This is metric
fluctuation rather than monotonic degradation. Against mature M12 epoch 395,
M16 epoch 159 is still 13.5% lower in temporal curvature, 66.8% lower in spatial
Laplacian roughness, 75.0% lower in temporal high-frequency energy, and 90.6%
lower in spatial high-frequency energy. Its session-0 subset BPS increased to
0.6584, behavior permutation costs 0.0221 BPS, and 99.1% of readout widths are
at the intended floor. The first-layer spatial high-frequency fraction remains
83.5% below M12 epoch 395. M16 therefore continues; the doubled-spatial branch
remains a controlled fallback rather than a necessary rescue.

The matched doubled-spatial branch (M19) was stopped after its preserved epoch-
159 checkpoint. Relative to M16 at the same epoch, identical units, and
identical stimulus contexts, it reduced exact-Jacobian spatial Laplacian
roughness by 14.4% and spatial high-frequency energy by 21.7%. Its first-layer
spatial Laplacian and high-frequency metrics fell by 10.1% and 22.8%,
respectively. The gain was selective rather than global: Jacobian temporal
curvature rose 6.2%, temporal high-frequency energy rose 20.1%, and session-0
BPS fell from 0.6584 to 0.6521. The paired rotating validation score at epoch
159 fell from 0.54775 to 0.54439 (-0.00336 BPS). Because M16 is already 67--91%
smoother than mature M12 and continued to gain accuracy, this is not a useful
trade: M19 is rejected and M16 remains the selected long trajectory.

M16 epoch 239 is the first late checkpoint to close a substantial part of the
accuracy gap: rotating validation reached 0.57790 BPS and the fixed session-0
subset reached 0.68287. Its exact Jacobians remain localized and structured,
although they are visibly finer than at epoch 159. Relative to epoch 159,
temporal curvature increased 6.2%, spatial Laplacian roughness 15.9%, temporal
high-frequency energy 73.3%, and spatial high-frequency energy 62.2%. Against
the mature no-prior M12 epoch 395, however, epoch-239 M16 remains 8.1% lower in
temporal curvature, 61.6% lower in spatial Laplacian roughness, 56.7% lower in
temporal high-frequency energy, and 84.8% lower in spatial high-frequency
energy. The checkpoint is preserved; this confirms that later accuracy gains
do gradually buy finer structure, but have not erased the regularization
benefit.

The resumed half-smoothness M17 checkpoint at epoch 63 is the first promising
accuracy/structure midpoint. On the identical rotating subset it scores
0.51439 BPS, versus 0.51850 for no-prior M12 and 0.51298 for full-smoothness
M16. Its fixed session-0 subset score is 0.60318. The exact Jacobians are
localized and mostly oriented; against mature no-prior M12 epoch 395, their
mean temporal curvature is 15.3% lower, spatial Laplacian roughness 67.6%
lower, temporal high-frequency energy 52.3% lower, and spatial high-frequency
energy 92.1% lower. Context rank remains 8/12. Relative to M16 epoch 159, M17
epoch 63 has comparable curvature and spatial Laplacian roughness, 15.1% less
spatial high-frequency energy, but 91% more temporal high-frequency energy.
Its first-layer spectra nevertheless place essentially all power below 60 Hz;
the temporal metric is therefore the main late-checkpoint watch item, not
evidence of Nyquist aliasing. The checkpoint is preserved for later matched
selection.

M16 epoch 315 sets a new rotating-validation high of 0.59227, above M12's
matched epoch-315 score of 0.58538, and its fixed session-0 subset improves to
0.69681. The gain is not structurally free. Relative to M16 epoch 239, exact-
Jacobian temporal curvature rises 18.3%, spatial Laplacian roughness 2.9%,
temporal high-frequency energy 51.5%, and spatial high-frequency energy 18.7%.
Against mature M12 epoch 395 it is now 8.7% *higher* in temporal curvature, but
remains 60.5% lower in spatial Laplacian roughness, 34.4% lower in temporal
high-frequency energy, and 82.0% lower in spatial high-frequency energy. The
images remain localized but visibly more mottled than epoch 239. Epoch 315 is
preserved as an accuracy-leading candidate, not automatically preferred over
the cleaner earlier checkpoint.

## Deterministic validation and FixRSVP generalization

The corrected exhaustive evaluator scores each example exactly once rather
than drawing replacement batches. M12 epoch 415 scores **0.597433 BPS** over
the complete 30-session validation split (the rotating 10% checkpoint score
was 0.6008). The completed epoch-487 endpoint scores **0.598220 BPS**, an
increase of only 0.000787. It improves 25/30 paired sessions; a deterministic
100,000-draw session bootstrap gives a mean-difference 95% interval of
[0.000425, 0.001124]. Epoch 487 is therefore the mature no-prior accuracy
ceiling, but the last 72 epochs bought less than one thousandth of a bit per
spike. Structured candidates are compared against this endpoint.

`evaluate_dekel_fixrsvp.py` implements a deliberately sparse external-
generalization gate matching `paper/fig3/_fig3_data.py`. FixRSVP was absent
from training. The evaluator retains the Dekel model's native 240-Hz,
60-frame history and causal 120-Hz targets, but aligns predictions to the
published Figure-3 trial/time grid. It reuses the exact Figure-3 cell
population and canonical observed-response/data-filter arrays, and verifies
that independently extracted causal spike-count bins match them exactly. It
then applies Ryan's positive per-cell gain plus positive offset, fitted by
Poisson LBFGS, followed by the same 120-bin window, >200-spike threshold,
two-seed 500-split normalized correlation, PSTH correlation, and single-trial
variance-explained calculations. Detailed traces are cached so a checkpoint is
never recomputed merely to make a later plot.

The protocol is exact on all 24 sessions: every independently extracted causal
spike-count tensor matches the Figure-3 reference on its finite mask. Full
FixRSVP evaluation covers 1,703 cells, including 398 reliable cells selected by
the shared `ccmax > 0.85` criterion. Mature no-prior M12 epoch 487 reaches an
all-cell median rho of **0.4590**, versus **0.4468** for Figure 3; the paired
mean difference is -0.0020 and 49.0% of cells favor M12. Thus the feed-forward
Dekel architecture itself generalizes at essentially the Figure-3 level on the
raw PSTH correlation. For reliable cells the medians are 0.6251 versus 0.6484.
The remaining baseline deficit is concentrated in noise-normalized correlation
(all-cell medians 0.5936 versus 0.6356; paired mean -0.0457) and model variance
explained (0.0185 versus 0.0224; paired mean -0.0039).
Across-session bootstrap intervals make the distinction precise. The mean
session-level M12-minus-Figure-3 difference has a 95% interval of
[-0.0319, 0.0054] for rho and [-0.00481, 0.00062] for variance explained, both
including zero, but [-0.0808, -0.0164] for normalized correlation. Thus M12 is
tied on raw correlation and variance explained at the session level, but not on
the noise-normalized metric.

Full-smoothness M16 epoch 223 is worse on the same cached gate: all-cell median
rho is **0.4235**, normalized correlation is 0.5228, and model variance
explained is 0.0136. Their paired mean differences from Figure 3 are -0.0283,
-0.1039, and -0.0095, respectively. Reliable-cell medians are 0.5681 rho,
0.5990 normalized correlation, and 0.0588 variance explained. The identical
`ccmax` distributions across all three evaluations confirm that this is a
model difference rather than an alignment or reliability change. Together
with M16 epoch 223's exhaustive in-distribution deficit (0.571405 versus M12
epoch 487's 0.598220 BPS), this establishes that the full Laplacian dose has a
real generalization cost even though it produces substantially cleaner
Jacobians. M17 is therefore resumed as the predeclared half-strength
smoothness-only compromise.
Direct M12-minus-M16 session bootstraps are positive for all three metrics:
[0.00075, 0.02428] for rho, [0.01514, 0.05494] for normalized correlation, and
[0.00079, 0.00491] for variance explained.

FixRSVP is an expensive sparse decision gate, not a per-checkpoint metric. The
M12 and M16 traces are cached permanently. It will next be run only for a
mature candidate that is competitive on exhaustive validation and passes the
filter/Jacobian structural screen.

### Fixed exemplar unit and shared-context Jacobians

For qualitative inspection without cherry-picking, the session-0 exemplar is
chosen from Figure-3-reliable cells (`ccmax > 0.85`) by maximizing the minimum
raw PSTH correlation across Figure 3, M12 epoch 487, M16 epoch 315, and M17
epoch 63. This selects Allen 2022-02-16 recorded unit 105. Its correlations are
0.8132, 0.5819, 0.6129, and 0.5911, respectively; normalized correlations are
0.8940, 0.8274, 0.8275, and 0.8140. Ryan remains clearly better on this
particular trace, while M16 is the strongest of the three candidates.

Exact log-rate stimulus Jacobians for unit 105 were then computed at identical
held-out stimulus and behavior examples for all three Dekel candidates. The
contexts are selected once at the 20th, 50th, and 80th percentiles of the mean
standardized log rate across the models. M12 is context-fragile: its low-drive
map is diffuse and pixel-scale, and its spatial high-frequency energy ranges
from 0.0148 to 0.0534 across contexts. M16 ranges from 0.0024 to 0.0068 and M17
from 0.0012 to 0.0052; both retain localized center/flank or center/surround
organization across all three contexts. Temporal sensitivity peaks around
29-46 ms and none of the structured candidates shows energy accumulating near
Nyquist. The comparison is cached under
`outputs/dekel240_exemplar/allen_2022-02-16_unit105/`.

Ryan was subsequently added through a physical-context comparison on FixRSVP.
The three examples are aligned by recorded trial ID and 120-Hz PSTH bin, while
each model retains its native input lattice: Ryan receives 33 x 51 x 51 at 120
Hz and the Dekel candidates receive 60 x 35 x 35 at 240 Hz. Ryan's displayed
spatial gradient is center-cropped to 35 x 35, but the full 51 x 51 Jacobian is
preserved in the cache. On these shared low-, median-, and high-drive moments,
Ryan's spatial high-frequency fractions are 0.1128, 0.3328, and 0.3845; M12's
are 0.0099, 0.0192, and 0.0173; M16's are 0.0055, 0.0024, and 0.0035; and
M17's are 0.0017, 0.0017, and 0.0011. Thus Ryan's end-to-end sensitivity is
substantially more fragmented, M12 is intermediate, and M16/M17 are the most
localized and context-stable. Temporal roughness statistics are not compared
numerically across Ryan and the candidates because their sampling grids
differ. The aligned artifacts, including one compact PNG, one PDF, and the full
NPZ per model, live under
`outputs/dekel240_exemplar/allen_2022-02-16_unit105/fixrsvp_aligned/`.

For this exemplar, single-trial FixRSVP variance explained after the same
positive affine calibration used in Figure 3 is 0.1478 for Ryan, 0.1326 for
M12 epoch 487, 0.1512 for M16 epoch 315, and 0.1368 for M17 epoch 63. M16
therefore narrowly exceeds Ryan on single-trial variance explained even though
Ryan has the much better trial-averaged PSTH correlation (0.8132 versus
0.6129). This is plausible because the single-trial metric can reward
trial-specific eye-position and behavior modulation that is averaged away in
the PSTH.

A live repeat for M17 epoch 139 at the same three validation-batch rows remains
localized and temporally compact. Its spatial high-frequency fractions are
0.0028, 0.0014, and 0.0073, versus 0.0019, 0.0012, and 0.0052 at epoch 63.
Thus training has sharpened the high-drive map modestly without approaching
M12's diffuse 0.0534 failure case. Its temporal high-frequency fractions are
0.0021-0.0033 and its sensitivity peaks at 29-42 ms.
The epoch-139 first-layer filter bank likewise remains spectrally clean: mean
power at or above 60 Hz is 0.00269, and the temporal spectra show negligible
energy above roughly 55-60 Hz. The leading separable components explain 79.2%
of filter energy on average. Several components retain slow tails at the oldest
history position, but the exemplar's end-to-end Jacobian has little energy
there; this is monitored as support truncation rather than classified as
aliasing.

M16 epoch 375 later tied epoch 315 on rotating validation (0.59220 versus
0.59227) and was preserved for deterministic comparison. On the same exemplar
contexts its spatial structure remains localized and its spatial
high-frequency fractions (0.0043, 0.0028, 0.0045) are comparable to epoch 315.
Its temporal high-frequency fractions rise to 0.0063, 0.0044, and 0.0118 from
0.0037, 0.0038, and 0.0069 at epoch 315. With no evident predictive advantage,
epoch 315 is currently the cleaner M16 checkpoint; exhaustive validation will
decide whether the rotating-score tie is real. The completed deterministic
all-session validation score for M16 epoch 315 is 0.58777 BPS, versus 0.59822
for M12 epoch 487. Full smoothness therefore recovers much of the early M16
deficit but remains 0.01045 BPS behind the unsmoothed M12 reference.

M16 epoch 415 set a new rotating-validation high of 0.5999 and was preserved
before top-k rotation. Its first-layer mean temporal power at or above 60 Hz is
0.00672. On the fixed exemplar validation contexts, temporal high-frequency
fractions are 0.0061, 0.0043, and 0.0123, while spatial high-frequency
fractions are 0.0060, 0.0029, and 0.0114. The maps remain localized and
center/flank structured, although the high-drive map is more textured than at
epoch 315 and one context has a small rise at the oldest 250-ms boundary.
Epoch 415 therefore passes the structural screen while being somewhat more
textured than epoch 315. A deterministic session-0 check confirms that most of
the predictive gain is real: Allen
2022-02-16 exhaustive validation rises from 0.66852 at M16 epoch 315 to 0.68118
at epoch 415. The matched M12 epoch-415 value is 0.68593, leaving only a
0.00475-BPS gap on this session (0.00659 versus mature M12 epoch 487).

The completed exhaustive 30-session score is **0.597272 BPS**, versus
**0.598220** for mature M12: a difference of only 0.000948 BPS (0.16%). Full
matched FixRSVP evaluation covers the same 1,703 cells and 24 sessions as
Figure 3. M16 reaches median raw PSTH correlation **0.4436**, essentially the
Figure-3 twin's **0.4468**, but normalized correlation is **0.5457** versus
**0.6356** and model variance explained is **0.0169** versus **0.0224**. On the
fixed unit-105 exemplar, M16 reaches 0.6324 raw correlation, 0.8414 CCnorm, and
0.1526 variance explained; the last quantity is slightly above Ryan's 0.1478.

The existing Ryan behavior decomposition localizes the population deficit:
Ryan's visual-only FixRSVP median CCnorm is 0.5429, almost exactly M16's intact
0.5457, while Ryan's intact behavior path raises it to 0.6351 and velocity
history alone reaches 0.6382. This makes an explicit nonrecurrent
behavior-residual head the primary successor experiment; simply increasing
generic visual capacity is not the first intervention. The predeclared design
and gates are in `paper/model_selection/M16_SUCCESSOR_SPEC.md`.

An isolated Figure-4 response-subspace pilot on the same 12 predeclared RR100
units produces localized, oriented M16 modes. Rank-4 held-out response R2 is
0.222 versus 0.240 in the existing Ryan pilot; neither model passes the 0.8
low-rank-surrogate gate. The visual interpretability problem is therefore
substantially improved, while compact response reduction remains an analysis
problem rather than a discriminator between the twins.

## Resume integrity

The first D240M16c continuation exposed a resume-only sampler issue: Lightning
restored model/optimizer epoch 64, but the recreated homogeneous batch sampler
restarted its private draw counter at epoch zero. Epochs 64-71 from that attempt
were discarded. `ByDatasetBatchSampler.set_epoch()` and the model's
`on_train_epoch_start` hook now synchronize the batch draw with restored
`current_epoch`. A regression assertion verifies that directly jumping a new
sampler to epoch N produces the same batches as an uninterrupted sampler's
epoch N. Long training is restarted from the untouched epoch-63 checkpoint.
The invalid epoch-67/71 and `last.ckpt` files were moved, not deleted, to
`invalid_resume_sampler_reset/`. Runtime losses diverged immediately between
the corrected and replayed trajectories, and the corrected epoch-67 checkpoint
scored 0.5199 BPS.

The same synchronization belongs in `on_validation_epoch_start`. A later audit
found that this Lightning hook had been defined twice in the model class;
Python silently retained only the later accumulator-reset definition, so the
first validation-sync implementation was inactive. The definitions are now
merged into one hook, with an AST regression assertion enforcing uniqueness
and the sync call. M16 was stopped after epoch 103 and resumed with the fixed
hook from its intact `last.ckpt`. Consequently the training trajectory is
valid throughout, but post-resume partial-validation scores through epoch 99
used a restarted validation-subset sequence; deterministic exhaustive
validation remains the final model-selection metric.

The exhaustive evaluator was also audited before final selection. Its original
implementation reused the homogeneous batch sampler, whose dataset and sample
draws are with replacement across batches; exhausting that loader does not
exhaust the underlying split. It now iterates each session dataset directly in
fixed order with `drop_last=False`, so every validation or test example is
visited exactly once. A synthetic regression assertion checks the final partial
batch and exact index coverage.

When D240M19c resumed from the preserved top-k epoch-127 checkpoint, Lightning
restored all states and then replayed a short epoch-127 validation loop before
starting epoch 128. That replay took 19 seconds rather than a normal 57-second
partial validation and produced a misleading `0.3076` checkpoint. It is marked
invalid and excluded from every comparison. Training began at epoch 128 with
the restored optimizer/scheduler; the first normal post-resume validation at
epoch 131 scored 0.5438 BPS. This artifact affects neither weights nor the
paired post-127 training sequence.

Behavior permutation at epoch 31 reduced held-out BPS by 0.0107 on the same
examples; zeroing behavior reduced it by 0.0075. Behavior is therefore making a
measurable contribution without recurrence.

M20b adds a zero-initialized, neuron-specific bounded-gain plus additive
behavior residual to the frozen M16 epoch-415 model. Its selected epoch-35
checkpoint scores **0.600893 BPS** on exhaustive 30-session validation, a
+0.003621 improvement over the exact inherited prediction. The mean and median
paired per-cell changes are +0.003999 and +0.002856 BPS, and 80.8% of 2,790
cells improve. Additive-only and gain-only ablations score 0.599723 and
0.600109. Shifting residual behavior within each batch lowers the score to
0.594379, establishing that the gain is behavior-aligned. A one-session
FixRSVP smoke is neutral relative to M16, so the full held-out generalization
gate is reserved for the best successor after readout co-adaptation and the
51-pixel surround branch.

## Native-240 normalization selection: M77

The native-240 sweep keeps both input and likelihood supervision at 240 Hz,
uses a causal 60-frame (250-ms) visual history, and applies the frequency mask
on that native lattice. The normalization comparison was stopped after a
12-validation-check plateau. M77 epoch 279, with GroupNorm followed by a weak
local-response-normalization term (`alpha=0.1`), is selected over the
historical-strength GN+LRN M76 and the weaker-LRN M78. M76 has a marginally
higher rotating validation score (0.5922 versus 0.5912), but M77 is decisively
better on held-out Allen 2022-04-13 FixRSVP: rho 0.4874 versus 0.4615, CCnorm
0.6292 versus 0.6172, and single-trial R2 0.0330 versus 0.0262. M78's rotating
score is 0.5927 but the same external gate is also worse than M77. Thus M77 is
the best external-generalizing normalization setting rather than the winner
of a noisy partial-validation ranking.

The final Ryan comparison uses exact common validation support and one
data-only FixRSVP reliability ceiling per neuron. The independent 24-session
audit reconstructs the finite support, CCabs, 500-split CCmax, stability mask,
and `CCnorm = CCabs / CCmax` identity to machine precision. On 30-session
validation, M77 is still below Ryan: overall BPS is 0.49541 versus 0.54616 and
the paired per-neuron median difference is -0.04976 (hierarchical 95% interval
[-0.05725, -0.04281]). This paired gate contains 204,122 common 120-Hz source
bins, 13.39% and 13.55% of the candidate and reference validation geometries;
the small fraction is the expected intersection of two independently formed
held-out splits, not a prediction-defined mask. On FixRSVP the gap is much
smaller. Across the 1,641
stable common units, median CCabs is 0.44277 versus 0.44730 and median CCnorm is
0.62868 versus 0.64023; the paired CCnorm difference is -0.01830 with interval
[-0.04587, 0.01875]. Across all 1,703 units, affine-calibrated single-trial R2
is tied: 0.02245 versus 0.02236, paired difference -0.00023 with interval
[-0.00137, 0.00131]. M77 therefore does not beat Ryan on in-distribution BPS,
but the audited FixRSVP correlations and single-trial variance are statistically
compatible with it at the session-bootstrap level.

The Poisson-likelihood comparison tells the same generalization story with a
useful nuance. On the identical 1,703-unit FixRSVP support, unadjusted
training-style aggregate BPS is effectively tied (Ryan 0.10817, M77 0.10781),
while the per-unit medians are 0.07501 and 0.06650 and the paired median
difference is -0.00562 (hierarchical 95% interval [-0.02111, 0.00881]). After
the same per-unit positive affine calibration, M77's aggregate is slightly
higher (0.16991 versus 0.15868), the per-unit medians are 0.14616 versus
0.14343, and the paired median difference is -0.00184 ([-0.00756, 0.00501]).
This is likelihood parity, not a declared M77 win: the unit-paired interval
crosses zero, but it rules out the earlier impression of a large FixRSVP
generalization deficit.

M77 retains the desired mechanistic structure. In the fixed population
Jacobian audit its median temporal second-difference ratio is 0.327, median
power at or above 60 Hz is 0.0118, and median spatial high-frequency fraction
is 0.0353. The corresponding Ryan exemplar Jacobians are visibly fragmented
and roughly an order of magnitude rougher on the shared physical contexts.
The respaced, cycle-valid periodic probe uses nine half-octave SFs from 1.07 to
15.94 cpd and fourteen dynamic TFs from 1 to 90.5 Hz. Preferred SF and TF are
now estimated with a local joint quadratic in log2 frequency around the sampled
two-dimensional maximum; the earlier global Gaussian center is retained only
as a bias diagnostic. Delete-one-local-sample stability, RMS/F1 agreement, and
explicit boundary censoring leave 22 trusted peaks. The M77 gradient subspace
is localized and oriented: rank 8 explains a median 0.3204 of held-out response
variance (minimum 0.2189) and captures 0.3578 of Jacobian energy. It is useful
for interpretation, though it does not satisfy the deliberately stringent 0.8
low-rank-surrogate gate.

A corrected causal Figure-4 production replay expands each retained 120-Hz eye
trace to the true 240-Hz model grid before rendering, holds the initial gaze
through the 59-frame history prefix, and integrates expected spikes with
`dt=1/240`. On 100 images by 1,000 traces, measured retinal motion increases
expected-spike-weighted pooled SSI by 6.86% (paired image-bootstrap 95% interval
[5.84%, 7.88%]); 96/100 images and all 1,000 trace-pooled effects are positive.
Exact layerwise replay on a separate 8-image by 8-trace subset shows that
temporal-filter drive grows with motion, the temporal stem alone changes
spatial information by -0.39%, and spatial stages 1/2/3 change it by +0.22%,
+4.54%, and +7.17%, respectively, before the RR100 output reaches +2.88%.
About 59.3% of exact stem drive lies in channels whose measured median TF is at
least 10 Hz. The original framewise
`TF = |k dot v(t)|` histogram is withdrawn as a retinal power estimate because
it discards temporal ordering and displacement correlations. The corrected
Rucci calculation estimates the two-sided DPSS spectrum of the complete
trajectory carrier `exp(-i 2 pi k.X(t))`, weighted by image Fourier power. It
matches directly rendered retinal-movie spectra well at aggregate scale
(cosine similarity 0.988; total-variation distance 0.073 in a 4-image by
8-trace validation).

The renderer-faithful production analysis measures the spatial and temporal FFT
of the actual 151-pixel retinal movies and uses the ideal trajectory-phase
spectrum only as a validated cross-check. Across 100 images and 16 evenly
sampled traces, total dynamic power has median within-unit image correlation
rho 0.366 with the causal rate change (95% interval [0.319, 0.396]); raw joint
passband-weighted drive is 0.358 ([0.310, 0.374]). After division by total
dynamic power, normalized joint alignment is -0.178 ([-0.258, -0.105]) and the
normalized TF-marginal score is -0.147 ([-0.186, -0.112]). Extra image
selectivity from joint versus separable alignment trends with SSI gain over all
units (rho 0.188, p=0.064, bootstrap interval [-0.004, 0.363]) but is absent in
the 22 trusted-peak units (rho -0.016, p=0.942). This establishes the claim
boundary: motion supplies dynamic power that enters the measured passband and
drives rate, and later spatial nonlinearities create the SSI increase; the
fraction of power in fine joint tuning does not explain which neurons benefit
most. Controlled trajectory scaling remains a qualitative 8-image by 8-trace
dose check rather than a production-scale causal estimate.

The Figure-3 ablation cache is schema-v7 and writes an atomic per-session
progress sidecar before publishing a completed cache. A production audit
rejects partial caches, requires the exact intact neuron order/noise ceiling,
checks `CCnorm = CCabs / CCmax` for every condition, and verifies that the
data-only term of the Figure-2-matched FEM decomposition is identical across
conditions. M77's completed 24-session/1,703-unit cache passes every invariant
with zero CCnorm identity error. In the production Figure 3 population (19
sessions after the canonical session floor), median held-out CCnorm is 0.656
full, 0.655 with the separate extraretinal branch zeroed, and 0.599 with the
retinal movie stabilized. Median Figure-2-normalized captured rate variance is
0.247 full and 0.059 stabilized (median paired change -0.163, 66% of full).
The empirical, full-model, and stabilized FEM-fraction medians are 0.736,
0.675, and 0.210; full M77 passes the predeclared paired TOST equivalence gate
at margin +/-0.1, while the stabilized condition does not. The focused native-
240/Figure-3/Figure-4 regression suite currently passes 205 tests.

The production Figure-4 trace bank is complete at 100 images by 1,000 traces,
with a separately scored image-matched stabilized baseline and passed
finite/nonnegative/shape/coordinate integrity checks. Its unit metadata never
inherits Ryan's coarse-grid SF labels. M77's cycle-valid response-weighted SF
centers are divided into stable tertile tails (32 lower, 34 middle, 32 upper;
two inactive RR100 channels remain explicit), and orientation preference/OSI
are remeasured from M77's native probe. The exact compositor and all
low/middle/high selection stages consume these labels without reverting to the
historical sub-cycle 0.5/0.75-cpd thresholds. Production outputs are under
`outputs/dekel240_paper/m77_epoch279/figure4_real_trace_production_corrected/`.

## Selection rule

1. Reject any run with readout-width collapse, unstable training, visibly
   aliased/checkerboard Jacobians, diffuse pixel-scale sensitivity, or temporal
   energy accumulating near Nyquist. Localized high-frequency structure,
   oriented flanks, and surrounds are not failures and may be mechanistically
   important.
2. Compare candidates on deterministic full validation, then report the final
   selected checkpoint on the untouched test split.
3. Among candidates that support mechanistic interpretation and are not badly
   aliased, prefer the best predictive model. Smoothness is a validity gate and
   a tie-breaker, not an objective to minimize without limit.
4. Re-run exact filters, readout widths, behavior ablation, and Jacobians on the
   selected checkpoint before using it as a subspace teacher.

## Commands

The current compatibility run passes 433 tests (one skipped) after excluding
the baseline-known `test_covariance.py`, the script-style
`test_frozencore_pipeline.py`, and `test_config_builds.py`, whose module-level
smoke code unconditionally allocates on `cuda:1` during collection. Executed
directly after the GPUs were released, `test_config_builds.py` completes its
adapter/frontend/convnet/modulator/recurrent/readout forward smoke with exit
status zero. Figure 3's
explainable-variance tests now use the same fixed three-bin matching history as
the shared Figure-2 contract and pass in the compatibility run. The new mixed-rate test-split stubs initially
exposed an assumption that every dataset wrapper has a `.dsets` attribute; the
optional output-behavior dtype check now uses a guarded lookup, and all four of
those tests pass.

```bash
conda run -n yatesfv python paper/model_selection/launch_dekel.py D240M15c --gpu 0
conda run -n yatesfv python paper/model_selection/launch_dekel.py D240M16c --gpu 1
conda run -n yatesfv python paper/model_selection/diagnose_dekel.py CHECKPOINT --gpu 0
conda run -n yatesfv python paper/model_selection/evaluate_dekel_split.py CHECKPOINT --split val --gpu 0
conda run -n yatesfv python paper/model_selection/compare_dekel_split_evaluations.py REFERENCE.json CANDIDATE.json
conda run -n yatesfv python paper/model_selection/compare_fixrsvp_exemplar_jacobians.py --help
conda run -n yatesfv python paper/model_selection/launch_dekel.py D240M38a --gpu 1 --pretrained-checkpoint M26_E95.ckpt
conda run -n yatesfv python paper/model_selection/launch_dekel.py D240M39a --gpu 0 --pretrained-checkpoint M26_E95.ckpt
conda run -n yatesfv python paper/model_selection/launch_dekel.py D240M40a --gpu 0 --pretrained-checkpoint SELECTED_M38.ckpt
conda run -n yatesfv python paper/model_selection/launch_dekel.py D240M45a --gpu 0 --pretrained-checkpoint SELECTED_M40.ckpt
```
