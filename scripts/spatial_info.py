#%% Imports
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def embed_time_lags(movie, n_lags=32):
    """
    Embed time lags into a movie tensor.
    
    Input: movie (T, H, W) or (T, 1, H, W)
    Output: (T - n_lags + 1, 1, n_lags, H, W)
    """
    if movie.dim() == 3:
        movie = movie.unsqueeze(1)  # (T, 1, H, W)
    
    T, C, H, W = movie.shape
    # Create lagged indices: for each output frame t, we want frames [t, t+1, ..., t+n_lags-1]
    # But stim uses negative lags (past frames), so we want [t-n_lags+1, ..., t]
    out_frames = T - n_lags + 1
    
    # Build lagged tensor
    lagged = torch.zeros(out_frames, C, n_lags, H, W, dtype=movie.dtype, device=movie.device)
    for lag in range(n_lags):
        # lag 0 = current frame, lag 1 = 1 frame ago, etc.
        lagged[:, :, lag] = movie[n_lags - 1 - lag : T - lag]
    
    return lagged

def spatial_ssi_population(y, dt=1.0, eps=1e-8, log_base=2.0, spike_weighted=True):
    """
    Spatial single-spike information from a rate map.
    y: rates, shape [T, N, H, W]. Must be >= 0.
    Returns: ispikepop (bits/spike), iratepop (bits/sec), I_tn (T, N)
    """
    T, N, H, W = y.shape
    # T = time, N = units, H = height, W = width
    P = H * W # number of spatial bins
    r = y.reshape(T, N, P)
    rbar = r.mean(dim=2) # mean across space
    g = r / (rbar[..., None] + eps) # r/rbar
    logg = torch.log2(g + eps) if log_base == 2.0 else torch.log(g + eps) # log(r/rbar)
    
    I_tn = (g * logg).mean(dim=2) # expectation over space 
    
    # rescale to get bits per spike and per bin
    spikes_tn = rbar * dt                    # (T, N) expected spikes in bin

    # bits/sec per (time, neuron)
    bits_per_sec_tn = rbar * I_tn            # (T, N)

    if spike_weighted:
        # population bits/spike at each time t: sum_n spikes*I / sum_n spikes
        bits_t = (spikes_tn * I_tn).sum(dim=1)                 # (T,)
        spikes_t = spikes_tn.sum(dim=1)                        # (T,)
        ispike_t = bits_t / (spikes_t + eps)                   # (T,)
    else:
        # equal-weight neuron average
        ispike_t = I_tn.mean(dim=1)                            # (T,)

    # population bits/sec at each time t: sum_n rbar*I  (i.e., total bits/sec across neurons)
    irate_t = bits_per_sec_tn.sum(dim=1) 
    return ispike_t, irate_t, I_tn

class PopulationReadout(nn.Module):
    def __init__(
        self,
        feat_weights,
        biases,
        space_weights,
        logit_gains=None,
        logit_offsets=None,
        auxiliary_feat_weights=None,
        auxiliary_space_weights=None,
        residual_feat_weights=None,
        residual_space_weights=None,
        auxiliary_residual_feat_weights=None,
        auxiliary_residual_space_weights=None,
        residual_visual_feat_weights=None,
        residual_visual_space_weights=None,
    ):
        super().__init__()
        self.features = nn.Conv2d(feat_weights.shape[1], feat_weights.shape[0], kernel_size=1, bias=False)
        self.features.weight = nn.Parameter(feat_weights, requires_grad=False)
        self.bias = nn.Parameter(biases, requires_grad=False)
        self.space_weights = nn.Parameter(space_weights[:, None, :, :], requires_grad=False)
        self.n_units = space_weights.shape[0]
        if logit_gains is None:
            logit_gains = torch.zeros(self.n_units, dtype=feat_weights.dtype)
        if logit_offsets is None:
            logit_offsets = torch.zeros(self.n_units, dtype=feat_weights.dtype)
        self.logit_gains = nn.Parameter(logit_gains, requires_grad=False)
        self.logit_offsets = nn.Parameter(logit_offsets, requires_grad=False)
        if (auxiliary_feat_weights is None) != (auxiliary_space_weights is None):
            raise ValueError(
                "Auxiliary feature and spatial weights must be supplied together"
            )
        if auxiliary_feat_weights is None:
            self.auxiliary_features = None
            self.register_parameter("auxiliary_space_weights", None)
        else:
            if auxiliary_feat_weights.shape[0] != self.n_units:
                raise ValueError("Auxiliary feature/unit count mismatch")
            if auxiliary_space_weights.shape[0] != self.n_units:
                raise ValueError("Auxiliary spatial/unit count mismatch")
            self.auxiliary_features = nn.Conv2d(
                auxiliary_feat_weights.shape[1],
                auxiliary_feat_weights.shape[0],
                kernel_size=1,
                bias=False,
            )
            self.auxiliary_features.weight = nn.Parameter(
                auxiliary_feat_weights,
                requires_grad=False,
            )
            self.auxiliary_space_weights = nn.Parameter(
                auxiliary_space_weights[:, None, :, :],
                requires_grad=False,
            )
        if (residual_feat_weights is None) != (residual_space_weights is None):
            raise ValueError(
                "Residual feature and spatial weights must be supplied together"
            )
        if residual_feat_weights is None:
            self.residual_features = None
            self.register_parameter("residual_space_weights", None)
        else:
            if residual_feat_weights.shape[0] != self.n_units:
                raise ValueError("Residual feature/unit count mismatch")
            if residual_space_weights.shape[0] != self.n_units:
                raise ValueError("Residual spatial/unit count mismatch")
            self.residual_features = nn.Conv2d(
                residual_feat_weights.shape[1],
                residual_feat_weights.shape[0],
                kernel_size=1,
                bias=False,
            )
            self.residual_features.weight = nn.Parameter(
                residual_feat_weights,
                requires_grad=False,
            )
            self.residual_space_weights = nn.Parameter(
                residual_space_weights[:, None, :, :],
                requires_grad=False,
            )
        if (
            (auxiliary_residual_feat_weights is None)
            != (auxiliary_residual_space_weights is None)
        ):
            raise ValueError(
                "Auxiliary residual feature and spatial weights must be "
                "supplied together"
            )
        if auxiliary_residual_feat_weights is None:
            self.auxiliary_residual_features = None
            self.register_parameter("auxiliary_residual_space_weights", None)
        else:
            if auxiliary_residual_feat_weights.shape[0] != self.n_units:
                raise ValueError("Auxiliary residual feature/unit count mismatch")
            if auxiliary_residual_space_weights.shape[0] != self.n_units:
                raise ValueError("Auxiliary residual spatial/unit count mismatch")
            self.auxiliary_residual_features = nn.Conv2d(
                auxiliary_residual_feat_weights.shape[1],
                auxiliary_residual_feat_weights.shape[0],
                kernel_size=1,
                bias=False,
            )
            self.auxiliary_residual_features.weight = nn.Parameter(
                auxiliary_residual_feat_weights,
                requires_grad=False,
            )
            self.auxiliary_residual_space_weights = nn.Parameter(
                auxiliary_residual_space_weights[:, None, :, :],
                requires_grad=False,
            )
        if (
            (residual_visual_feat_weights is None)
            != (residual_visual_space_weights is None)
        ):
            raise ValueError(
                "Residual-visual feature and spatial weights must be "
                "supplied together"
            )
        if residual_visual_feat_weights is None:
            self.residual_visual_features = None
            self.register_parameter("residual_visual_space_weights", None)
        else:
            if residual_visual_feat_weights.shape[0] != self.n_units:
                raise ValueError("Residual-visual feature/unit count mismatch")
            if residual_visual_space_weights.shape[0] != self.n_units:
                raise ValueError("Residual-visual spatial/unit count mismatch")
            self.residual_visual_features = nn.Conv2d(
                residual_visual_feat_weights.shape[1],
                residual_visual_feat_weights.shape[0],
                kernel_size=1,
                bias=False,
            )
            self.residual_visual_features.weight = nn.Parameter(
                residual_visual_feat_weights,
                requires_grad=False,
            )
            self.residual_visual_space_weights = nn.Parameter(
                residual_visual_space_weights[:, None, :, :],
                requires_grad=False,
            )

    @staticmethod
    def _center_crop_spatial_map(value, height, width):
        current_h, current_w = value.shape[-2:]
        if current_h < height or current_w < width:
            raise ValueError(
                "Cannot center-crop spatial map "
                f"{(current_h, current_w)} to {(height, width)}"
            )
        top = (current_h - height) // 2
        left = (current_w - width) // 2
        return value[..., top:top + height, left:left + width]
    
    def forward(self, x, auxiliary_x=None, residual_visual_x=None):
        feat = self.features(x)
        
        space = F.conv2d(feat, self.space_weights, groups=self.n_units, padding="valid")
        if self.residual_features is not None:
            # The mature readout may consume behavior channels appended after
            # the visual core, whereas the localized residual is explicitly
            # visual-only.  Mirror ordinary forward by taking its declared
            # leading channel slice.
            residual_feat = self.residual_features(
                x[:, :self.residual_features.in_channels]
            )
            residual_space = F.conv2d(
                residual_feat,
                self.residual_space_weights,
                groups=self.n_units,
                padding="valid",
            )
            target_h = min(space.shape[-2], residual_space.shape[-2])
            target_w = min(space.shape[-1], residual_space.shape[-1])
            space = self._center_crop_spatial_map(space, target_h, target_w)
            residual_space = self._center_crop_spatial_map(
                residual_space, target_h, target_w
            )
            space = space + residual_space
        if self.auxiliary_features is not None:
            if auxiliary_x is None:
                raise ValueError(
                    "This population readout requires auxiliary visual features"
                )
            auxiliary_feat = self.auxiliary_features(auxiliary_x)
            auxiliary_space = F.conv2d(
                auxiliary_feat,
                self.auxiliary_space_weights,
                groups=self.n_units,
                padding="valid",
            )
            target_h = min(space.shape[-2], auxiliary_space.shape[-2])
            target_w = min(space.shape[-1], auxiliary_space.shape[-1])
            space = self._center_crop_spatial_map(space, target_h, target_w)
            auxiliary_space = self._center_crop_spatial_map(
                auxiliary_space,
                target_h,
                target_w,
            )
            space = space + auxiliary_space
        if self.auxiliary_residual_features is not None:
            if auxiliary_x is None:
                raise ValueError(
                    "This population readout requires auxiliary visual features"
                )
            auxiliary_residual_feat = self.auxiliary_residual_features(
                auxiliary_x
            )
            auxiliary_residual_space = F.conv2d(
                auxiliary_residual_feat,
                self.auxiliary_residual_space_weights,
                groups=self.n_units,
                padding="valid",
            )
            target_h = min(
                space.shape[-2], auxiliary_residual_space.shape[-2]
            )
            target_w = min(
                space.shape[-1], auxiliary_residual_space.shape[-1]
            )
            space = self._center_crop_spatial_map(space, target_h, target_w)
            auxiliary_residual_space = self._center_crop_spatial_map(
                auxiliary_residual_space, target_h, target_w
            )
            space = space + auxiliary_residual_space
        elif auxiliary_x is not None and self.auxiliary_features is None:
            raise ValueError(
                "Auxiliary features were supplied to a base-only readout"
            )
        if self.residual_visual_features is not None:
            if residual_visual_x is None:
                raise ValueError(
                    "This population readout requires residual visual features"
                )
            residual_visual_feat = self.residual_visual_features(
                residual_visual_x
            )
            residual_visual_space = F.conv2d(
                residual_visual_feat,
                self.residual_visual_space_weights,
                groups=self.n_units,
                padding="valid",
            )
            target_h = min(space.shape[-2], residual_visual_space.shape[-2])
            target_w = min(space.shape[-1], residual_visual_space.shape[-1])
            space = self._center_crop_spatial_map(space, target_h, target_w)
            residual_visual_space = self._center_crop_spatial_map(
                residual_visual_space, target_h, target_w
            )
            space = space + residual_visual_space
        elif residual_visual_x is not None:
            raise ValueError(
                "Residual visual features were supplied to a readout without "
                "that component"
            )
        out = space + self.bias[None, :, None, None]
        out = (
            out * (1.0 + self.logit_gains[None, :, None, None])
            + self.logit_offsets[None, :, None, None]
        )

        return out
    
def _output_cids_used(output, n_output_rows):
    """Return the cell identity associated with each McFarland result row."""
    for key in ("cids_used", "cids"):
        values = np.asarray(output.get(key, []))
        if values.ndim == 1 and values.size == int(n_output_rows):
            return values.astype(np.int64, copy=False)
    raise ValueError(
        f"McFarland output {output.get('sess', '<unknown>')!r} has "
        f"{n_output_rows} ccnorm rows but no equally sized cids_used/cids array."
    )


def get_spatial_readout(model, outputs, return_unit_rows=False):
    """
    Combine readouts from multiple datasets into a single readout.

    If return_unit_rows=True, also return per-channel provenance rows derived
    from the same session matching and ccnorm filter used to build the readout.
    """
    sessions = [outputs[i]['sess'] for i in range(len(outputs))]
    if len(set(sessions)) != len(sessions):
        raise ValueError("McFarland outputs contain duplicate session names.")

    # Keep the historical canonical channel ordering: model dataset order,
    # then McFarland ccnorm-row order within each session.  The ccnorm rows are
    # *not* readout row numbers.  They are aligned to outputs['cids_used'],
    # whereas readout rows are aligned to the dataset config's `cids` list.
    model_dataset_idx = [i for i, name in enumerate(model.names) if name in sessions]

    # make single readout

    convnet = getattr(model.model, "convnet", None)
    readout_mask_size = int(getattr(convnet, "scaffold_size", 14))
    feat_weights = []
    biases = []
    space_weights = []
    logit_gains = []
    logit_offsets = []
    auxiliary_readouts = getattr(model.model, "auxiliary_readouts", None)
    residual_readouts = getattr(model.model, "residual_readouts", None)
    auxiliary_residual_readouts = getattr(
        model.model, "auxiliary_residual_readouts", None
    )
    residual_visual_readouts = getattr(
        model.model, "residual_visual_readouts", None
    )
    auxiliary_convnet = getattr(model.model, "auxiliary_convnet", None)
    auxiliary_feat_weights = [] if auxiliary_readouts is not None else None
    auxiliary_space_weights = [] if auxiliary_readouts is not None else None
    residual_feat_weights = [] if residual_readouts is not None else None
    residual_space_weights = [] if residual_readouts is not None else None
    auxiliary_residual_feat_weights = (
        [] if auxiliary_residual_readouts is not None else None
    )
    auxiliary_residual_space_weights = (
        [] if auxiliary_residual_readouts is not None else None
    )
    residual_visual_feat_weights = (
        [] if residual_visual_readouts is not None else None
    )
    residual_visual_space_weights = (
        [] if residual_visual_readouts is not None else None
    )
    if auxiliary_readouts is not None:
        auxiliary_mask_size = int(
            getattr(auxiliary_convnet, "scaffold_size", readout_mask_size)
        )
    residual_convnet = getattr(model.model, "residual_convnet", None)
    if residual_visual_readouts is not None:
        residual_visual_mask_size = int(
            getattr(residual_convnet, "scaffold_size", readout_mask_size)
        )
    unit_rows = []
    channel = 0
    for model_readout_idx in model_dataset_idx:
        session = model.names[model_readout_idx]
        output_index = sessions.index(session)
        all_ccnorm = np.asarray(outputs[output_index]['ccnorm']['ccnorm'], dtype=np.float32)
        output_cids = _output_cids_used(outputs[output_index], all_ccnorm.size)
        selected_output_rows = np.flatnonzero(all_ccnorm > .5)

        model_cids = np.asarray(model.cfgs[model_readout_idx].get("cids", []), dtype=np.int64)
        if model_cids.ndim != 1:
            raise ValueError(f"Dataset cids for {session!r} must be one-dimensional.")
        if np.unique(model_cids).size != model_cids.size:
            raise ValueError(f"Dataset cids for {session!r} contain duplicates.")
        model_row_by_cid = {int(cid): int(row) for row, cid in enumerate(model_cids)}

        readout = model.model.readouts[model_readout_idx]
        feat_weight = readout.features.weight.detach().cpu()
        bias = readout.bias.detach().cpu()
        space_weight = readout.compute_gaussian_mask(
            readout_mask_size,
            readout_mask_size,
            model.device,
        ).detach().cpu()
        if auxiliary_readouts is not None:
            auxiliary_readout = auxiliary_readouts[model_readout_idx]
            if auxiliary_readout.features is None:
                raise TypeError(
                    "Figure 4 requires an independent auxiliary feature projection"
                )
            auxiliary_feat_weight = (
                auxiliary_readout.features.weight.detach().cpu()
            )
            auxiliary_space_weight = auxiliary_readout.compute_gaussian_mask(
                auxiliary_mask_size,
                auxiliary_mask_size,
                model.device,
                auxiliary_readout.features.weight.dtype,
                readout,
            ).detach().cpu()
        if residual_readouts is not None:
            residual_readout = residual_readouts[model_readout_idx]
            if residual_readout.features is None:
                raise TypeError(
                    "Figure 4 requires an independent residual feature projection"
                )
            residual_feat_weight = residual_readout.features.weight.detach().cpu()
            residual_space_weight = residual_readout.compute_gaussian_mask(
                readout_mask_size,
                readout_mask_size,
                model.device,
                residual_readout.features.weight.dtype,
                readout,
            ).detach().cpu()
        if auxiliary_residual_readouts is not None:
            auxiliary_residual_readout = auxiliary_residual_readouts[
                model_readout_idx
            ]
            if auxiliary_residual_readout.features is None:
                raise TypeError(
                    "Figure 4 requires an independent auxiliary residual "
                    "feature projection"
                )
            auxiliary_residual_feat_weight = (
                auxiliary_residual_readout.features.weight.detach().cpu()
            )
            auxiliary_residual_space_weight = (
                auxiliary_residual_readout.compute_gaussian_mask(
                    auxiliary_mask_size,
                    auxiliary_mask_size,
                    model.device,
                    auxiliary_residual_readout.features.weight.dtype,
                    readout,
                ).detach().cpu()
            )
        if residual_visual_readouts is not None:
            residual_visual_readout = residual_visual_readouts[
                model_readout_idx
            ]
            if residual_visual_readout.features is None:
                raise TypeError(
                    "Figure 4 requires an independent residual-visual "
                    "feature projection"
                )
            residual_visual_feat_weight = (
                residual_visual_readout.features.weight.detach().cpu()
            )
            residual_visual_space_weight = (
                residual_visual_readout.compute_gaussian_mask(
                    residual_visual_mask_size,
                    residual_visual_mask_size,
                    model.device,
                    residual_visual_readout.features.weight.dtype,
                    readout,
                ).detach().cpu()
            )

        output_modulator = getattr(model.model, "output_modulator", None)
        if output_modulator is None:
            output_gain = torch.zeros_like(bias)
            output_offset = torch.zeros_like(bias)
        else:
            behavior = torch.zeros(
                1,
                output_modulator.behavior_dim,
                device=model.device,
                dtype=next(model.model.parameters()).dtype,
            )
            with torch.no_grad():
                output_gain, output_offset = output_modulator.gain_offset(
                    behavior,
                    model_readout_idx,
                )
            output_gain = output_gain[0].detach().cpu()
            output_offset = output_offset[0].detach().cpu()

        if feat_weight.shape[0] != model_cids.size:
            raise ValueError(
                f"Readout for {session!r} has {feat_weight.shape[0]} rows but "
                f"its dataset config has {model_cids.size} cids."
            )

        for source_output_row in selected_output_rows:
            source_cid = int(output_cids[source_output_row])
            model_readout_row = model_row_by_cid.get(source_cid)
            available = model_readout_row is not None
            if available:
                row = int(model_readout_row)
                feat_weights.append(feat_weight[row : row + 1])
                biases.append(bias[row : row + 1])
                space_weights.append(space_weight[row : row + 1])
                logit_gains.append(output_gain[row : row + 1])
                logit_offsets.append(output_offset[row : row + 1])
                if auxiliary_readouts is not None:
                    auxiliary_feat_weights.append(
                        auxiliary_feat_weight[row : row + 1]
                    )
                    auxiliary_space_weights.append(
                        auxiliary_space_weight[row : row + 1]
                    )
                if residual_readouts is not None:
                    residual_feat_weights.append(
                        residual_feat_weight[row : row + 1]
                    )
                    residual_space_weights.append(
                        residual_space_weight[row : row + 1]
                    )
                if auxiliary_residual_readouts is not None:
                    auxiliary_residual_feat_weights.append(
                        auxiliary_residual_feat_weight[row : row + 1]
                    )
                    auxiliary_residual_space_weights.append(
                        auxiliary_residual_space_weight[row : row + 1]
                    )
                if residual_visual_readouts is not None:
                    residual_visual_feat_weights.append(
                        residual_visual_feat_weight[row : row + 1]
                    )
                    residual_visual_space_weights.append(
                        residual_visual_space_weight[row : row + 1]
                    )
            else:
                # Preserve the exact 756-channel production coordinate system.
                # An unavailable cell is an explicitly inactive placeholder;
                # downstream RR population adaptation can substitute another
                # available member of its redundancy group where possible.
                feat_weights.append(torch.zeros_like(feat_weight[:1]))
                biases.append(torch.full_like(bias[:1], -50.0))
                space_weights.append(torch.zeros_like(space_weight[:1]))
                logit_gains.append(torch.zeros_like(output_gain[:1]))
                logit_offsets.append(torch.zeros_like(output_offset[:1]))
                if auxiliary_readouts is not None:
                    auxiliary_feat_weights.append(
                        torch.zeros_like(auxiliary_feat_weight[:1])
                    )
                    auxiliary_space_weights.append(
                        torch.zeros_like(auxiliary_space_weight[:1])
                    )
                if residual_readouts is not None:
                    residual_feat_weights.append(
                        torch.zeros_like(residual_feat_weight[:1])
                    )
                    residual_space_weights.append(
                        torch.zeros_like(residual_space_weight[:1])
                    )
                if auxiliary_residual_readouts is not None:
                    auxiliary_residual_feat_weights.append(
                        torch.zeros_like(auxiliary_residual_feat_weight[:1])
                    )
                    auxiliary_residual_space_weights.append(
                        torch.zeros_like(auxiliary_residual_space_weight[:1])
                    )
                if residual_visual_readouts is not None:
                    residual_visual_feat_weights.append(
                        torch.zeros_like(residual_visual_feat_weight[:1])
                    )
                    residual_visual_space_weights.append(
                        torch.zeros_like(residual_visual_space_weight[:1])
                    )

            unit_rows.append(
                {
                    "channel": int(channel),
                    "session": str(session),
                    # Retain the old field name for existing consumers, but
                    # make its coordinate system explicit in the new fields.
                    "source_unit_index": int(source_output_row),
                    "source_output_row": int(source_output_row),
                    "source_cid": int(source_cid),
                    "model_readout_row": (
                        int(model_readout_row) if model_readout_row is not None else None
                    ),
                    "available": bool(available),
                    "ccnorm": float(all_ccnorm[source_output_row]),
                    "model_readout_index": int(model_readout_idx),
                    "mcfarland_output_index": int(output_index),
                }
            )
            channel += 1

    if not feat_weights:
        raise ValueError("No overlapping McFarland/model sessions produced canonical readout channels.")

    feat_weights = torch.cat(feat_weights, dim=0)
    biases = torch.cat(biases, dim=0)
    space_weights = torch.cat(space_weights, dim=0)
    logit_gains = torch.cat(logit_gains, dim=0)
    logit_offsets = torch.cat(logit_offsets, dim=0)
    if auxiliary_readouts is not None:
        auxiliary_feat_weights = torch.cat(
            auxiliary_feat_weights,
            dim=0,
        )
        auxiliary_space_weights = torch.cat(
            auxiliary_space_weights,
            dim=0,
        )
    if residual_readouts is not None:
        residual_feat_weights = torch.cat(residual_feat_weights, dim=0)
        residual_space_weights = torch.cat(residual_space_weights, dim=0)
    if auxiliary_residual_readouts is not None:
        auxiliary_residual_feat_weights = torch.cat(
            auxiliary_residual_feat_weights, dim=0
        )
        auxiliary_residual_space_weights = torch.cat(
            auxiliary_residual_space_weights, dim=0
        )
    if residual_visual_readouts is not None:
        residual_visual_feat_weights = torch.cat(
            residual_visual_feat_weights, dim=0
        )
        residual_visual_space_weights = torch.cat(
            residual_visual_space_weights, dim=0
        )

    # print(feat_weights.shape, biases.shape, space_weights.shape)
    readout = PopulationReadout(
        feat_weights,
        biases,
        space_weights,
        logit_gains=logit_gains,
        logit_offsets=logit_offsets,
        auxiliary_feat_weights=auxiliary_feat_weights,
        auxiliary_space_weights=auxiliary_space_weights,
        residual_feat_weights=residual_feat_weights,
        residual_space_weights=residual_space_weights,
        auxiliary_residual_feat_weights=auxiliary_residual_feat_weights,
        auxiliary_residual_space_weights=auxiliary_residual_space_weights,
        residual_visual_feat_weights=residual_visual_feat_weights,
        residual_visual_space_weights=residual_visual_space_weights,
    )
    if return_unit_rows:
        return readout, unit_rows
    return readout

def compute_rate_map(
    model,
    readout,
    stim,
    behavior=None,
    output_behavior=None,
):
    """Compute rate map from stimulus and optional behavior.

    ``behavior`` (N, n_vars) is required for behavior-conditioned twins
    (e.g. concat/FiLM modulators); leave it None for none-modulator twins.
    """
    resolve_behavior = getattr(
        model.model, "resolve_feature_behavior", None
    )
    # Legacy spatial probes expose one behavior tensor.  A zero-behavior
    # counterfactual is identical under both contracts, so let that explicit
    # tensor serve the selected feature route when a separate lower-rate
    # tensor was not supplied.  Nonzero dual-contract analyses should pass
    # output_behavior explicitly.
    feature_output_behavior = output_behavior
    if (
        feature_output_behavior is None
        and getattr(model.model, "feature_behavior_source", "behavior")
        == "output_behavior"
    ):
        feature_output_behavior = behavior
    feature_behavior = (
        resolve_behavior(behavior, feature_output_behavior)
        if resolve_behavior is not None
        else behavior
    )
    spatial_forward = getattr(model.model, "core_forward_spatial_map", None)
    x = (
        spatial_forward(stim, feature_behavior)
        if spatial_forward is not None
        else model.model.core_forward(stim, feature_behavior)
    )
    auxiliary_spatial_forward = getattr(
        model.model,
        "auxiliary_visual_forward_spatial_map",
        None,
    )
    auxiliary_x = (
        auxiliary_spatial_forward(stim)
        if auxiliary_spatial_forward is not None
        else None
    )
    residual_visual_spatial_forward = getattr(
        model.model,
        "residual_visual_forward_spatial_map",
        None,
    )
    residual_visual_x = (
        residual_visual_spatial_forward(stim)
        if residual_visual_spatial_forward is not None
        else None
    )
    y_batch = readout(
        x[:, :, -1],
        auxiliary_x[:, :, -1] if auxiliary_x is not None else None,
        (
            residual_visual_x[:, :, -1]
            if residual_visual_x is not None
            else None
        ),
    )

    return model.model.activation(y_batch)

def compute_rate_map_batched(
    model,
    readout,
    stim,
    batch_size=32,
    behavior=None,
    output_behavior=None,
):
    """Compute rate map from stimulus and optional behavior in batches.

    When ``behavior`` (N, n_vars) is provided it is chunked in lockstep with
    ``stim`` along the leading (time) axis.
    """
    device = next(model.model.parameters()).device
    T = stim.shape[0]
    y_chunks = []

    model.model.eval()
    readout.eval()

    with torch.no_grad():
        for t_start in range(0, T, batch_size):
            t_end = min(t_start + batch_size, T)

            # Move batch to GPU
            x = stim[t_start:t_end].to(device)
            beh = behavior[t_start:t_end].to(device) if behavior is not None else None
            output_beh = (
                output_behavior[t_start:t_end].to(device)
                if output_behavior is not None
                else None
            )

            y_batch = compute_rate_map(
                model,
                readout,
                x,
                behavior=beh,
                output_behavior=output_beh,
            )

            # Move to CPU immediately
            y_chunks.append(y_batch.cpu())
            del y_batch
            torch.cuda.empty_cache()

    return torch.cat(y_chunks, dim=0)

def make_movie(y, save_path='', n_units_to_show=100):
    from torchvision.utils import make_grid
    from matplotlib.animation import FFMpegWriter

    # if n_units_to_show is list or array, use it as index
    if isinstance(n_units_to_show, (list, np.ndarray)):
        units_to_show = np.array(n_units_to_show)
        n_units_to_show = len(units_to_show)
    else:
        units_to_show = np.arange(n_units_to_show)
    
    y_subset = y[:, units_to_show].detach().cpu()  # (T, N, H, W)
    
    # normalize each unit to [0,1]
    # miny = torch.tensor(np.array([y_subset[:,i].min() for i in range(n_units_to_show)]))
    # maxy = torch.tensor(np.array([y_subset[:,i].max() for i in range(n_units_to_show)]))
    # y_subset = (y_subset - miny[None,:,None,None]) / (maxy[None,:,None,None] - miny[None,:,None,None] + 1e-8)

    # normalize each unit to 0 mean, 1 std (better)
    std = y_subset.std(dim=(0, 2, 3), keepdim=True)
    mu = y_subset.mean(dim=(0, 2, 3), keepdim=True)
    y_subset = (y_subset - mu) / (std + 1e-8)
    
    T = y_subset.shape[0]
    nrow = int(np.ceil(np.sqrt(n_units_to_show)))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.axis('off')

    writer = FFMpegWriter(fps=15, codec='libx264', bitrate=8000)

    save_path = f'../figures/{save_path}.mp4'
    with writer.saving(fig, save_path, dpi=100):
        for t in range(T):
            ax.clear()
            # make_grid expects (N, C, H, W), add channel dim
            frames = y_subset[t].unsqueeze(1)  # (N, 1, H, W)
            grid = make_grid(frames, nrow=nrow, normalize=False, padding=1, pad_value=0.0)
            ax.imshow(grid[0].numpy(), cmap='gray', vmin=-6, vmax=6)
            ax.set_title(f'Spatial Activations - Frame {t}/{T}', fontsize=14)
            ax.axis('off')
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved spatial activations movie to {save_path}")

# Reconstruct stimulus
def make_stimulus_stack(type='fixrsvp', frame=None, frames_per_im=6, num_frames=500):
    """
    Make a stimulus stack for a given type. 
    Input:
        type: 
            'fixrsvp': fixrsvp images
            'face': marmoset face images
            'nat': natural images
        frame: frame number to use for all time points (None flashes frames at framerate specified by frames_per_im)
        frames_per_im: number of frames to show each image for (if frame is None)
            This specifies the frame rate (if frames_per_im = 1, then the update is the screen rate (e.g., 120Hz), if 2, then 60Hz)
        num_frames: number of frames to generate (if frame is None)
    """

    from mcfarland_sim import get_fixrsvp_stack

    if type == 'fixrsvp':
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='im')
    elif type == 'face':
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='face')
    elif type == 'nat':
        full_stack = get_fixrsvp_stack(frames_per_im=frames_per_im, prefix='nat')

    if frame is not None:
        full_stack = full_stack[[frame]].repeat(num_frames, axis=0)

    return full_stack

def make_counterfactual_stim(full_stack, eyepos,
                            ppd = 37.50476617,
                            scale_factor = 1.0,
                            n_lags = 32,
                            out_size = (101, 101)):
    '''
    Reconstruct stimulus from eye positions.
    
    Input:
        eyepos: [T, 2] eye positions in degrees
        type: 'fixrsvp', 'face', 'nat'
        frame: frame number to use for all time points (None flashes frames at framerate specified by frames_per_im)
        frames_per_im: number of frames to show each image for (if frame is None)
        ppd: pixels per degree
        scale_factor: scale factor for stimulus (1.0 is no scaling)
        n_lags: number of time lags to use
        out_size: (H, W) size of output stimulus
    '''

    from mcfarland_sim import eye_deg_to_norm, shift_movie_with_eye

    eye_norm = eye_deg_to_norm(torch.fliplr(eyepos), ppd, full_stack.shape[1:3])

    eye_movie = shift_movie_with_eye(
        torch.from_numpy(full_stack[:eyepos.shape[0] + n_lags]).float(),
        torch.cat([eye_norm[:n_lags], eye_norm], dim=0),  # pad beginning
        out_size=out_size,
        center=(0.0, 0.0),
        scale_factor=scale_factor,
        mode="bilinear"
    )

    # Embed time lags to match stim shape
    eye_stim = embed_time_lags(eye_movie, n_lags=n_lags)

    return eye_stim
# %%
