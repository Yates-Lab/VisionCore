import torch

from .conv_layers import StandardConv, DepthwiseConv
from .initializers import gaussian_steerable_kernels_2d, init_conv3d_with_2d_kernels

def make_steerable_conv(*,
    in_channels:int,
    kernel_hw:int, kt:int=1,
    sigmas=(1.6,2.8,5.0), n_orient=8, orders=(1,2),
    aa=True, wn=False, unit_norm=False,
    padding_hw=None, temporal="delta",
    **conv_kwargs
):
    if padding_hw is None:
        padding = (0 if kt==1 else kt//2, kernel_hw//2, kernel_hw//2)
    else:
        padding = (0 if kt==1 else kt//2, *( (padding_hw, padding_hw) if isinstance(padding_hw,int) else padding_hw ))

    # build the bank FIRST → defines out_channels
    # (device/dtype taken from a temp tensor to avoid instantiating the conv first)
    tmp = torch.empty(1, dtype=torch.float32)  # will be ignored; just for signature parity
    bank = gaussian_steerable_kernels_2d(
        kernel_size=kernel_hw, sigmas=sigmas,
        n_orient=n_orient, orders=orders,
        # device/dtype will be adjusted after conv is created; it’s fine to leave defaults here
    )

    out_channels = bank.shape[0]

    conv = StandardConv(
        in_channels, out_channels,
        kernel_size=(kt, kernel_hw, kernel_hw), padding=padding,
        aa_signal=aa, use_weight_norm=wn, keep_unit_norm=unit_norm,
        **conv_kwargs
    )
    # move bank onto conv’s weight dtype/device, then init
    bank = bank.to(device=conv.conv.weight.device, dtype=conv.conv.weight.dtype)
    init_conv3d_with_2d_kernels(conv.conv, bank, temporal=temporal)
    return conv

def make_steerable_depthwise(*,
    in_channels:int,
    kernel_hw:int, kt:int=1,
    sigmas=(1.6,2.8,5.0), n_orient=8, orders=(1,2),
    aa=True, wn=False, unit_norm=False,
    padding_hw=None, temporal="delta",
    **conv_kwargs
):
    if padding_hw is None:
        padding = (0 if kt==1 else kt//2, kernel_hw//2, kernel_hw//2)
    else:
        padding = (0 if kt==1 else kt//2, *( (padding_hw, padding_hw) if isinstance(padding_hw,int) else padding_hw ))

    bank = gaussian_steerable_kernels_2d(
        kernel_size=kernel_hw, sigmas=sigmas,
        n_orient=n_orient, orders=orders,
    )
    out_channels = bank.shape[0]

    conv = DepthwiseConv(
        in_channels, out_channels,
        kernel_size=(kt, kernel_hw, kernel_hw), padding=padding,
        aa_signal=aa, use_weight_norm=wn, keep_unit_norm=unit_norm,
        **conv_kwargs
    )
    bank = bank.to(device=conv.depthwise.weight.device, dtype=conv.depthwise.weight.dtype)
    init_conv3d_with_2d_kernels(conv.depthwise, bank, temporal=temporal)
    return conv
