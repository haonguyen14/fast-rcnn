from torch.utils.ffi import create_extension

ffi = create_extension(
    name="_ext.smooth_l1_loss",
    headers=["SmoothL1Loss.h"],
    sources=["SmoothL1Loss.c"],
    with_cuda=False
)

ffi.build()
