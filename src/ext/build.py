import os
from torch.utils.ffi import create_extension

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = [os.path.join(this_file, "src/smooth_l1_loss_cuda.cu.o")]

ffi = create_extension(
    name="_ext.smooth_l1_loss",
    headers=["src/SmoothL1Loss.h", "src/SmoothL1Loss_cuda.h"],
    sources=["src/SmoothL1Loss.c", "src/SmoothL1Loss_cuda.c"],
    extra_objects=extra_objects,
    relative_to=__file__,
    define_macros=[("WITH_CUDA", None)],
    with_cuda=True,
)

ffi.build()
