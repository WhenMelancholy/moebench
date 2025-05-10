# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .impls.fast_dispatch import (
    extract_critical,
    fast_decode,
    fast_dispatcher,
    fast_encode,
)

# Low-level Ops
from .jit_kernels.gating import fast_cumsum_sub_one

top_k_routing = extract_critical

# High-level Ops
from .impls.moe_layer import moe_layer
