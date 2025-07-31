"""
Megatron-LM main module.
This is the main entry point for the Megatron-LM library.
"""

# Import commonly used modules
from .training import arguments, global_vars
from .core import mpu, enums, parallel_state
from .legacy import module, fused_kernels

# Make these available at the megatron level
__all__ = ["arguments", "global_vars", "mpu", "enums", "module", "fused_kernels", "parallel_state"]
