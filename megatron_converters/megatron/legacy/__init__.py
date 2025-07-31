"""
Legacy module for Megatron-LM.
This module contains legacy components that are still used by conversion scripts.
"""

# Import commonly used components
from .model import module

# For conversion scripts, we don't need fused_kernels
fused_kernels_load = None

# Make these available at the legacy level
__all__ = ["module", "fused_kernels_load"]
