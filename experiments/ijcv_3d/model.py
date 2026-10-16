"""Compatibility import for the journal training and evaluation entry points.

The canonical unified model lives with the other lane models instead of under
the experiment runner.
"""

from modeling.models.models.lane.cond_lstr_3d import (
    AttentionBlock,
    CondLSTR3D,
    build_model,
)

__all__ = ['AttentionBlock', 'CondLSTR3D', 'build_model']
