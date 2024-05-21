from enum import Enum
class NopeNerfDepthLossType(Enum):
    """Types of depth losses for depth supervision."""

    DS_NERF = 1
    URF = 2
    SPARSENERF_RANKING = 3
    NOPE_NERF = 4
    RELATIVE_LOSS = 5
    