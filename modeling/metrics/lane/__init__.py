from .lane_det_2d import LaneDet2DMetric
from .lane_det_3d import LaneDet3DMetric


__factory = {
    'lane_det_2d': LaneDet2DMetric,
    'lane_det_3d': LaneDet3DMetric,
}
