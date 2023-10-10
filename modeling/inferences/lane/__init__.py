from .lane_det_2d import LaneDet2DInference
from .lane_det_3d import LaneDet3DInference


__factory = {
    'lane_det_2d': LaneDet2DInference,
    'lane_det_3d': LaneDet3DInference,
}
