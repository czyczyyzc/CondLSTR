from .culane import CULaneDataset
from .tusimple import TusimpleDataset
from .curvelanes import CurveLanesDataset
from .openlane import OpenLaneDataset
from .apollo_sim import ApolloSimDataset
from .once_3dlanes import ONCE3DLanesDataset
from .lane_test import LaneTestDataset


__factory = {
    'culane':       CULaneDataset,
    'tusimple':     TusimpleDataset,
    'curvelanes':   CurveLanesDataset,
    'openlane':     OpenLaneDataset,
    'apollo_sim':   ApolloSimDataset,
    'once_3dlanes': ONCE3DLanesDataset,
    'lane_test':    LaneTestDataset,
}
