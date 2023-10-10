from .culane import culane_transforms
from .tusimple import tusimple_transforms
from .curvelanes import curvelanes_transforms
from .openlane import openlane_transforms
from .apollo_sim import apollo_sim_transforms
from .once_3dlanes import once_3dlanes_transforms
from .lane_test import lane_test_transforms


__factory = {
    'culane':       culane_transforms,
    'tusimple':     tusimple_transforms,
    'curvelanes':   curvelanes_transforms,
    'openlane':     openlane_transforms,
    'apollo_sim':   apollo_sim_transforms,
    'once_3dlanes': once_3dlanes_transforms,
    'lane_test':    lane_test_transforms,
}
