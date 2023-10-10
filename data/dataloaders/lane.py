from .dataloaders import simple_dataloader

__factory = {
    'culane':       simple_dataloader,
    'tusimple':     simple_dataloader,
    'curvelanes':   simple_dataloader,
    'openlane':     simple_dataloader,
    'apollo_sim':   simple_dataloader,
    'once_3dlanes': simple_dataloader,
    'lane_test':    simple_dataloader,
}
