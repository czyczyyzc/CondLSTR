from __future__ import print_function, absolute_import


def get_params(model, name=''):
    if hasattr(model.module, name):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    return param_groups
