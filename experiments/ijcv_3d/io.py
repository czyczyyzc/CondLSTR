"""Small shared I/O helpers for journal experiments."""

import json

import torch


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")
