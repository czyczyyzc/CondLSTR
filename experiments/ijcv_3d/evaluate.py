"""Evaluate a journal 3D checkpoint with the official OpenLane evaluator."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets.lane.openlane.openlane import OpenLaneDataset
from experiments.ijcv_3d.data import openlane_transforms_3d, validate_geometry
from experiments.ijcv_3d.geometry import bev_points_to_ego
from experiments.ijcv_3d.io import autocast_dtype, move, write_json
from modeling.models.models.lane.cond_lstr_3d import build_model
from tools.metrics.lane.openlane.lane3d.eval_3D_lane import LaneEval


def export_predictions(model, loader, output, device):
    output.mkdir(parents=True, exist_ok=False)
    identities = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            metadata = batch["img_metas"][0]
            with torch.autocast("cuda", dtype=autocast_dtype()):
                prediction = model(move(batch, device))
            identity = "validation/" + metadata["filename"].replace(",", "/")
            identities.append(identity)
            lanes = []
            triples = zip(
                prediction["lane_points"][0],
                prediction["lane_attris"][0],
                prediction["lane_scores"][0],
            )
            for lane, category, score in triples:
                if prediction.get("lane_points_frame") == "ego":
                    xyz = np.asarray(lane)
                else:
                    points, _ = lane
                    xyz = bev_points_to_ego(points, metadata["calibration"]["trans_bev2ego"])
                lanes.append({"xyz": xyz.tolist(), "category": int(category), "score": float(score)})
            path = output / Path(identity).with_suffix(".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"file_path": identity, "lane_lines": lanes}))
    listing = output / "test_list.txt"
    listing.write_text("\n".join(identities) + "\n")
    return listing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "complete.json")
    parser.add_argument("--cache-version", default="3d")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    validate_geometry(config)
    transform, collate = openlane_transforms_3d(train=False)
    dataset = OpenLaneDataset(str(args.data_root), "val", transform, version=args.cache_version)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate)
    device = torch.device("cuda:0")
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config")
    if saved_config is not None and saved_config != config:
        raise ValueError("Checkpoint and requested configuration differ")
    saved_model = checkpoint.get("model", checkpoint)
    saved_model = {key.removeprefix("module."): value for key, value in saved_model.items()}
    model.load_state_dict(saved_model)
    listing = export_predictions(model, loader, args.output, device)

    evaluator = LaneEval(SimpleNamespace(
        dataset_dir=str(args.data_root / "lane3d_1000"),
        pred_dir=str(args.output),
        test_list=str(listing),
    ))
    values = evaluator.bench_one_submit(
        str(args.output) + "/",
        str(args.data_root / "lane3d_1000") + "/",
        str(listing),
    )
    keys = [
        "F1", "recall", "precision", "category_accuracy", "x_error_close", "x_error_far",
        "z_error_close", "z_error_far", "recall_matches", "precision_matches",
        "category_matches", "gt_lanes", "predicted_lanes", "matches",
    ]
    metrics = dict(zip(keys, [float(value) if np.isfinite(value) else None for value in values]))
    metrics["validation_samples"] = len(dataset)
    write_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
