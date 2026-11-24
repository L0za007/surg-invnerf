"""Evaluate the depth rendered by InvNeRF on a SCARED keyframe against the ground truth.

For every fitted frame, the depth of each pixel is rendered at the resolution of the staged frames and
compared with the ground-truth depth (in mm) over the pixels where the ground truth is known.
"""

import argparse, json, re, sys, traceback
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import torch

sys.path.append(str(Path(__file__).parent.parent))
from config.config import get_config, get_saving_path
from DataManager import get_dataset
from DataManager.ray_manager import RayManager
from model import get_model
from engine import BaseTrainer


@torch.no_grad()
def render_depth(exp_dir: Path) -> np.ndarray:
    """Render the depth of every fitted frame.

    Returns:
        (n_imgs, H, W) depth in mm, at the resolution of the staged frames.
    """
    config = get_config(str(exp_dir / "config.yaml"), mk_out_dir=False)
    saving_path = get_saving_path(config)
    dataset, _ = get_dataset(config, saving_path, loaders=False, eval=True)
    handler = RayManager(config, dataset)
    models = get_model(config, handler.device, load=True)
    framework = BaseTrainer(config, models, criterion=None, label_manager=handler, save_path=str(exp_dir)).to(handler.device)
    framework.eval()
    grid = handler.gen_grid(normalize=False).float()
    grid_shape = grid.shape[:2]
    grid = grid.reshape(-1, 2)
    depths = []
    for img_id in torch.arange(dataset.num_imgs, device=handler.device):  # ids on the model device, as in vis.py
        depth, _ = framework.get_depthmap_and_rgb(img_id, grid)
        assert depth.numel() == grid.shape[0], f"solver/SCARED_eval.py|render_depth|mismatch shape depth {tuple(depth.shape)} against {(grid.shape[0],)}"
        depths.append(depth.reshape(*grid_shape).cpu().numpy() * 1000.0)  # m to mm
    return np.stack(depths)


def read_gt_depth_maps(depth_dir: Path, n_imgs: int) -> np.ndarray:
    """Read the ground-truth depth maps written by processes/SCARED.py: (n_imgs, H, W) in mm, NaN where unknown."""
    if not depth_dir.exists():
        raise FileNotFoundError(f"Ground-truth depth directory {depth_dir} does not exist")
    return np.stack([np.load(depth_dir / f"{idx:04d}.npy") for idx in range(n_imgs)])


def depth_metrics(gt_depth: np.ndarray, pred_depth: np.ndarray) -> dict:
    """Depth errors in mm over the pixels with a valid ground truth.

    Args:
        gt_depth: (n, H, W) ground-truth depth in mm, NaN or <= 0 where unknown.
        pred_depth: (n, H, W) rendered depth in mm.

    Returns:
        Mean absolute error, RMSE and log-RMSE, and the fraction of pixels with an error below 1, 2 and 5 mm.
    """
    assert gt_depth.shape == pred_depth.shape, f"solver/SCARED_eval.py|depth_metrics|mismatch shape pred_depth {pred_depth.shape} against {gt_depth.shape}"
    mask = np.isfinite(gt_depth) & (gt_depth > 0)
    error = np.abs(gt_depth - pred_depth)[mask]
    log_error = np.log(gt_depth[mask] + 1e-6) - np.log(np.clip(pred_depth[mask], 1e-6, None))
    return {
        "l1_error": float(error.mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "rmse_log": float(np.sqrt((log_error**2).mean())),
        "a1": float((error < 1).mean()),
        "a2": float((error < 2).mean()),
        "a3": float((error < 5).mean()),
        "valid_pixels": int(mask.sum()),
    }


def extract_total_time(log_file_path: Path):
    """Training time in hours, read from the training log."""
    with open(log_file_path, "r") as file:
        for line in file:
            match = re.search(r"Total time:\s*(\d{2}):(\d{2}):(\d{2})", line)
            if match:
                hours, minutes, seconds = map(int, match.groups())
                return round(hours + minutes / 60 + seconds / 3600, 2)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the rendered depth of InvNeRF on a SCARED keyframe")
    parser.add_argument("--out_dir", "-o", type=str, required=True, help="Processed keyframe folder, e.g. <DATA_DIR>/SCARED_InvNeRF/DS02_KF01")
    parser.add_argument("--experiment", "-e", type=int, default=1, help="Experiment number")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    failed = []
    for experiment_dir in sorted((out_dir / "out").glob(f"Test_{args.experiment}_*")):
        print(f"Evaluating {experiment_dir}")
        try:
            pred_depth = render_depth(experiment_dir)
            gt_depth = read_gt_depth_maps(out_dir / "depth", n_imgs=pred_depth.shape[0])
            metrics = depth_metrics(gt_depth, pred_depth)
            print(" | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()))
            exp_stem = experiment_dir.stem.split("_")
            result = {
                "Experiment": f"{exp_stem[0]} {exp_stem[1]}",
                "Date (aa-mm-dd)": exp_stem[2],
                "Starting time(hrs-min-sec)": exp_stem[3].replace("-", ":"),
                "Training time(hrs)": extract_total_time(experiment_dir / "log_screen.txt"),
                "Keyframe": out_dir.name,
                "Frames": int(pred_depth.shape[0]),
                **metrics,
            }
            with open(experiment_dir / "evaluation.json", "w") as f:
                json.dump(result, f, indent=4)
        except Exception:
            print(f"Error evaluating {experiment_dir}")
            with open(experiment_dir / "eval_error_log.txt", "w") as f:
                f.write(traceback.format_exc())
            failed.append(experiment_dir)
    if failed:
        print(f"Evaluation failed for {len(failed)} fit(s), see eval_error_log.txt in: " + ", ".join(str(d) for d in failed))
        sys.exit(1)
