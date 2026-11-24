"""Script to execute the complete pipeline on SCARED keyframes
1. Unpack the frame data (calibration and camera poses) and the scene points of the keyframe
2. Rectify the stereo frames and compute the ground-truth disparity with MICCAI_challenge_preprocess
3. Stage every second rectified frame with the rectified calibration, the camera poses and the ground-truth depth
4. Run preprocessing (correspondences with Preprocessing4tracker)
5. Run tracker
6. Run evaluation (rendered depth against the ground truth)
7. Clean up
"""

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import numpy as np
from PIL import Image
import os, sys, subprocess, json, argparse, imageio, tarfile, tempfile
import cv2, yaml
import matplotlib.pyplot as plt

#: Repository root, the working directory for the training and evaluation calls.
REPO_DIR = Path(__file__).resolve().parents[1]
#: Flow folders written by the preprocessing, deleted when a keyframe is preprocessed again with other settings.
FLOW_FOLDERS = [
    "raft_exhaustive",
    "raft_exhaustive_stereo",
    "raft_masks",
    "cotracker",
    "cotracker_masks",
    "features",
    "count_maps",
    "3D_info",
    "of_vis",
    "flow_stats.json",
]
#: Image size of the SCARED frames (H, W). The video stacks the left frame on top of the right one.
FRAME_SIZE = (1024, 1280)
#: Width in pixels, at full resolution, of the band on the left of the rectified frames without image content.
RECTIFICATION_BORDER = 48


def env_path(key: str) -> Path:
    """Read a path from the .env file in the parent directory of the repository."""
    load_dotenv(find_dotenv(usecwd=True))
    if os.getenv(key) is None:
        raise ValueError(f"{key} is not defined. Add it to the .env file in the parent directory of the repository")
    return Path(os.getenv(key))


def get_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Process SCARED dataset")
    parser.add_argument("--exp_n", type=int, default=1, help="Experiment number, selects config/SCARED/Exp<exp_n>.yaml")
    parser.add_argument("--flow", type=str, default="mft", choices=["raft", "mft"], help="Short-term correspondences computed in the preprocessing")
    parser.add_argument(
        "--keyframes", type=str, nargs="+", default=None, help="Process only these keyframes, e.g. dataset_2/keyframe_1. All keyframes if not given"
    )
    parser.add_argument("--dataset", type=str, default="scared2019", help="Folder of the extracted SCARED dataset, relative to DATA_DIR or absolute")
    parser.add_argument(
        "--workspace", type=str, default="SCARED_InvNeRF", help="Folder for the processed keyframes and results, relative to DATA_DIR or absolute"
    )
    parser.add_argument("--preprocessing_dir", type=str, default=str(REPO_DIR.parent / "Preprocessing4tracker"), help="Clone of Preprocessing4tracker")
    parser.add_argument("--no_reprocess", action="store_true", help="Skip the keyframes that already have results")
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep frames, flows and caches after the evaluation")
    return parser.parse_args()


def get_keyframe_list(scared_dir: Path, workspace: Path, keyframes=None):
    """List the keyframe folders (dataset_<d>/keyframe_<k>) and their output folders (DS<d>_KF<k>)."""
    kf_dirs = sorted(scared_dir.glob("dataset_*/keyframe_*"), key=lambda p: (int(p.parent.name.split("_")[-1]), int(p.name.split("_")[-1])))
    kf_dirs = [kf for kf in kf_dirs if (kf / "data" / "rgb.mp4").exists()]
    if keyframes is not None:
        names = {f"{kf.parent.name}/{kf.name}" for kf in kf_dirs}
        missing = set(keyframes) - names
        if missing:
            raise ValueError(f"Keyframes not found in {scared_dir}: {sorted(missing)}")
        kf_dirs = [kf for kf in kf_dirs if f"{kf.parent.name}/{kf.name}" in keyframes]
    out_folders = [workspace / f"DS{int(kf.parent.name.split('_')[-1]):02d}_KF{int(kf.name.split('_')[-1]):02d}" for kf in kf_dirs]
    return kf_dirs, out_folders


def unpack_keyframe(kf_dir: Path):
    """Extract frame_data.tar.gz and scene_points.tar.gz into data/frame_data and data/scene_points."""
    for name in ["frame_data", "scene_points"]:
        out_dir = kf_dir / "data" / name
        archive = kf_dir / "data" / f"{name}.tar.gz"
        if out_dir.exists() and any(out_dir.iterdir()):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive}")
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():  # the archives are flat, keep only the file names
                    member.name = Path(member.name).name
                    tar.extract(member, out_dir)


def import_miccai_preprocess(miccai_dir: Path):
    """Import the modules of MICCAI_challenge_preprocess (https://github.com/EikoLoki/MICCAI_challenge_preprocess)."""
    if not miccai_dir.exists():
        raise FileNotFoundError(f"MICCAI_challenge_preprocess not found in {miccai_dir}. Clone it into THIRD_PARTY_DIR")
    sys.path.insert(0, str(miccai_dir))
    import video_parser, image_scissor, stereo_rectify, depth_to_disp

    return video_parser, image_scissor, stereo_rectify, depth_to_disp


def rectify_keyframe(kf_dir: Path, miccai_dir: Path) -> int:
    """Split the stacked video into left and right frames and rectify them with MICCAI_challenge_preprocess.

    Its functions process every keyframe of a dataset folder, so they are given a temporary folder that
    links only this keyframe. Returns the number of frames.
    """
    video_parser, image_scissor, stereo_rectify, _ = import_miccai_preprocess(miccai_dir)
    n_frames = int(cv2.VideoCapture(str(kf_dir / "data" / "rgb.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
    if len(list((kf_dir / "data" / "left_finalpass").glob("*.png"))) == n_frames:
        return n_frames
    # The scripts write with cv2.imwrite, which does not create folders
    for name in ["rgb_data", "left", "right", "left_finalpass", "right_finalpass", "reprojection_data", "disparity"]:
        (kf_dir / "data" / name).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as root:
        os.symlink(kf_dir.resolve(), Path(root) / kf_dir.name)
        video_parser.video_parser(root)
        image_scissor.image_scissor(root)
        stereo_rectify.stereo_rectify(root)
    assert len(list((kf_dir / "data" / "left_finalpass").glob("*.png"))) == n_frames, f"Rectification of {kf_dir} is incomplete"
    for name in ["rgb_data", "left", "right"]:  # unrectified copies, not needed any more
        subprocess.run(["rm", "-rf", str(kf_dir / "data" / name)])
    return n_frames


def compute_disparity(kf_dir: Path, miccai_dir: Path, frame_ids):
    """Ground-truth disparity of the rectified left frames, from the scene points, with MICCAI_challenge_preprocess."""
    _, _, _, depth_to_disp = import_miccai_preprocess(miccai_dir)
    (kf_dir / "data" / "disparity").mkdir(parents=True, exist_ok=True)
    for i in frame_ids:
        disp_file = kf_dir / "data" / "disparity" / f"frame_data{i:06d}.tiff"
        if disp_file.exists():
            continue
        Q = depth_to_disp.read_Q(str(kf_dir / "data" / "reprojection_data" / f"frame_data{i:06d}.json"))
        img_l, _ = depth_to_disp.tiff_reader(str(kf_dir / "data" / "scene_points" / f"scene_points{i:06d}.tiff"))
        disp = depth_to_disp.coor_to_disp(img_l, Q)
        cv2.imwrite(str(disp_file), disp.astype(np.float32))


def read_reprojection(kf_dir: Path, frame_id: int = 0):
    """Focal length, principal point and baseline (mm) of the rectified cameras, from the reprojection matrix Q."""
    Q = np.array(json.load(open(kf_dir / "data" / "reprojection_data" / f"frame_data{frame_id:06d}.json"))["reprojection-matrix"])
    return Q[2, 3], -Q[0, 3], -Q[1, 3], 1.0 / Q[3, 2]


def rectification_rotation(kf_dir: Path) -> np.ndarray:
    """Rotation R1 from the original to the rectified left camera, computed as in the rectification."""
    calib = json.load(open(kf_dir / "data" / "frame_data" / "frame_data000000.json"))["camera-calibration"]
    R1 = cv2.stereoRectify(
        np.array(calib["KL"]),
        np.array(calib["DL"]),
        np.array(calib["KR"]),
        np.array(calib["DR"]),
        FRAME_SIZE[::-1],
        np.array(calib["R"]),
        np.reshape(np.array(calib["T"]), (3, 1)),
    )[0]
    return R1


def stage_frames(kf_dir: Path, out_folder: Path, frame_ids, scale: float):
    """Copy the rectified left and right frames to color/ and color_r/, resized by scale."""
    for side, folder in [("left_finalpass", "color"), ("right_finalpass", "color_r")]:
        (out_folder / folder).mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(frame_ids):
            frame = Image.open(kf_dir / "data" / side / f"frame_data{i:06d}.png").convert("RGB")
            if scale < 1.0:
                new_size = ((int(scale * frame.width) // 16) * 16, (int(scale * frame.height) // 16) * 16)
                frame = frame.resize(new_size, Image.LANCZOS)
            frame.save(out_folder / folder / f"{n:04d}.png")


def write_calibration(kf_dir: Path, out_folder: Path):
    """calib.json of the rectified stereo pair: same intrinsics for both cameras, right camera at +baseline along x."""
    fl, cx, cy, baseline = read_reprojection(kf_dir)
    K = [[fl, 0.0, cx], [0.0, fl, cy], [0.0, 0.0, 1.0]]
    calib = {
        "leftcameramat": K,
        "rightcameramat": K,
        "translation": [abs(baseline) / 1000.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "img_size": list(FRAME_SIZE),
    }
    with open(out_folder / "calib.json", "w") as f:
        json.dump(calib, f, indent=4)


def write_camera_poses(kf_dir: Path, out_folder: Path, frame_ids):
    """cam_RT.json: camera-to-world pose of the rectified left camera for each staged frame, in metres.

    SCARED stores the world-to-camera transform of the original left camera in mm. The world is set to the
    rectified left camera of the first frame of the keyframe video.
    """
    R1 = np.eye(4)
    R1[:3, :3] = rectification_rotation(kf_dir)

    def world_to_cam(i):
        return np.array(json.load(open(kf_dir / "data" / "frame_data" / f"frame_data{i:06d}.json"))["camera-pose"], dtype=float)

    P0 = world_to_cam(0)
    output = {"num_poses": len(frame_ids), "RT": {}, "time_step": {}}
    for n, i in enumerate(frame_ids):
        pose = R1 @ P0 @ np.linalg.inv(world_to_cam(i)) @ R1.T  # rectified camera i -> rectified camera of frame 0
        pose[:3, 3] /= 1000.0  # mm to m
        output["RT"][f"{n:04d}"] = pose.tolist()
        output["time_step"][f"{n:04d}"] = i
    with open(out_folder / "cam_RT.json", "w") as f:
        json.dump(output, f, indent=4)


def depth_to_heatmap(arr, cmap="viridis"):
    """Convert a 2D depth array into an RGB heatmap image."""
    mask = np.isnan(arr) | np.isinf(arr) | (arr <= 0)
    norm = plt.Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    rgb = (plt.get_cmap(cmap)(norm(arr))[:, :, :3] * 255).astype(np.uint8)
    rgb[mask] = 0  # Set invalid values to black
    return Image.fromarray(rgb)


def write_depth_maps(kf_dir: Path, out_folder: Path, frame_ids, scale: float):
    """depth/<n>.npy: ground-truth depth in mm of each staged frame at the staged resolution, NaN where unknown.

    Depth is recovered from the disparity as fl * baseline / disparity. The 3 % closest and the 0.5 % farthest
    values of the keyframe are discarded as outliers of the structured-light reconstruction.
    """
    fl, _, _, baseline = read_reprojection(kf_dir)
    depth_folder = out_folder / "depth"
    depth_folder.mkdir(parents=True, exist_ok=True)
    depth_maps = []
    for i in frame_ids:
        disp = cv2.imread(str(kf_dir / "data" / "disparity" / f"frame_data{i:06d}.tiff"), cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth = np.full(disp.shape, np.nan)
        depth[disp > 0] = fl * abs(baseline) / disp[disp > 0]
        depth_maps.append(depth)
    depth_maps = np.stack(depth_maps)
    near, far = np.nanpercentile(depth_maps, 3.0), np.nanpercentile(depth_maps, 99.5)
    depth_maps[(depth_maps < near) | (depth_maps > far)] = np.nan
    print(f"Ground-truth depth range: {near:.1f} - {far:.1f} mm")
    for n, depth in enumerate(depth_maps):
        valid = ~np.isnan(depth)
        if scale < 1.0:  # scatter every valid pixel to the staged resolution
            new_h, new_w = (int(scale * depth.shape[0]) // 16) * 16, (int(scale * depth.shape[1]) // 16) * 16
            grid_y, grid_x = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij")
            grid_y = np.clip((grid_y * scale).astype(np.int32), 0, new_h - 1)
            grid_x = np.clip((grid_x * scale).astype(np.int32), 0, new_w - 1)
            small = np.full((new_h, new_w), np.nan, dtype=np.float32)
            small[grid_y[valid], grid_x[valid]] = depth[valid]
            depth = small
        np.save(depth_folder / f"{n:04d}.npy", depth.astype(np.float32))
        imageio.imwrite(depth_folder / f"{n:04d}_heatmap.png", depth_to_heatmap(depth))


def preprocess(out_folder: Path, settings: dict, preprocessing_dir: Path) -> None:
    """Compute the correspondences of a keyframe with Preprocessing4tracker.

    The settings are recorded in preprocessing.json once the preprocessing completes. Every flow source writes
    to the same folders, so a completed run with the same settings is reused and any other outputs are deleted.
    """
    record = out_folder / "preprocessing.json"
    if record.exists() and json.load(open(record)) == settings:
        print(f"Correspondences of {out_folder.name} already computed with these settings")
        return
    # Outputs of an interrupted run or of other settings are not reused
    subprocess.run(["rm", "-f", str(record)])
    for name in FLOW_FOLDERS:
        subprocess.run(["rm", "-rf", str(out_folder / name)])
    cmd = [
        sys.executable,
        "main_processing.py",
        "--data_dir",
        str(out_folder),
        "--max_pair",
        str(settings["pairs_win"]),
        "--gap",
        str(settings["pairs_gap"]),
        "--stereo",
    ]
    if settings["flow"] == "mft":
        cmd.append("--MFT")
    if settings["cotracker"]:
        cmd.append("--CoTracker")
    subprocess.run(cmd, cwd=preprocessing_dir, check=True)
    with open(record, "w") as f:
        json.dump(settings, f)


def update_mask(out_folder: Path, scale: float = 1.0):
    """Mask out the band on the left of the rectified frames, which has no image content."""
    pixels_to_remove = int(RECTIFICATION_BORDER * scale)
    for f in (out_folder / "raft_masks").glob("*.npy"):
        os.remove(f)
    for f in (out_folder / "raft_masks").glob("*.png"):
        mask = imageio.v2.imread(f) > 0
        mask[:, :pixels_to_remove] = False
        imageio.imwrite(f, mask.astype(np.uint8) * 255)
    _mask_shape = mask.shape[:2]
    for file_name in (out_folder / "3D_info").glob("*.npz"):
        data = np.load(file_name, allow_pickle=True)
        map_3D, mask = data["map_3D"], data["mask"]
        mask_shape = mask.shape
        mask = mask.reshape(*_mask_shape, -1)
        mask[:, :pixels_to_remove] = False
        mask = mask.reshape(mask_shape)
        np.savez_compressed(file_name, map_3D=map_3D, mask=mask)


def clean_up(out_folder: Path):
    """Delete the frames, flows and caches of a processed keyframe, keeping the last checkpoint, the config, the logs and the results"""
    for name in FLOW_FOLDERS + ["preprocessing.json", "color", "color_r", "depth"]:
        subprocess.run(["rm", "-rf", str(out_folder / name)])
    for exp_folder in (out_folder / "out").glob("Test*"):
        if exp_folder.is_dir():
            for name in ["cache", "lightning_logs", "vid", "eval_error_log.txt"]:
                subprocess.run(["rm", "-rf", str(exp_folder / name)])
            for img in exp_folder.glob("*.png"):
                if img.is_file():
                    os.remove(img)
            for ckpt in sorted(exp_folder.glob("model_*.pth"))[:-1]:
                os.remove(ckpt)


if __name__ == "__main__":
    print("SCARED dataset processing")
    args = get_args()
    data_dir = env_path("DATA_DIR")
    miccai_dir = env_path("THIRD_PARTY_DIR") / "MICCAI_challenge_preprocess"
    config_file = f"config/SCARED/Exp{args.exp_n}.yaml"
    with open(REPO_DIR / config_file, "r") as f:
        exp_cfg = yaml.safe_load(f)
    settings = {
        "flow": args.flow,
        "pairs_win": exp_cfg.get("pairs_win", 5),
        "pairs_gap": exp_cfg.get("pairs_gap", 2),
        "cotracker": exp_cfg.get("add_CoTracker", False),
    }

    kf_dirs, out_folders = get_keyframe_list(data_dir / args.dataset, data_dir / args.workspace, args.keyframes)
    print(f"Number of keyframes: {len(kf_dirs)}")
    (data_dir / args.workspace).mkdir(parents=True, exist_ok=True)
    log = open(REPO_DIR / "processes" / "SCARED_processed_vids.txt", "w")
    failed = []

    scale = 1.0 / 2.0
    for kf_dir, out_folder in zip(kf_dirs, out_folders):
        if (out_folder / "out").exists() and args.no_reprocess:
            print(f"Output folder already exists: {out_folder.name}")
            continue
        print(f"Processing {kf_dir.parent.name}/{kf_dir.name} -> {out_folder}")
        out_folder.mkdir(parents=True, exist_ok=True)

        # 1-2 Unpack, rectify and compute the ground-truth disparity of the training frames (every second frame)
        unpack_keyframe(kf_dir)
        n_frames = rectify_keyframe(kf_dir, miccai_dir)
        frame_ids = list(range(0, n_frames, 2))
        compute_disparity(kf_dir, miccai_dir, frame_ids)

        # 3 Stage frames, calibration, camera poses and ground-truth depth
        stage_frames(kf_dir, out_folder, frame_ids, scale)
        write_calibration(kf_dir, out_folder)
        write_camera_poses(kf_dir, out_folder, frame_ids)
        write_depth_maps(kf_dir, out_folder, frame_ids, scale)

        # 4 Run preprocessing, then mask out the band lost by the rectification
        preprocess(out_folder, settings, Path(args.preprocessing_dir))
        subprocess.run(
            [sys.executable, "train.py", "--configuration", config_file, "--data_dir", str(out_folder), "--data_initialization"], cwd=REPO_DIR, check=True
        )
        update_mask(out_folder, scale=scale)

        # 5 Run tracker
        train_run = subprocess.run([sys.executable, "train.py", "--configuration", config_file, "--data_dir", str(out_folder)], cwd=REPO_DIR)
        if train_run.returncode != 0:
            print(f"Training failed for {out_folder.name} - the intermediates are kept for inspection")
            log.write(f"{kf_dir}\ttraining failed\n")
            failed.append(out_folder.name)
            continue

        # 6 Run evaluation
        eval_run = subprocess.run([sys.executable, "solver/SCARED_eval.py", "--out_dir", str(out_folder), "--experiment", str(args.exp_n)], cwd=REPO_DIR)
        if eval_run.returncode != 0:
            print(f"Evaluation failed for {out_folder.name} - the intermediates are kept for inspection")
            log.write(f"{kf_dir}\tevaluation failed\n")
            failed.append(out_folder.name)
            continue

        # 7 Clean up
        if not args.keep_intermediate:
            clean_up(out_folder)
        log.write(f"{kf_dir}\n")
    log.close()
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
