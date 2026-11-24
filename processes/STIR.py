"""Script to execute the complete pipeline on the STIR dataset
1. Filter videos with empty or crowded masks and with less than 24 frames
2. Extract frames from videos
3. Run preprocessing (RAFT or MFT optical flow, optional CoTracker, flow filtering, SAM2 tool masks)
4. Run tracker
5. Run evaluation
6. Clean up (get rid of RAFT optical flow files)
"""

from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import numpy as np
from PIL import Image
import os, sys, cv2, subprocess, json, torch, yaml
import argparse, imageio, tempfile, traceback

#: Repository root, the working directory for the training and evaluation calls.
REPO_DIR = Path(__file__).resolve().parents[1]
#: Folder of the raw STIR clips under DATA_DIR. Overwritten by ``--dataset``.
STIR_DATASET = "STIRChallenge_2024"
#: Folder where the processed clips and results are written, relative to DATA_DIR or absolute. Overwritten by ``--workspace``.
WORKSPACE = "STIR_InvNeRF"
#: Flow folders written by the preprocessing, deleted when a clip is preprocessed again with other settings.
FLOW_FOLDERS = ["raft_exhaustive", "raft_exhaustive_stereo", "raft_masks", "cotracker", "cotracker_masks", "features", "count_maps", "3D_info", "of_vis", "flow_stats.json"]


def data_root() -> Path:
    """DATA_DIR from the .env file in the parent directory of the repository: the folder that holds the datasets."""
    load_dotenv(find_dotenv(usecwd=True))
    if os.getenv("DATA_DIR") is None:
        raise ValueError("DATA_DIR is not defined. Add it to the .env file in the parent directory of the repository")
    return Path(os.getenv("DATA_DIR"))


def videos_criteria_selection():
    """Filter videos with empty or crowded masks and with less than 24 frames"""
    data_dir = data_root() / STIR_DATASET
    folders = [f for f in os.listdir(data_dir) if (data_dir / f / "left").is_dir()]  # collection folders only
    # sort folders
    folders = sorted(folders)
    count = 0
    threshold = 0.1
    video_list = []
    outfoledr_list = []
    video_type = []  # short, medium, long
    for f in folders:
        scene_dir = data_dir / f
        videos_dir = os.listdir(scene_dir / "left")
        # sort videos
        videos_dir = sorted(videos_dir)
        for vd in videos_dir:
            init_mask_dir = scene_dir / "left" / vd / "segmentation" / "icgstartseg.png"
            end_mask_dir = scene_dir / "left" / vd / "segmentation" / "icgendseg.png"
            # read files
            init_mask, end_mask = Image.open(init_mask_dir), Image.open(end_mask_dir)
            # Image size
            W, H = init_mask.size
            # print(f"Folder: {f}, Video: {vd}, Image size: {W}x{H}")
            init_mask_s = (np.array(init_mask) / 255).sum() / (W * H)
            end_mask_s = (np.array(end_mask) / 255).sum() / (W * H)
            empty_mask = (np.array(init_mask) == 0).all() and (np.array(end_mask) == 0).all()
            crowded_mask = (init_mask_s > threshold) and (end_mask_s > threshold)

            if empty_mask or crowded_mask:
                if empty_mask:
                    print(f"Empty video: {f}/{vd}, Masks value: ({init_mask_s},{end_mask_s})")
                if crowded_mask:
                    print(f"Crowded video: {f}/{vd}, Masks value: ({init_mask_s},{end_mask_s})")
                # print(f"Discarted video: {f}/{vd}, Masks value: ({init_mask},{end_mask})")
            else:
                video = scene_dir / "left" / vd / "frames"
                video = video / os.listdir(video)[0]
                vid = cv2.VideoCapture(str(video))
                frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                vid.release()
                if frame_count > 24:
                    # print(f"{video}")
                    video_list.append(video)
                    output_folder = data_root() / WORKSPACE / ("case" + str(f) + "_" + str(vd))
                    # print(f"Output folder: {output_folder}")
                    outfoledr_list.append(output_folder)
                    if frame_count <= 75:
                        video_type.append(f"short\t({frame_count})")
                    elif frame_count <= 175:
                        video_type.append(f"medium\t{frame_count}")
                    else:
                        video_type.append(f"long\t{frame_count}")
                    count += 1
    print(f"Number of folders: {len(folders)}, Number of videos: {count}")
    # save list of videos
    assert len(video_list) == len(outfoledr_list), "Different number of videos and output folders"
    (data_root() / WORKSPACE).mkdir(parents=True, exist_ok=True)
    video_list_file = data_root() / WORKSPACE / "videos_list.txt"
    output_folder_file = data_root() / WORKSPACE / "output_folders.txt"
    video_type_file = data_root() / WORKSPACE / "video_type.txt"
    video_list_file = open(video_list_file, "w")
    output_folder_file = open(output_folder_file, "w")
    video_type_file = open(video_type_file, "w")
    for i in range(len(video_list)):
        video_list_file.write(f"{video_list[i]}\n")
        output_folder_file.write(f"{outfoledr_list[i]}\n")
        video_type_file.write(f"{video_type[i]}\n")
    video_list_file.close()
    output_folder_file.close()
    video_type_file.close()

    return video_list, outfoledr_list, video_type


def load_vid_list_txt():
    video_list_file = data_root() / WORKSPACE / "videos_list.txt"
    output_folder_file = data_root() / WORKSPACE / "output_folders.txt"
    video_type_file = data_root() / WORKSPACE / "video_type.txt"
    video_list = open(video_list_file, "r").readlines()
    output_folders = open(output_folder_file, "r").readlines()
    video_type = open(video_type_file, "r").readlines()
    video_list = [v.strip() for v in video_list]
    output_folders = [o.strip() for o in output_folders]
    video_type = [t.strip() for t in video_type]
    return video_list, output_folders, video_type


def extract_frames(vid_dir, output_folder, camera="left", scale=1.0, one_frame_only=False):
    """Extract frames from video"""
    output_folder = Path(output_folder)
    output_folder = output_folder / "color" if camera == "left" else output_folder / "color_r"
    print(f"Extracting frames from {str(vid_dir).split('/')[5:-2]} to {str(output_folder).split('/')[5:]}")
    output_folder.mkdir(parents=True, exist_ok=True)
    vid = cv2.VideoCapture(str(vid_dir))
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if one_frame_only:
        frame_count = 1
    if frame_count > 1000:
        print(f"Warning: Video {vid_dir} has more than 1000 frames.")
        # return (0, 0) # Return dummy size
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f"Video:{str(output_folder).split('/')[-1]}, Frame count: {frame_count}, FPS: {fps}")
    for i in range(frame_count):
        ret, frame = vid.read()
        if ret:  # save frame
            np_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np_frame)
            if scale < 1.0:  # resize frame
                new_width = int(scale * frame.width)
                new_height = int(scale * frame.height)
                new_width = (new_width // 16) * 16
                new_height = (new_height // 16) * 16
                frame = frame.resize((new_width, new_height), Image.LANCZOS)
            frame.save(output_folder / f"{i:04d}.png")
    vid.release()
    # Return original size of the frames
    return (np_frame.shape[1], np_frame.shape[0])


def extract_cam_params(vid_dir, output_folder, img_size):
    """Extract camera parameters from video"""
    calb_file = str(Path(vid_dir).parents[3] / "calib.json")
    # read the calibration file into a dictionary
    calib = json.load(open(calb_file))
    calib["img_size"] = img_size[::-1]  # (H, W)
    # Convert the transformation to fo from left to right
    if calib["translation"][0] < 0:
        calib["translation"] = [-item for item in calib["translation"]]
        rot_mat = cv2.Rodrigues(np.array(calib["rotation"]).reshape(3, 1))[0]
        calib["rotation"] = cv2.Rodrigues(rot_mat.T)[0].ravel().tolist()
    # save the calibration parameters
    calib_file = str(Path(output_folder) / "calib.json")
    with open(calib_file, "w") as f:
        json.dump(calib, f)


def get_video_lists(selected_subsets=["short", "medium", "long"]):
    """Get the list of videos and output folders"""
    # Paths and config
    data_folder = data_root() / WORKSPACE
    # 1 Get list of videos and output folders to process
    if (data_folder / "videos_list.txt").exists() and (data_folder / "output_folders.txt").exists() and (data_folder / "video_type.txt").exists():
        videos_list, out_folders, video_type = load_vid_list_txt()
    else:
        videos_list, out_folders, video_type = videos_criteria_selection()
    print(f"Number of videos: {len(videos_list)}")
    video_type = [v.split("\t")[0] for v in video_type]  # Remove frame count from video type
    print(f"Short videos: {video_type.count('short')}, Medium videos: {video_type.count('medium')}, Long videos: {video_type.count('long')}")
    assert len(videos_list) == len(out_folders), "Different number of videos and output folders. Delete lists and run again"
    videos_list, out_folders, video_type = np.array(videos_list), np.array(out_folders), np.array(video_type)
    # Filter videos by type
    idxs = np.where(np.isin(video_type, selected_subsets))[0]
    videos_list, out_folders, video_type = videos_list[idxs], out_folders[idxs], video_type[idxs]
    print(f"Selected subsets{selected_subsets}. Total videos {len(videos_list)}")
    return videos_list, out_folders, video_type


def bring_gt_labels(video_dir, output_dir, scale=1.0):
    """Copy ground truth labels to output directory
    Args:
        video_dir (str): Path to video directory
        output_dir (str): Path to output directory
    """
    labels_dir = Path(video_dir).parent.parent / "segmentation"
    output_dir = Path(output_dir) / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in labels_dir.glob("*seg.png"):
        file_name = "start" if "start" in str(label) else "final"
        label_img = Image.open(label)
        label_img.save(output_dir / f"original_{file_name}.png")
        if scale < 1.0:  # resize label
            new_width = int(scale * label_img.width)
            new_height = int(scale * label_img.height)
            new_width = (new_width // 16) * 16
            new_height = (new_height // 16) * 16
            label_mask = np.array(label_img) > 0  # Convert to boolean mask
            pts = np.argwhere(label_mask)  # Get coordinates of non-zero pixels # pts.shape = (N, 2) where 2 is (y, x)
            if pts.size == 0:  # If no points, skip resizing the points
                label_img = label_img.resize((new_width, new_height), Image.LANCZOS)
            else:
                x_scale, y_scale = new_width / label_img.width, new_height / label_img.height
                pts[:, 0] = np.clip(pts[:, 0] * y_scale, 0, int(scale * label_img.height) - 1)  # Clip y-coordinates
                pts[:, 1] = np.clip(pts[:, 1] * x_scale, 0, int(scale * label_img.width) - 1)  # Clip x-coordinates
                pts = pts.astype(int)  # Scale the points
                # Create
                label_img = np.zeros((new_height, new_width), dtype=np.uint8)  # size order is (H, W) for numpy
                label_img[pts[:, 0], pts[:, 1]] = 255  # Set the points in the new image
                label_img = Image.fromarray(label_img)
        label_img.save(output_dir / f"{file_name}.png")
    assert len(list(output_dir.glob("*.png"))) > 0, "No labels were copied"
    # assert len(list(output_dir.glob('*.png'))) == 2, "There should be only two labels per video"


def get_args():
    """Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Process STIR dataset")
    parser.add_argument("--exp_n", type=int, default=6, help="Experiment number, selects config/STIR/STIR_test<exp_n>.yaml")
    parser.add_argument("--flow", type=str, default="raft", choices=["raft", "mft"], help="Short-term correspondences computed in the preprocessing")
    parser.add_argument("--v_idx", type=int, default=None, help="Index of a single video to process. All selected videos if not given")
    parser.add_argument(
        "--subsets", type=str, nargs="+", default=["short", "medium", "long"], help="Video lengths to process: short (<=75 frames), medium (<=175), long"
    )
    parser.add_argument("--clips", type=str, nargs="+", default=None, help="Process only these clips, named <collection>_<seq>, e.g. 02_seq09")
    parser.add_argument("--dataset", type=str, default=STIR_DATASET, help="Folder of the raw STIR clips under DATA_DIR")
    parser.add_argument("--workspace", type=str, default=WORKSPACE, help="Folder for the processed clips and results, relative to DATA_DIR or absolute")
    parser.add_argument("--preprocessing_dir", type=str, default=str(REPO_DIR.parent / "Preprocessing4tracker"), help="Clone of Preprocessing4tracker")
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep frames, flows and caches after the evaluation")
    args = parser.parse_args()
    return args


def update_tool_masks(data_dir: Path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[SAM2]: using device: {device}")
    # delete .npy files to regeneratem when dataset objects is created
    for f in (data_dir / "raft_masks").glob("*.npy"):
        os.remove(f)
    for f in (data_dir / "cotracker_masks").glob("*.npy"):
        os.remove(f)
    # read data
    frames_dir = data_dir / "color"
    frame_files = [f for f in (data_dir / "color").glob("*.png")]
    frame_files = sorted(frame_files)
    # (data_dir/'temp').mkdir(parents=True,exist_ok=True)
    # read prompts
    # Prompts in the workspace take precedence over the ones shipped with the repository for this dataset
    save_path = data_dir.parent / "points_for_masking.json"
    if not save_path.exists():
        save_path = REPO_DIR / "processes" / "sam2_prompts" / f"{STIR_DATASET}.json"
    with open(save_path, "r") as f:
        labels = json.load(f)
    if data_dir.name not in labels:
        raise KeyError(f"No SAM2 prompts for {data_dir.name} in {save_path}")
    prompt = {}
    for idx, n_pts in enumerate(labels[data_dir.name]):
        prompt[idx] = np.array(n_pts, dtype=np.float32), np.ones(len(n_pts), np.int32)
    print(f"[SAM2]: Generating tool masks for the folder {data_dir.name} with the prompt{prompt}")
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # The SAM2 video predictor only reads folders of JPEG frames named by frame
        # index, so it gets a temporary JPEG copy of the PNG frames. Same size and
        # order, so the prompts and frame indices still refer to the PNG frames.
        # init_state loads every frame up front, so the copy can go straight after.
        with tempfile.TemporaryDirectory() as jpg_dir:
            for i, f in enumerate(frame_files):
                Image.open(f).convert("RGB").save(Path(jpg_dir) / f"{i:05d}.jpg", quality=95)
            state = predictor.init_state(jpg_dir)
        # Make inference in initial mask
        for key, val in prompt.items():
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=key, points=val[0], labels=val[1])
        # Propagate mask
        seg_frames = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            tool_mask = [(out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)]
            tool_mask = np.concatenate(tool_mask, axis=0)
            tool_mask = np.any(tool_mask, axis=0)
            _image = imageio.imread(frame_files[out_frame_idx])
            _image[..., 1] = np.clip((_image[..., 1].astype(np.int32) + tool_mask * 40), a_min=0, a_max=255).astype(np.uint8)
            seg_frames.append(_image)
            # _file = data_dir/'temp'/frame_files[out_frame_idx].name
            # imageio.imwrite(_file, _image)
            # Update files
            for f in (data_dir / "raft_masks").glob(f"{frame_files[out_frame_idx].stem}_*.png"):
                mask = imageio.imread(f) > 0
                mask[..., -1] = ~tool_mask
                imageio.imwrite(f, mask.astype(np.uint8) * 255)
            # Add the mask also in cotracker_masks folder
            for f in (data_dir / "cotracker_masks").glob(f"{frame_files[out_frame_idx].stem}_*.png"):
                mask = imageio.imread(f) > 0
                mask[..., -1] = ~tool_mask
                imageio.imwrite(f, mask.astype(np.uint8) * 255)
        _file = data_dir / "tool_masks.mp4"
        imageio.mimsave(_file, seg_frames, quality=8, fps=20)


def preprocess(out_folder: Path, settings: dict, preprocessing_dir: Path) -> None:
    """Compute the correspondences of a clip with Preprocessing4tracker.

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
    cmd = [sys.executable, "main_processing.py", "--data_dir", str(out_folder), "--max_pair", str(settings["pairs_win"]), "--gap", str(settings["pairs_gap"]), "--stereo"]
    if settings["flow"] == "mft":
        cmd.append("--MFT")
    if settings["cotracker"]:
        cmd.append("--CoTracker")
    subprocess.run(cmd, cwd=preprocessing_dir, check=True)
    with open(record, "w") as f:
        json.dump(settings, f)


def clean_up(out_folder: Path) -> None:
    """Delete the frames, flows and caches of a processed clip, keeping the last checkpoint, the config, the logs and evaluation.json."""
    for name in FLOW_FOLDERS + ["preprocessing.json", "color_r", "color", "labels", "calib.json"]:
        subprocess.run(["rm", "-rf", str(out_folder / name)])
    for exp_folder in (out_folder / "out").glob("Test*"):
        if exp_folder.is_dir():
            for name in ["cache", "lightning_logs", "vid", "eval_error_log.txt"]:
                subprocess.run(["rm", "-rf", str(exp_folder / name)])
            for img in exp_folder.glob("*.png"):
                if img.is_file():
                    os.remove(img)
            checkpoints = sorted(exp_folder.glob("model_*.pth"))
            for ckpt in checkpoints[:-1]:
                os.remove(ckpt)


if __name__ == "__main__":
    print("STIR dataset processing")
    args = get_args()
    STIR_DATASET, WORKSPACE = args.dataset, args.workspace
    config_file = f"config/STIR/STIR_test{args.exp_n}.yaml"
    with open(REPO_DIR / config_file, "r") as f:
        exp_cfg = yaml.safe_load(f)
    # The flow is computed for exactly the frame pairs the experiment trains on
    pairs_win, pairs_gap = exp_cfg.get("pairs_win", 5), exp_cfg.get("pairs_gap", 2)
    add_cotracker, mask_tools = exp_cfg.get("add_CoTracker", False), exp_cfg.get("mask_tools", False)
    if add_cotracker and args.flow == "mft":
        raise ValueError(f"{config_file} trains on CoTracker correspondences, which are computed on the RAFT path. Use --flow raft.")

    videos_list, out_folders, video_type = get_video_lists(selected_subsets=args.subsets)
    video_indices = list(range(len(videos_list)))
    if args.clips is not None:
        video_indices = [i for i in video_indices if Path(out_folders[i]).name.replace("case", "", 1) in args.clips]
        missing = set(args.clips) - {Path(out_folders[i]).name.replace("case", "", 1) for i in video_indices}
        if missing:
            raise ValueError(f"Clips not found in the selection: {sorted(missing)}")
    if args.v_idx is not None:
        video_indices = video_indices[args.v_idx : args.v_idx + 1]
    log = open(REPO_DIR / "processes" / "STIR_processed_vids.txt", "w")
    failed = []

    for j, i in enumerate(video_indices):
        print(f"[{j}:{i}] Processing video ({video_type[i]}):", Path(out_folders[i]).stem)

        try:
            # 2 Extract frames and camera parameters
            scale = 1.0 / 2
            img_size = extract_frames(videos_list[i], str(out_folders[i]), scale=scale)  # Default camera is left
            r_img_size = extract_frames(Path(str(videos_list[i]).replace("left", "right")), str(out_folders[i]), scale=scale, camera="right")
            assert img_size == r_img_size, "Different image sizes between left and right cameras"
            extract_cam_params(videos_list[i], out_folders[i], img_size)
            bring_gt_labels(videos_list[i], out_folders[i], scale)

            # 3 Run preprocessing
            settings = {"flow": args.flow, "pairs_win": pairs_win, "pairs_gap": pairs_gap, "cotracker": add_cotracker}
            preprocess(Path(out_folders[i]), settings, Path(args.preprocessing_dir))

            # 3.1 Update tool masks
            if mask_tools:
                update_tool_masks(Path(out_folders[i]))
        except Exception:
            traceback.print_exc()
            print(f"[{j}:{i}] Data preparation failed for", Path(out_folders[i]).stem, "- the intermediates are kept for inspection")
            log.write(f"{videos_list[i]}\tdata preparation failed\n")
            failed.append(Path(out_folders[i]).stem)
            continue

        # 4 Run tracker
        train_run = subprocess.run([sys.executable, "train.py", "--configuration", config_file, "--data_dir", str(out_folders[i])], cwd=REPO_DIR)
        if train_run.returncode != 0:
            print(f"[{j}:{i}] Training failed for", Path(out_folders[i]).stem, "- the intermediates are kept for inspection")
            log.write(f"{videos_list[i]}\ttraining failed\n")
            failed.append(Path(out_folders[i]).stem)
            continue

        # 5 Run evaluation
        eval_run = subprocess.run(
            [sys.executable, "solver/STIR_eval.py", "--vid_dir", str(videos_list[i]), "--out_dir", str(out_folders[i]), "--vid_type", video_type[i], "--experiment", str(args.exp_n)],
            cwd=REPO_DIR,
        )
        if eval_run.returncode != 0:
            print(f"[{j}:{i}] Evaluation failed for", Path(out_folders[i]).stem, "- the intermediates are kept for inspection")
            log.write(f"{videos_list[i]}\tevaluation failed\n")
            failed.append(Path(out_folders[i]).stem)
            continue

        # 6 Clean up
        if not args.keep_intermediate:
            clean_up(Path(out_folders[i]))

        log.write(f"{videos_list[i]}\n")
    log.close()
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
