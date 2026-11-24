# Dataset preparation

The scripts in this folder run the full pipeline for one dataset, clip by clip: they stage the frames and the camera parameters, compute the short-term correspondences with [Preprocessing4tracker](https://github.com/L0za007/Preprocessing4tracker), fit InvNeRF with `train.py` and evaluate the fit with the scripts in `solver/`.

| Script | Dataset | Configurations |
| ------ | ------- | -------------- |
| `STIR.py`   | STIR Challenge 2024 / STIR | `config/STIR/STIR_test{6,8,9}.yaml` |
| `SCARED.py` | SCARED                     | `config/SCARED/Exp1.yaml` |

## Paths
Every path is resolved from `DATA_DIR`, defined in a `.env` file in the **parent** directory of this repository (see the main [README](../README.md#paths)). The raw datasets are expected in `DATA_DIR`, and the processed clips and results are written next to them:
```
DATA_DIR
|-- STIRChallenge_2024     # raw STIR clips (input)
|-- STIR_InvNeRF           # processed STIR clips and results (output)
|-- scared2019             # extracted SCARED keyframes (input)
`-- SCARED_InvNeRF         # processed SCARED keyframes and results (output)
```
`--dataset` and `--workspace` change these folders; they take a path relative to `DATA_DIR` or an absolute one. Preprocessing4tracker is called as a subprocess from a sibling folder of this repository, or from `--preprocessing_dir`.

---

## STIR

### 1. Download
The 2D tracking comparison against feed-forward methods and the 3D tracking results were obtained on the [STIR Challenge 2024 dataset](https://zenodo.org/records/14803158). The same format is used by the original [STIR dataset](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared), so either can be processed with `--dataset <folder name>`.

Extract the data to `DATA_DIR/STIRChallenge_2024`. The script expects the original layout of the release:
```
STIRChallenge_2024
|-- 02
|   |-- calib.json                 # stereo calibration of the collection
|   |-- left
|   |   |-- seq00
|   |   |   |-- frames/*.mp4       # rectified left video
|   |   |   `-- segmentation
|   |   |       |-- icgstartseg.png
|   |   |       `-- icgendseg.png
|   |   `-- ...
|   `-- right
|       `-- seq00/frames/*.mp4     # rectified right video
|-- 03
`-- ...
```

### 2. Ground-truth point locations
The tracked points and their ground truth are read from the JSON files generated with the evaluation code of the challenge organisers, [STIRMetrics](https://github.com/athaddius/STIRMetrics). Follow its installation instructions, point it at the same copy of the dataset, and export the start and end locations in 2D and 3D:
```
cd STIRMetrics
python datatest/write2dgtjson.py --num_data -1 --jsonsuffix test
python datatest/write3dgtjson.py --num_data -1 --jsonsuffix test
```
Copy the four files it writes into the workspace folder:
```
cp results/gt_positions_start_all_test.json    results/gt_positions_end_all_test.json \
   results/gt_3d_positions_start_all_test.json results/gt_3d_positions_end_all_test.json \
   <DATA_DIR>/STIR_InvNeRF/
```
The keys of these files are the paths of the clips. `solver/STIR_eval.py` matches each clip by the part of the path `<dataset>/<collection>/left/<seq>`, so the dataset folder must have the same name in both repositories.

### 3. Tool-mask prompts (`InvNeRF_mask` only)
The masked configuration removes the surgical tools from the optimisation with [SAM2](https://github.com/facebookresearch/sam2) (`facebook/sam2-hiera-large`, downloaded from HuggingFace on first use). Install it in the same environment:
```
pip install git+https://github.com/facebookresearch/sam2.git huggingface_hub==2.1.1
```
SAM2 is prompted on the first frame of each clip. The prompts we used for the STIR Challenge 2024 clips are included in [`sam2_prompts/STIRChallenge_2024.json`](sam2_prompts/STIRChallenge_2024.json) and are used by default. To use your own, write them to `DATA_DIR/STIR_InvNeRF/points_for_masking.json`, which takes precedence. The keys are the names of the processed clip folders (`case<collection>_<seq>`), and each value holds one list of `[x, y]` points per tool, in pixels of the half-resolution frames (640x512) the pipeline extracts:
```
{
  "case02_seq08": [[[282, 31], [301, 107], [313, 152], [308, 197], [303, 245]], [[626, 126], [597, 129], [559, 140]]],
  ...
}
```
To process other clips, add an entry per clip in the same format.

### 4. Run
Each configuration reported in the paper is run by selecting its experiment number and the correspondences it was trained on:

| Paper | Config | Correspondences | Tool masks | Command |
| ----- | ------ | --------------- | ---------- | ------- |
| InvNeRF              | `STIR_test6.yaml` | RAFT + CoTracker | no  | `python processes/STIR.py --exp_n 6 --flow raft` |
| InvNeRF<sub>mask</sub> | `STIR_test8.yaml` | RAFT + CoTracker | yes | `python processes/STIR.py --exp_n 8 --flow raft` |
| InvNeRF<sub>MFT</sub>  | `STIR_test9.yaml` | MFT              | no  | `python processes/STIR.py --exp_n 9 --flow mft` |

Run the commands from the root of this repository. For each clip, the script:
1. Selects the clips, discarding those with empty or crowded ICG segmentations (more than 10 % of the image) and those with 24 frames or fewer. The selection is cached in `videos_list.txt`, `output_folders.txt` and `video_type.txt` inside the workspace; delete them to select again.
2. Extracts the left and right frames at half resolution, the calibration (`calib.json`) and the ICG segmentations (`labels/`).
3. Computes the correspondences between the frame pairs the configuration trains on (`pairs_win` and `pairs_gap` in its config), and the SAM2 tool masks when the configuration sets `mask_tools`.
4. Fits InvNeRF with `train.py`.
5. Evaluates the fit with `solver/STIR_eval.py`, which writes `evaluation.json` to the experiment folder.
6. Deletes the frames, flows and caches. Use `--keep_intermediate` to keep them.

Useful options:
- `--clips 02_seq09 05_seq51` processes only the named clips (`<collection>_<seq>`).
- `--v_idx N` processes only the N-th selected clip, e.g. one clip per job on a cluster.
- `--subsets short medium long` filters the clips by length: short (<= 75 frames), medium (<= 175 frames) and long.
- `--keep_intermediate` keeps the frames and correspondences. The settings of the correspondences are recorded in `preprocessing.json`, and they are recomputed when another configuration needs different ones.

The script exits with an error when the training or the evaluation of any clip fails; the intermediates of that clip are kept for inspection.

Each processed clip ends up with the following structure:
```
STIR_InvNeRF/case02_seq09
|-- tool_masks.mp4                     # SAM2 masks overlaid on the frames (InvNeRF_mask only)
`-- out
    `-- Test_6_<date>_<time>
        |-- config.yaml                # resolved configuration
        |-- log_screen.txt             # training log
        |-- model_<step>.pth           # last checkpoint
        |-- evaluation.json            # predicted and ground-truth points, errors, PSNR and SSIM
        `-- *.mp4                      # rendered tracks, RGB and depth
```
Finally, `solver/gather_info.py` collects the `evaluation.json` files of all clips into `evaluation_summary.csv`:
```
python solver/gather_info.py --out_dir <DATA_DIR>/STIR_InvNeRF --all
```

---

## SCARED

### 1. Download
Request access to the [SCARED dataset](https://endovissub2019-scared.grand-challenge.org/) (EndoVis 2019) and extract the keyframes you want to process into `DATA_DIR/scared2019`, keeping the structure of the release:
```
scared2019
|-- dataset_2
|   |-- keyframe_1
|   |   `-- data
|   |       |-- rgb.mp4                # left and right frames, stacked vertically
|   |       |-- frame_data.tar.gz      # calibration and camera pose per frame
|   |       `-- scene_points.tar.gz    # 3D points per frame (ground truth)
|   `-- ...
`-- ...
```
For example, `unzip train_dataset_2.zip "dataset_2/keyframe_1/*" -d <DATA_DIR>/scared2019`.

The frames are rectified, and the ground-truth disparity is computed, with [MICCAI_challenge_preprocess](https://github.com/EikoLoki/MICCAI_challenge_preprocess). Clone it into `THIRD_PARTY_DIR` (see the main [README](../README.md#paths)); the script calls it, nothing has to be run by hand:
```
cd <THIRD_PARTY_DIR>
git clone https://github.com/EikoLoki/MICCAI_challenge_preprocess.git
```

### 2. Run
```
python processes/SCARED.py --exp_n 1
python processes/SCARED.py --exp_n 1 --keyframes dataset_2/keyframe_1    # a single keyframe
```
For each keyframe, the script:
1. Extracts the frame data and the scene points from their archives.
2. Splits the video into left and right frames and rectifies them with MICCAI_challenge_preprocess (`data/left_finalpass`, `data/right_finalpass` and the reprojection matrix in `data/reprojection_data`), and computes the ground-truth disparity of the training frames from the scene points (`data/disparity`).
3. Stages every second frame at half resolution in `DATA_DIR/SCARED_InvNeRF/DS<d>_KF<k>`, with the rectified calibration (`calib.json`), the camera poses from the robot kinematics (`cam_RT.json`) and the ground-truth depth in mm (`depth/`). The poses are expressed for the rectified left camera, relative to the first frame of the keyframe.
4. Computes the MFT correspondences, and masks out the band on the left of the rectified frames, which has no image content.
5. Fits InvNeRF with `config/SCARED/Exp1.yaml`, using the camera poses.
6. Renders the depth of every fitted frame and compares it with the ground truth over the pixels where it is known, with `solver/SCARED_eval.py`. `evaluation.json` in the experiment folder holds the mean absolute error and the RMSE in mm, and the fraction of pixels with an error below 1, 2 and 5 mm.
7. Deletes the frames, flows and caches. Use `--keep_intermediate` to keep them.

`--no_reprocess` skips the keyframes that already have results, and `--dataset`, `--workspace` and `--preprocessing_dir` work as for STIR. As for STIR, the script exits with an error when the training or the evaluation of a keyframe fails.
