# Surg-InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision
<center> <h3>
 <a href="">Paper</a>.
 <a href="https://github.com/L0za007/Preprocessing4tracker">Preprocessing code</a>.
 <a href="https://github.com/athaddius/STIRMetrics">STIR evaluation code</a>.
</h3></center>

PyTorch implementation of InvNeRF, a test-time optimisation (TTO) approach for long-term 2D and 3D point tracking in surgical scenes. A NeRF-based architecture with an invertible deformation network aggregates short-term 2D correspondences into a bidirectional mapping between the deformable workspace and a static canonical space, so every pixel can be tracked in 2D and 3D while the scene is reconstructed. The framework includes multi-scale HexPlanes for fast inference, ray-density guidance from stereo triangulation, an error-map pixel sampler that avoids redundant samples, and the integration of camera poses from a robotic system.

Gerardo Loza, Junlei Hu, Dominic Jones, [Sharib Ali](https://eps.leeds.ac.uk/computing/staff/11465/dr-sharib-ali), [Pietro Valdastri](https://eps.leeds.ac.uk/electronic-engineering/staff/863/professor-pietro-valdastri)

Work done at the [STORM Lab UK](https://www.stormlabuk.com/) and the [AI in Medicine and Surgery (AIMS) group](https://artificial-intelligence.leeds.ac.uk/aims/), University of Leeds.

## Installation
We recommend using conda for the environment and pip for the dependencies. Create the environment, then install PyTorch 2.0.0 with CUDA 11.7 following the official [documentation](https://pytorch.org/get-started/previous-versions/).
```
git clone https://github.com/L0za007/surg-invnerf.git
cd surg-invnerf
conda env create -f env.yml
conda activate InvNeRF
```
The proposal-network sampler additionally needs [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn#requirements) and [NeRFAcc](https://github.com/nerfstudio-project/nerfacc#installation). The configurations reported in the paper use the fixed sampler and run without them.

### Preprocessing
The short-term correspondences (RAFT, CoTracker or MFT) are computed with [Preprocessing4tracker](https://github.com/L0za007/Preprocessing4tracker). Clone it next to this repository and follow its README to download the model weights:
```
cd ..
git clone https://github.com/L0za007/Preprocessing4tracker.git
```
The masked configuration also needs [SAM2](https://github.com/facebookresearch/sam2) for the tool masks; the prompts we used are included in [`processes/sam2_prompts/`](processes/sam2_prompts/), see [`processes/README.md`](processes/README.md#3-tool-mask-prompts-invnerf_mask-only).

### Paths
Machine-specific paths are kept out of the repository in a `.env` file placed in the **parent** directory of the repo. It must define `ROOT_PATH`, the directory that contains the `Datasets/` folder. `PREPROCESSING_DIR` is optional and defaults to a sibling `Preprocessing4tracker/` folder.
```
# ../.env
ROOT_PATH=/absolute/path/to/data/root
PREPROCESSING_DIR=/absolute/path/to/Preprocessing4tracker
```

## Data
We evaluated 2D and 3D point tracking on the [STIR Challenge 2024 dataset](https://zenodo.org/records/14803158), using the ground-truth locations generated with [STIRMetrics](https://github.com/athaddius/STIRMetrics), the evaluation code of the challenge organisers. The [SCARED dataset](https://endovissub2019-scared.grand-challenge.org/) was used to evaluate the integration of camera poses from the robot kinematics and the rendered depth.

The download, the expected folder structure and the preparation of both datasets are described in [`processes/README.md`](processes/README.md).

## Configurations
This repo includes the configurations of the experiments reported in the paper. The rest of the options are documented in `config/config.py`.

| Paper | Config | Correspondences | Tool masks |
| ----- | ------ | --------------- | ---------- |
| InvNeRF                | `config/STIR/STIR_test6.yaml` | RAFT + CoTracker | no  |
| InvNeRF<sub>mask</sub> | `config/STIR/STIR_test8.yaml` | RAFT + CoTracker | yes |
| InvNeRF<sub>MFT</sub>  | `config/STIR/STIR_test9.yaml` | MFT              | no  |
| InvNeRF on SCARED      | `config/SCARED/Exp1.yaml`     | MFT              | no  |

All of them use the MLP-CaDeX++ deformation network, multi-scale HexPlanes for colour and density, and the fixed sampler.

## Code usage
The full pipeline (frame extraction, preprocessing, fitting and evaluation) is run per dataset from the root of the repo:
```
python processes/STIR.py --exp_n 6 --flow raft     # InvNeRF
python processes/STIR.py --exp_n 8 --flow raft     # InvNeRF_mask
python processes/STIR.py --exp_n 9 --flow mft      # InvNeRF_MFT
python processes/SCARED.py --exp_n 1
```

### Training
A clip that has already been preprocessed can be fitted on its own. `--data_dir` is relative to `ROOT_PATH`:
```
python train.py -c config/STIR/STIR_test6.yaml -dd Datasets/STIR_InvNeRF/case02_seq09
```
The results are written to `<data_dir>/out/<expname>_<date>_<time>/`: the resolved `config.yaml`, the training log, the checkpoints and videos of the rendered tracks, RGB and depth.

### Evaluation
```
python solver/STIR_eval.py --vid_dir <path to the clip's left .mp4> --out_dir <ROOT_PATH>/Datasets/STIR_InvNeRF/case02_seq09 --experiment 6
python solver/gather_info.py --out_dir <ROOT_PATH>/Datasets/STIR_InvNeRF --all
```
`STIR_eval.py` evaluates every fit of the experiment found in the clip folder and writes `evaluation.json` with the predicted and ground-truth points, the 2D and 3D end-point errors, and the PSNR and SSIM of the rendered right views. `gather_info.py` summarises all clips into `evaluation_summary.csv`. The threshold accuracies reported in the paper are computed with [STIRMetrics](https://github.com/athaddius/STIRMetrics).

## Layout
```
surg-invnerf/
|-- config/        # config.py (defaults and documentation of every option), experiment .yaml files
|-- DataManager/   # dataset, data module, camera models and ray manager
|-- model/         # invertible deformation networks, HexPlanes and the colour/density models
|-- solver/        # loss functions and evaluation scripts
|-- processes/     # end-to-end pipelines per dataset and their README
|-- engine.py      # Lightning training framework
|-- train.py       # fits InvNeRF on one clip
`-- vis.py         # rendered videos of tracks, RGB, depth and canonical space
```

## Acknowledgments
Our preprocessing builds on the code of [OmniMotion](https://github.com/qianqianwang68/omnimotion), and the correspondences come from [RAFT](https://github.com/princeton-vl/RAFT), [CoTracker](https://github.com/facebookresearch/co-tracker) and [MFT](https://github.com/serycjon/MFT). The invertible deformation networks follow [CaDeX](https://github.com/JiahuiLei/CaDeX), and the tool masks are generated with [SAM2](https://github.com/facebookresearch/sam2).

## Contact & Support
**Maintainer:** Gerardo Loza
**Contact:** scgelg@leeds.ac.uk

## Cite
```
@article{Loza2026InvNeRF,
  author  = {Gerardo Loza and Junlei Hu and Dominic Jones and Sharib Ali and Pietro Valdastri},
  title   = {Surg-InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision},
  year    = {2026},
}
```
