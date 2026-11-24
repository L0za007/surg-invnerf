# STORMLAB config
from yacs.config import CfgNode as CN
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys, os, time

_C = CN()
# Exp description
_C.name = "Default set-up for tracker"
_C.expname     = ''  # Experiment name

# Paths 
_C.root_dir    = ''  # The data root, read from DATA_DIR in the .env file
_C.data_dir    = ''  # The directory for the video sequence, absolute or relative to root_dir
_C.save_dir    = ''  # Output directory

# Data config
_C.dataset_type  = 'mono' # The type of dataset: mono, stereo.
_C.pairs_win     = 5      # Window size,max number of img pairs per img id.
_C.pairs_gap     = 2     # Gap between the image pairs.
_C.max_num_imgs  = 5000   # Max number of images to use during trainig.
_C.num_imgs      = 0      # Holds the number of images in the dataset. It is updated after reading dataset.
_C.num_pts       = 128    # Number pts to match in each sample during training ( a sample is a pair of images).
_C.em_step       = 16     # Grid step for the error map.
_C.use_error_map = True   # If set to True, the error map will be used to select the samples and their matching points.
_C.scale_factor  = 1.0    # Scale factor for the images and flow maps
_C.max_scale     = 1.0    # Maximum scale factor for the images and flow maps
_C.coord_system  = 'cartesian' # The coordinate system for the images: 'cartesian' or 'polar'
_C.mask_tools    = False
_C.add_CoTracker = False # If set to True, and optical flow updated by Cotracker will be used (cotracker folder should be generated in the preprocessing stage)

# Dataloader config
_C.batch_size     = 8  # Number image pairs in each batch. Batch size
_C.val_batch_size = 16 # Number image pairs in each validation batch. Batch size
_C.loader_seed = 12 # The random seed used for DataLoader
_C.num_workers = 8  # Number of workers in data loader

# Ray manager
_C.projection_type   = 'orthogonal' # orthogonal or focal.
_C.ray_size     = 32    # Number of samples per ray
_C.aabb          = [[-1,-1, 0, 1,1,1]] # Workspace (min_xyz,max_xyz)
_C.fit_aabb      = False  # Fit the workspace to the clip's own stereo cloud instead of using aabb
_C.static_camera = True  # If the camera is static. The extrinsic parameters will be fixed during training
_C.min_depth     = 0.0   # The minimum depth value
_C.max_depth     = 1.0   # The maximum depth value
_C.estimator     = None      # If especified. the model'll use an estimatior to sample the rays: None, Density estimator (density_est), Density guided estimator (density_guided)
_C.ray_std    = 0.02      # The std for the guided estimator (density_guided)

_C.load_chckpt = False  # If loading checkpoint
_C.ckpt_path   = ''     # Checkpoint path
_C.init_step   = 1      # Initial step

# Training 
_C.device        = 'gpu'   # Device
_C.max_steps     = 100000  # Maximum number of steps to train
_C.canonical_contraction = False
_C.depth_mask    = False  # Mask the depth values outside the range. Useful when rays are projected across the workspace as in orthogonal projection
_C.grad_clip          = 0     # Clip the gradient to avoid training instability
# Learning rate
_C.lr_feature         = 1e-3  # Learning rate for feature mlp
_C.lr_deform          = 1e-3  # Learning rate for deform mlp
_C.lr_color           = 5e-4  # Learning rate for color mlp
## learing rate scheduler
_C.lr_scheduler       = 'step' # Learning rate scheduler: step, cosine, plateau
_C.lrate_decay_steps  = 100   # Decay learning rate by a factor every specified number of training epochs if using step scheduler
_C.lrate_patience     = 3     # Patience for reducing the learning rate if using plateau scheduler
_C.lrate_cooldown     = 2     # Cold down period for the plateau scheduler
_C.lrate_decay_factor = 0.8   # factor to decay the learning rate
_C.min_lr             = 1e-6  # Minimum learning rate
## Scale scheduler
_C.scale_cool_down = 2  # Cool down period for the scale scheduler
_C.scale_patience  = 2  # Patience for reducing the scale factor
## Early stopping
_C.patience      = 5       # Patience for early stopping
_C.min_delta     = 0.001    # Minimum delta for early stopping

# Validation 
_C.val_freq      = 200     # Run a validation epoch every specifiend number of training epochs


# Models' configurations 
_C.merge_before_deform=False
## Proposal network:
_C.samples_per_net = [256]#,96] # The number of samples for each proposal net/ density estimator
## Deform MLP: ## Feature MLP: 
_C.invertible_model = 'InvertNeRF' # The type of model: InvertNeRF, HexaInvertNeRF
_C.n_layers  = 3            # Number of layers in the invertible network. Coupling layers
_C.density_on_canonical = True # If set to True, the density and/or color network sample the canonical space instead the deformable space
_C.pe_freq    = 4   # The freq for pe used in the affine coupling layers
_C.use_affine = False # If using additional 2D affine transformation layers for x, y in the invertible network
_C.plane_inv_f = 'NVP' # The function used for the invertible network: NVP, NVPnonlin
## Extra time encoder. By default it uses GaborNet to encode time when using InvertNeRF 
_C.t_encoder = 'GaborNet' # The type of time encoder: GaborNet, FeatTimeLine, None. 
## Color MLP:
_C.color_model = 'GaborNet'
_C.color_input = 3 # Number of input channels for the color mlp
_C.add_time = False
## Density MLP:
_C.density_model = None # DT_plane, HexaPlane
_C.density_n_levels = 6 # Number of levels/resolutions in the hexaplane-based architecture
# loss
# See solver/critera.py for the full descrition of the available losses
_C.losses = ['optical_flow_loss', 
             'rgb_loss',
             'depth_range_loss',
             'distortion_loss']
             #'canonical_sphere_loss']
             #'smoth_flow_loss']
             #'cross_consistency_loss']
             
# The weights for the losses. If a loss is a single value, it is the weight for the loss.
# If a loss is a tuple of 4 values, it is start_step, weight per step, min weight, max weight
_C.w_losses = [ 1,
               [0,2e-4,0,0.1],
                10,
               [7500, 5e-4, 0, 10]]
                #0]
                #20]
                #1]
_C.max_padding       = 0.0   # If predicted pixel locs exceed this padding, mask them out for training

# log
_C.i_print  = 20    # Frequency for printing losses
_C.i_weight = 1000  # Frequency for saving ckpts
_C.txt_log  = False # If saving the log to a txt file

# inference
_C.chunk_size  = 10000 # Chunk size for rendering depth and rgb
_C.use_max_loc = False # For visualisation only. If True, only the sample with maximum blending weight on the ray
                       # is used to compute correspondence. If set to False, the correspondences will be computed in
                       # the same way as training, i.e., compositing all samples along the ray.
_C.query_frame_id = 0     # The id of the query frame
_C.vis_occlusion  = True # If marking occluded pixels as crosses for visualization
_C.occlusion_th   = 0.99  # To determine if a mapped 3d location in the target frame is occluded or not
                          # we look at the fraction of light absorbed by samples in front of this location
                          # on the ray in the target frame (i.e., 1 - transmittance
                          # if that value is higher than this threshold, the mapped point is considered as occluded
_C.foreground_mask_path = '' # Providing the path for foreground mask file for generating trails
_C.test = True

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def get_config(path_cfg_data: Path=None, mk_out_dir=True, data_dir=None):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_cfg_data: path to path_cfg_data files
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data=Path(path_cfg_data)
        if not path_cfg_data.exists():
            raise FileNotFoundError(f"Cannot find {path_cfg_data}")

    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    cfg_base.merge_from_file(path_cfg_data.absolute())
    if data_dir is not None:
        cfg_base.data_dir = data_dir
    print("Loaded configuration: ", cfg_base.name)
    
    manage_directories(cfg_base)
    if mk_out_dir:
        make_output_dir(cfg_base)

    return cfg_base

def make_output_dir(config):
    """ Create the output directory for the experiment"""
    if config.load_chckpt: 
        # If loading checkpoint, the output directory should be provided in the config file
        print("Output directory: ", config.save_dir)
        return
    
    # Create the output directory for experiment
    out_dir = (config.expname+time.strftime("_%Y-%m-%d_%H-%M-%S"))
    if config.save_dir == '': # 
        out_dir = Path(config.data_dir) / Path('out/') / out_dir
        config.save_dir = str(out_dir)
        out_dir = Path(config.root_dir) / out_dir
    else: #
        out_dir = Path(config.save_dir) / out_dir
        config.save_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Created output directory: ", str(out_dir))
    # include the cache folder in the output directory
    cache_dir = out_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

def manage_directories(config):
    """ Set the data root from DATA_DIR in the .env file and check the data directory.
    A relative data_dir is resolved under DATA_DIR; an absolute one is used as given."""
    load_dotenv(find_dotenv(usecwd=True))
    config.root_dir = os.getenv("DATA_DIR")
    if config.root_dir is None:
        raise ValueError("DATA_DIR is not defined. Add it to the .env file in the parent directory of the repository")
    print("DATA_DIR: ", config.root_dir)

    # Check that data directory exist
    data_dir = Path(config.root_dir) / config.data_dir
    if not data_dir.exists():
        raise ValueError("Data directory does not exist: ", data_dir)
    print("Data directory: ", str(data_dir))

def get_saving_path(config):
    """ Get the saving path for the experiment"""
    if config.data_dir in config.save_dir:
        # The output is being stored in the data directory
        exp_path = Path(config.root_dir) / config.save_dir
    else: # The output is being stored in the execution folder
        exp_path = Path(config.save_dir)
    return exp_path

def update_log_screen(config_log, file_name = 'log_screen', mode = 'a'):
    """ Update the log screen to save the output of the model"""
    if config_log.txt_log:
        exp_path = get_saving_path(config_log)
        log_file = f"{file_name}.txt"
        screen_path = exp_path/log_file
        if mode == 'a' and sys.stdout != sys.__stdout__:
            sys.stdout.close()  
        f = open(screen_path,mode)
        sys.stderr = f
        sys.stdout = f