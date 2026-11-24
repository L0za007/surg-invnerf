import warnings, os
warnings.filterwarnings("ignore")

import torch.multiprocessing as mp
mp.set_start_method('fork', force=True)
import argparse, torch, time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from config.config import get_config, update_log_screen, get_saving_path
from DataManager import get_dataset, RAFTDataModule
from DataManager.ray_manager import RayManager
from model import get_model

from solver.criteria import InvNeRFLoss
from engine import BaseTrainer

from vis import get_video_correspondences, get_rgb_and_depth_video, get_gt_depth_video, get_density_progression_video, get_canonical_space_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training script for point tracker.")
    parser.add_argument("-c", "--configuration", type=str, default=False, help = "Path to experiment specifications (.YAML file)")
    parser.add_argument("-dd", "--data_dir",     type=str, default=None,  help = "Data directory")
    parser.add_argument("-di", "--data_initialization", action="store_true", help = "Initialize the data set to create 3D structures. This will not train the model.")
    args = parser.parse_args()

    # Configurations
    if args.configuration:
        config = get_config(args.configuration, data_dir=args.data_dir, mk_out_dir=not args.data_initialization)
    else:
        raise ValueError("Please provide a configuration file.")
    if args.data_initialization:
        config.txt_log = False
    update_log_screen(config, mode='a' if config.load_chckpt else 'w')
    print("Loaded configuration: ", config.name)
    print("Experiment name: ", config.expname)
    print("Starting time: ", time.ctime())
    if torch.cuda.is_available():
        print(f"Active GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("No GPU available. Using CPU.")
    initial_time = time.time()
    saving_path = get_saving_path(config)
    update_log_screen(config)

    # Dataset
    print("\n-> Loading dataset. ", config.data_dir)
    train_dataset, eval_dataset= get_dataset(config, saving_path,  loaders=False)
    DataModule = RAFTDataModule(config, saving_path, train_dataset, eval_dataset)
    train_dataset = DataModule.train_dataset
    train_loader = DataModule.train_dataloader()
    eval_loader = DataModule.val_dataloader()
    print(f"Size: {train_dataset.num_imgs} imgs. Sampling window: {train_dataset.pairs_win} frms. Gap between frames {train_dataset.pairs_gap} frms.") 
    if args.data_initialization:
        exit(0)
    
    update_log_screen(config)
   
    # Data handler
    print("\n-> Creating data handler")
    handler = RayManager(config, train_dataset)
    print(f"Camera model: {config.projection_type}")
    print(f"Static camera: {config.static_camera}")
    print(f"Workspace (min_xyz,max_xyz): {config.aabb}")
    print(f"Rays range: {config.min_depth} - {config.max_depth}")
    print(f"Samples per ray: {config.ray_size}")
    update_log_screen(config)
 
    # Model
    print("\n-> Creating model.")
    models = get_model(config, handler.device, load = config.load_chckpt)
    print(f"Estimator: {config.estimator}")
    update_log_screen(config)

    # Criterion
    print("\n-> Creating criterion.")
    crit = InvNeRFLoss(config, train_dataset, handler)
    for w,l in zip(config.w_losses, config.losses):
        print(f"Loss: {l} | Weight: {w}")
    update_log_screen(config)
    
    print(f"\n-> Output directory: {saving_path}")
    vid_folder = saving_path / 'vid'
    vid_folder.mkdir(parents=True, exist_ok=True)
    # Save configuration
    config_file = saving_path / 'config.yaml'
    with open(config_file, 'w') as f:
        f.write(config.dump())
    
    # Accelerator
    accelerator = config.device
    if accelerator == 'gpu' and not torch.cuda.is_available():
        accelerator = 'cpu'
    print(f"Training on: {accelerator}\n")
    update_log_screen(config)

    # Trainer
    training_framework = BaseTrainer(config, models, 
                                     criterion=crit, 
                                     label_manager=handler, 
                                     save_path=saving_path)

    

    # Callbacks
    early_stop_CB = EarlyStopping(monitor="_V_emap", min_delta=config.min_delta, 
                                  patience=config.patience, strict=False, verbose=True)

    trainer = Trainer(accelerator=accelerator, devices=1, strategy='ddp_find_unused_parameters_true', 
        max_steps           = config.max_steps - config.init_step+1, 
        gradient_clip_val   = config.grad_clip,
        callbacks           = [early_stop_CB],  
        default_root_dir     = str(saving_path),
        log_every_n_steps    = len(train_loader),
        check_val_every_n_epoch = config.val_freq,
        num_sanity_val_steps = 0,
        enable_progress_bar = False,
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs = config.val_freq*config.scale_patience,
        #logger=False,
        )
    
    trainer.fit(model      = training_framework, 
                datamodule = DataModule
                )
    
    print("End time: ", time.ctime())
    final_time = time.time()    
    training_time = final_time - initial_time
    training_time = time.strftime("%H:%M:%S", time.gmtime(training_time))
    print(f"Training finished. Total time: {training_time}")
    training_framework.save_checkpoint()
    
    # Qualitative evaluation
    training_framework.to(handler.device)
    training_framework.eval()

    get_video_correspondences(training_framework, config.query_frame_id, train_dataset, interval=10, radius=2, out_dir=saving_path)
    update_log_screen(config)
    get_gt_depth_video(handler, grid_step=2, colorise=True, out_dir=str(saving_path))
    get_density_progression_video(training_framework, grid_step=80, out_dir=str(saving_path), fps=10)
    get_canonical_space_video(training_framework, grid_step=20, out_dir=str(saving_path), progress_bar=True)
    get_rgb_and_depth_video(training_framework, grid_step=4, colorise=True, out_dir=saving_path)