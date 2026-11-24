import torch, time, os
from pathlib import Path
import torch.nn.functional as F
import pytorch_lightning as pl
from config.config import update_log_screen
from solver.criteria import InvNeRFLoss, get_in_range_mask, weight_scheduler
from DataManager.ray_manager import RayManager
try: 
    from nerfacc.volrend import render_weight_from_density
    from nerfacc.estimators.prop_net import PropNetEstimator
except Exception as e:
    print("NERFACC not installed.")


# Data types
from typing import Dict, List, Tuple
from torch import Tensor


class BaseTrainer(pl.LightningModule):
    """ Base trainer class
    Meaning of the complete out dictionary in the training_step function:
        rays1 (torch.Tensor):        Sampled points in frame 1.                                [n_imgs, n_pts, n_samples, 3] 3D space             -> Not used by loss 
        rays1_depth (torch.Tensor):  Sampled points depth in frame 1.                          [n_imgs, n_pts, n_samples]    3D space            
        canon_pts1 (torch.Tensor):      Canonical points generated from frame 1.               [n_imgs, n_pts, n_samples, 3] 3D canonical space  
        p_rays2 (torch.Tensor):   Prediction for each sample in canon_pts1.                    [n_imgs, n_pts, n_samples, 3] 3D space             -> Not used by loss 
        color1 (torch.Tensor):           Color for each sample in canon_pts1                   [n_imgs, n_pts, n_samples, 3] RGB                  -> Not used by loss 
        density1 (torch.Tensor):         Density for each sample in canon_pts1                 [n_imgs, n_pts, n_samples]    -                    -> Not used by loss  
        m_wgt1 (torch.Tensor):       Blending weights                                          [n_imgs, n_pts, n_samples]    -                    
        p_pts2 (torch.Tensor):       Projection of the blended predicted points(pred_spl_pts2) [n_imgs, n_pts, 2]            Image space           
        p_pts2_depth (torch.Tensor): Depth of the blended predicted points(pred_spl_pts2)      [n_imgs, n_pts]               3D space             -> Used for occ score
        p_rgb1 (torch.Tensor):       Predicted blended color                                   [n_imgs, n_pts, 3]            RGB  

        p_rays2_depth (torch.Tensor):  Depth of NON blended predicted points(pred_spl_pts2)    [n_imgs, n_pts, n_samples]    3D space            
        p_rays2_onimgPlane (torch.T):Projection of NON blended predicted points(pred_spl_pts2) [n_imgs, n_pts, n_samples, 2] Image space          -> Not used by loss 

    if calculate flow scene loss we also add:
        flow_prev (torch.Tensor):  Scene flow from the previous frame [n_imgs, n_pts, 3]
        flow_after (torch.Tensor): Scene flow from the after frame    [n_imgs, n_pts, 3]
        
    if cross consistency loss we also add:
        canon_pts2 (torch.Tensor): Canonical points generated from frame 2.                  [n_imgs, n_pts, n_samples, 3] 3D canonical space  
        m_wgt2 (torch.Tensor):     Blending weights                                          [n_imgs, n_pts, n_samples]    -          
    
    """
    def __init__(self, config:Dict, models:Dict[str,torch.nn.Module], criterion:InvNeRFLoss, label_manager:RayManager, save_path:Path):
        super(BaseTrainer, self).__init__()
        ## Saving config
        self.config = config
        
        ## Creating models
        self.Feature_mlp = models['Feature_mlp']
        self.Deform_mlp  = models['Deform_mlp']
        self.Color_mlp   = models['Color_mlp']
        self.Density_mlp = models['Density_mlp'] if 'Density_mlp' in models.keys() else None
        self.Rho_mlp = None
        self.Estimator   = None
        if config.estimator == 'density_est':
            self.Rho_mlp = models['Rho_mlp']
            self.Estimator = PropNetEstimator()
            self.transmitance_from_density = render_weight_from_density
        if self.config.merge_before_deform:
            print("Render before deformation, the Density/Color is calculated over deformable space")
        elif self.config.density_on_canonical:
            print("Density/Color is calculated over canonical space")
        else:
            print("Density/Color is calculated over deformable space")

        ## Variables for training
        self.Lbl_mgr = label_manager
        # TODO: find a better way to generate a time step
        self.time_step = 1. / label_manager.num_imgs
        #self.grid = label_manager.gen_grid(normalize=False, homogeneous=True).float()
        self.step = max(config.init_step, 1)
        self.epoch = 1
        self.surface_found = True # if False it bypasses the flow loss until the surface is found

        ## Losses
        self.Criterion = criterion
        ## Trackers
        self.Val_trackers = {'V_loss':0, 'V_floss': 0, 'V_RGBloss':0, 'V_gloss': 0, 'V_epoch':0}
        self.train_trackers = {}
        self.nb=0
        self.last_lr = None
        self.last_val_loss = float('inf')
        self.upscale_counter = -self.config.scale_cool_down
        ## Output directory
        self.out_dir = save_path
        self.cache_dir = Path(save_path) / 'cache'
    
    def configure_optimizers(self):
        """ Configure the optimizer and the learning rate scheduler
        Returns:
            optimizer (torch.optim): Optimizer
            lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        """
        param_dict = [{'params': self.Deform_mlp.parameters() , 'lr': self.config.lr_deform , 'name': 'Deform_mlp'}]
        if self.Color_mlp is not None:
            param_dict += [{'params': self.Color_mlp.parameters(), 'lr': self.config.lr_color, 'name': 'Color_mlp'}]
        if self.Density_mlp is not None:
            param_dict += [{'params': self.Density_mlp.parameters(), 'lr': self.config.lr_color, 'name': 'Density_mlp'}]
        if self.Feature_mlp is not None:
            param_dict += [{'params': self.Feature_mlp.parameters(), 'lr': self.config.lr_feature, 'name': 'Feature_mlp'}]
        if self.Rho_mlp is not None:
            param_dict += [{'params': self.Rho_mlp.parameters(), 'lr': 1e-4, 'name': 'Rho_mlp'}]
        optimizer = torch.optim.AdamW(param_dict, weight_decay=1e-3)
        
        if self.config.lr_scheduler == 'step':
            sched = torch.optim.lr_scheduler.StepLR(optimizer,
                        step_size=self.config.lrate_decay_steps,
                        gamma=self.config.lrate_decay_factor)
            shedparams = {'scheduler': sched}
        elif self.config.lr_scheduler == 'plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                        factor=self.config.lrate_decay_factor , 
                        patience=self.config.lrate_patience,
                        threshold=0.01, 
                        cooldown=self.config.lrate_cooldown,
                        min_lr=self.config.min_lr)
            shedparams = {'scheduler': sched, 
                          'monitor': 'V_emap', 
                          'strict': False, 
                          'interval': 'epoch', 
                          'frequency': self.config.val_freq}
        else: 
            raise ValueError(f"Learning rate scheduler {self.config.lr_scheduler} not implemented")

        return {"optimizer": optimizer, 
                "lr_scheduler": shedparams,
                }

    def training_step(self, batch, batch_idx):
        """ Training step
        Args:
            batch (dict): Batch data
            batch_idx (int): Batch index
        Returns:
            loss (torch.Tensor): Loss value
        """
        ids1, pts1 = batch['ids1'], batch['pts1'], 
        raysOD1 = (batch['ro1'], batch['rd1']) if 'ro1' in batch.keys() else None
        pts1_3D = batch['pts1_3D'] #if 'pts1_3D' in batch.keys() else None
        ids2, pts2 = batch['ids2'], batch['pts2']
        raysOD2 = (batch['ro2'], batch['rd2']) if 'ro2' in batch.keys() else None
        pts2_3D = batch['pts2_3D'] #if 'pts2_3D' in batch.keys() else None

        prediction_1_to_2 = self.sampling_and_inference(pts1, ids1, ids2, raysODa=raysOD1, pts3D_a=pts1_3D, det=False)
        prediction_2_to_1 = self.sampling_and_inference(pts2, ids2, ids1, raysODa=raysOD2, pts3D_a=pts2_3D, det=True, id_tags=(2,1), return_T=False) 
        prediction = {**prediction_1_to_2, **prediction_2_to_1} # Combine the predictions
        
        if 'smoth_flow_loss' in self.config.losses:
            input = prediction_1_to_2['rays1'][:, :int(self.config.num_pts / 4)]
            if self.config.merge_before_deform:
                input = prediction_1_to_2['p_3Dpts1'].unsqueeze(-2).detach().clone()[:, :int(self.config.num_pts / 4)]
            flow = self.predict_scene_flow(input, 
                                           prediction_1_to_2['canon_pts1'][:, :int(self.config.num_pts / 4)], ids1)
            prediction.update(flow)
        # Computing the loss
        loss, losses = self.Criterion(output=prediction, target=batch, 
            step=self.epoch, device=self.Lbl_mgr.device, bypass_flow=not self.surface_found)
        if self.config.estimator == 'density_est':
            loss_e = self.Estimator.compute_loss(prediction_1_to_2['transmitance'], loss_scaler=100)
            losses['estimator'] = loss_e
            loss += loss_e

        # if loss is nan print the output dictionaty
        if torch.isnan(loss):
            self.get_nan_values(losses, batch, prediction)
            self.save_checkpoint(f'model_NAN_{self.epoch:06d}.pth')
            raise ValueError("Loss is nan")
        
        # Logging
        for k,v in losses.items():
            if k not in self.train_trackers:
                self.train_trackers[k] = 0
            self.train_trackers[k] += v.item()
        if "T_loss" not in self.train_trackers:
            self.train_trackers["T_loss"] = 0
        self.train_trackers["T_loss"] += loss.item()
        self.log('T_loss', loss.item())
        self.log_dict(losses)
        self.step += 1
        self.nb += 1
        return loss
        
    def on_train_epoch_start(self, *args, **kwargs):
        self.epoch_timer = time.time()

    def on_train_epoch_end(self, opt=None, *args, **kwargs):
        losses = {k: v/self.nb for k,v in self.train_trackers.items()}
        self.nb = 0
        loss = losses['T_loss']
        losses.pop('T_loss')
        self.log_txt(time.time()-self.epoch_timer, losses=losses, loss=loss)
        self.train_trackers = {}
        curent_lr = self.trainer.optimizers[0].param_groups[0]['lr'] if opt is None else opt.param_groups[0]['lr']
        self.log(f'lr', curent_lr)
        if self.last_lr is None:
            self.last_lr = curent_lr
        elif curent_lr != self.last_lr:
                self.last_lr = curent_lr
                sched = self.lr_schedulers()
                sched.best = float('inf')
        
        # get_density_progression_video(self, grid_step=180, out_dir=str(self.out_dir), fps=10)
        self.epoch += 1
            
    def validation_step(self, batch, batch_idx):
        ids1, pts1 = batch['ids1'], batch['pts1'], 
        raysOD1 = (batch['ro1'], batch['rd1']) if 'ro1' in batch.keys() else None
        pts1_3D = batch['pts1_3D'] #if 'pts1_3D' in batch.keys() else None
        ids2, pts2 = batch['ids2'], batch['pts2']
        pts2_3D = batch['pts2_3D'] #if 'pts2_3D' in batch.keys() else None

        output = self.sampling_and_inference(pts1, ids1, ids2, raysODa=raysOD1, pts3D_a=pts1_3D,
                use_max_loc=self.config.use_max_loc)

        # Criteria
        h,w = self.Lbl_mgr.h, self.Lbl_mgr.w
        total_loss = 0
        cache_error = 0
        ## Flow error
        if self.surface_found:
            in_range_mask = get_in_range_mask(pts2, w,h, self.config.max_padding)
            flow_error = F.l1_loss(output['p_pts2'], pts2, reduction='none').mean(dim=-1)
            cicle_mask = (batch['c_mask'] | batch['occ_mask'])
            flow_error *= in_range_mask * cicle_mask * batch['pair_weight']
            self.Val_trackers['V_floss'] += flow_error.mean()
            total_loss += flow_error.mean()
            cache_error += flow_error
        ## RGB error
        if 'rgb_loss' in self.config.losses:
            rgb_error = F.l1_loss(output['p_rgb1'], batch['gt_rgb1'], reduction='none').mean(dim=-1)
            rgb_mean_error = rgb_error.mean()
            self.Val_trackers['V_RGBloss'] += rgb_mean_error
            total_loss += rgb_mean_error
            cache_error += rgb_error 

        if True:#'geo_consistency_loss' in self.config.losses: ### geometric reconstruction error
            rays1 = self.Lbl_mgr.expand_from_unisphere(output['rays1'])
            pts1_3D = torch.sum(output['m_wgt1'].unsqueeze(-1) * rays1, dim=-2)
            geo_error = torch.norm(pts1_3D - batch['pts1_3D'], dim=-1) * 10
            geo_error *= batch['geo_mask1'] * (geo_error > 1e-2) * batch['c_mask'] * batch['pair_weight']
            self.Val_trackers['V_gloss'] += geo_error.mean()
            total_loss += geo_error.mean()
            cache_error += geo_error

        # Accumulate error map updates directly in the shared memory
        self.Lbl_mgr.update_error_map(ids1, ids2, cache_error, self.epoch)

        # Update trackers
        self.Val_trackers['V_loss'] += total_loss
        self.Val_trackers['V_epoch'] += 1
        return total_loss
    
    def on_validation_epoch_end(self, *args, **kwargs):
        # Get epoch time
        val_time = time.time() - self.val_timer
        # Save the cache stats
        self.Lbl_mgr.save_error_log(self.epoch)
        # Get the losses
        n_scale_flags = len(list((self.out_dir/'cache').glob('upscale*.txt')))
        scale = self.config.scale_factor * (2**n_scale_flags)
        early_stop_flag = scale > self.config.max_scale
        scale = min(scale, self.config.max_scale)
        val_loss = {}
        for k,v in self.Val_trackers.items():
            if k != 'V_epoch':
                val_loss[k] = v/self.Val_trackers['V_epoch']
        
        emap_mean = self.Lbl_mgr.dataset.error_map.mean()
        emap_std = self.Lbl_mgr.dataset.error_map.std()
        emap_metric = emap_std+emap_mean
        val_loss['emap_avg'] = emap_mean
        val_loss['emap_std'] = emap_std
        val_loss['V_emap'] = emap_metric 
        val_loss['scale'] = scale
        # Upscale schedulers
        lr_sched = self.lr_schedulers()
        if self.upscale_counter <0:
            self.upscale_counter += 1
        else:
            self.upscale_function(emap_metric)
        if val_loss['V_gloss']<=2e-2 and self.epoch>1 and not self.surface_found:
            self.surface_found=True
            self.last_val_loss =lr_sched.best= float('inf')
            self.upscale_counter =lr_sched.num_bad_epochs= 0
        elif not self.surface_found:
            self.last_val_loss =lr_sched.best= float('inf')
            self.upscale_counter =lr_sched.num_bad_epochs= 0
        print("Scale Scheduler:",f"Counter: {self.upscale_counter}, Best:{self.last_val_loss:.4f}")
        print("LRate Scheduler:",f"Counter: {lr_sched.num_bad_epochs-lr_sched.cooldown_counter}, Best: {lr_sched.best:.4f}")
        # Log
        self.log_txt(val_time, val_loss, l_type='Val')
        if early_stop_flag:
            val_loss['_V_emap'] = emap_metric 
        self.log_dict(val_loss)
        # Reset the trackers
        self.Val_trackers = {'V_loss':0, 'V_floss': 0, 'V_RGBloss':0, 'V_gloss': 0, 'V_epoch':0}
        self.val_timer = 0
        if self.epoch % self.config.i_weight == 0:
            self.save_checkpoint(f'model_{self.epoch:06d}.pth')
    
    def on_validation_epoch_start(self, *args, **kwargs):
        # Reset the validation timer
        self.val_timer = time.time()

    def upscale_function(self, val_loss):
        if self.last_val_loss-val_loss < 0.01:
            self.upscale_counter += 1
            n_scale_flags = len(list(self.cache_dir.glob('upscale*.txt')))
            scale = min(self.config.scale_factor * (2**n_scale_flags), self.config.max_scale)
            if self.upscale_counter >= self.config.scale_patience:
                upscale_flag = self.cache_dir / f'upscale_{n_scale_flags:02d}.txt'
                file = open(upscale_flag, 'w')
                file.close()
                self.last_val_loss = float('inf')
                self.upscale_counter = -self.config.scale_cool_down
                if scale < self.config.max_scale: # Reset the learning rate scheduler
                    sched = self.lr_schedulers()
                    sched.cooldown_counter = self.config.scale_cool_down
                    #sched.num_bad_epochs = 0
                    #sched.best = float('inf')
        else:
            self.last_val_loss = val_loss
            self.upscale_counter = 0
            # Reset the learning rate scheduler
            # sched = self.lr_schedulers()
            # sched.num_bad_epochs = 0
            # sched.best = float('inf')
            # sched.cooldown_counter = 2

##########################################################################################################################################
# Functions for inference
##########################################################################################################################################

    def sampling_and_inference(self, ptsa, ida, idb, id_tags=(1,2), raysODa=None, pts3D_a=None,
                      det=True, use_max_loc=False, return_T=True, camera_side='left', return_depth=True):
        a,b = id_tags[0], id_tags[1]
        rays_a_data = self.Lbl_mgr.sample_rays(ptsa, ida, raysOD=raysODa, 
            det=det, return_depth=True, time_step = self.time_step, 
            est = self.Estimator, rho_mlp = self.Rho_mlp, side=camera_side)
        rays_a = rays_a_data['rays']
        # rays_a = rays_a.requires_grad_(True)
        rays_a = self.Lbl_mgr.contract_to_unisphere(rays_a) # unit cube
        if pts3D_a is not None: pts3D_a = self.Lbl_mgr.contract_to_unisphere(pts3D_a) 
        if self.config.merge_before_deform:
            sig_a, r_wgt_a, rgb_a, p_3Dpts_a, canon_a, p_3Dpts_b = self.model_inference_render_first(rays_a, ida, idb,
                                                                    use_max_loc=use_max_loc) #, pts3D1=pts3D_a)
            # rays_a = self.Lbl_mgr.expand_from_unisphere(rays_a)
            p_3Dpts_a = self.Lbl_mgr.expand_from_unisphere(p_3Dpts_a)
            _,p_pts_a_depth = self.Lbl_mgr.camera_model.project_to_image(
                p_3Dpts_a, ids=ida, return_depth=True) if return_depth else (None,None)
            p_rays_b = None # The input rays were never deformed
            if idb is not None: 
                p_3Dpts_b = self.Lbl_mgr.expand_from_unisphere(p_3Dpts_b)
                p_pts_b, p_pts_b_depth = self.Lbl_mgr.camera_model.project_to_image(
                        p_3Dpts_b, ids=idb, return_depth=True) if return_depth else (None,None)
            else: p_pts_b=p_pts_b_depth=None
        else:
            canon_a, sig_a, r_wgt_a, rgb_a, p_rays_b = self.model_inference(rays_a, ida, idb, use_max_loc=use_max_loc)#, pts3D1=pts3D_a)
            if idb is not None: 
                p_rays_b = self.Lbl_mgr.expand_from_unisphere(p_rays_b)
                p_3Dpts_b = self.Lbl_mgr.merge_samples(p_rays_b, r_wgt_a, use_max_loc=use_max_loc, detach=True)
                p_pts_b, p_pts_b_depth = self.Lbl_mgr.camera_model.project_to_image(
                    p_3Dpts_b, ids=idb, return_depth=True) if return_depth else (None,None)
            else: p_3Dpts_b=p_pts_b=p_pts_b_depth=None
            # rays_a = self.Lbl_mgr.expand_from_unisphere(rays_a)
            p_3Dpts_a = self.Lbl_mgr.merge_samples(self.Lbl_mgr.expand_from_unisphere(rays_a), 
                    r_wgt_a, use_max_loc=use_max_loc)
            _,p_pts_a_depth = self.Lbl_mgr.camera_model.project_to_image(
                p_3Dpts_a, ids=ida, return_depth=True) if return_depth else (None,None)

        out = { # Rendering of the first image
                f'rays_org_{a}': rays_a_data['rays_o'],
                f'rays_dir_{a}': rays_a_data['rays_d'],
                f'rays{a}': rays_a, 
                f'canon_pts{a}': canon_a, 
                f'density{a}': sig_a,
                f'm_wgt{a}': r_wgt_a,
                f'p_pts{a}_depth': p_pts_a_depth,
                f'p_3Dpts{a}': p_3Dpts_a,
                f'p_rgb{a}': rgb_a,
                f'p_rays{b}': p_rays_b, 
                f'p_3Dpts{a}to{b}': p_3Dpts_b,
                f'p_pts{b}': p_pts_b, 
                f'p_pts{b}_depth': p_pts_b_depth,
                }
        
        if self.config.estimator == 'density_est' and return_T:
            t_starts,t_ends = rays_a_data['t_starts'],rays_a_data['t_ends']
            sigmas = sig_a.reshape(t_starts.shape[0], sig_a.shape[2]) # [n_imgs*n_pts, n_samples]
            _, trans, _ = self.transmitance_from_density(t_starts, t_ends, sigmas)
            out['transmitance'] = trans
            # out['prop_net_T'] = rays_a_data["hidden_sigmas"]
        return out
    
    def model_inference(self, rays1, ids1, ids2=None, use_max_loc=False):
        """ Inference of the newral networks. 
        Args:
            rays1 (torch.Tensor): Sampled points [n_imgs, n_pts, n_samples, 3]
            ids1 (torch.Tensor): Image 1 ids [n_imgs]
            ids2 (torch.Tensor): Image 2 ids [n_imgs]
        """
        canon_pts1 = self.get_prediction_one_way(rays1, ids1)
        if self.config.canonical_contraction:
            canon_pts1 =  self.canonical_contraction(canon_pts1)
        if self.config.density_on_canonical:
            color, density = self.get_canonical_color_and_density(canon_pts1, ids1) # DO NOT detach the canon_pts1 here
        else:
            color, density = self.get_canonical_color_and_density(rays1, ids1)
          
        r_wgt = self.Lbl_mgr.get_rendering_weights(density=density)
        p_rgb = self.Lbl_mgr.merge_samples(color, r_wgt) if color is not None else None

        if ids2 is not None:
            p_rays2 = self.get_prediction_one_way(canon_pts1, ids2, inverse=True)
            return canon_pts1, density, r_wgt, p_rgb, p_rays2
        
        return canon_pts1, density, r_wgt, p_rgb, None

    def model_inference_render_first(self, rays1, ids1, ids2=None, use_max_loc=False, pts3D1=None):
        """ Inference of the newral networks. 
        Args:
            rays1 (torch.Tensor): Sampled points [n_imgs, n_pts, n_samples, 3]
            ids1 (torch.Tensor): Image 1 ids [n_imgs]
            ids2 (torch.Tensor): Image 2 ids [n_imgs]
        """
        color, density = self.get_canonical_color_and_density(rays1, ids1)
        r_wgt = self.Lbl_mgr.get_rendering_weights(density=density)
        p_rgb = self.Lbl_mgr.merge_samples(color, r_wgt) if color is not None else None
        
        if pts3D1 is None:
            # raise NotImplementedError("pts3D1 is not supported in this function.")
            p_3Dpts1 = self.Lbl_mgr.merge_samples(rays1, r_wgt, use_max_loc=use_max_loc)
            p_3Dpts1 = p_3Dpts1.unsqueeze(-2)
            canon_pts1 = self.get_prediction_one_way(p_3Dpts1.detach().clone(), ids1)
        else:
            raise NotImplementedError("pts3D1 is not supported in this function.")
            p_3Dpts1 = pts3D1.unsqueeze(-2)
            canon_pts1 = self.get_prediction_one_way(p_3Dpts1, ids1)
            
        if ids2 is not None:
            p_3Dpts2 = self.get_prediction_one_way(canon_pts1, ids2, inverse=True).squeeze(-2)
            return density, r_wgt, p_rgb, p_3Dpts1.squeeze(-2), canon_pts1.squeeze(-2), p_3Dpts2
        
        return density, r_wgt, p_rgb, p_3Dpts1.squeeze(-2), canon_pts1.squeeze(-2),  None

    def get_prediction_one_way(self, rays, ids, inverse=False):
        """ Mapping 3d points from local to canonical or from canonical to local (inverse=True)
        Args:
            rays (torch.Tensor): 3D points [n_imgs, n_pts, n_samples, 3] (they should be canonical if inverse=True)
            ids (torch.Tensor): Image ids [n_imgs]
            inverse (bool): If True, map from canonical to local
        Returns:
            out (torch.Tensor): Predicted 3D points [n_imgs, n_pts, n_samples, 3]
        """
        assert len(ids.shape) == 1, f"The time shape must be 1D. Got shape {ids.shape}"
        assert ids.shape[0] == rays.shape[0], "The number of images and the number of time steps must be the same. Got {} and {}".format(ids.shape[0], rays.shape[0])
        time = ((ids+1) * self.time_step)[:,None] # [n_imgs, 1]
        
        if self.config.invertible_model == 'InvertNeRF' or self.config.invertible_model == 'InvertNeRFv2' or self.config.invertible_model == 'fastInvertNeRF':
            feature = self.Feature_mlp(time)  # [n_images, n_feat]
            if inverse:
                out = self.Deform_mlp.inverse(time, feature, rays) # from canonical to local
            else:
                out = self.Deform_mlp.forward(time, feature, rays) # from local to canonical

        elif self.config.invertible_model == 'HexaInvertNeRF':
            if inverse:
                out = self.Deform_mlp.inverse(rays, time)  # from canonical to local
            else:
                out = self.Deform_mlp.forward(rays, time)  # from local to canonical

        elif self.config.invertible_model == 'NVPnonlin':
            if inverse:
                out = self.Deform_mlp.inverse(y=rays, t=time,f=None)
            else:
                out = self.Deform_mlp.forward(x=rays, t=time,f=None)
        if torch.isnan(out).any():
            print("NAN detected in the output of the Deform_mlp")
            print(f"Input rays shape: {rays.shape}, ids shape: {ids.shape}, inverse: {inverse}")
            print(f"Output shape: {out.shape}")
            raise ValueError("NAN detected in the output of the Deform_mlp")
        return out 
    
    def canonical_contraction(self, pts):
        # Move points to the centre of the coordinate system
        pts = (pts - 0.5) * 2
        # Find the points that are outside the unit sphere
        x_norm = pts.norm(dim=-1, keepdim=True).detach()
        mask = (x_norm > 1).squeeze(-1)
        # Calculate the contraction factor for each point
        c_factor = torch.ones_like(pts)
        c_factor[mask] = ((2. - 1. / x_norm) * (1 / x_norm))[mask]
        # Apply the contraction factor
        pts = pts * c_factor
        # Move the points back to the original position
        pts = pts / 2 + 0.5
        return pts

    def get_canonical_color_and_density(self, x_canonical, time=None):
        """ Query the nerf network to get the color and density of the canonical 3D locations.
        Args:
            x_canonical (torch.Tensor): Canonical 3D locations [n_imgs, n_pts, n_samples, 3]
            time (torch.Tensor): Time [n_imgs]
        Returns:
            color (torch.Tensor): Color [n_imgs, n_pts, n_samples, 3]
            density (torch.Tensor): Density [n_imgs, n_pts, n_samples]
        """
        time = (time+1) * self.time_step
        if self.config.color_model == 'GaborNet':
            if self.config.add_time:
                time = torch.ones_like(x_canonical[...,:1])*time[...,None,None,None]
                x_canonical = torch.cat([x_canonical, time], dim=-1)
            out_canonical = self.Color_mlp(x_canonical)
            color   = torch.sigmoid(out_canonical[..., :3]) # [n_imgs, n_pts, n_samples, 3]
            density = F.softplus(out_canonical[..., -1] - 1.) # [n_imgs, n_pts, n_samples]
            return color, density
        elif self.config.color_model == 'HexaPlane':
            color = self.Color_mlp(x_canonical, time[:,None])
            color = torch.sigmoid(color.squeeze(-1)) # [n_imgs, n_pts, n_samples, 3]
        elif self.config.color_model == None or self.config.color_model == 'None':
            color = None
        else:
            raise ValueError(f"Color model {self.config.color_model} not implemented")

        if self.config.density_model == 'HexaPlane':
            density = self.Density_mlp(x_canonical, time[:,None])
            density = F.softplus(density.squeeze(-1) - 1.)
        else:
            raise ValueError("No color or density MLP found")

        return color, density
    
    def predict_scene_flow(self, rays, canon_rays, ids):
        """ Predict scene flow for a set of 3D points
        Args:
            ids (torch.Tensor): Image ids [n_imgs,]
            rays (torch.Tensor): 3D points [n_imgs, n_pts, n_samples, 3]
        Returns:
            flow_prev (torch.Tensor): Scene flow from the previous frame [n_imgs, n_pts, 3]
            flow_after (torch.Tensor): Scene flow from the after frame [n_imgs, n_pts, 3]
        """
        # Prepare the data
        mask_valid = (ids >= 1) * (ids < self.Lbl_mgr.num_imgs - 1)
        ids = ids[mask_valid]
        if len(ids) == 0:
            if len(ids) > 1: # ignore the last frame but print if multiple frames are ignored
                print("No valid ids step", self.step)
            return {'flow_prev': torch.zeros(1,1, 3), 'flow_after': torch.zeros(1,1, 3)}
        rays = rays[mask_valid]
        canon_rays = canon_rays[mask_valid].unsqueeze(-2) if self.config.merge_before_deform else canon_rays[mask_valid]
        assert rays.shape == canon_rays.shape, f"The number of rays and the number of canonical rays must be the same. Got {rays.shape} and {canon_rays.shape}"
        ids_prev = ids - 1
        ids_after = ids + 1
        # Predict points' locations in the previous and after frames 
        pts_prev_after = self.get_prediction_one_way(torch.cat([canon_rays, canon_rays], dim=0), 
            torch.cat([ids_prev, ids_after]), inverse=True)
        # Bring the points to the image plane if ray projection is focal
        # if self.config.projection_type == 'focal':
            # rays = self.Lbl_mgr.camera_model.project_to_image(rays, ids=ids)
            # pts_prev_after = self.Lbl_mgr.expand_from_unisphere(pts_prev_after)
            # pts_prev_after = self.Lbl_mgr.camera_model.project_to_image(pts_prev_after, ids=torch.cat([ids_prev, ids_after]))

        # Calculate scene flow
        pts_prev, pts_after = torch.split(pts_prev_after, split_size_or_sections=len(rays), dim=0)
        flow_prev = rays - pts_prev
        flow_after = pts_after - rays
        return {'flow_prev': flow_prev, 'flow_after': flow_after}   
    
    ########################################################################################
    # Functions for the log
    ########################################################################################

    def log_txt(self, duration: float, losses:Dict[str,Tensor], loss:Tensor=None, l_type:str='Train'):
        """ Log the losses and the duration of the step
        Args:
            duration (float): Duration of the step
            losses (dict): Dictionary with the losses
            loss (torch.Tensor): Loss value
            l_type (str): Type of the loss
            TB (bool): If True, log the losses in tensorboard
        """
        try:
            lr = [f'{gp["lr"]:.1e}' for gp in self.trainer.optimizers[0].param_groups]
            self.cap_lr(self.config.min_lr)
        except Exception as e:
            lr = "---"

        if True:
            text = l_type[0] + f' Epoch: {self.epoch:6d}'
            if l_type == 'Train':
                text += f' lr: ({lr[0]})'
            if loss is not None:
                _loss =loss.item()  if isinstance(loss, torch.Tensor) else loss
                text += f' | loss: {_loss:.4f}'
            for k, v in losses.items():
                _v = v.item() if isinstance(v,torch.Tensor) else v
                text += f' | {k}: {_v:.4f}'
            print(text + f' | time: {duration:.2f}s')
        
        update_log_screen(self.config) 

    def save_checkpoint(self, filename=None):
        if filename is None:
            filename = f'model_{self.epoch:06d}.pth'
        to_save = {'Deform_mlp': self.Deform_mlp.state_dict()}
        if self.Color_mlp is not None:
            to_save.update({'Color_mlp': self.Color_mlp.state_dict()})
        if self.Density_mlp is not None:
            to_save.update({'Density_mlp': self.Density_mlp.state_dict()})
        if self.Feature_mlp is not None:
            to_save.update({'Feature_mlp': self.Feature_mlp.state_dict()})
        if self.Rho_mlp is not None:
            to_save.update({'Rho_mlp': self.Rho_mlp.state_dict()})
        torch.save(to_save, os.path.join(self.out_dir, filename))

    def cap_lr(self, min_lr):
        """ Cap the learning rate to a minimum value"""
        for pg_id, param_group in enumerate(self.trainer.optimizers[0].param_groups):
            if param_group['lr'] < min_lr:
                param_group['lr'] = max(param_group['lr'], min_lr)
                print(f"Learning rate in group {pg_id} capped to {min_lr}")

    ################################################################################################
    # Functions for visualization
    ################################################################################################

    def get_density_progression(self, ids:torch.Tensor, pts:torch.Tensor):
        """ Visualize the density progression of the model
        """
        output = self.sampling_and_inference(pts[None], ids[None], None)
        rays_o, _ = self.Lbl_mgr.camera_model.get_rays(pts[None], ids=ids[None])
        rays = self.Lbl_mgr.expand_from_unisphere(output['rays1'])
        depth = torch.linalg.norm(rays - rays_o.unsqueeze(-2), dim=-1, ord=2)
        r_wgt = output['m_wgt1']

        NP = rays.shape[1]
        if 'prop_net_T' in output.keys():
            _depths, _transmitance = [],[]
            for inter,trans in output['prop_net_T'][:-1]:
                # Transmitance
                NS = trans.shape[-1]
                trans = trans.reshape(1, NP, NS)
                # Depth
                init, end = inter.vals[:,:-1], inter.vals[:,1:]
                h_depth = (end - init) / 2 + init
                h_depth = h_depth * (self.Lbl_mgr.camera_model.max_depth - self.Lbl_mgr.camera_model.min_depth) + self.Lbl_mgr.camera_model.min_depth
                h_depth = h_depth.reshape(1, NP, NS)
                _depths.append(h_depth.squeeze()), _transmitance.append(trans.squeeze())
            _depths.append(depth.squeeze()), _transmitance.append(r_wgt.squeeze())
        else:
            _depths, _transmitance = [depth.squeeze()], [r_wgt.squeeze()]

        return _depths, _transmitance, rays_o

    def get_depthmap_and_rgb(self, img_id:int, pts:Tensor, camera_side:str='left'):
        """"""
        rgbs,depths = [],[]
        # Iterate over the grid points to get the depth and color
        img_id = torch.tensor(img_id)[None]
        for _pts in torch.split(pts, split_size_or_sections=self.config.chunk_size, dim=0):
            output = self.sampling_and_inference(_pts[None], img_id, None, use_max_loc=self.config.use_max_loc, camera_side=camera_side)
            _,_depth = self.Lbl_mgr.camera_model.project_to_image(output['p_3Dpts1'],
                                    ids=img_id, return_depth=True)
            _rgb= output['p_rgb1']
            if _rgb is not None:
                rgbs.append(_rgb[0])
            depths.append(_depth[0])
        rgb = torch.cat(rgbs, dim=0) if _rgb is not None else None
        depthmap = torch.cat(depths, dim=0)
        
        return depthmap, rgb
    
    def get_pair_flow(self, id1, id2):
        """ Get the flow between two images
        Args:
            id1 (torch.Tensor): Image 1 id [1]
            id2 (torch.Tensor): Image 2 id [1]
        Returns:
            flow (torch.Tensor): Optical flow [1, h, w, 2]
        """
        grid = self.Lbl_mgr.gen_grid(normalize=False, homogeneous=True).float() # a pt per pixel [h, w, 3]
        grid = grid[..., :2].reshape(-1, 2) # remove z coordinate & flat -> [n_samples, 2] where n_samples = h * w

        pred_loc = []
        for pts in torch.split(grid, split_size_or_sections=self.config.chunk_size, dim=0):
            prediction = self.sampling_and_inference(pts[None,], id1, id2, use_max_loc=self.config.use_max_loc)
            pred_loc.append(prediction["p_pts2"][0])
        pred_loc = torch.cat(pred_loc, dim=0)
        flow = (pred_loc - grid).reshape(self.Lbl_mgr.h, self.Lbl_mgr.w, 2)
        return flow
    
    def get_corr_pair(self, id1, id2, pts, occ=None, return3D=False):
        """ Get the correspondence between two images for a set of 2D points in the image plane
            of the first image.
        Args:
            id1 (torch.Tensor): Image 1 id [1]
            id2 (torch.Tensor): Image 2 id [1]
            pts (torch.Tensor): Image 1 points [n_pts, 2]
            occ (bool): If True, return the occlusion mask
        Returns:
            pred_pts (torch.Tensor): Predicted points [n_pts, 2]
            occ_mask (np.array): Occlusion mask [n_pts]
        """
        if occ is None:
            occ = self.config.vis_occlusion

        if id1 == id2:
            pred_pts = pts
            occ_mask = None
            return pred_pts, occ_mask, torch.ones_like(pred_pts[...,0], dtype=torch.bool)

        pts = pts[None]
        id1,id2 = id1[None], id2[None]
        out = self.sampling_and_inference(pts, id1, id2, use_max_loc=self.config.use_max_loc)
        VF_mask = self.Lbl_mgr.get_in_3D_range_mask(out['p_pts2'],out['p_pts2_depth'])[0]
        if occ:
            occlusion_score = self.get_occlusion_score(
                pts=out['p_pts2'], pts_depth=out['p_pts2_depth'], t_id=id2, v_mask=VF_mask)
            occ_mask = occlusion_score > self.config.occlusion_th
            pred_pts = torch.cat([out['p_pts2'][0], occ_mask.float()], dim=-1)
            occ_mask = occ_mask.squeeze(-1).cpu().numpy()
        else:
            pred_pts = out['p_pts2'][0]
            occ_mask = None
        
        if return3D:
            return pred_pts, occ_mask, VF_mask, out['p_3Dpts1'][0] ,out['p_3Dpts1to2'][0]

        return pred_pts, occ_mask, VF_mask

    def get_occlusion_score(self, pts, pts_depth, t_id, v_mask):
        """ Get the occlusion score for a set of 2D points in the image plane 
            of the target image. 
        Args:
            pts (torch.Tensor): predicted 2D points [1, n_pts, 2]
            pts_depth (torch.Tensor): predicted depth of the pts [1, n_pts]
            t_id (torch.Tensor): Query image id [1,]
            v_mask (torch.Tensor): Visibility mask [n_pts]
        Returns:
            occlusion_score (torch.Tensor): Occlusion score [1, n_pts, 1] (where 1 means occluded)
        """
        if v_mask.sum() == 0:
            return torch.zeros_like(v_mask, dtype=torch.float)[..., None]
        t_id = t_id[0]
        _pts = pts[0][v_mask]
        _pts_depth = pts_depth[0][v_mask]
        out = self.sampling_and_inference(_pts[None], t_id[None], None,
                    use_max_loc=self.config.use_max_loc)
        _,rays_depth = self.Lbl_mgr.camera_model.project_to_image(out['rays1'],
                                    ids=t_id[None],return_depth=True)
        m_wgt = out['m_wgt1']
        eps = 1.1 * (self.Lbl_mgr.camera_model.max_depth - self.Lbl_mgr.camera_model.min_depth
                     ) / self.config.ray_size
        mask_zero = rays_depth.squeeze(-1) >= (_pts_depth[None].expand(-1, -1, 
                        self.config.ray_size) - eps)
        m_wgt[mask_zero] = 0.
        _occlusion_score = m_wgt.sum(dim=-1, keepdim=True)
        occlusion_score = torch.zeros_like(v_mask, dtype=torch.float)
        occlusion_score[v_mask] = _occlusion_score.squeeze()
        return occlusion_score[..., None]
    
    def get_nan_values(self, losses, batch, output):
        """ Check if the networks have nan values
        Returns:
            nan_values (dict): Dictionary with the name of the network and a boolean value
        """
        print(f"NAN Values. Epoch: {self.epoch}, Step: {self.step}")
        for k,v in losses.items():
            print(k,': ', v.item())
        print('\n')

        print("Status of all the networks:")
        nets = {'Deform': self.Deform_mlp}
        if self.Color_mlp is not None:
            nets['Color'] = self.Color_mlp
        if self.Feature_mlp is not None:
            nets['Feature'] = self.Feature_mlp
        if self.Rho_mlp is not None:
            nets['Density'] = self.Rho_mlp
        for name, net in nets.items():
            is_nan = torch.stack([torch.isnan(p).any() for p in net.parameters()]).any().item()
            print(f"Parameters in {name} net have nan values: ", is_nan)
        print('\n')

        for k,v in output.items():
            is_nan = torch.isnan(v).any().item()
            print(k,'has nan values: ', is_nan, ' with shape: ', v.shape)
        print('\n')
        for k,v in batch.items():
            is_nan = torch.isnan(v).any().item()
            print(k,'has nan values: ', is_nan, ' with shape: ', v.shape)
