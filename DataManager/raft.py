import json, imageio, torch
from pathlib import Path
import numpy as np
# Data types
from typing import Dict, Tuple, List
from torch import Tensor

class RAFT_Base_Dataset(torch.utils.data.Dataset):
    def __init__(self,config: dict, out_path:Path, images:Tuple[List[np.ndarray],List[np.ndarray]]=None, eval: bool = False) -> None:
        """ Initialize the RAFT dataset.  """
        # General configuration
        self.config = config
        # Directories
        self.seq_dir  = Path(config.root_dir) / config.data_dir
        self.img_dir  = Path(self.seq_dir) / 'color'
        self.flow_dir = Path(self.seq_dir) / 'raft_exhaustive'
        self.mask_dir = Path(self.seq_dir) / 'raft_masks'
        if config.add_CoTracker:
            self.flow_dir = Path(self.seq_dir) / 'cotracker'
            self.mask_dir = Path(self.seq_dir) / 'cotracker_masks'
        self.cache_dir = Path(out_path) / 'cache'
        self.stereo_dir = Path(self.seq_dir) / 'raft_exhaustive_stereo'
        self.info_3D_dir = Path(self.seq_dir) / '3D_info'
        
        # General configuration
        self.pairs_gap = config.pairs_gap
        self.mask_tools = config.mask_tools
        if config.mask_tools: print("Masking tools")
        self.scale = config.scale_factor
        self.em_step = config.em_step

        # Load images if training, otherwise get specific config from config file
        if eval: print("Running in evaluation mode, loading data description from config")
        img_names = sorted([f.stem for f in self.img_dir.glob('*.png')])
        h, w, _ = imageio.imread(str(self.img_dir / (img_names[0]+'.png'))).shape
        self.h, self.w = h, w
        self.num_imgs = config.num_imgs if eval else min(self.config.max_num_imgs, len(img_names))
        self.img_names = img_names[:self.num_imgs]
        self.pairs_win = min(config.pairs_win, self.num_imgs - 1) # max number of pairs in a window

        # if eval:
        #     self.left_imgs, self.right_imgs = None, None
        if images is not None:
            self.left_imgs, self.right_imgs = images
        else:
            self.left_imgs = self.load_all_images(side='left')
            self.right_imgs = self.load_all_images(side='right') if config.dataset_type == 'stereo' else None

        if (self.seq_dir / 'flow_stats.json').exists():
            flow_stats = json.load(open(str(self.seq_dir / 'flow_stats.json')))
            self.sample_weights = self.get_sample_weights(flow_stats) # Dict of dict. [img_name_1][img_name_2] = num_matches_between_them

        # Camera model
        if (config.dataset_type == 'stereo') and not (Path(self.seq_dir)/'color_r').exists():
            config.dataset_type = 'mono'
            print("WARNING: Stereo dataset requested but not found. Switching to monocular configuration")
        if config.projection_type == 'focal':
            from .camera_models import PinholeCam
            self.camera_model = PinholeCam(config, self)
        else: 
            self.camera_model = None

        # 3D info
        self.add_3D_info = not eval
        if self.add_3D_info:
            self.info_3D_dir.mkdir(parents=True, exist_ok=True)
            self.gen_3D_info()
            self.info_3D = {}  # Dict of npz files with 3D information
            for npz_file in self.info_3D_dir.glob('*.npz'):
                self.info_3D[npz_file] = self.load_3D_info(npz_file)
            if config.fit_aabb:
                self.fit_workspace()

    def fit_workspace(self, quantile: float = 0.005, margin: float = 0.2):
        """Shrink the workspace onto the clip's own triangulated points.

        The configured box is one size for every clip, and a clip occupies a fraction of
        a percent of it, so the grids spend their resolution on space the scene never
        reaches. The box is replaced here by the extent of the stereo cloud, taken
        between quantiles so a handful of bad triangulations cannot stretch it, and
        padded by a share of each span. The ray range follows the same points, with
        enough room above the far face that the ray manager's outlier test still passes
        the real surface.
        """
        points = []
        for map_3D, _, mask in self.info_3D.values():
            kept = map_3D[0].permute(1, 2, 0)[mask[0, 0]]
            if kept.numel():
                points.append(kept.reshape(-1, 3))
        if not points:
            print("No triangulated points: keeping the configured workspace")
            return
        cloud = torch.cat(points).float()
        # torch.quantile refuses very large inputs, and the extent does not need them.
        limit = 2_000_000
        if cloud.shape[0] > limit:
            cloud = cloud[torch.randperm(cloud.shape[0])[:limit]]
        low = torch.quantile(cloud, quantile, dim=0)
        high = torch.quantile(cloud, 1 - quantile, dim=0)
        pad = (high - low) * margin
        low, high = low - pad, high + pad

        distance = cloud.norm(dim=-1)
        near = torch.quantile(distance, quantile).item()
        far = torch.quantile(distance, 1 - quantile).item()

        self.config.aabb = [[round(v, 6) for v in low.tolist() + high.tolist()]]
        self.config.min_depth = max(1e-3, round(near * (1 - margin), 6))
        self.config.max_depth = round(far * 1.5, 6)
        print(f"Fitted workspace from {cloud.shape[0]} points: {self.config.aabb}")
        print(f"Fitted rays range: {self.config.min_depth} - {self.config.max_depth}")

    def get_ids_pair(self, id1: int) -> Tuple[Tensor, Tensor]:
        """Select a second image id for a given first image id and the sample weights"""
        img_name1 = self.img_names[id1]
        matches_with_id1 = self.sample_weights[img_name1] 
        low_cand  = list(range(id1 - 1, max(-1, id1-1-self.pairs_win*self.pairs_gap), -1*self.pairs_gap))[::-1] 
        high_cand =list(range(id1 + 1, min(self.num_imgs, id1 +1+ self.pairs_win*self.pairs_gap), self.pairs_gap))
        cand = low_cand + high_cand
        img2_cand = [f'{i:04d}' for i in cand if i != id1]
        img2_cand = [c for c in img2_cand if c in matches_with_id1.keys()]

        id2s = torch.tensor([self.img_names.index(n) for n in img2_cand])
        sample_weights = torch.tensor([matches_with_id1[i] for i in img2_cand])
        sample_weights /= sample_weights.sum()
        sample_weights[torch.abs(id2s - id1) <= 1] += sample_weights.sum()/sample_weights.numel()
        sample_weights /= sample_weights.sum()

        id2 = torch.multinomial(sample_weights, 1)
        img_name2 = img2_cand[id2]
        id2 = self.img_names.index(img_name2)
        id2 = torch.tensor(id2)
        if not isinstance(id1, Tensor):
            id1 = torch.tensor(id1)

        return id1, id2
    
    def load_all_images(self, side='left') -> List[Tensor]:
        images = []
        for idx in range(self.num_imgs):
            img_name = self.img_names[idx] + '.png'
            img_file = str(self.img_dir / img_name)
            if side == 'right':
                img_file = img_file.replace('color', 'color_r')
            img = imageio.imread(img_file) / 255. # [H, W, 3]
            img = torch.from_numpy(img).float()
            images.append(img)
        return images
    
    def load_mask(self, id1, id2) -> Tensor:
        img_name1, img_name2 = self.img_names[id1], self.img_names[id2]
        mask_file = self.mask_dir / f'{img_name1}_{img_name2}.npy'
        if not Path(mask_file).exists():
            mask_file = str(mask_file).replace('.npy', '.png')
            mask = imageio.imread(mask_file)
            mask = mask > 0
            # save mask in npy format for fast loading
            mask_file = mask_file.replace('.png', '.npy')
            np.save(mask_file, mask)
        else:
            mask = np.load(mask_file)
        return torch.from_numpy(mask)
    
    def get_image(self, idx, side='left', np=False, scale=1) -> Tensor:
        assert side in ['left', 'right'], "side should be either 'left' or 'right'"
        if side == 'right':
            img = self.right_imgs[idx] # [H, W, 3]
        else:
            img = self.left_imgs[idx]

        if np:
            return img.numpy()
        
        if scale < 1:
            size = (int(self.h*scale), int(self.w*scale))
            img = torch.nn.functional.interpolate(img.permute(2,0,1)[None], size=size, 
                        mode='nearest').squeeze().permute(1,2,0)
        return img

    def get_flow(self, id1, id2, scale=1) -> Tensor:
        """ Given two ids, returns the optical flow between the two images"""
        size = (int(self.h*scale), int(self.w*scale))
        img_name1, img_name2 = self.img_names[id1], self.img_names[id2]
        flow_file = str(self.flow_dir / '{}_{}.npy'.format(img_name1, img_name2))
        flow = torch.from_numpy(np.load(flow_file))#, mmap_mode='r'))
        if scale < 1:
            flow = torch.nn.functional.interpolate(flow.permute(2,0,1)[None], size=size, 
                        mode='nearest').squeeze()
            flow = flow.permute(1,2,0) * scale
        return flow
    
    def get_consistency_mask(self, id1, id2, scale=1) -> Tuple[Tensor, bool, Tensor]:
        """ Given two ids, returns a mask of the matches that are consistent between the two images"""
        # Read the masks
        masks = self.load_mask(id1, id2)
        consistency_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0
        # if id1 - id2 == 1: # If the images are consecutive, all pairs are consistent
        #     consistency_mask = torch.ones_like(consistency_mask, dtype=torch.bool) & ~occlusion_mask
        if self.mask_tools:
            masked_tool = masks[..., 2] > 0
            if masked_tool.sum() == 0: 
                print("WARNING: skiping empty tool mask")
                masked_tool = torch.ones_like(masked_tool, dtype=torch.bool)
            consistency_mask = consistency_mask & masked_tool
            occlusion_mask = occlusion_mask & masked_tool

        if scale < 1:
            size = (int(self.h*scale), int(self.w*scale))
            consistency_mask = torch.nn.functional.interpolate(
                                consistency_mask[None,None].float(), 
                                size=size, mode='nearest').squeeze() > 0.5
            occlusion_mask = torch.nn.functional.interpolate(
                                occlusion_mask[None, None].float(), 
                                size=size, mode='nearest').squeeze() > 0.5

        return consistency_mask, occlusion_mask
    
    def get_sample_weights(self, flow_stats):
        sample_weights = {}
        for k in flow_stats.keys():
            sample_weights[k] = {}
            total_num = np.array(list(flow_stats[k].values())).sum()
            for j in flow_stats[k].keys():
                sample_weights[k][j] = 1. * flow_stats[k][j] / total_num
        return sample_weights

    def gen_grid(self, h=None, w=None, normalize=False, homogeneous=False) -> Tensor:
        """ Generate a grid with the position of each pixel
        Args:
            h (int): Height of the grid
            w (int): Width of the grid
            normalize (bool): Normalize the grid between -1 and 1
            homogeneous (bool): Add a homogeneous coordinate to the grid
        Returns:
            grid (Tensor): Grid with the position of each pixel"""
        if h is None: h = self.h
        if w is None: w = self.w
        if normalize:
            lin_y = torch.linspace(-1., 1., steps=h)
            lin_x = torch.linspace(-1., 1., steps=w)
        else:
            lin_y = torch.arange(0, h)
            lin_x = torch.arange(0, w)
        grid_y, grid_x = torch.meshgrid((lin_y, lin_x), indexing='ij') # Addressing torch warning regarding indexing
        grid = torch.stack((grid_x, grid_y), -1)
        if homogeneous:
            grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
        return grid  # [h, w, 2 or 3]
    
    def normalize_coords(self, coords, h, w, no_shift=False):
        assert coords.shape[-1] == 2
        if no_shift:
            return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
        else:
            return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.

######################################################################################################################

    def get_scale(self) -> float:
        upscale_factor = len(list(self.cache_dir.glob('upscale*.txt')))
        scale = min(self.scale * (2**upscale_factor), self.config.max_scale)
        return scale

    def get_stereo_corr(self,idx, ptsL:Tensor=None, scale=1, step=None): 
        """ Given an image id, returns the corresponding stereo matches
        Args:
            idx (int): Image id
            ptsL (Tensor): Points in the left image to get the corresponding stereo matches in the right image. coords (y,x)"""
        assert self.config.dataset_type == 'stereo', "This function is only for stereo datasets"
        size = (int(self.h*scale), int(self.w*scale))
        img_name = self.img_names[idx]
        flow_file = str(self.stereo_dir / '{}L_{}R.npy'.format(img_name, img_name))
        flow = torch.from_numpy(np.load(flow_file)).float()
        if scale < 1:
            flow = torch.nn.functional.interpolate(flow.permute(2,0,1)[None], size=size, 
                        mode='nearest').squeeze()
            flow = flow.permute(1,2,0) * scale
        # if len(flow.shape) == 2: # If the flow is a single flow field assume it is the disparity
        #     y_flow = np.zeros_like(flow)
        #     flow = np.stack([flow, y_flow], axis=-1)
        if ptsL is None:
            step = self.em_step if step is None else step
            coordL = self.gen_grid(*size)
            coordR = self.gen_grid(*size) + flow
            ptsL = coordL[::step, ::step, :].reshape(-1, 2)
            ptsR = coordR[::step, ::step, :].reshape(-1, 2)
        else:
            flow = flow.to(ptsL.device)
            ptsL = ptsL * scale
            ptsR = ptsL + flow[ptsL[:, 1].int(), ptsL[:, 0].int()]
        return ptsL/scale, ptsR/scale

    def rays_generator(self, ids1:Tensor,     ids2:Tensor,
                             pts1:Tensor=None,pts2:Tensor=None,
                             scale:float=1, stereo_pair:bool=False,  **kwargs):
        """ Generates rays for given points and IDs based on the dataset type.

            Parameters:
            ids1 (Tensor): Tensor containing the first set of IDs.
            ids2 (Tensor): Tensor containing the second set of IDs.
            pts1 (Tensor): Tensor containing the first set of points.
            pts2 (Tensor): Tensor containing the second set of points.
            **kwargs: Additional keyword arguments.

            Returns:
            tuple: Depending on the dataset type, returns either:
                - For 'stereo' dataset type: (rays_oL, rays_dL, rays_oR, rays_dR)
                - For other dataset types: (rays_o1, rays_d1, rays_o2, rays_d2)
            
            Each element in the tuple is a Tensor of shape [n_pts, 3] representing the origin and direction of the rays.
        """
        if pts1 is None:
            pts1 = self.gen_grid().reshape(-1, 2)
        if stereo_pair:
            ptsL, ptsR = self.get_stereo_corr(ids1.int(), pts1, scale=scale)

            # Get origin and direction of the rays
            rays_oL, rays_dL = self.camera_model.get_rays(ptsL[None], ids1.int(), side='left')
            rays_oR, rays_dR = self.camera_model.get_rays(ptsR[None], ids1.int(), side='right') 

            # Reshape rays to [n_pts, 3]
            rays_oL, rays_dL = rays_oL.reshape(-1, 3), rays_dL.reshape(-1, 3)
            rays_oR, rays_dR = rays_oR.reshape(-1, 3), rays_dR.reshape(-1, 3)

            return rays_oL, rays_dL, rays_oR, rays_dR

        else:
            if pts2 is None:
                flow_1to2 = self.get_flow(ids1.int(), ids2.int()).to(pts1.device)
                pts2 = pts1 + flow_1to2[pts1[:, 1].int(), pts1[:, 0].int()]
            rays_o1, rays_d1 = self.camera_model.get_rays(pts1[None], ids1.int())
            rays_o2, rays_d2 = self.camera_model.get_rays(pts2[None], ids2.int())

            # Reshape rays to [n_pts, 3]
            rays_o1, rays_d1 = rays_o1.reshape(-1, 3), rays_d1.reshape(-1, 3)
            rays_o2, rays_d2 = rays_o2.reshape(-1, 3), rays_d2.reshape(-1, 3)

            return rays_o1, rays_d1, rays_o2, rays_d2

    def rays_triangulation(self, rO_1, rD_1, rO_2, rD_2):
        """ Triangulates the intersection point of two rays in 3D space.
        Args:
            rO_1 (torch.Tensor): Origin of the first ray, shape [n_pts, 3].
            rD_1 (torch.Tensor): Direction of the first ray, shape [n_pts, 3].
            rO_2 (torch.Tensor): Origin of the second ray, shape [n_pts, 3].
            rD_2 (torch.Tensor): Direction of the second ray, shape [n_pts, 3].

        Returns:
            tuple: A tuple containing:
            - intersection (torch.Tensor): The intersection points, shape [n_pts, 3].
            - tol1 (torch.Tensor): The tol1 of the intersection, shape [n_pts, 3].
            - geo_mask (torch.Tensor): A mask indicating the valid intersections, shape [n_pts].
        Raises:
            ValueWarning: If all coefficients are zero, indicating no valid intersection.
        """
        # Generate equation system to solve the intersection
        # rO_1 +c_a*rD_1 = rO_2 + c_b*rD_2 +c_c*unit_d
        # c_b*rD_2 +c_c*unit_d - c_a*rD_1 = rO_1 - rO_2
        # h = rO_1 - rO_2 # hypotenuse
        ca = torch.cross(rD_1, rD_2, dim=-1) # Vector perpendicular to rays' direction (planes) 
        unit_d = ca / torch.norm(ca, dim=-1, keepdim=True)
        A = torch.stack([rD_2, unit_d, -rD_1], dim=-1) # A [n_pts, 3, 3] 
        B = rO_1 - rO_2 # B [n_pts, 3]

        # Find matched parallel rays
        coef = torch.zeros_like(B)
        det_A = torch.linalg.det(A)
        non_zero_det_mask = det_A != 0

        # Solve the equation system for non-parallel rays
        if non_zero_det_mask.any():
            coef[non_zero_det_mask] = torch.linalg.solve(A[non_zero_det_mask], 
                                                         B[non_zero_det_mask])

        # Remove intersections out of the depth range
        invalid_mask = coef[..., 2] < self.config.min_depth
        coef[..., 2][invalid_mask] = 0
        invalid_mask = coef[..., 2] > self.config.max_depth
        coef[..., 2][invalid_mask] = 0
        
        geo_mask = coef[..., 2] != 0  # Valid intersections mask
        # Warning if all coefficients are zero
        if (~geo_mask).all():
            print(f"Warning: All coefficients are zero.")
        #print(f"num 0 det {(~non_zero_det_mask).sum()}, num neg coef {neg_coef_mask.sum()}")

        # interpolate the coefficient values from non zero values if enough valid intersections
        missing_coef = (~geo_mask).sum()
        valid_coef = (geo_mask).sum()
        if (missing_coef > 0) and ((valid_coef/2) > missing_coef) and not (self.config.dataset_type == 'mono'):
            coef[..., 2:] = self.coef_interpolation(rD_1, coef[..., 2:], geo_mask)
            geo_mask = coef[..., 2] != 0
        
        # Calculate the intersection and tol1
        intersection = rO_1 + coef[..., 2:] * rD_1 # [n_pts, 3]
        tol1 = coef[..., 1:2] * unit_d
        

        return intersection, tol1, geo_mask

    def coef_interpolation(self, pts, coef, mask):
        """ Interpolates the coefficients of invalid intersections based on the coefficients of valid intersections.
        Args:
            pts (torch.Tensor): Coordinates of the cooficients in a 2D space, shape [n_pts, 2].
            coef (torch.Tensor): The coefficients' values, shape [n_pts, 1].
            mask (torch.Tensor): A mask indicating the valid intersections, shape [n_pts].

        Returns:
            torch.Tensor: The interpolated coefficients, shape [n_pts, 1].
        """
        # Get the valid and invalid coefficients
        valid_coef = coef[mask]
        invalid_coef = coef[~mask]
        neighbors = min(8, valid_coef.numel())

        # Interpolate the invalid coefficients
        if invalid_coef.numel() > 0:
            #print(f"Warning: Interpolating {invalid_coef.numel()} invalid coefficients")
            invalid_pts = pts[~mask]
            valid_pts = pts[mask]
            dists = torch.cdist(invalid_pts, valid_pts)
            topk_dists, topk_indices = torch.topk(dists, neighbors, largest=False)
            topk_dists = topk_dists / (topk_dists.sum(dim=-1, keepdim=True) + 1e-6)
            valid_coef = valid_coef[topk_indices].squeeze()
            interpolated_coef = (topk_dists * valid_coef).sum(dim=-1, keepdim=True)
            coef[~mask] = interpolated_coef

        return coef
    
    def gen_3D_info(self):
        if self.config.dataset_type == 'stereo':
            for i in range(self.num_imgs):
                file_name = self.info_3D_dir / f'{self.img_names[i]}.npz'
                if not file_name.exists():
                    pstL, ptsR = self.get_stereo_corr(i, step=1)
                    ro1,rd1 = self.camera_model.get_rays(pstL[None],ids=i,side='left')
                    ro2,rd2 = self.camera_model.get_rays(ptsR[None],ids=i,side='right')
                    map_3D, map_tol, mask = self.rays_triangulation(ro1,rd1,ro2,rd2)
                    map_3D, map_tol, mask = map_3D.squeeze(), map_tol.squeeze(), mask.squeeze()
                    # Save 3D information in reference to the origin of the workspace
                    param1 = self.camera_model.get_cam_params(i)['cam_ex'][0]
                    R,T = param1[:3, :3], param1[:3, 3]
                    map_3D = (R.T @ (map_3D.T - T[:, None])).T
                    self.save_3D_info(file_name, map_3D, map_tol, mask)
        elif self.config.dataset_type == 'mono':
            for i in range(self.num_imgs):
                file_name = self.info_3D_dir / f'{self.img_names[i]}.npz'
                if not file_name.exists():
                    if self.config.static_camera:
                        raise ValueError("3D information generation is not supported for static camera in mono datasets.")
                    if i == self.num_imgs - 1: i -= 1 # Last image has no next image
                    flow_1to2 = self.get_flow(i, i+1).reshape(-1, 2)
                    pts1 = self.gen_grid().reshape(-1, 2)
                    pts2 = pts1 + flow_1to2
                    ro1,rd1 = self.camera_model.get_rays(pts1[None],ids=i)
                    ro2,rd2 = self.camera_model.get_rays(pts2[None],ids=i+1)
                    map_3D, map_tol, mask = self.rays_triangulation(ro1,rd1,ro2,rd2)
                    map_3D, map_tol, mask = map_3D.squeeze(), map_tol.squeeze(), mask.squeeze()
                    # Save 3D information in reference to the origin of the workspace
                    param1 = self.camera_model.get_cam_params(i)['cam_ex'][0]
                    R,T = param1[:3, :3], param1[:3, 3]
                    map_3D = (R.T @ (map_3D.T - T[:, None])).T

                    invalid_depth_mask = torch.norm(map_3D, dim=-1)
                    mask = mask & (invalid_depth_mask > self.config.min_depth) & (invalid_depth_mask < self.config.max_depth)
                    self.save_3D_info(file_name, map_3D, map_tol, mask)
        else:
            raise ValueError(f"Unknown dataset type: {self.config.dataset_type}. Supported types are 'stereo' and 'mono'.")
    
    def save_3D_info(self, file_name:Path, map_3D:Tensor, tol:Tensor, mask:Tensor):
        map_3D = map_3D.numpy()
        tol = tol.numpy()
        mask = mask.numpy()
        np.savez_compressed(file_name, map_3D=map_3D, tol=tol, mask=mask)
    
    def load_3D_info(self, file_name:Path) -> Tuple[Tensor, Tensor, Tensor]:
        data = np.load(file_name, allow_pickle=True)
        map_3D = torch.from_numpy(data['map_3D'])
        # tol = torch.from_numpy(data['tol'])
        mask = torch.from_numpy(data['mask'])

        map_3D = map_3D.reshape(self.h, self.w, 3)
        mask = mask.reshape(self.h, self.w)

        map_3D = map_3D.permute(2, 0, 1).unsqueeze(0)  # [3, H, W] -> [1, 3, H, W]
        mask = mask[None, None]

        return map_3D, None, mask

    def get_3Dpts(self, ids1, ids2, pts1, pts2, scale=1, **kw):
        if self.config.dataset_type == 'stereo':
            map1_3D, map1_tol, mask1 = self.info_3D[self.info_3D_dir / f'{self.img_names[ids1.int()]}.npz']
            map2_3D, map2_tol, mask2 = self.info_3D[self.info_3D_dir / f'{self.img_names[ids2.int()]}.npz']
            # mask = mask1 & mask2

        elif self.config.dataset_type == 'mono':
            map1_3D, map1_tol, mask1 = self.info_3D[self.info_3D_dir / f'{self.img_names[ids1.int()]}.npz']
            map2_3D, map2_tol, mask2 = self.info_3D[self.info_3D_dir / f'{self.img_names[ids2.int()]}.npz']
            # mask = mask1 & mask2

        # Sample pts1_3D using the pts1 and the interpolation function from pytorch
        pts1_normed = self.normalize_coords(pts1, self.h, self.w)[None, None]  # [1, 1, n_pts, 2]
        pts1_3D = torch.nn.functional.grid_sample(map1_3D, pts1_normed.float(), align_corners=True).squeeze().T  # [n_pts, 3]
        # tol1 = torch.nn.functional.grid_sample(map1_tol, pts1_normed.float(), align_corners=True).squeeze().T
        sampled_mask1 = torch.nn.functional.grid_sample(mask1.float(), pts1_normed.float(), align_corners=True).squeeze()
        sampled_mask1 = sampled_mask1 > 0.5

        pts2_normed = self.normalize_coords(pts2, self.h, self.w)[None, None]  # [1, 1, n_pts, 2]
        pts2_3D = torch.nn.functional.grid_sample(map2_3D, pts2_normed.float(), align_corners=True).squeeze().T  # [n_pts, 3]
        # tol2 = torch.nn.functional.grid_sample(map2_tol, pts2_normed.float(), align_corners=True).squeeze().T
        sampled_mask2 = torch.nn.functional.grid_sample(mask2.float(), pts2_normed.float(), align_corners=True).squeeze()
        sampled_mask2 = sampled_mask2 > 0.5

        # Move the points to the right camera location.
        if not self.config.static_camera: # If I move the camera, I need to move the points that were obtained at the origin of the workspace
            param1 = self.camera_model.get_cam_params(ids1.int())['cam_ex'][0]
            R,T = param1[:3, :3], param1[:3, 3]
            pts1_3D = (R @ pts1_3D.T + T[:, None]).T
            param2 = self.camera_model.get_cam_params(ids2.int())['cam_ex'][0]
            R,T = param2[:3, :3], param2[:3, 3]
            pts2_3D = (R @ pts2_3D.T + T[:, None]).T

        data = {'pts1_3D': pts1_3D, 'pts2_3D': pts2_3D, 'geo_mask1': sampled_mask1, 'geo_mask2': sampled_mask2}
        return data

    def get_single_3Dpts(self, ids1, pts1):
        if self.config.dataset_type == 'stereo':
            map1_3D, _, _ = self.info_3D[self.info_3D_dir / f'{self.img_names[ids1.int()]}.npz']

        elif self.config.dataset_type == 'mono':
            map1_3D, _, _ = self.info_3D[self.info_3D_dir / f'{self.img_names[ids1.int()]}.npz']

        map1_3D = map1_3D.to(pts1.device)
        # Sample pts1_3D using the pts1 and the interpolation function from pytorch
        pts1_normed = self.normalize_coords(pts1, self.h, self.w)[None, None]  # [1, 1, n_pts, 2]
        pts1_3D = torch.nn.functional.grid_sample(map1_3D, pts1_normed.float(), align_corners=True).squeeze().T  # [n_pts, 3]
        if not self.config.static_camera: # If I move the camera, I need to move the points that were obtained at the origin of the workspace
            param1 = self.camera_model.get_cam_params(ids1.int())['cam_ex'][0].to(pts1.device)
            R,T = param1[:3, :3], param1[:3, 3]
            pts1_3D = (R @ pts1_3D.T + T[:, None]).T

        data = {'pts1_3D': pts1_3D}
        return data
########################################################################################################

########################################################################################################

class RAFTDataset(RAFT_Base_Dataset):
    def __init__(self, config:dict, out_path:Path, error_map, eval=False) -> None:
        super().__init__(config, out_path, eval=eval)
        # Training helpers
        self.num_pts = self.config.num_pts
        self.error_map = error_map

    def __len__(self) -> int:
        return self.num_imgs

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        
        scale = self.get_scale()
        cache_files = list(self.cache_dir.glob('error_log*.txt'))
        use_error_map = self.config.use_error_map and (np.random.rand() < 0.75)

        # Select Ids
        id1 = torch.tensor(idx % self.num_imgs)
        if use_error_map and (len(cache_files) >= 1):
            id1 = self.get_id_with_high_error(idx)
    
        id1, id2 = self.get_ids_pair(id1)
        frame_interval = abs(id1 - id2)

        # Get consistency mask
        mask, occ_mask = self.get_consistency_mask(id1, id2, scale)

        # Get matching points
        _mask = torch.ones_like(mask, dtype=torch.bool)
        if use_error_map and (len(cache_files) >= 1):
            pts1, pts2, grid, flow1to2 = self.err_point_sampler(id1, id2, _mask, scale)
        else:
            pts1, pts2, grid, flow1to2 = self.count_point_sampler(id1, id2, _mask, scale)

        # Get the RGB values
        img1 = self.get_image(id1, scale=scale)
        img2 = self.get_image(id2, scale=scale)
        h,w = int(self.h*scale), int(self.w*scale)
        pts2_normed = self.normalize_coords(pts2, h, w)[None, None]
        gt_rgb1 = img1[pts1[:, 1], pts1[:, 0]] # [n_pts, 3]
        gt_rgb2 = torch.nn.functional.grid_sample(img2.permute(2, 0, 1)[None], 
                    pts2_normed.float(), align_corners=True).squeeze().T
        
        # Get the pair weight
        weight_factor = torch.tensor(torch.pi / 2) * (frame_interval - 1.) /(self.pairs_win * self.pairs_gap)
        pair_weight = torch.cos(weight_factor)[None]
        mask = mask[pts1[:, 1].int(), pts1[:, 0].int()]
        occ_mask = occ_mask[pts1[:, 1].int(), pts1[:, 0].int()]

        # Add extra data
        rays = self.rays_generator(ids1=id1.int(), ids2=id2.int(),
                                pts1=pts1.float()/scale, pts2=pts2.float()/scale)
        rays_o1, rays_d1, rays_o2, rays_d2 = rays

        # formatr output
        id1, id2 = id1.float(), id2.float() # [1]
        pts1, pts2 = pts1.float()/scale, pts2.float()/scale # [n_pts, 2]
        data = {'ids1': id1, 'pts1': pts1, 'gt_rgb1': gt_rgb1,
                'ids2': id2, 'pts2': pts2, 'gt_rgb2': gt_rgb2,
                'ro1': rays_o1, 'rd1': rays_d1, 'ro2': rays_o2, 'rd2': rays_d2, # [n_pts, 3]
                'c_mask': mask, 'occ_mask': occ_mask,"pair_weight": pair_weight # [n_pts] [n_pts] [1]
                }
        if self.add_3D_info:
            data.update(self.get_3Dpts(id1, id2, pts1, pts2, scale))
       
        return data
        
    def get_id_with_high_error(self, idx) -> Tensor:
        """ Get the id with the highest error in the error map"""
        err_img = torch.ones(self.num_imgs) 
        for i in range(self.num_imgs):
            low_cand = max(0, i-self.pairs_win) - i + self.pairs_win
            high_cand = min(self.num_imgs-1, i+self.pairs_win) - i + self.pairs_win
            err_img[i] = self.error_map[i, low_cand:high_cand].mean()
        err_img = err_img/(err_img.sum()+1e-6)
        id1 = torch.multinomial(err_img+1e-6, 1)
        id1 = id1.squeeze()
        return id1
    
    def quick_item(self, idx) -> Dict[str, Tensor]:
        """ Get a quick item for debugging purposes and plotting sampled 
        points on visualisation of the error map"""
        scale = self.get_scale()
        cache_files = list(self.cache_dir.glob('error_log*.txt'))
        use_error_map = self.config.use_error_map
        # Select Ids
        id1 = torch.tensor(idx % self.num_imgs)
        id1, id2 = self.get_ids_pair(id1)
        # Get consistency mask
        mask, _ = self.get_consistency_mask(id1, id2, scale)

        # Get matching points
        if use_error_map and len(cache_files) >= 1:
            pts1, pts2, grid, flow1to2 = self.err_point_sampler(id1, id2, mask, scale)
        else:
            pts1, pts2, grid, flow1to2 = self.count_point_sampler(id1, id2, mask, scale)
        
        return {'ids1': id1.float(), 'pts1': pts1.float()/scale, 
                'ids2': id2.float(), 'pts2': pts2.float()/scale}
    
    def get_correspondences(self, id1, id2, query_pts) -> Dict[str, Tensor]:
        id1 = torch.tensor(id1)
        id2 = torch.tensor(id2)
        if query_pts is not None:
            query_pts = torch.tensor(query_pts) if not isinstance(query_pts, torch.Tensor) else query_pts
            assert query_pts.shape[-1] == 2, "query_pts should have shape [n_pts, 2]"
        frame_interval = abs(id1 - id2)

        # Get consistency mask
        mask, occ_mask = self.get_consistency_mask(id1, id2)

        # Get matching points
        _mask = torch.ones_like(mask, dtype=torch.bool)
        if query_pts is None:
            pts1, pts2, grid, flow1to2 = self.count_point_sampler(id1, id2, _mask)
        else:
            pts1 = query_pts
            flow_1to2 = self.get_flow(id1, id2)
            pts2 = pts1 + flow_1to2[pts1[:, 1].int(), pts1[:, 0].int()]
            pts2 = pts2.float()
            pts1 = pts1.float()

        # Get the RGB values
        img1 = self.get_image(id1)
        img2 = self.get_image(id2)
        h,w = self.h, self.w
        pts2_normed = self.normalize_coords(pts2, h, w)[None, None]
        gt_rgb1 = img1[pts1[:, 1].int(), pts1[:, 0].int()] # [n_pts, 3]
        gt_rgb2 = torch.nn.functional.grid_sample(img2.permute(2, 0, 1)[None], 
                    pts2_normed.float(), align_corners=True).squeeze().T
        
        # Get the pair weight
        weight_factor = torch.tensor(torch.pi / 2) * (frame_interval - 1.) /(self.pairs_win * self.pairs_gap)
        pair_weight = torch.cos(weight_factor)[None]
        mask = mask[pts1[:, 1].int(), pts1[:, 0].int()]
        occ_mask = occ_mask[pts1[:, 1].int(), pts1[:, 0].int()]

        # Add extra data
        rays = self.rays_generator(ids1=id1.int(), ids2=id2.int(),
                                pts1=pts1.float(), pts2=pts2.float())
        rays_o1, rays_d1, rays_o2, rays_d2 = rays

        # formatr output
        id1, id2 = id1.float(), id2.float() # [1]
        pts1, pts2 = pts1.float(), pts2.float() # [n_pts, 2]
        data = {'ids1': id1, 'pts1': pts1, 'gt_rgb1': gt_rgb1,
                'ids2': id2, 'pts2': pts2, 'gt_rgb2': gt_rgb2,
                'ro1': rays_o1, 'rd1': rays_d1, 'ro2': rays_o2, 'rd2': rays_d2,
                'c_mask': mask, 'occ_mask': occ_mask,"pair_weight": pair_weight
                }
        return data

    def err_point_sampler(self, id1, id2, mask:Tensor, scale:float=1) -> Tuple[Tensor, Tensor]:
        # Get error map for the pair
        low_cand  = list(range(id1 - 1, max(-1, id1-1-self.pairs_win*self.pairs_gap), -1*self.pairs_gap))[::-1] 
        high_cand =list(range(id1 + 1, min(self.num_imgs, id1 +1+ self.pairs_win*self.pairs_gap), self.pairs_gap))
        cand = low_cand + high_cand
        id2_idx = cand.index(id2)
        grid_error = self.error_map[id1, id2_idx].squeeze() # [h//grid_step, w//grid_step] == [h_grid, w_grid]
        
        # Get grid points
        # h,w = (int(self.h*scale), int(self.w*scale))
        # grid_step = self.config.em_step
        # grid = self.gen_grid(h,w)
        # grid_corners = grid[::grid_step, ::grid_step, :]
        # assert grid_error.shape == grid_corners.shape[:2], f"Error grid shape {grid_error.shape} != {grid_corners.shape}"

        # Normalize
        grid_error /= grid_error.max()

        # Expand grid_error to match the size of mask
        h,w = (int(self.h*scale), int(self.w*scale))
        grid_error_expanded = torch.nn.functional.interpolate(
            grid_error.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze().squeeze()

        # Apply grid_error to mask
        mask = mask.float()
        mask *= grid_error_expanded
        mask = mask.reshape(-1)/ mask.max()
        n_valid_pts = (mask > 0).sum().item()

        # Flatten grid
        grid = self.gen_grid(h, w)

        # Select points on the grid based on the error map and mask
        pts_on_grid = torch.multinomial(mask, self.num_pts, 
                replacement=n_valid_pts<self.num_pts)
        pts1 = grid.view(-1, 2)[pts_on_grid] # [n_pts, 2]

        # Get corresponding points on the second image
        flow_1to2 = self.get_flow(id1, id2, scale=scale)
        pts2 = pts1 + flow_1to2[pts1[:, 1], pts1[:, 0]]

        return pts1, pts2, grid, flow_1to2
    
    def count_point_sampler(self, id1, id2, mask, scale=1) -> Tuple[Tensor, Tensor]:
        img_name1, img_name2 = self.img_names[id1], self.img_names[id2]
        count_map = imageio.imread(str(self.seq_dir / 'count_maps' / (img_name1+'.png')))

        pixel_sample_weight = 1 / np.sqrt(count_map + 1.)
        pixel_sample_weight = torch.from_numpy(pixel_sample_weight).float()
        if scale < 1:
            size = (int(self.h*scale), int(self.w*scale))
            pixel_sample_weight = torch.nn.functional.interpolate(pixel_sample_weight[None, None], 
                        size=size, mode='bilinear').squeeze()
        pixel_sample_weight = pixel_sample_weight[mask]
        pixel_sample_weight /= pixel_sample_weight.sum()
        select_ids = torch.multinomial(pixel_sample_weight, self.num_pts, 
                        replacement=(mask.sum() < self.num_pts).item())
        
        flow_1to2 = self.get_flow(id1, id2, scale=scale)
        size = (int(self.h*scale), int(self.w*scale))
        #Bypassing count sampler for quick test
        # step=5
        # grid = self.gen_grid(*size)[::step,::step,:]
        # pts1 = grid.reshape(-1, 2)
        grid = self.gen_grid(*size)
        pts1 = grid[mask][select_ids]
        pts2 = pts1 + flow_1to2[pts1[:, 1], pts1[:, 0]]
        return pts1, pts2, grid, flow_1to2

########################################################################################################

########################################################################################################

class RAFTDatasetEval(RAFT_Base_Dataset):
    def __init__(self, config: dict, out_path: Path, images:Tuple[List[np.ndarray],List[np.ndarray]]=None, eval=False) -> None:
        super().__init__(config, out_path, images=images, eval=eval)
        self.cache_dir = Path(out_path) / 'cache'

        all_pairs = []
        for i in range(self.num_imgs):
            low_cand  = list(range(i - 1, max(-1, i-1-self.pairs_win*self.pairs_gap), -1*self.pairs_gap))[::-1] 
            high_cand =list(range(i + 1, min(self.num_imgs, i +1+ self.pairs_win*self.pairs_gap), self.pairs_gap))
            cand = low_cand + high_cand
            for j in cand:
                all_pairs.append((i, j))
        self.all_pairs = all_pairs
    
    def __len__(self) -> int:
        return len(self.all_pairs)
    
    def __getitem__(self, idx:int) -> Dict[str, Tensor]:
       
        scale = self.get_scale()
        size = (int(self.h*scale), int(self.w*scale))
        evaluation_grid = self.gen_grid(*size)[::self.em_step , ::self.em_step , :]
        evaluation_grid = evaluation_grid.reshape(-1, 2)

        # Select Ids
        id1, id2 = self.all_pairs[idx]
        id1, id2 = torch.tensor(id1), torch.tensor(id2)
        
        # Sample points on img1, at least on point per cell in the validation grid
        em_step = self.em_step # max(1,int(self.em_step*scale))
        shift = torch.randint(0, em_step, evaluation_grid.shape)
        pts1 = evaluation_grid + shift
        w,h = int(self.w*scale), int(self.h*scale)
        pts1[:, 0] = torch.clamp(pts1[:, 0], 0, w-1)
        pts1[:, 1] = torch.clamp(pts1[:, 1], 0, h-1)

        # Get matching points
        flow_1to2 = self.get_flow(id1, id2, scale=scale)
        pts2 = pts1 + flow_1to2[pts1[:, 1], pts1[:, 0]]

        # Get RGB values
        img1, img2 = self.get_image(id1, scale=scale), self.get_image(id2, scale=scale)
        gt_rgb1 = img1[pts1[:, 1], pts1[:, 0]]
        pts2_normed = self.normalize_coords(pts2, self.h, self.w)[None, None]
        gt_rgb2 = torch.nn.functional.grid_sample(img2.permute(2, 0, 1)[None], 
                    pts2_normed.float(), align_corners=True).squeeze().T

        # Get consistency mask
        mask, occ_mask = self.get_consistency_mask(id1, id2, scale)

        # Get the pair weight 
        frame_interval = abs(id1 - id2)
        weight_factor = torch.tensor(torch.pi / 2) * (frame_interval - 1.) /(self.pairs_win * self.pairs_gap)
        pair_weight = torch.cos(weight_factor)[None]
        mask = mask[pts1[:, 1], pts1[:, 0]]
        occ_mask = occ_mask[pts1[:, 1], pts1[:, 0]]

        # formatr output
        id1, id2 = id1.float(), id2.float() # [1]
        pts1, pts2 = pts1.float()/scale, pts2.float()/scale # [n_pts, 2]
        gt_rgb1, gt_rgb2 = gt_rgb1.float(), gt_rgb2.float()
        data = {'ids1': id1, 'pts1': pts1, 'gt_rgb1': gt_rgb1,
                'ids2': id2, 'pts2': pts2, 'gt_rgb2': gt_rgb2,
                'c_mask': mask, 'occ_mask': occ_mask, "pair_weight": pair_weight,
                }

        # Add extra data
        if self.add_3D_info:
            data.update(self.get_3Dpts(id1, id2, pts1, pts2, scale))
        
        return data

