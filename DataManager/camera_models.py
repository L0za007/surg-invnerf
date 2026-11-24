import torch, os, json
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

from typing import Union, Dict
from torch import Tensor
from .raft import RAFTDataset

class CamModelBase(torch.nn.Module):
    def __init__(self, config: Dict[str,Union[int,str]], dataset:RAFTDataset, device:str=None) -> None:
        """ Base class for camera models. It defines the basic methods for ray sampling and projection.
        Args:
            config(dictionay): Config object
            dataset (torch.Dataset): Dataset object with the images
            device (torch.device): Device where the tensors will be allocated
        """
        super().__init__()
        self.config = config
        self.device = device
        self.h, self.w = dataset.h, dataset.w
        self.num_imgs  = dataset.num_imgs
        self.img_files = dataset.img_names
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        if self.min_depth <= 0:
            self.min_depth = 1e-3
            config.min_depth = 1e-3
        self.seq_dir = dataset.seq_dir

    def sample_rays(self, pix_coord, ids=None, return_depth=False, 
                    det=True, raysOD=None, side='left', *args, **kargs):
        """ Samples rays from the camera model using uniform sampling.
        Args:
            pix_coord (torch.Tensor): Pixel coordinates [N,2]
            ids (int, list): Camera ids to use for sampling
            return_depth (bool): Return the depth of the samples
            det (bool): Deterministic sampling
        Returns:
            samples (torch.Tensor): Sampled rays [N,ray_size,3]
            depth (torch.Tensor): Depth of the samples [N,ray_size]
        """
        if raysOD is None:
            rays_o, rays_d = self.get_rays(pix_coord, ids, side=side)
        else:
            # print("Using provided rays_o and rays_d")
            rays_o, rays_d = raysOD
        scales = torch.linspace(self.config.min_depth, 
                                self.config.max_depth, 
                                self.config.ray_size, 
                                device=self.device)
        samples = rays_d[..., None, :] * scales[..., None]
        if not det:
            mids = .5 * (samples[..., 1:,:] + samples[..., :-1,:])
            upper = torch.cat([mids, samples[..., -1:, :]], dim=-2)
            lower = torch.cat([samples[..., 0:1,:], mids] , dim=-2)
            t_rand = torch.rand(samples.shape, device=self.device)
            samples = lower + (upper - lower) * t_rand

        if return_depth:
            depth = torch.clone(samples[...,-1])
            samples += rays_o[..., None, :]
            return {'rays':samples, 'rays_depth':depth, 'rays_d':rays_d, 'rays_o':rays_o}
        else:
            samples += rays_o[..., None, :]
            return {'rays':samples, 'rays_d':rays_d, 'rays_o':rays_o}

    def get_rays(self, pix_coord:Tensor, ids:Union[int,list], side='left', *args, **kargs): # To be implemented in the derived classes
        """Get the rays direction and origin for the given pixel coordinates."""
        raise NotImplementedError
    
    def project_to_image(self): # To be implemented in the derived classes
        """Project the 3D points to the image plane."""
        raise NotImplementedError    

    def get_cam_params(self, idxs, side='left'):
        """ Get the camera parameters for the given indices."""
        assert side in ['left', 'right'], f"Invalid camera side: {side}"
        if isinstance(idxs, torch.Tensor): idxs = idxs.tolist()
        if isinstance(idxs, int): idxs = [idxs]
        #print(idxs)
        idxs = [int(i) for i in idxs]

        cam_params = {'cam_in': [], 'cam_ex': []}
        for idx in idxs:
            K = self.camL_K if side == 'left' else self.camR_K
            cam_params['cam_in'].append(torch.tensor(K, device=self.device))
            RT = self.extrinsics[idx] if side=='left' else self.extrinsics[idx] @ self.calib_M
            cam_params['cam_ex'].append(torch.tensor(RT, device=self.device))
        cam_params['cam_in'] = torch.stack(cam_params['cam_in'], dim=0).float().to(self.device)
        cam_params['cam_ex'] = torch.stack(cam_params['cam_ex'], dim=0).float().to(self.device)

        return cam_params
    
    def normalize_coords(self, coords, h, w, no_shift=False):
        """Nomilise coordinates X,Y between -1 and 1"""
        assert coords.shape[-1] == 2
        if no_shift:
            return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
        else:
            return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.
        
    def denormalize_coords(self, coords, h, w, no_shift=False):
        """Denomilise coordinates X,Y between 0 and W-1, 0 and H-1"""
        assert coords.shape[-1] == 2
        if no_shift:
            return coords * torch.tensor([w-1., h-1.], device=coords.device) / 2.
        else:
            return (coords + 1.) * torch.tensor([w-1., h-1.], device=coords.device) / 2.
        
    def get_camera_pose(self, time_step=0, side='left'):
        """Get the camera pose at a given time step in the format of origin and direction vectors."""
        assert side in ['left', 'right'], f"Invalid camera side: {side}"
        pixel_center = torch.tensor([self.w / 2, self.h / 2], device=self.device).float()
        pixel_center = pixel_center.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        rays_o, rays_d = self.get_rays(pixel_center, ids=time_step, side=side)
        rays_o = rays_o.squeeze()  # [1, 3] -> [3]
        rays_d = rays_d.squeeze()  # [1, 3] -> [3]
        return rays_o, rays_d
        
####################################################################################################################################

class PinholeCam(CamModelBase):
    """ Pinhole camera model. It implements the methods for ray sampling and projection.
    In this model the values of pix_coord are the pixel coordinates in the image plane and they are
    rescaled to go from the real camera resolution to the low resolution used in the model and biseversa.
    The variables h and w are the resolution of the low resolution image and the variables orig_h and orig_w
    are the resolution of the real camera image."""
    def __init__(self, config, dataset, device=None) -> None:
        super().__init__(config, dataset, device)
        self.load_camera_parameters(ds_type=config.dataset_type)

    def get_rays(self, pix_coord, ids, side='left', *args, **kargs):
        assert ids is not None, "Camera ids must be provided"
        assert side in ['left', 'right'], f"Invalid camera side: {side}"
        # Scale the pixel coordinates to the real camera resolution
        pix_coord = self.normalize_coords(pix_coord, self.h, self.w)
        pix_coord = self.denormalize_coords(pix_coord, self.orig_h, self.orig_w)
        # Get the camera parameters
        cam_parms = self.get_cam_params(ids, side=side)
        # Image plane to camera coordinates
        cam_in = cam_parms['cam_in'].to(pix_coord.device)
        dirs = torch.stack([(pix_coord[...,0]-cam_in[:,0,2][:,None])/cam_in[:,0,0][:,None], 
                            (pix_coord[...,1]-cam_in[:,1,2][:,None])/cam_in[:,1,1][:,None], 
                            torch.ones_like(pix_coord[:,:, 0])], -1).float()
        # Camera coordinates to world coordinates.
        cam_ex = cam_parms['cam_ex'].to(pix_coord.device)
        rays_d = torch.einsum('ijk,ilk->ijl', dirs, cam_ex[...,:3,:3])
        rays_o = cam_ex[:,:3,-1].unsqueeze(1).expand_as(rays_d)
        rays_d = rays_d / torch.linalg.vector_norm(rays_d, dim=-1, keepdim= True)
        return rays_o, rays_d

    def project_to_image(self, points, ids, return_depth=False, *args, **kargs):
        cam_parms = self.get_cam_params(ids)
        in_dim = len(points.shape)
        # World coordinates to camera coordinates
        cam_ex = cam_parms['cam_ex']
        translation = cam_ex[:,:3,-1].view(cam_ex.shape[0],*((1,)*(in_dim-2)),3)
        points = points - translation

        # Define the opration based on the number of dimensions
        eins_notation = 'ijk,ilk->ijl' if in_dim == 3 else 'ijqk,ilk->ijql'
        points = torch.einsum(eins_notation, points, torch.inverse(cam_ex[:,:3,:3]))
        
        '''if in_dim == 3:
            # Reshape points to match the einsum notation 'ijk,ilk->ijl'
            points = points.permute(0, 2, 1).contiguous()
            points = torch.matmul(points, torch.inverse(cam_ex[:, :3, :3]))
            points = points.permute(0, 2, 1).contiguous()
        else:
            # Reshape points to match the einsum notation 'ijqk,ilk->ijql'
            points = points.permute(0, 1, 3, 2).contiguous()
            points = torch.matmul(points, torch.inverse(cam_ex[:, :3, :3]))
            points = points.permute(0, 1, 3, 2).contiguous()'''

        if return_depth:
            depth = torch.clone(points[...,-1:])
        x_y, z = points[...,:2], points[...,2:].expand_as(points[...,:2]) 
        z = torch.clamp(z, min=self.config.min_depth)#.detach().clone()
        pixels = x_y / z

        #Camera coordinates to image plane
        cam_in = cam_parms['cam_in']
        size = (cam_in.shape[0],*((1,)*(in_dim-2)))
        pixel_coords = torch.stack([pixels[...,0]*cam_in[:,0,0].view(*size)+cam_in[:,0,2].view(*size),
                                    pixels[...,1]*cam_in[:,1,1].view(*size)+cam_in[:,1,2].view(*size)], -1)
        
        # Scale the pixel coordinates to the low resolution
        pixel_coords = self.normalize_coords(pixel_coords, self.orig_h, self.orig_w)
        pixel_coords = self.denormalize_coords(pixel_coords, self.h, self.w)
        if return_depth:
            return pixel_coords, depth
        else:
            return pixel_coords
    
    def project_to_camera(self, points, ids, *args, **kargs):
        cam_parms = self.get_cam_params(ids)
        in_dim = len(points.shape)
        # World coordinates to camera coordinates
        cam_ex = cam_parms['cam_ex']
        translation = cam_ex[:,:3,-1].view(cam_ex.shape[0],*((1,)*(in_dim-2)),3)
        points = points - translation

        # Define the opration based on the number of dimensions
        eins_notation = 'ijk,ilk->ijl' if in_dim == 3 else 'ijqk,ilk->ijql'
        points = torch.einsum(eins_notation, points, torch.inverse(cam_ex[:,:3,:3]))
        return points
    
    def load_camera_parameters(self, ds_type='mono'):
        """ Load the camera parameters from the dataset files."""
        # Load the camera intrinsics
        calib_file = Path(self.seq_dir) / 'calib.json'
        if not calib_file.exists():
            raise FileNotFoundError(f"Camera intrinsics file not found: {calib_file}")
        calib = json.load(open(calib_file))
        img_size = calib['img_size']
        self.orig_h, self.orig_w = img_size[0], img_size[1]
        self.camL_K = np.array(calib['leftcameramat'])
        if ds_type == 'stereo':
            self.camR_K = np.array(calib['rightcameramat'])
            T,R = np.array(calib['translation']), np.array(calib['rotation'])
            assert T.shape[0] == 3, "Invalid translation vector in calibration parameters"
            assert R.shape[0] == 3, "Invalid rotation vector in calibration parameters. It should be rotation vec"
            self.calib_M = np.eye(4)
            R = Rotation.from_rotvec(R)
            self.calib_M[:3,:3] = R.as_matrix()
            self.calib_M[:3,-1] = T
        # Load the camera extrinsics
        if self.config.static_camera:
            print("Using static camera for ray projection")
            self.extrinsics = [np.eye(4) for _ in range(self.num_imgs)]
        else:
            cam_RT_file = Path(self.seq_dir) / 'cam_RT.json'
            if not cam_RT_file.exists():
                print("WARNING: Camera extrinsics file not found. Using identity matrix (Static camera)")
                self.extrinsics = [np.eye(4) for _ in range(self.num_imgs)]
            else:
                cam_RT_file = json.load(open(cam_RT_file))
                # print(f"Loading pos file {cam_RT_file['pos_file']} and rot file {cam_RT_file['rot_file']}") Only SimCol hast those keys
                RT = cam_RT_file['RT']
                self.extrinsics = [np.array(RT[i]) for i in RT.keys()]   
        
####################################################################################################################################

class NormalCam(CamModelBase):
    def __init__(self, config, dataset, device=None) -> None:
        super().__init__(config, dataset, device)

    def get_rays(self, pix_coord, *args, **kargs):
        rays_o = self.normalize_coords(pix_coord, self.h, self.w)
        rays_d = torch.zeros_like(rays_o, device=self.device)
        # Add the depth dimension
        rays_o = torch.cat([rays_o, torch.zeros_like(rays_o[..., :1])], dim=-1)
        rays_d = torch.cat([rays_d, torch.ones_like( rays_d[..., :1])], dim=-1)

        return rays_o, rays_d

    def project_to_image(self, points, return_depth=False, *args, **kargs):
        samples, depth = torch.split(points, dim=-1, split_size_or_sections=[2, 1])
        pixels = self.denormalize_coords(samples, self.h, self.w)
        if return_depth:
            return pixels, depth
        else:
            return pixels