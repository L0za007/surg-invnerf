import torch
from kornia import morphology as morph
from config.config import get_saving_path

# Data types
from typing import Dict, Tuple
from torch import Tensor
from .raft import RAFTDataset

try:
    from nerfacc.estimators.prop_net import PropNetEstimator
except Exception as e:
    print("NERFACC not installed.")
# Video error
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw


class RayManager(torch.nn.Module):
    def __init__(self, config: Dict[str, Tensor], dataset: RAFTDataset):
        super().__init__()
        self.config = config
        self.h, self.w = dataset.h, dataset.w
        self.num_imgs = dataset.num_imgs
        config.num_imgs = self.num_imgs
        self.img_files = dataset.img_names
        self.dataset = dataset
        self.error_map_size = tuple(dataset.error_map.shape[-2:])
        self.coord_system = config.coord_system
        print(f"Using coordinate system: {self.coord_system}")

        if config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.aabb = torch.tensor(config.aabb, dtype=torch.float32).to(self.device)

        if config.projection_type == "focal":
            from .camera_models import PinholeCam

            self.camera_model = PinholeCam(config, dataset, self.device)
        elif config.projection_type == "orthogonal":
            from .camera_models import NormalCam

            self.camera_model = NormalCam(config, dataset, self.device)
        else:
            raise ValueError("Unknown ray projection type")

    def sample_rays(self, pts, img_id, return_depth=False, det=True, **kargs):
        if self.config.estimator == "density_est":
            return self.density_est_sampler(pts, img_id, return_depth=return_depth, det=det, **kargs)
        if self.config.estimator == "density_guided":
            return self.guided_sampler(pts, img_id, return_depth=return_depth, det=det, **kargs)
        else:
            return self.camera_model.sample_rays(pts, img_id, return_depth=return_depth, det=det, **kargs)

    def guided_sampler(self, pts, img_id, intersection, raysOD=None, return_depth=True, det=True, side="left", **kargs):
        if raysOD is None:
            rays_o, rays_d = self.camera_model.get_rays(pts, ids=img_id, side=side)
        else:
            rays_o, rays_d = raysOD
        rays_hd = intersection - rays_o
        # compress to max depth of the camera
        rays_hd_norm = torch.norm(rays_hd, dim=-1, keepdim=False)
        outlier_mask = rays_hd_norm > self.config.max_depth * 0.8
        rays_hd[outlier_mask] = rays_hd[outlier_mask] / rays_hd_norm[outlier_mask][..., None] * self.config.max_depth * 0.8

        # Expand rHD to accommodate the number of samples
        rays_hd = rays_hd.unsqueeze(2).expand(-1, -1, self.config.ray_size, 3)  # [B, N_pts, num_samples, 3]

        # Sample offsets from a Gaussian distribution
        std = self.config.ray_std
        if det:
            offsets = torch.linspace(0, 1, self.config.ray_size, device=self.device)
            offsets = (offsets[..., None] - 0.5) * std * 2
            offsets = offsets[None, None].repeat(rays_hd.shape[:2] + (1, 1))
        else:
            offsets = torch.randn_like(rays_hd[..., :1], device=self.device) * std

        # Handle invalid offsets
        rays = rays_hd + rays_d.unsqueeze(2) * offsets
        ## Flip the direction of the rays if they are antiparallel
        rays_d_exp = rays_d.unsqueeze(2).expand(-1, -1, self.config.ray_size, 3)
        antiparallel = torch.sum(rays * rays_d_exp, dim=-1, keepdim=False) < 0
        offsets[antiparallel] = -offsets[antiparallel]
        ## Push the samples infront of the image plane
        rays_norm = torch.norm(rays, dim=-1, keepdim=False)
        mask = rays_norm < self.config.min_depth
        offsets[mask] += self.config.min_depth
        ## Sort the offsets
        offsets, _ = torch.sort(offsets, dim=-2)

        # Add the offsets to the high-density points
        rays = rays_hd + rays_d.unsqueeze(2) * offsets

        rays = rays_o.unsqueeze(2).expand(-1, -1, self.config.ray_size, 3) + rays
        if return_depth:
            _, rays_depth = self.camera_model.project_to_image(rays, ids=img_id, return_depth=True)
            return {"rays": rays, "rays_depth": rays_depth[..., -1]}
        return {"rays": rays}

    def density_est_sampler(self, pts, img_id, time_step, est: "PropNetEstimator", rho_mlp, raysOD=None, det=True, return_depth=True, **kargs):
        if raysOD is None:
            rays_o, rays_d = self.camera_model.get_rays(pts, ids=img_id)
        else:
            rays_o, rays_d = raysOD
        time_stamp = (img_id + 1) * time_step
        # Re-shaping rays' parameters
        shape = rays_o.shape
        n_rays = torch.prod(torch.tensor(shape[:-1]))
        rays_o, rays_d = rays_o.reshape(n_rays, 3), rays_d.reshape(n_rays, 3)
        time_stamp = time_stamp.repeat_interleave(shape[1])[..., None, None]

        def prop_sigma_fn(t_starts, t_ends, proposal_network):
            t_origins, t_dirs = rays_o[..., None, :], rays_d[..., None, :]
            positions = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
            positions = torch.cat([positions, time_stamp.expand(-1, positions.shape[1], -1)], dim=-1)
            sigmas = proposal_network(positions)
            return sigmas.squeeze(-1)

        t_starts, t_ends, hidden_sigmas = est.sampling(
            prop_sigma_fns=[lambda *args: prop_sigma_fn(*args, p) for p in rho_mlp],
            prop_samples=self.config.samples_per_net,
            num_samples=self.config.ray_size,
            n_rays=rays_o.shape[0],
            near_plane=self.camera_model.min_depth,
            far_plane=self.camera_model.max_depth,
            sampling_type="uniform",
            stratified=not det,
            requires_grad=not det,
        )
        nr = self.config.ray_size  # + self.config.ray_size//4
        t_origins = rays_o[..., None, :]
        t_dirs = rays_d[..., None, :].repeat_interleave(nr, dim=-2)

        rays = t_origins + t_dirs * (t_starts + t_ends)[..., None] / 2.0
        rays = rays.reshape(*shape[:-1], -1, 3)
        rays_d = rays_d.reshape(*shape[:-1], 3)
        if return_depth:
            _, rays_depth = self.camera_model.project_to_image(rays, ids=img_id, return_depth=True)
            return {
                "rays": rays,
                "rays_depth": rays_depth[..., -1],
                "rays_o": rays_o,
                "rays_d": rays_d,
                "t_starts": t_starts,
                "t_ends": t_ends,
                "hidden_sigmas": hidden_sigmas,
            }

        return {"rays": rays, "rays_o": rays_o, "rays_d": rays_d, "t_starts": t_starts, "t_ends": t_ends, "hidden_sigmas": hidden_sigmas}

    def intersection_check(self, batch):
        check = ("pts1_3D" in batch.keys()) and ("tol1" in batch.keys())
        assert check, f"Intersection points are not available in the batch:{batch.keys()}"
        # Bring intersection points to the unisphere
        rays_intersection = batch["pts1_3D"]
        rays_intersection = self.contract_to_unisphere(rays_intersection)
        batch["pts1_3D"] = rays_intersection
        # scale and norm tol1
        tol1 = batch["tol1"]
        aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
        batch["tol1"] = (tol1 / (aabb_max - aabb_min)).norm(dim=-1, keepdim=True)

        return batch

    #######################################################################################
    # functions to process the output of the network
    #######################################################################################
    def get_depthrage_mask(self, depth):

        max_depth = self.camera_model.max_depth
        min_depth = self.camera_model.min_depth
        mask = (depth >= min_depth) * (depth <= max_depth)
        return mask[..., 0]

    def get_in_3D_range_mask(self, pts_2d, pts_z, max_padding=0):
        w, h = self.w, self.h
        mask = (
            (pts_2d[..., 0] >= -max_padding)
            * (pts_2d[..., 0] <= w - 1 + max_padding)
            * (pts_2d[..., 1] >= -max_padding)
            * (pts_2d[..., 1] <= h - 1 + max_padding)
        )
        mask = mask * (pts_z >= 0)[..., 0]
        return mask

    def merge_samples(self, spl_pts, weights, use_max_loc=False, detach=False):
        """Blend ray sampled 3D points using the weights
        Args:
            spl_pts (torch.Tensor): Sampled 3D points [n_imgs, n_pts, n_samples, n]
            weights (torch.Tensor): Weights [n_imgs, n_pts, n_samples]
            use_max_loc (bool): Use the max location instead of the weighted average
        Returns:
            pts (torch.Tensor): Blended 3D points [n_imgs, n_pts, n]
        """
        _weights = weights.detach().clone() if detach else weights
        if use_max_loc:
            indices = torch.max(_weights, dim=-1, keepdim=True)[1]
            pts = torch.gather(spl_pts, 2, indices[..., None].repeat(1, 1, 1, spl_pts.shape[-1])).squeeze(-2)
        else:
            pts = torch.sum(_weights.unsqueeze(-1) * spl_pts, dim=-2)
        return pts

    def get_rendering_weights(self, density):

        alpha = 1.0 - torch.exp(-density)  # [n_imgs, n_pts, n_samples]
        # T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[..., :-1]  # [n_imgs, n_pts, n_samples-1]
        # T = torch.cat((torch.ones_like(T[..., 0:1], device=self.device), T), dim=-1)  # [n_imgs, n_pts, n_samples]
        T = torch.cumsum(torch.cat([torch.zeros_like(density[..., :1]), density[..., :-1]], dim=-1), dim=-1)
        T = torch.exp(-T)  # [n_imgs, n_pts, n_samples]

        weights = T * alpha  # [n_imgs, n_pts, n_samples]
        return weights

    def contract_to_unisphere(self, x):
        aabb = self.aabb
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        if self.coord_system == "cartesian":
            x = (x - aabb_min) / (aabb_max - aabb_min)
        elif self.coord_system == "spherical":
            x = self.cartesian_to_spherical(x)
            r, theta, phi = torch.split(x, 1, dim=-1)
            # r = r/torch.norm(aabb_max - aabb_min)
            r = r / (self.config.max_depth - self.config.min_depth)
            theta = theta / (2 * torch.pi)  # range [0, 2*pi]
            phi = phi / (torch.pi / 3)  # range [0, pi/2]
            x = torch.cat([r, theta, phi], dim=-1)
        else:
            raise ValueError(f"Unknown coordinate system: {self.coord_system}")
        return x

    def expand_from_unisphere(self, x):
        aabb = self.aabb
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        if self.coord_system == "cartesian":
            x = x * (aabb_max - aabb_min) + aabb_min
        elif self.coord_system == "spherical":
            r, theta, phi = torch.split(x, 1, dim=-1)
            # r = r * torch.norm(aabb_max - aabb_min)
            r = r * (self.config.max_depth - self.config.min_depth)
            theta = theta * (2 * torch.pi)
            phi = phi * (torch.pi / 3)
            x = torch.cat([r, theta, phi], dim=-1)
            x = self.spherical_to_cartesian(x)
        else:
            raise ValueError(f"Unknown coordinate system: {self.coord_system}")
        return x

    def cartesian_to_spherical(self, x):
        """Convert Cartesian coordinates to spherical coordinates
        Args:
            x (torch.Tensor): Cartesian coordinates [n, 3]
        Returns:
            spherical (torch.Tensor): Spherical coordinates [n, 3] (r, theta, phi)
        """
        r = torch.norm(x, dim=-1, keepdim=True)
        r = torch.clamp(r, min=1e-6)  # Avoid division by zero
        theta = torch.atan2(x[..., [1]], x[..., [0]])
        theta = torch.remainder(theta, 2 * torch.pi)  # Ensure theta is in [0, 2*pi]
        phi = torch.acos(x[..., [2]] / r)
        return torch.cat([r, theta, phi], dim=-1)

    def spherical_to_cartesian(self, x):
        """Convert spherical coordinates to Cartesian coordinates
        Args:
            x (torch.Tensor): Spherical coordinates [n, 3] (r, theta, phi)
        Returns:
            cartesian (torch.Tensor): Cartesian coordinates [n, 3]
        """
        r, theta, phi = torch.split(x, 1, dim=-1)
        x_cart = r * torch.sin(phi) * torch.cos(theta)
        y_cart = r * torch.sin(phi) * torch.sin(theta)
        z_cart = r * torch.cos(phi)
        return torch.cat([x_cart, y_cart, z_cart], dim=-1)

    #######################################################################################
    # Log helper
    #######################################################################################

    def get_eval_grid_shape(self, shift=0):
        _upscale_factor = len(list(self.dataset.cache_dir.glob("upscale*.txt")))
        if shift > 0:
            print(f"Upscale factor: {_upscale_factor} (number of found files in {self.dataset.cache_dir})")
        _upscale_factor -= shift
        _scale = min(self.dataset.scale * (2**_upscale_factor), self.config.max_scale)
        _size = (int(self.h * _scale), int(self.w * _scale))
        _evaluation_grid = self.gen_grid(*_size)
        step = self.config.em_step
        # step = max(1,int(step*_scale))
        _evaluation_grid = _evaluation_grid[::step, ::step, :]
        return _evaluation_grid.shape[:-1]

    def update_error_map(self, ida: torch.Tensor, idb: torch.Tensor, error: torch.Tensor, epoch: torch.Tensor):
        """Update error map updates during validation."""
        win = self.dataset.pairs_win  # clamped to the number of frames, as the error map was built
        gap = self.config.pairs_gap
        high_pairs = torch.arange(1, win * gap + 1, gap)
        low_pairs = -torch.arange(1, win * gap + 1, gap).flip(dims=[0])
        options = torch.cat([low_pairs, high_pairs]).to(ida.device)
        options = ida[..., None] + options[None]
        _, idb = torch.where(options == idb[..., None])
        try:
            grid_shape = self.get_eval_grid_shape()
            error = error.detach().reshape(ida.shape[0], *grid_shape).squeeze()
        except Exception as e:
            print(f"Warning: adjusting error map shape: {grid_shape} != {error.shape}")
            grid_shape = self.get_eval_grid_shape(shift=1)
            error = error.detach().reshape(ida.shape[0], *grid_shape).squeeze()
        ida = ida.cpu().int()
        idb = idb.cpu().int()
        error = error.cpu().detach()
        if error.shape != self.dataset.error_map[ida, idb].shape:
            new_size = self.dataset.error_map[ida, idb].shape[1:]
            error = torch.nn.functional.interpolate(error.unsqueeze(1), size=new_size, mode="nearest").squeeze().squeeze()
            assert error.shape == self.dataset.error_map[ida, idb].shape, f"Shape mismatch: error {error.shape}, map {self.dataset.error_map[ida, idb].shape}"
        if epoch == self.config.val_freq:  # First time
            self.dataset.error_map.index_put_((ida, idb), error)
        else:
            existing = self.dataset.error_map[ida, idb]
            updated = existing * 0.3 + error * 0.7
            self.dataset.error_map.index_put_((ida, idb), updated)

    def save_error_log(self, step=0):
        error_map = self.dataset.error_map
        output_dir = get_saving_path(self.config)
        output_dir = output_dir / "cache"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"error_log_{step:06d}.txt"

        avg_error = error_map.mean(dim=(1, 2, 3))

        with open(file_path, "w") as file:
            file.write(f"Error map shape: {error_map.shape},  Step: {step}\n")
            for i in range(error_map.shape[0]):
                max_error = error_map[i].max()
                min_error = error_map[i].min()
                file.write(f"Image {i:4d}: Avg: {avg_error[i].item():0.3f}, Max:{max_error.item():0.3f}, Min:{min_error.item():0.3f}\n")
        # self.generate_video_error(step, draw_pts=True)

    def generate_video_error(self, step=0, draw_pts=False, dot_size=4):
        error_map = self.dataset.error_map
        error_video = []
        for i in range(error_map.shape[0]):
            error = error_map[i].mean(dim=0)
            error = error.numpy() if error.device == "cpu" else error.cpu().numpy()
            cmap = plt.get_cmap("YlOrRd")
            norm = plt.Normalize(vmin=0, vmax=2)
            error = cmap(norm(error))[:, :, :3]
            # error = cmap(error)[:, :, :3]
            error = (error * 255).astype("uint8")
            error = Image.fromarray(error)
            if draw_pts:
                # scale = self.dataset.get_scale()
                size = (int(self.w), int(self.h))
                error = error.resize(size)
                draw = ImageDraw.Draw(error)
                sample = self.dataset.quick_item(i)
                pts = sample["pts1"]
                # print("image size",size, "pts: x max,min", pts[:,0].max(), pts[:,0].min(), "y max,min", pts[:,1].max(), pts[:,1].min())
                # print("id:", i, "time_step", sample['ids1'])
                for pt in pts:
                    ptx, pty = int(pt[0]), int(pt[1])
                    draw.ellipse([(ptx - dot_size, pty - dot_size), (ptx + dot_size, pty + dot_size)], fill="blue")
            error = error.resize((self.w // 1, self.h // 1))
            error_video.append(error)
        output_dir = get_saving_path(self.config)
        output_dir = output_dir / "cache" / f"error_video_{step:06d}.mp4"
        imageio.mimsave(output_dir, error_video, quality=8, fps=10)

    #######################################################################################
    # samplers for the image plane
    #######################################################################################

    def sample_pts_on_image(self, mask=None, reverse_mask=False, interval=10):
        """Sample points on the image plane
        Args:
            mask (torch.Tensor): Mask of the image [h, w]
            reverse_mask (bool): Reverse the mask
            interval (int): Interval between points
        Returns:
            coords (torch.Tensor): Pixel coordinates of the sampled points
        """
        grid = self.gen_grid(self.h, self.w, homogeneous=True)
        if mask is not None:
            if reverse_mask:
                mask = ~mask
            kernel = torch.ones(7, 7, device=self.device)
            mask = morph.erosion(mask.float()[None, None], kernel).bool().squeeze()  # Erosion
        else:
            mask = torch.ones_like(grid[..., 0], dtype=torch.bool)
        coords = grid[::interval, ::interval, :2][mask[::interval, ::interval]]

        return coords

    def gen_grid(self, h=None, w=None, normalize=False, homogeneous=False):
        """Generate a grid with the position of each pixel
        Args:
            h (int): Height of the grid
            w (int): Width of the grid
            normalize (bool): Normalize the grid between -1 and 1
            homogeneous (bool): Add a homogeneous coordinate to the grid
        Returns:
            grid (torch.Tensor): Grid with the position of each pixel [h,w,2 or 3]"""
        if h is None:
            h = self.h
        if w is None:
            w = self.w
        if normalize:
            lin_y = torch.linspace(-1.0, 1.0, steps=h, device=self.device)
            lin_x = torch.linspace(-1.0, 1.0, steps=w, device=self.device)
        else:
            lin_y = torch.arange(0, h, device=self.device)
            lin_x = torch.arange(0, w, device=self.device)
        grid_y, grid_x = torch.meshgrid((lin_y, lin_x), indexing="ij")  # Addressing torch warning regarding indexing
        grid = torch.stack((grid_x, grid_y), -1)
        if homogeneous:
            grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
        return grid  # [h, w, 2 or 3]

    def set_device(self, device):
        self.device = device
