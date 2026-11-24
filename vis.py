import matplotlib.pyplot as plt
import numpy as np, cv2
import torch, os, imageio, util, io, time
from tqdm import tqdm
from pathlib import Path

# Data types
from typing import List, Tuple, Dict
from torch import Tensor
from engine import BaseTrainer

#########################################################
# region Dataset vis

from PIL import Image, ImageDraw
from DataManager.raft import RAFTDataset


def see_batch(batch: Dict[str, Tensor], dataset: RAFTDataset):
    """Visualise the dataset batch"""
    # Display batch info
    print("Batch keys: ", batch.keys())
    print("Batch sizes: ", {k: v.shape for k, v in batch.items()})
    print("Batch d_types: ", {k: v.dtype for k, v in batch.items()})
    print("ids1: ", batch["ids1"])
    print("ids2: ", batch["ids2"])
    # Load images and draw matches
    img_pairs = []
    for i, (id1, id2) in enumerate(zip(batch["ids1"], batch["ids2"])):
        img1, img2 = dataset.get_image(id1.int(), np=True) * 255, dataset.get_image(id2.int(), np=True) * 255
        img1, img2 = img1.astype(np.uint8), img2.astype(np.uint8)
        img1, img2 = Image.fromarray(img1), Image.fromarray(img2)
        matchs_colours = create_palette(batch["pts1"][i])
        img1n2 = draw_matches_pil(img1, img2, batch["pts1"][i], batch["pts2"][i], matchs_colours)
        img_pairs.append(img1n2)
    # Show the images
    num_images = len(img_pairs)
    num_columns = min(num_images, 4)
    num_rows = (num_images + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(40, 5 * num_rows))
    axes = axes.flatten()

    for i in range(num_columns * num_rows):
        if i < num_images:
            img = img_pairs[i]
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"id 1: {batch['ids1'][i].int()}, id 2: {batch['ids2'][i].int()}")
        else:
            axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def see_stereo_pair(dataset: RAFTDataset, idx: int, ptsL: Tensor = None):
    # Load the sample and its stereo pair (left and right cameras)
    scale = 1  # dataset.scale
    h, w = int(dataset.h * scale), int(dataset.w * scale)
    ptsL, ptsR = dataset.get_stereo_corr(idx, ptsL, scale=scale)
    ptsL, ptsR = ptsL * scale, ptsR * scale
    imgL, imgR = dataset.get_image(idx, np=True) * 255, dataset.get_image(idx, side="right", np=True) * 255
    imgL, imgR = imgL.astype(np.uint8), imgR.astype(np.uint8)
    imgL, imgR = Image.fromarray(imgL), Image.fromarray(imgR)
    imgL, imgR = imgL.resize((w, h)), imgR.resize((w, h))
    print("ImageL: ", imgL.size, "ImageR: ", imgR.size)
    # Create a draw object
    colours = create_palette(ptsL)
    combined_img = draw_matches_pil(imgL, imgR, ptsL, ptsR, colours)

    return combined_img


def see_sample(dataset: RAFTDataset, idx: int, add_mask=False, query_pts=None):
    """Visualise the dataset samples"""
    # Load the sample
    if isinstance(idx, int):
        sample = dataset[idx]
    elif isinstance(idx, tuple):
        sample = dataset.get_correspondences(idx[0], idx[1], query_pts=query_pts)
    id1, pts1, gt_rgb1 = sample["ids1"], sample["pts1"], sample["gt_rgb1"]
    id2, pts2, gt_rgb2 = sample["ids2"], sample["pts2"], sample["gt_rgb2"]
    ro1, rd1, ro2, rd2 = sample["ro1"], sample["rd1"], sample["ro2"], sample["rd1"]
    c_mask = sample["c_mask"]
    occ_mask = sample["occ_mask"]
    # weights = sample['weights']
    print(f"Id1 info: {id1}, pts1: {pts1.shape}, gt_rgb1: {gt_rgb1.shape}, ro1: {ro1.shape}, rd1: {rd1.shape}")
    print(f"Id2 info: {id2}, pts2: {pts2.shape}, gt_rgb2: {gt_rgb2.shape}, ro2: {ro2.shape}, rd2: {rd2.shape}")
    print(f"c_mask: {c_mask.shape}, occ_mask: {occ_mask.shape}")
    if "masks" in sample:
        print(f"Masks: {sample['masks'].shape}")
    # Load the images
    img1, img2 = dataset.get_image(id1.int(), np=True) * 255, dataset.get_image(id2.int(), np=True) * 255
    img1, img2 = img1.astype(np.uint8), img2.astype(np.uint8)
    img1, img2 = Image.fromarray(img1), Image.fromarray(img2)
    print("Image1: ", img1.size, "Image2: ", img2.size)
    # Create a draw object
    colours = create_palette(pts1)
    combined_img = draw_matches_pil(img1, img2, pts1, pts2, colours, mask=c_mask)
    if add_mask:
        masks, occ_mask = dataset.get_consistency_mask(id1.int(), id2.int())
        print("Masks: ", masks.shape)
        mask = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
        mask[..., 1] = masks.numpy().astype(np.uint8) * 255
        mask[..., 0] = occ_mask.numpy().astype(np.uint8) * 255
        combined_img = Image.fromarray(np.concatenate((mask, np.array(combined_img)), axis=1))

    return combined_img, sample


def see_point_cloud(pts1_3D, pts2_3D, mask, **kw):
    fig = plt.figure(figsize=(18, 6))
    # First subplot: pts1_3D
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.scatter(pts1_3D[0, :, 0].cpu(), pts1_3D[0, :, 1].cpu(), pts1_3D[0, :, 2].cpu(), c="b", label="pts1_3D", s=5)
    ax1.scatter(pts2_3D[0, :, 0].cpu(), pts2_3D[0, :, 1].cpu(), pts2_3D[0, :, 2].cpu(), c="y", label="pts2_3D", s=5)
    ax1.scatter(
        pts2_3D[~mask][:, 0].cpu(),
        pts2_3D[~mask][:, 1].cpu(),
        pts2_3D[~mask][:, 2].cpu(),
        edgecolors="r",
        facecolors="none",
        marker="o",
        s=10,
        label="Invalid_point",
    )
    ax1.set_title("Preprocessed point cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()


def see_pred_and_target(prediction, target, config, e_mask=None, mask_pts=False):
    print(
        f"Operation mode:\n Render before deformation: {config.merge_before_deform}\n",
        f"Density predicted over canonical space: {config.density_on_canonical}\n",
        f"Deforming model: {config.invertible_model}, Inverible function: {config.plane_inv_f}\n",
        f"Color model: {config.color_model}, Density Model: {config.density_model}\n",
        f"Sampling mechanism: {config.estimator}\n",
    )
    fig = plt.figure(figsize=(18, 10))
    mask = target["c_mask"].cpu().numpy()[0]
    if e_mask is not None:
        mask = mask & e_mask.cpu().numpy()[0]
    print(f"Number of valid points: {mask.sum()}/{mask.size}")
    pts1_3D = target["pts1_3D"].cpu()[0]
    pts2_3D = target["pts2_3D"].cpu()[0]
    p_pts1_3D = prediction["p_3Dpts1"].cpu()[0] if "p_3Dpts1" in prediction else None
    p_pts2_3D = prediction["p_3Dpts2"].cpu()[0] if "p_3Dpts2" in prediction else None
    canon_pts1 = prediction["canon_pts1"].cpu()[0]
    canon_pts2 = prediction["canon_pts2"].cpu()[0]
    p_3Dpts1to2 = prediction["p_3Dpts1to2"].cpu()[0]
    p_3Dpts2to1 = prediction["p_3Dpts2to1"].cpu()[0]
    # if True:
    #     pts1_3D = cartesian_to_spherical(pts1_3D, cfg=config)
    #     p_pts1_3D = cartesian_to_spherical(p_pts1_3D, cfg=config)
    #     p_pts2_3D = cartesian_to_spherical(p_pts2_3D, cfg=config)
    #     p_3Dpts1to2 = cartesian_to_spherical(p_3Dpts1to2, cfg=config)
    #     canon_pts1 = spherical_to_cartesian(canon_pts1, cfg=config)
    #     canon_pts2 = spherical_to_cartesian(canon_pts2, cfg=config)
    if mask_pts:
        pts1_3D = pts1_3D[mask]
        pts2_3D = pts2_3D[mask]
        if p_pts1_3D is not None:
            p_pts1_3D = p_pts1_3D[mask]
        if p_pts2_3D is not None:
            p_pts2_3D = p_pts2_3D[mask]
        canon_pts1 = canon_pts1[mask]
        canon_pts2 = canon_pts2[mask]
        p_3Dpts1to2 = p_3Dpts1to2[mask]
        p_3Dpts2to1 = p_3Dpts2to1[mask]
        mask = mask[mask]  # Update mask to reflect the filtered points

    col1a = "darkturquoise"  # target 1
    col1b = "b"  # prediction 1>2
    col2a = "yellowgreen"  # target 2
    col2b = "darkgreen"  # prediction 2>1
    # First row: preprocessed 3D points1 vs pred 3D points1 | canonical space pts1 | processed 3D points2 vs pred 3D points2 (from 1 to 2)
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    ax1.scatter(pts1_3D[:, 0], pts1_3D[:, 1], pts1_3D[:, 2], c=col1a, label="p1", s=5)
    if p_pts1_3D is not None:
        ax1.scatter(p_pts1_3D[:, 0], p_pts1_3D[:, 1], p_pts1_3D[:, 2], c=col1b, label="^p1", s=5)
        for i in range(pts1_3D.shape[0]):
            ax1.plot(
                [pts1_3D[i, 0], p_pts1_3D[i, 0]], [pts1_3D[i, 1], p_pts1_3D[i, 1]], [pts1_3D[i, 2], p_pts1_3D[i, 2]], c="r" if mask[i] else "k", linewidth=1
            )
    ax1.set_title("3D Points1")
    ax1.set_xlabel("X/r")
    ax1.set_ylabel("Y/theta")
    ax1.set_zlabel("Z/phi")
    ax1.legend()
    #
    ax1b = fig.add_subplot(2, 3, 4, projection="3d")
    ax1b.scatter(p_3Dpts2to1[:, 0], p_3Dpts2to1[:, 1], p_3Dpts2to1[:, 2], c=col2b, label="^p(2-1)", s=5)
    if p_pts1_3D is not None:
        ax1b.scatter(p_pts1_3D[:, 0], p_pts1_3D[:, 1], p_pts1_3D[:, 2], c=col1b, label="^p_1", s=5)
        for i in range(p_pts1_3D.shape[0]):
            ax1b.plot(
                [p_pts1_3D[i, 0], p_3Dpts2to1[i, 0]],
                [p_pts1_3D[i, 1], p_3Dpts2to1[i, 1]],
                [p_pts1_3D[i, 2], p_3Dpts2to1[i, 2]],
                c="r" if mask[i] else "k",
                linewidth=1,
            )
    ax1b.set_title("3D Points 2 to 1")
    ax1b.set_xlabel("X")
    ax1b.set_ylabel("Y")
    ax1b.set_zlabel("Z")
    ax1b.legend()

    #
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(canon_pts1[..., 0], canon_pts1[..., 1], canon_pts1[..., 2], c=col1b, label="cp1", s=5)
    ax2.scatter(canon_pts2[..., 0], canon_pts2[..., 1], canon_pts2[..., 2], c=col2b, label="cp2", s=5)
    for i in range(canon_pts1.shape[0]):
        ax2.plot([canon_pts1[i, 0], canon_pts2[i, 0]], [canon_pts1[i, 1], canon_pts2[i, 1]], [canon_pts1[i, 2], canon_pts2[i, 2]], c="r" if mask[i] else "k")
    ax2.set_title("Canonical Space")
    ax2.set_xlabel("X/r")
    ax2.set_ylabel("Y/theta")
    ax2.set_zlabel("Z/phi")
    ax2.legend()
    # ax2.set_xlim(0, 1);    ax2.set_ylim(0, 1);    ax2.set_zlim(0, 1)

    # Third subplot: pts2_3D (blue), p_3Dpts2 (green), red lines between pairs
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    ax3.scatter(p_3Dpts1to2[:, 0].cpu(), p_3Dpts1to2[:, 1].cpu(), p_3Dpts1to2[:, 2].cpu(), c="b", label="^p(1-2)", s=5)
    if p_pts2_3D is not None:
        ax3.scatter(p_pts2_3D[:, 0], p_pts2_3D[:, 1], p_pts2_3D[:, 2], c=col2b, label="^p2", s=5)
        for i in range(p_pts2_3D.shape[0]):
            ax3.plot(
                [p_pts2_3D[i, 0], p_3Dpts1to2[i, 0]],
                [p_pts2_3D[i, 1], p_3Dpts1to2[i, 1]],
                [p_pts2_3D[i, 2], p_3Dpts1to2[i, 2]],
                c="r" if mask[i] else "k",
                linewidth=1,
            )
    ax3.set_title("3D Points 1 to 2")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend()
    #
    ax3b = fig.add_subplot(2, 3, 6, projection="3d")
    ax3b.scatter(pts2_3D[:, 0], pts2_3D[:, 1], pts2_3D[:, 2], c=col2a, label="p2", s=5)
    if p_pts2_3D is not None:
        ax3b.scatter(p_pts2_3D[:, 0], p_pts2_3D[:, 1], p_pts2_3D[:, 2], c=col2b, label="^p2", s=5)
        for i in range(pts2_3D.shape[0]):
            ax3b.plot(
                [pts2_3D[i, 0], p_pts2_3D[i, 0]], [pts2_3D[i, 1], p_pts2_3D[i, 1]], [pts2_3D[i, 2], p_pts2_3D[i, 2]], c="r" if mask[i] else "k", linewidth=1
            )
    ax3b.set_title("3D Points 2")
    ax3b.set_xlabel("X")
    ax3b.set_ylabel("Y")
    ax3b.set_zlabel("Z")
    ax3b.legend()

    plt.tight_layout()
    plt.show()


def spherical_to_cartesian(x, scale=True, cfg=None):
    """Convert spherical coordinates to Cartesian coordinates
    Args:
        x (torch.Tensor): Spherical coordinates [n, 3] (r, theta, phi)
    Returns:
        cartesian (torch.Tensor): Cartesian coordinates [n, 3]
    """
    r, theta, phi = torch.split(x, 1, dim=-1)
    # if scale:
    #     aabb = torch.tensor(cfg.aabb, dtype=torch.float32)
    #     aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    #     r = r * torch.norm(aabb_max - aabb_min)
    #     theta = theta * (2 * torch.pi)
    #     phi = phi * (torch.pi/3)
    theta = theta * (2 * torch.pi)
    phi = phi * (2 * torch.pi)
    x_cart = r * torch.sin(phi) * torch.cos(theta)
    y_cart = r * torch.sin(phi) * torch.sin(theta)
    z_cart = r * torch.cos(phi)
    return torch.cat([x_cart, y_cart, z_cart], dim=-1)


def cartesian_to_spherical(x, scale=True, cfg=None):
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
    if scale:
        aabb = torch.tensor(cfg.aabb, dtype=torch.float32)
        # aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        # r = r/torch.norm(aabb_max - aabb_min)
        r = r / (cfg.max_depth - cfg.min_depth)
        theta = theta / (2 * torch.pi)  # range [0, 2*pi]
        phi = phi / (torch.pi / 3)  # range [0, pi/4]
    return torch.cat([r, theta, phi], dim=-1)


def see_pred_and_target_2D(prediction, target, mask_pts=False):
    pts1_np = target["pts1"][0].cpu().numpy()
    pts2_np = target["pts2"][0].cpu().numpy()
    p_pts2_np = prediction["p_pts2"][0].cpu().numpy()
    mask_np = (target["c_mask"] | target["occ_mask"])[0].cpu().numpy()
    mask_occ = target["occ_mask"][0].cpu().numpy()
    if mask_pts:
        pts2_np[~mask_np] = np.nan
        pts1_np[~mask_np] = np.nan
        p_pts2_np[~mask_np] = np.nan

    plt.figure(figsize=(8, 8))
    plt.scatter(pts1_np[:, 0], pts1_np[:, 1], c="b", label="p1", s=40)
    plt.scatter(pts2_np[:, 0], pts2_np[:, 1], c="y", label="p2", s=40)
    if not mask_pts:
        plt.scatter(pts2_np[~mask_np, 0], pts2_np[~mask_np, 1], edgecolors="r", facecolors="none", marker="o", s=45, label="Invalid")
    plt.scatter(pts2_np[mask_occ, 0], pts2_np[mask_occ, 1], edgecolors="k", facecolors="none", marker="o", s=90, label="Occluded")
    plt.scatter(p_pts2_np[:, 0], p_pts2_np[:, 1], c="g", marker="x", label="^p2", s=40)
    for i in range(pts2_np.shape[0]):
        plt.plot([pts2_np[i, 0], p_pts2_np[i, 0]], [pts2_np[i, 1], p_pts2_np[i, 1]], "k--", linewidth=1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Projection: pts2 (GT) vs p_pts2 (Pred)")
    plt.legend(loc="lower right")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# endregion
#########################################################

#########################################################
# region Tools


def draw_matches_pil(img1, img2, kp1, kp2, colours, mask=None, draw_lines=False):
    """
    Draw matches between two images using PIL package.
    Args:
        img1 (PIL.Image): First image
        img2 (PIL.Image): Second image
        kp1 (list): List of keypoints in the first image
        kp2 (list): List of keypoints in the second image
        colours (list): List of colours for each keypoint
    """
    if mask is None:
        mask = torch.ones_like(kp1[:, 0], dtype=torch.bool)
    # Create a new image with both images side by side
    width = img1.width + img2.width
    height = max(img1.height, img2.height)
    combined_img = Image.new("RGB", (width, height))
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    # Create a draw object
    draw = ImageDraw.Draw(combined_img)

    # Draw lines between matching points
    pt_size = 3
    for idx, (pt1, pt2, colour) in enumerate(zip(kp1, kp2, colours)):
        colour = tuple(int(255 * c) for c in colour) if mask[idx] else (0, 0, 0)
        pt1_x, pt1_y = int(pt1[0]), int(pt1[1])
        pt2_x, pt2_y = int(pt2[0] + img1.width), int(pt2[1])
        if draw_lines:
            draw.line([(pt1_x, pt1_y), (pt2_x, pt2_y)], fill="black", width=1)
        in_scope = (0 <= pt1_x < img1.width) and (0 <= pt1_y < img1.height)
        if in_scope:
            draw.ellipse([(pt1_x - pt_size, pt1_y - pt_size), (pt1_x + pt_size, pt1_y + pt_size)], fill=colour, width=3)
        in_scope = (img1.width <= pt2_x < img2.width + img1.width) and (0 <= pt2_y < img2.height)
        if in_scope:
            draw.ellipse([(pt2_x - pt_size, pt2_y - pt_size), (pt2_x + pt_size, pt2_y + pt_size)], fill=colour, width=3)

    return combined_img


def create_palette(points):
    set_max = range(128)
    colors = {m: i for i, m in enumerate(set_max)}
    colors = {m: (255 * np.array(plt.cm.hsv(i / float(len(colors))))[:3][::-1]).astype(np.int32) for m, i in colors.items()}
    center = np.median(points, axis=0)
    selected_colors = []
    for i, pt in enumerate(points):
        coord_angle = np.arctan2(pt[1] - center[1], pt[0] - center[0])
        corr_color = np.int32(64 * coord_angle / np.pi) % 128
        color = tuple((colors[corr_color] / 255).tolist())
        selected_colors.append(color)
    return selected_colors


# endregion
#########################################################

##########################################################
# region Model output vis

from DataManager.ray_manager import RayManager
from solver.criteria import weight_scheduler


def get_gt_depth_video(data_Handler: RayManager, grid_step=1, colorise=True, out_dir=None):
    """Generates a video of ground truth depth maps from the dataset.
    Args:
        dataset (RAFTDataset): Dataset object
    """
    dataset = data_Handler.dataset
    vid_depth = []
    valid_masks = []
    for img in dataset.img_names:
        file = dataset.info_3D_dir / f"{img}.npz"
        map3D, _, mask = dataset.load_3D_info(file)
        # print(f"Loading depth map from: {file}, shape: {map3D.shape}, mask shape: {mask.shape}")
        map3D = map3D.squeeze(0).permute(1, 2, 0)  # Change axes from [c, h, w] to [h, w, c]
        mask = mask.squeeze(0).squeeze(0)
        vid_depth.append(map3D[..., -1])
        valid_masks.append(mask)

    vid_depth = torch.stack(vid_depth, dim=0)
    valid_masks = torch.stack(valid_masks, dim=0)
    print(f" - Depth range: {vid_depth.min():.2f} - {vid_depth.max():.2f}, valid masks shape: {valid_masks.shape}")

    vid_depth = (vid_depth - vid_depth.min()) / (vid_depth.max() - vid_depth.min())
    vid_depth = vid_depth.cpu().numpy() if vid_depth.is_cuda else vid_depth.numpy()
    valid_masks = valid_masks.cpu().numpy() if valid_masks.is_cuda else valid_masks.numpy()
    print("-> Depth maps generated. ", vid_depth.shape)

    if colorise:
        print("Colorising depth maps...")
        colored_depth = []
        for i, frame in enumerate(vid_depth):
            cmap = plt.get_cmap("jet")
            norm = plt.Normalize(vmin=vid_depth.min(), vmax=vid_depth.max())
            colored_depth_map = cmap(norm(frame))[:, :, :3]
            colored_depth_map = (colored_depth_map * 255).astype(np.uint8)
            colored_depth_map = np.where(valid_masks[i][..., None], colored_depth_map, 0)
            # print("Number of invalid pixels in frame ", i, ": ", (~valid_masks[i]).sum())
            pil_image = Image.fromarray(colored_depth_map)
            colored_depth.append(pil_image)
        vid_depth = colored_depth
    else:
        vid_depth = (vid_depth * 255).astype(np.uint8)
    mode = dataset.config.dataset_type
    out_vid = os.path.join(str(out_dir), "vid", f"gt_{mode}_depth.mp4")
    print("-> Saving depth video to: ", out_vid, "...")
    imageio.mimsave(out_vid, vid_depth, quality=8, fps=20)


def get_rgb_and_depth_video(
    framework: BaseTrainer, grid_step=1, colorise=True, out_dir=None, progress_bar=False, return_depth=False, save_frames=False, camera_side="left"
):
    """Generates a video of depth maps and rendered images from the dataset.
    Args:
        framework (BaseTrainer): Trainer object
    """
    min_depth = framework.config.min_depth
    max_depth = framework.config.max_depth
    chkpt_n = os.path.basename(framework.config.ckpt_path).split(".")[0].split("_")[-1]
    chkpt_n = chkpt_n if len(chkpt_n) > 0 else f"{framework.epoch:06d}"
    with torch.no_grad():  # Run the model
        num_imgs = framework.Lbl_mgr.num_imgs - 1
        grid_pts = framework.Lbl_mgr.gen_grid(normalize=False).float()
        grid_pts = grid_pts[::grid_step, ::grid_step]  # downsample
        _grid_shape = grid_pts.shape[:2]
        grid_pts = grid_pts.reshape(-1, 2)  # flat -> [n_samples, 2]
        vid_depth = []
        vid_img = []
        print("Generating depth maps and rendered images for video...")
        if progress_bar:
            pbar = tqdm(total=num_imgs)
        starting_time = time.time()
        for id1 in torch.arange(num_imgs, device=framework.Lbl_mgr.device):
            depthmap, rgb_img = framework.get_depthmap_and_rgb(id1, grid_pts, camera_side=camera_side)
            depthmap = depthmap.reshape(*_grid_shape, 1)
            vid_depth.append(depthmap)
            if rgb_img is not None:
                rgb_img = rgb_img.reshape(*_grid_shape, 3)
                vid_img.append(rgb_img)
            if progress_bar:
                pbar.update(1)
        if progress_bar:
            pbar.close()
        end_time = time.time()
        print(
            f"-> Depth maps and rendered images generated in {end_time - starting_time:.2f} seconds, {num_imgs} images processed, frame rate: {num_imgs/(end_time - starting_time):.2f} fps."
        )
        vid_depth = torch.stack(vid_depth, dim=0)
        vid_img = torch.stack(vid_img, dim=0) if len(vid_img) > 0 else None
        print("-> Depth maps and rendered images generated. Image resolution: ", _grid_shape)
    if return_depth:
        return vid_depth, vid_img
    if grid_step > 1:
        _depth = vid_depth.permute(0, 3, 1, 2)
        _depth = torch.nn.functional.interpolate(_depth, size=(framework.Lbl_mgr.h, framework.Lbl_mgr.w), mode="bilinear", align_corners=True)
        vid_depth = _depth.permute(0, 2, 3, 1)
        if vid_img is not None:
            _rgb = vid_img.permute(0, 3, 1, 2)  # Change axes from [n, w, h, c] to [n, c, w, h]
            _rgb = torch.nn.functional.interpolate(_rgb, size=(framework.Lbl_mgr.h, framework.Lbl_mgr.w), mode="bilinear", align_corners=True)
            vid_img = _rgb.permute(0, 2, 3, 1)  # Change axes back to [n, w, h, c]
        print("-> Upscaled RGB and depth maps:")

    vid_depth = vid_depth.cpu().numpy() if vid_depth.is_cuda else vid_depth.numpy()
    if colorise:
        print("Colorising depth maps...")
        colored_depth = []
        for frame in vid_depth:
            frame = frame.squeeze(-1)
            cmap = plt.get_cmap("jet")
            norm = plt.Normalize(vmin=vid_depth.min(), vmax=vid_depth.max())
            frame = norm(frame)
            colored_depth_map = cmap(frame)[:, :, :3]
            colored_depth_map = (colored_depth_map * 255).astype(np.uint8)
            pil_image = Image.fromarray(colored_depth_map)
            colored_depth.append(pil_image)
        vid_depth = colored_depth
    else:
        vid_depth = (vid_depth * 255).astype(np.uint8)
    out_vid = os.path.join(str(out_dir), "vid", f"depth_{chkpt_n}.mp4")
    print(f"-> Saving depth video to: ", out_vid, "...")
    imageio.mimsave(out_vid, vid_depth, quality=8, fps=20)

    if vid_img is not None:
        vid_img = vid_img.cpu().numpy() if vid_img.is_cuda else vid_img.numpy()
        vid_img = (vid_img * 255).astype(np.uint8)
        out_vid = os.path.join(str(out_dir), "vid", f"rend_{chkpt_n}.mp4")
        print(f"-> Saving rgb video to: ", out_vid, "...")
        imageio.mimsave(out_vid, vid_img, quality=8, fps=20)
    else:
        print("-> No rendered images to save.")

    # save frames too
    if save_frames:
        out_vid = os.path.join(str(out_dir), "vid", f"depth_{chkpt_n}")
        out_vid2 = os.path.join(str(out_dir), "vid", f"rend_{camera_side}_{chkpt_n}")
        if not os.path.exists(out_vid):
            os.makedirs(out_vid)
        if not os.path.exists(out_vid2):
            os.makedirs(out_vid2)
        for i, frame in enumerate(vid_depth):
            if i % 10 != 0:
                continue  # Save every 10th frame
            frame.save(os.path.join(out_vid, f"depth_{i:04d}.png"))
            if vid_img is not None:
                Image.fromarray(vid_img[i]).save(os.path.join(out_vid2, f"{i:04d}.png"))


def get_video_correspondences(
    framework: BaseTrainer,
    query_id,
    dataset,
    query_pts=None,
    interval=10,
    radius=2,
    out_dir=None,
    save_first_last=False,
    progress_bar=False,
    save_frames=False,
    target_pts=None,
):
    """Generates a video of point correspondences between the query image and
        all other images in the dataset.
    Args:
        framework (BaseTrainer): Trainer object
        query_id (int): Query image id
        dataset (Dataset): Dataset object
        interval (int): Interval between sampled points
        radius (int): Radius of the circle to draw around each point
        target_pts (Tensor, optional): [N, 2] reference positions in the last image, in the
            same pixel coordinates as query_pts. Drawn on the last frame only, as hollow
            white squares, so the predicted end points can be compared against them.
    """
    handler = framework.Lbl_mgr
    chkpt_n = os.path.basename(framework.config.ckpt_path).split(".")[0].split("_")[-1]
    chkpt_n = chkpt_n if len(chkpt_n) > 0 else f"{framework.epoch:06d}"
    fmp = handler.config.foreground_mask_path
    print("Generating video correspondences for query id: ", query_id)
    with torch.no_grad():
        if len(fmp) > 0 and os.path.exists(fmp):
            foreground_mask = handler.load_foreground_mask(fmp)
            print("Using foreground mask: ", fmp)
        else:
            foreground_mask = None
        query_id = torch.tensor(query_id, device=handler.device)
        if query_pts is None:
            query_pts = handler.sample_pts_on_image(interval=interval, mask=foreground_mask)
        else:
            chkpt_n = f"{chkpt_n}_custom"
        num_pts = query_pts.shape[0]
        query_img = dataset.get_image(query_id, np=True)
        frames = []
        if progress_bar:
            pbar = tqdm(total=handler.num_imgs)
        for target_id in torch.arange(handler.num_imgs, device=handler.device):
            pred_pts, occ_mask, VF_mask = framework.get_corr_pair(id1=query_id, id2=target_id, pts=query_pts)
            VF_mask = VF_mask.cpu().numpy()  # Visual field mask
            img_i = dataset.get_image(target_id, np=True)
            image = util.drawMatches(
                query_img, img_i, query_pts.cpu().numpy(), pred_pts.cpu().numpy(), t_VF=VF_mask, num_vis=num_pts, mask=occ_mask, radius=radius
            )
            if target_pts is not None and target_id == handler.num_imgs - 1:
                # The target image is the right half of the side-by-side frame
                x_shift = query_img.shape[1]
                for pt in torch.as_tensor(target_pts).cpu().numpy():
                    image = cv2.drawMarker(
                        image,
                        (int(round(pt[0])) + x_shift, int(round(pt[1]))),
                        (255, 255, 255),
                        markerType=cv2.MARKER_SQUARE,
                        markerSize=int(6 * radius),
                        thickness=2,
                        line_type=cv2.LINE_AA,
                    )
            frames.append(image)
            if progress_bar:
                pbar.update(1)
        if progress_bar:
            pbar.close()
        out_vid = os.path.join(str(out_dir), "vid", f"corr_{chkpt_n}.mp4")
        print("-> Saving video to: ", out_vid)
        imageio.mimsave(out_vid, frames, quality=8, fps=20)
        # Save the first and last frame as PNG images
        if save_first_last:
            final_frame = Image.fromarray(frames[-1])
            final_frame.save(os.path.join(str(out_dir), "init_final.png"))
        if save_frames:
            out_vid = os.path.join(str(out_dir), "vid", f"corr_{chkpt_n}")
            if not os.path.exists(out_vid):
                os.makedirs(out_vid)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(out_vid, f"corr_{i:04d}.png"))


def get_density_progression_video(framework: BaseTrainer, grid_step=50, out_dir=None, fps=20):
    chkpt_n = os.path.basename(framework.config.ckpt_path).split(".")[0].split("_")[-1]
    chkpt_n = chkpt_n if len(chkpt_n) > 0 else f"{framework.epoch:06d}"
    out_vid = os.path.join(str(out_dir), "vid", f"density_{chkpt_n}.mp4")
    std_weight = weight_scheduler(int(chkpt_n), 0, -1e-4, 0.25, 1)
    ray_std = framework.config.ray_std * std_weight

    with torch.no_grad():  # Run model
        num_imgs = framework.Lbl_mgr.num_imgs - 1
        grid_pts = framework.Lbl_mgr.gen_grid(normalize=False).float()
        grid_pts = grid_pts[::grid_step, ::grid_step]  # downsample
        grid_pts = grid_pts.reshape(-1, 2)  # flat -> [n_samples, 2]

        depths, weights, gt_pts = [], [], []
        for id1 in torch.arange(num_imgs, device=framework.Lbl_mgr.device):
            _depths, _weights, _ro = framework.get_density_progression(id1, grid_pts)
            depths.append(_depths)
            weights.append(_weights)
            _gt_pts = framework.Lbl_mgr.dataset.get_single_3Dpts(id1, grid_pts)["pts1_3D"]
            gt_pts.append(_gt_pts - _ro[0])
        gt_pts = torch.stack(gt_pts, dim=0)  # [n, n_p, 3]
        gt_depth = torch.linalg.norm(gt_pts, dim=-1)  # [n, n_p]

    frames = []
    n = gt_depth.shape[0]
    n_p = gt_depth.shape[1]
    for t in range(n):  # Plotting
        # Plot 2D depth vs density
        fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

        gt_z = gt_depth[t][..., None]  # [n_p]
        z = torch.linspace(0, framework.config.max_depth, steps=200, device=gt_z.device).unsqueeze(0).expand(n_p, 200)  # [n_p, 200]
        gt_sig = torch.exp(-0.5 * ((z - gt_z) / ray_std) ** 2) / torch.sqrt(2 * torch.pi * ray_std**2)  # [n, 200]
        gt_alpha = 1 - torch.exp(-gt_sig)
        gt_trans = torch.cumsum(torch.cat([torch.zeros_like(gt_sig[..., :1]), gt_sig[..., :-1]], dim=-1), dim=-1)
        gt_trans = torch.exp(-gt_trans)
        gt_w = gt_alpha * gt_trans
        # Plot 2D depth vs density
        fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
        axes[0].set_title(f"Step {t+1}/{n} - sig: {ray_std:.3f}")
        for i, ax in enumerate(axes):
            i = i * (n_p // 5) - 1
            if i >= n_p:  # Skip if there are fewer rays than subplots
                break
            # ideal transmitance, weight and depth
            ax.plot(z[i].cpu().numpy(), gt_w[i].cpu().numpy(), label="GT W", color="g")
            ax.plot(z[i].cpu().numpy(), gt_trans[i].cpu().numpy(), label="GT Trans", color="b")
            ax.axvline(x=gt_z[i].cpu().numpy(), color="red", linestyle="-", label="GT Z")

            # predicted transmitance, weight and depth
            colors = ["c", "m", "y"]
            for net, (_z, _trans) in enumerate(zip(depths[t], weights[t])):
                _z = _z.cpu().numpy()[i]
                _trans = _trans.cpu().numpy()[i]
                ax.plot(_z, _trans, linestyle="--", label=f"Net-{net+1} T", color=colors[net])

            pred_pts = framework.Lbl_mgr.merge_samples(depths[t][-1][..., None], weights[t][-1], use_max_loc=False)  # [n, n_p, 1]
            pred_z = pred_pts[i, 0].cpu().numpy()
            ax.axvline(x=pred_z, color="red", linestyle="--", label="Pred Z")
            ax.set_ylabel(f"Ray {i+1}:x_{grid_pts[i,0].int().item()},y_{grid_pts[i,1].int().item()}")
            ax.set_ylim(0, 1)
            ax.legend()

        axes[-1].set_xlabel("Z Coordinate")
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))

    print("Frame size: ", frames[-1].shape)
    # Save video
    out_vid = os.path.join(str(out_dir), "vid", f"density_{chkpt_n}.mp4")
    print("-> Saving density video to: ", out_vid, "...")
    imageio.mimsave(out_vid, frames, quality=8, fps=fps)


def get_canonical_space_video(framework: BaseTrainer, grid_step=10, out_dir=None, progress_bar=False):
    """Generates a video of canonical space points from the dataset.
    Args:
        framework (BaseTrainer): Trainer object
        grid_step (int): Step size for the grid
        out_dir (str): Output directory for the video
    """
    chkpt_n = os.path.basename(framework.config.ckpt_path).split(".")[0].split("_")[-1]
    chkpt_n = chkpt_n if len(chkpt_n) > 0 else f"{framework.epoch:06d}"
    with torch.no_grad():  # Run model
        num_imgs = framework.Lbl_mgr.num_imgs - 1
        grid_pts = framework.Lbl_mgr.gen_grid(normalize=False).float()
        grid_pts = grid_pts[::grid_step, ::grid_step]  # downsample
        grid_pts = grid_pts.reshape(-1, 2).unsqueeze(0)  # flat -> [n_samples, 2]
        aabb = framework.config.aabb[0]
        print("AABB: ", aabb)
        x1, y1, z1, x2, y2, z2 = aabb

        vid_canon = []
        print("Generating canonical points for video...")
        id1 = torch.tensor(0, device=framework.Lbl_mgr.device)
        prediction = framework.sampling_and_inference(grid_pts, id1.unsqueeze(0), None, pts3D_a=None, det=True)
        q_points3D = prediction["p_3Dpts1"].cpu()[0]  # [n_p, 3]
        if progress_bar:
            pbar = tqdm(total=num_imgs)
        for id1 in torch.arange(num_imgs, device=framework.Lbl_mgr.device):
            prediction = framework.sampling_and_inference(grid_pts, id1.unsqueeze(0), None, pts3D_a=None, det=True)
            if framework.Lbl_mgr.config.merge_before_deform:
                canon_pts1 = prediction["canon_pts1"].squeeze(-2)  # [n_p, 3]
            else:
                canon_pts1 = framework.Lbl_mgr.merge_samples(prediction["canon_pts1"], prediction["m_wgt1"])
            p_pts1_3D = prediction["p_3Dpts1"].cpu()[0]
            canon_pts1 = canon_pts1.cpu()[0]
            fig = plt.figure(figsize=(8, 5))
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            ax1.scatter(q_points3D[:, 0], q_points3D[:, 1], q_points3D[:, 2], c="k", label="p1", s=3)
            ax1.scatter(p_pts1_3D[:, 0], p_pts1_3D[:, 1], p_pts1_3D[:, 2], c="b", label="^p1", s=3)
            ax1.set_title(f"3D Points1, t={id1.item()}")
            ax1.set_xlabel("X/r")
            ax1.set_ylabel("Y/theta")
            ax1.set_zlabel("Z/phi")
            ax1.legend()
            ax1.set_xlim(x1, x2)
            ax1.set_ylim(y1, y2)
            ax1.set_zlim(z1, z2)
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            ax2.scatter(canon_pts1[:, 0], canon_pts1[:, 1], canon_pts1[:, 2], c="g", label="cp1", s=3)
            ax2.set_title(f"Canonical Space, t={id1.item()}")
            ax2.set_xlabel("X/r")
            ax2.set_ylabel("Y/theta")
            ax2.set_zlabel("Z/phi")
            ax2.legend()
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.set_zlim(0, 1)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            vid_canon.append(imageio.v2.imread(buf))

            # vid_canon.append(canon_pts)
            if progress_bar:
                pbar.update(1)
        if progress_bar:
            pbar.close()
    print("Frame size: ", vid_canon[-1].shape)
    # Save video
    out_vid = os.path.join(str(out_dir), "vid", f"canon_{chkpt_n}.mp4")
    print("-> Saving canonical space video to: ", out_vid, "...")
    imageio.mimsave(out_vid, vid_canon, quality=8, fps=20)


# endregion
##########################################################

##########################################################
# region Main

import argparse
from config.config import get_config, get_saving_path
from DataManager import get_dataset
from model import get_model


def extract_frames_from_video(video_path, output_folder, frame_interval=1):
    """Extract frames from a video and save them as images.
    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save the extracted frames.
        frame_interval (int): Interval between frames to extract.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:04d}.png"), frame)
        frame_count += 1
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training script for point tracker (omnimotion).")
    parser.add_argument("-c", "--configuration", type=str, default=None, help="Path to experiment specifications (.YAML file)")
    parser.add_argument("--all", action="store_true", help="Generate videos for all checkpoints in the folder.")
    parser.add_argument("--extract", action="store_true", help="extract frames from a video and save them as images.")
    parser.add_argument("--video_path", type=str, default=None, help="Path to the input video file.")
    args = parser.parse_args()

    if args.extract:
        if args.video_path is None:
            raise ValueError("Please provide a video path to extract frames from.")
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        output_folder = os.path.join(os.path.dirname(args.video_path), video_name)
        extract_frames_from_video(args.video_path, output_folder, frame_interval=10)
        print(f"Frames extracted to {output_folder}")
        exit(0)

    # Configurations
    if args.configuration:
        config = get_config(args.configuration, mk_out_dir=False)
    else:
        raise ValueError("Please provide a configuration file.")
    saving_path = get_saving_path(config)
    vid_folder = saving_path / "vid"
    vid_folder.mkdir(parents=True, exist_ok=True)

    # Dataset
    dataset, _ = get_dataset(config, saving_path, loaders=False, eval=True)
    print("-> Dataset loaded.")

    # Data handler
    handler = RayManager(config, dataset)
    print("-> Data handler created.")

    # Semi ground truth depth maps
    get_gt_depth_video(handler, grid_step=2, colorise=True, out_dir=str(saving_path))

    if args.all:
        # Checkpoints
        ckpts = [f for f in os.listdir(saving_path) if f.endswith(".pth")]
        ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    else:
        ckpts = [""]

    for ckpt in ckpts:
        config.ckpt_path = os.path.join(saving_path, ckpt) if len(ckpt) > 0 else ""

        # Model
        models = get_model(config, handler.device, True)
        print("-> Model created.")

        # Trainer
        framework = BaseTrainer(config, models, criterion=None, label_manager=handler, save_path=saving_path).to(handler.device)
        framework.eval()

        # Visualise
        # Point correspondences
        get_video_correspondences(framework, config.query_frame_id, dataset, interval=20, radius=3, out_dir=str(saving_path), progress_bar=True)

        # Depth maps and rendered images
        get_rgb_and_depth_video(framework, grid_step=6, colorise=True, out_dir=str(saving_path), progress_bar=True)

        # Density progression
        get_density_progression_video(framework, grid_step=25, out_dir=str(saving_path), fps=10)

        # Canonical space points
        get_canonical_space_video(framework, grid_step=20, out_dir=str(saving_path), progress_bar=True)

# endregion
##########################################################
