import torch, torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Union
from torch.utils.data import Dataset


class InvNeRFLoss(torch.nn.Module):
    def __init__(self, args:Dict, dataset: Dataset, data_mngr=None):
        super().__init__()
        self.args = args
        self.w = dataset.w
        self.h = dataset.h
        self.data_mngr = data_mngr
        for req_l in args.losses:
            assert hasattr(self, req_l), f"Loss function {req_l} not implemented"
        assert len(args.losses) > 0, "No loss function selected"
        assert len(args.losses) == len(args.w_losses), f"Number of losses and weights must be the same. {len(args.losses)} != {len(args.w_losses)}"

    def forward(self, output:Dict, target:Dict, step:Tensor, device:str, loss_fn:List=None, loss_w:List=None, bypass_flow:bool=False,):
        losses = {}
        self.device = device
        if loss_fn is not None or loss_w is not None: # Change the loss during traing time
            assert len(loss_fn) > 0, "No loss function selected"
            assert len(loss_fn) == len(loss_w), "Number of losses and weights must be the same"
        else:    
            loss_fn = self.args.losses
            loss_w = self.args.w_losses
        for loss_name, loss_weight in zip(loss_fn, loss_w):
            loss_weight = weight_scheduler(step,*loss_weight).to(device) if type(loss_weight) is list else loss_weight
            if loss_weight == 0:
                continue
            try:
                loss = getattr(self, loss_name)(**output, **target, step=step, device=device,bypass_flow=bypass_flow)
            except Exception as e:
                print(f"\nERROR in loss function: {loss_name}")
                raise e
            for k, v in loss.items():
                #print(f"Loss: {k} | Value: {v} | Weight: {loss_weight}")
                loss[k] = v * loss_weight
            losses.update(loss)
        total_loss = sum(l for l in losses.values() if not torch.isnan(l))
        return total_loss, losses
    
    def rgb_loss(self, pts2:Tensor, p_rgb1:Tensor, gt_rgb1:Tensor, **kwargs):
        mask = get_in_range_mask(pts2, self.w, self.h, self.args.max_padding)
        loss = {}
        if mask.sum() > 0:
            error = torch.linalg.norm(p_rgb1 - gt_rgb1, dim=-1, ord=2)
            loss['rgb_l'] = torch.mean(error)
        else:
            loss['rgb_l'] = torch.tensor(0., device=self.device)
        
        return loss
    
    def optical_flow_loss(self, p_pts2:Tensor, pts2:Tensor, p_pts1:Tensor, pts1:Tensor,
                          c_mask:Tensor, occ_mask:Tensor, pair_weight:Tensor, 
                          bypass_flow:bool=False, **kwargs):
        mask_1to2 = get_in_range_mask(pts2, self.w,self.h, self.args.max_padding)
        mask_1to2 = mask_1to2 & (c_mask | occ_mask)
        mask_2to1 = get_in_range_mask(pts1, self.w,self.h, self.args.max_padding)
        mask_2to1 = mask_2to1 & c_mask
        loss = {}
        assert p_pts2.shape == pts2.shape, f"Shape mismatch: gt[{pts2.shape}] != pred[{p_pts2.shape}]"
        if mask_1to2.sum() > 0 and not bypass_flow:
            distance1 = torch.linalg.norm((p_pts2 - pts2), dim=-1, ord=2)
            distance2 = torch.linalg.norm((p_pts1 - pts1), dim=-1, ord=2)
            flow_error_1to2 = (distance1 * pair_weight)
            focusmask12 = flow_error_1to2>2
            flow_error_2to1 = (distance2 * pair_weight)
            focusmask21 = flow_error_2to1>2
            
            # loss['flow_l'] = torch.tensor(0., device=p_pts2.device)
            # if (mask_1to2&~focusmask12).sum():
            #     loss['flow_l'] += torch.mean(flow_error_1to2[mask_1to2&~focusmask12]) 
            # if (mask_1to2&focusmask12).sum():
            #     loss['flow_l']+= torch.mean(flow_error_1to2[mask_1to2&focusmask12])
            # if (mask_2to1&~focusmask21).sum():
            #     loss['flow_l'] +=torch.mean(flow_error_2to1[mask_2to1&~focusmask21])
            # if (mask_2to1&focusmask21).sum():
            #     loss['flow_l']+=  torch.mean(flow_error_2to1[mask_2to1&focusmask21])
            loss['flow_l'] = torch.mean(flow_error_1to2[mask_1to2]) + torch.mean(flow_error_2to1[mask_2to1])
            # loss['flowG_l'] = gradient_loss(p_pts2, pts2, pair_weight, mask) * 0.001
        else:
            loss['flow_l'] = torch.tensor(0., device=p_pts2.device)
        return loss

    def cross_consistency_loss(self, pts1:Tensor, pts2:Tensor, p_3Dpts1:Tensor,
                               canon_pts1:Tensor, canon_pts2:Tensor,
                               m_wgt1:Tensor, m_wgt2:Tensor,
                               c_mask:Tensor, pair_weight:Tensor, 
                               bypass_flow=False,**kwargs):
        mask = get_in_range_mask(pts2, self.w,self.h, self.args.max_padding)
        mask = mask & get_in_range_mask(pts1, self.w, self.h, self.args.max_padding)
        mask = mask & c_mask
        assert canon_pts1.shape == canon_pts2.shape, f"Shape mismatch: gt[{canon_pts2.shape}] != pred[{canon_pts1.shape}]"
        if bypass_flow:
            loss = {'XC_l': torch.tensor(0., device=self.device)}
        else:
            # canon_pts1 = expand_from_unisphere(canon_pts1, self.aabb)
            # canon_pts2 = expand_from_unisphere(canon_pts2, self.aabb)
            if self.args.merge_before_deform:
                canon_pts1 = canon_pts1.squeeze(-2)
                canon_pts2 = canon_pts2.squeeze(-2)
            else:
                m_wgt1 = m_wgt1.detach().clone() # Do not propagate this error to the density/color network
                m_wgt2 = m_wgt2.detach().clone() 
                canon_pts1 = torch.sum(m_wgt1.unsqueeze(-1) * canon_pts1, dim=-2)
                canon_pts2 = torch.sum(m_wgt2.unsqueeze(-1) * canon_pts2, dim=-2)
            if self.args.coord_system == 'cartesian':
                distance = (canon_pts1 - canon_pts2)
                canon_error = torch.linalg.norm(distance, dim=-1, ord=2).squeeze(-1)*100 # [n_imgs, n_pts]
            else: # spherical coordinates
                rho = canon_pts1[..., :1] - canon_pts2[..., :1]
                angles1 = canon_pts1[..., 1:].sin() - canon_pts2[..., 1:].sin()
                angles2 = canon_pts1[..., 1:].cos() - canon_pts2[..., 1:].cos()
                canon_error = torch.linalg.norm(angles1, dim=-1, ord=2) + torch.linalg.norm(angles2, dim=-1, ord=2) + torch.linalg.norm(rho, dim=-1, ord=2)
            canon_error = (canon_error) * pair_weight
            focusmask = canon_error > 1
            # print("Number of outliers:", focusmask.sum().item(), "out of", mask.sum().item())
            loss = {'XC_l': torch.tensor(0., device=canon_error.device)}
            if (mask&~focusmask).sum() > 0:
                loss['XC_l'] += torch.mean(canon_error[mask]) #&~focusmask])
            if (mask&focusmask).sum() > 0:
                loss['XC_l'] += torch.mean(canon_error[mask]) #&focusmask])
            # 
            assert p_3Dpts1.shape == canon_pts1.shape, f"Shape mismatch: def[{p_3Dpts1.shape}] != canon[{canon_pts1.shape}]"
            grad_mask = mask & (canon_error < loss['XC_l']) # Use as reference qye points with low error
            p_3Dpts1 = p_3Dpts1.detach().clone() # Do not propagate this error to the density/color network
            p_3Dpts1 = self.data_mngr.contract_to_unisphere(p_3Dpts1)
            loss['XC_grad_l'] = gradient_loss(canon_pts1, p_3Dpts1, pair_weight, grad_mask)
        return loss
    
    def Jacobian_reg_loss(self, rays1:Tensor, rays2:Tensor,
                                pts1:Tensor,  pts2:Tensor,

                                canon_pts1:Tensor, canon_pts2:Tensor,
                                m_wgt1:Tensor, m_wgt2:Tensor,
                                c_mask:Tensor, **kwargs):
        """Compute the Jacobian regularization loss"""
        if not canon_pts1.requires_grad:
            return {}
        mask = get_in_range_mask(pts2, self.w,self.h, self.args.max_padding)
        mask = mask & get_in_range_mask(pts1, self.w, self.h, self.args.max_padding)
        mask = mask & c_mask
        canon1 = canon_pts1
        canon2 = canon_pts2
        if self.args.merge_before_deform:
            raise NotImplementedError("Jacobian regularization loss is not implemented for merge_before_deform=True")
        else:
            pts3D1 = rays1
            pts3D2 = rays2
            B, N, S, D = pts3D1.shape
            I = torch.eye(D, device=pts3D1.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # [1, 1, D, D]
            I = I.expand(B, N, S, D, D)

        assert canon_pts1.shape == canon_pts2.shape, f"Cannon shape mismatch: gt[{canon_pts2.shape}] != pred[{canon_pts1.shape}]"
        assert pts3D1.shape   == canon_pts1.shape, f"Shape mismatch: p_3Dpts[{pts3D1.shape}] != canon[{canon_pts1.shape}]"
        assert pts3D1.shape   == pts3D2.shape,   f"Shape mismatch: p_3Dpts1[{pts3D1.shape}] != p_3Dpts2[{pts3D2.shape}]"
        # Helper to compute Jacobian matrix [B, N, D, D]
        def compute_jacobian(p_pts, canon_pts):
            # assert p_pts.requires_grad, "p_pts must require gradients for Jacobian computation"
            # assert canon_pts.requires_grad, "canon_pts must require gradients for Jacobian computation"
            jacobian_parts = []
            for d in range(D):
                grad_outputs = torch.zeros_like(p_pts)
                grad_outputs[..., d] = 1.0
                grad = torch.autograd.grad(
                    outputs=canon_pts,
                    inputs=p_pts,
                    grad_outputs=grad_outputs,
                    retain_graph=True,
                    create_graph=True,
                    only_inputs=True
                )[0]  # [B, N, D]
                jacobian_parts.append(grad.unsqueeze(-2)) # [B, N, 1, D]
            J = torch.cat(jacobian_parts, dim=-2) # [B, N, D, D]
            return J

        J1 = compute_jacobian(pts3D1, canon1)
        J2 = compute_jacobian(pts3D2, canon2)

        JTJ = J1.transpose(-1, -2) @ J1                    # [B, N, D, D]
        diff = JTJ - I
        loss_1 = (diff ** 2).sum(dim=(-1, -2)).mean()   # scalar

        JTJ = J2.transpose(-1, -2) @ J2                   # [B, N, D, D]
        diff = JTJ - I
        loss_2 = (diff ** 2).sum(dim=(-1, -2)).mean()   # scalar

        jacobian_loss = loss_1 + loss_2
        return {'jacobian_loss': jacobian_loss}   

    def geo_consistency_loss(self, c_mask:Tensor, occ_mask:Tensor, pair_weight:Tensor, 
                             pts1:Tensor, p_3Dpts1to2:Tensor, p_3Dpts1:Tensor, 
                             pts2:Tensor, p_3Dpts2to1:Tensor, p_3Dpts2:Tensor, rd2:Tensor,
                            bypass_flow=False, **kwargs):
        mask_1to2 = get_in_range_mask(pts2, self.w,self.h, self.args.max_padding)
        mask_1to2 = mask_1to2 & c_mask  #| occ_mask)
        mask_2to1 = get_in_range_mask(pts1, self.w,self.h, self.args.max_padding)
        mask_2to1 = mask_2to1 & c_mask
        loss = {}
        if mask_1to2.sum() > 0 and not bypass_flow:
            # p_3Dpts2 = p_3Dpts2.detach().clone() # Do not propagate this error to the density/color network
            if p_3Dpts1to2.shape[-2]==1: p_3Dpts1to2 = p_3Dpts1to2.squeeze(-2)
            assert p_3Dpts2.shape == p_3Dpts1to2.shape, f"Shape mismatch: gt[{p_3Dpts2.shape}] != pred[{p_3Dpts1to2.shape}]"
            # p_rho2, p_rho12 = torch.norm(p_3Dpts2, dim=-1), torch.norm(p_3Dpts1to2, dim=-1)
            # geo_error_1to2 = torch.linalg.norm((p_rho2 - p_rho12).unsqueeze(-1), dim=-1, ord=2).squeeze(-1)
            geo_error_1to2 = torch.linalg.norm((p_3Dpts2 - p_3Dpts1to2), dim=-1, ord=2)
            geo_error_1to2 = (geo_error_1to2 * pair_weight) *1000
            focusmask12 = geo_error_1to2>5

            # p_3Dpts1 = p_3Dpts1.detach().clone() # Do not propagate this error to the density/color network
            if p_3Dpts2to1.shape[-2]==1: p_3Dpts2to1 = p_3Dpts2to1.squeeze(-2)
            assert p_3Dpts1.shape == p_3Dpts2to1.shape, f"Shape mismatch: gt[{p_3Dpts1.shape}] != pred[{p_3Dpts2to1.shape}]"
            # p_rho1, p_rho21 = torch.norm(p_3Dpts1, dim=-1), torch.norm(p_3Dpts2to1, dim=-1)
            # geo_error_2to1 = torch.linalg.norm((p_rho1- p_rho21).unsqueeze(-1), dim=-1, ord=2).squeeze(-1)
            geo_error_2to1 = torch.linalg.norm((p_3Dpts1 - p_3Dpts2to1), dim=-1, ord=2)
            geo_error_2to1 = (geo_error_2to1 * pair_weight)*1000
            focusmask21 = geo_error_2to1>5

            # loss['GeoC_l'] = torch.tensor(0., device=p_3Dpts1.device)
            # if (mask_1to2&~focusmask12).sum():
            #     loss['GeoC_l'] += torch.mean(geo_error_1to2[mask_1to2&~focusmask12]) 
            # if (mask_1to2&focusmask12).sum():
            #     loss['GeoC_l']+= torch.mean(geo_error_1to2[mask_1to2&focusmask12])
            # if (mask_2to1&~focusmask21).sum():
            #     loss['GeoC_l'] +=torch.mean(geo_error_2to1[mask_2to1&~focusmask21])
            # if (mask_2to1&focusmask21).sum():
            #     loss['GeoC_l']+=  torch.mean(geo_error_2to1[mask_2to1&focusmask21])
            loss['GeoC_l'] = torch.mean(geo_error_1to2[mask_1to2]) + torch.mean(geo_error_2to1[mask_2to1])
            # print(torch.mean(geo_error_1to2[mask_1to2&~error_mask12]).item(), torch.mean(geo_error_2to1[mask_2to1&~error_mask21]).item() )
            # print(torch.mean(geo_error_1to2[mask_1to2&error_mask12]).item(), torch.mean(geo_error_2to1[mask_2to1&error_mask21]).item())

            #occ_mask = occ_mask & (torch.linalg.norm((p_3Dpts1to2), dim=-1, ord=2) < torch.linalg.norm((p_3Dpts2), dim=-1, ord=2))
            if (occ_mask).sum() > 0:
                # Occlusion (Make sure the points that should be occluded are occluded) p_3Dpts1to2 p_3Dpts2
                target = p_3Dpts2 + rd2*0.25 # p_3Dpts2 should be closer than p_3Dpts1to2
                occ_error = torch.linalg.norm((p_3Dpts1to2 - target), dim=-1, ord=2)
                occ_error = (occ_error * pair_weight) # * 1000
                loss['Occ_l'] = torch.mean(occ_error[occ_mask])

            
        else:
            error = torch.tensor(0., device=pts2.device)
        
        return loss

    def gradient_loss(self,ids1, p_3Dpts1:Tensor,ids2, p_3Dpts2:Tensor, **kwargs):
        mask = torch.abs(ids1.int()-ids2.int())==1
        if mask.sum()>0:
            if self.args.merge_before_deform: 
                p_3Dpts2=p_3Dpts2.squeeze(-2)
                p_3Dpts1=p_3Dpts1.detach().clone() # Do not propagate this error to the density/color network
            error = gradient_loss(p_3Dpts1[mask], p_3Dpts2[mask])
        else: error = torch.tensor(0., device=self.device)
        return {'grad_l': error}

    def gaussian_target_loss(self, rays1:Tensor, m_wgt1:Tensor, pts1_3D:Tensor, ro1:Tensor, p_3Dpts1:Tensor,
                                   rays2:Tensor, m_wgt2:Tensor, pts2_3D:Tensor, ro2:Tensor, p_3Dpts2:Tensor,
                                   geo_mask1:Tensor, geo_mask2:Tensor, c_mask:Tensor, occ_mask:Tensor, **kwargs):
        """Compute the Gaussian target loss
        Args:
            prop_net_T (List): List of intervals and densities [intervals:[n_imgs X n_pts+1, NS],[densities:[n_imgs X n_pts, NS]]]
            pts1_3D (Tensor): 3D points on the surface of the object [n_imgs, n_pts, 3]
            geo_mask (Tensor): Mask discard non reliable points [n_imgs, n_pts]
            ro1 (Tensor): Rays rays origin [n_imgs, n_pts, 3]
        """
        losses={}
        std = self.args.ray_std

        gt_depth = pts1_3D - ro1
        gt_depth = torch.linalg.norm(gt_depth, dim=-1, ord=2) # [batch, n_pts]
        rays1 =self.data_mngr.expand_from_unisphere(rays1)
        depth = torch.linalg.norm(rays1-ro1.unsqueeze(-2), dim=-1, ord=2)
        gt_density = torch.exp(-0.5 * ((depth - gt_depth.unsqueeze(-1)) / std).pow(2))
        gt_density = gt_density / (gt_density.sum(dim=-1, keepdim=True) + 1e-8)
        rendering_error1 = (gt_density * ((gt_density + 1e-8) / (m_wgt1 + 1e-8)).log()).sum(dim=-1) # [batch, n_pts]
        # print(f"Rendering error range: {rendering_error1.min().item()} - {rendering_error1.max().item()}")
        mask1 = geo_mask1

        gt_depth = pts2_3D - ro2
        gt_depth = torch.linalg.norm(gt_depth, dim=-1, ord=2) # [batch, n_pts]
        rays2 = self.data_mngr.expand_from_unisphere(rays2)
        depth = torch.linalg.norm(rays2-ro2.unsqueeze(-2), dim=-1, ord=2)
        gt_density = torch.exp(-0.5 * ((depth - gt_depth.unsqueeze(-1)) / std).pow(2))
        gt_density = gt_density / (gt_density.sum(dim=-1, keepdim=True) + 1e-8)
        rendering_error2 = (gt_density * ((gt_density + 1e-8) / (m_wgt2 + 1e-8)).log()).sum(dim=-1) # [batch, n_pts]
        # print(f"Rendering error range: {rendering_error2.min().item()} - {rendering_error2.max().item()}")
        mask2 = geo_mask2

        # losses["GT1_l"] = torch.mean(rendering_error1[mask1]) 
        # losses["GT2_l"] = torch.mean(rendering_error2[mask2])
        losses["GT_l"] = torch.mean(rendering_error1[mask1]) + torch.mean(rendering_error2[mask2])
        return losses
    
    def canonical_sphere_loss(self, canon_pts1:Tensor, rays1:Tensor, m_wgt1:Tensor, ids1:Tensor, p_3Dpts1:Tensor=None, **kwargs):
        
        shift_canon_pts1 = (canon_pts1-0.5) * 2
        mask = shift_canon_pts1.norm(dim=-1) > 1
        loss = {}
        if mask.sum() > 0:
            p_3Dpts1=self.data_mngr.contract_to_unisphere(p_3Dpts1).detach().clone()
            if self.args.merge_before_deform:
                mask = mask.squeeze(-1)
                canon_pts1 = canon_pts1.squeeze(-2)
                loss['Canon_l'] = F.mse_loss(canon_pts1[mask], p_3Dpts1[mask])
            else:
                loss['Canon_l'] = F.mse_loss(canon_pts1[mask], rays1[mask])
        else:
            loss['Canon_l'] = torch.tensor(0., device=self.device)

        return loss
    
    def distortion_loss(self, rays1_depth, m_wgt1, **kwargs):
        """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
        t = torch.cat([rays1_depth, rays1_depth[..., -1:]], dim=-1)
        # The loss incurred between all pairs of intervals.
        ut = (t[..., 1:] + t[..., :-1]) / 2
        dut = torch.abs(ut[..., :, None] - ut[..., None, :])
        loss_inter = torch.sum(m_wgt1 * torch.sum(m_wgt1[..., None, :] * dut, dim=-1), dim=-1)

        # The loss incurred within each individual interval with itself.
        loss_intra = torch.sum(m_wgt1**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

        return {'distortion_l': (loss_inter + loss_intra).mean()}
    
    def smoth_flow_loss(self, flow_prev=None, flow_after=None, **kwargs):
        if flow_prev is None or flow_after is None:
            print("Flow is None, skipping smooth flow loss")
            return {}
        assert flow_prev.shape == flow_after.shape, "Flow shapes do not match"
        assert flow_prev.shape[-1] == 3, "Flow must be 3D"
        if (flow_prev.sum() == 0.) and (flow_after.sum() == 0.):
            return {'smoothFlow_l': torch.tensor(0., device=self.device)}
        error = torch.linalg.norm(flow_prev - flow_after, dim=-1, ord=2)
        return {'smoothFlow_l': error.mean()}

# region loss helpers

def gradient_loss(pred:Tensor, gt:Tensor, weight=None, mask = None, ):
    """Calculate gradient loss for selected points in the image
    Args:
        pred (torch.Tensor): Predicted points [n_imgs, n_pts, n]
        gt (torch.Tensor): Ground truth points [n_imgs, n_pts, n]
        mask (torch.Tensor, optional): Mask to select points [n_imgs, n_pts]
        weight (torch.Tensor, optional): Weight [n_imgs, 1]
    Returns:
        loss (torch.Tensor): Loss
    """
    pred_grad = pred[..., 1:, :] - pred[..., :-1, :]
    gt_grad = gt[..., 1:, :] - gt[..., :-1, :]
    mask = mask[..., 1:] & mask[..., :-1] if mask is not None else None
    loss = masked_l1_loss(pred_grad, gt_grad, weight, mask)
    return loss

def masked_l1_loss(pred, gt, weight=None, mask=None, normalize=True, quantile=1):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction='none').mean(dim=-1, keepdim=False) # [n_imgs, n_pts, 1]
        if quantile >= 1:
            #print("Quantile >= 1")
            error = sum_loss * weight if weight is not None else sum_loss
            error = error[mask] if mask is not None else error
            return torch.mean(error)
        loss_at_quantile = torch.quantile(sum_loss, quantile)
        quantile_mask = (sum_loss < loss_at_quantile).squeeze(-1)
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (ndim * torch.sum(mask[quantile_mask]) + 1e-8)
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])

def get_in_range_mask(x, w, h, max_padding=0):
    mask = (x[..., 0] >= -max_padding) * \
            (x[..., 0] <= w - 1 + max_padding) * \
            (x[..., 1] >= -max_padding) * \
            (x[..., 1] <= h - 1 + max_padding)
    is_nan = torch.isnan(x).any().item()
    if mask.sum() == 0 and not is_nan:
        raise ValueError("No points in range!!!")
    return mask

def get_in_depth_range_mask(rays, m_wgt, pts_target, pts_tol, p_3Dpts=None):
    if p_3Dpts is None:
        pts_3D = torch.sum(m_wgt.unsqueeze(-1) * rays, dim=-2)
    else:
        pts_3D = p_3Dpts
    depth_error = torch.linalg.norm(pts_3D - pts_target, dim=-1, ord=2)
    pts_tol = pts_tol.norm(dim=-1)*4+1e-3
    return depth_error < pts_tol # points that were sampled within the depth range

def trimmed_l1_loss(pred, gt, quantile=0.9):
    if quantile >= 1:
        return F.l1_loss(pred, gt)
    loss = F.l1_loss(pred, gt, reduction='none').mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss
       
def weight_scheduler(step, start_step, w, min_weight, max_weight):
    """Weight scheduler
    Args:
        step (int): Current step
        start_step (int): Start step
        w (float): Weight
        min_weight (float): Minimum weight
        max_weight (float): Maximum weight
    Returns:
        weight (float): Weight
    """
    if step <= start_step:
        weight = torch.tensor([0.0]) if w>0 else torch.tensor([max_weight])
    else:
        weight = torch.tensor(w * (step - start_step))
        weight = weight + min_weight if weight > 0 else weight + max_weight
        weight = torch.clip(weight, min=min_weight, max=max_weight)
    return weight
