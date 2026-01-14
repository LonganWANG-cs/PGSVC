#Trace Gaussians
from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan
from pytorch_msssim import ms_ssim
from torch.distributions import MultivariateNormal
import os

class GaussianSplats(nn.Module):
    def __init__(self, num_points_per_layer, scaling_factor, **kwargs):
        super().__init__()
        self.num_points_per_layer = num_points_per_layer.copy() #Number of added Gaussians in each layer (Delta Gaussians)
        self.initial_num_points_per_layer = num_points_per_layer.copy()
        self.cumulative_points = np.cumsum(num_points_per_layer).tolist() #Number of Gaussians in each layer
        self.total_points = self.cumulative_points[-1] #Total number of Gaussians
        self.scaling_factor = scaling_factor #Downsample scale in each layer
        self.layer_number =  len(self.num_points_per_layer) #Number of layers
        self.H, self.W = kwargs["H"], kwargs["W"] #Height and width
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.loss_type = kwargs["loss_type"]
        self.device = kwargs["device"]
        self.lr = kwargs["lr"]
        self.removal_rate = kwargs["removal_rate"]
        self.train_strategy = kwargs["training_strategy"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        self.is_GSP = kwargs["is_GSP"] #GSP button
        self.is_GSA = kwargs["is_GSA"] #GSA button
        self.is_GR = kwargs["is_GR"] #GR button
        self.is_frozen = kwargs["is_frozen"] #W button
        self.GSP_interval = 100 #GSP interval
        self._xyz_layers = nn.ParameterList([
            nn.Parameter(torch.atanh(2 * (torch.rand(n, 2) - 0.5))) for n in self.num_points_per_layer
        ])
        self._cholesky_layers = nn.ParameterList([
            nn.Parameter(torch.rand(n, 3)) for n in self.num_points_per_layer
        ])
        self._features_dc_layers = nn.ParameterList([
            nn.Parameter(torch.rand(n, 3)) for n in self.num_points_per_layer
        ])
        # self._features_dc_layers = nn.ParameterList([
        #     nn.Parameter(torch.rand(n, 3)) if i == 0 else nn.Parameter(0.01*torch.rand(n, 3))
        #     for i, n in enumerate(self.num_points_per_layer)
        # ])
        if self.is_frozen:
            self._rgb_W_layers = [torch.ones(n, 1) for n in self.num_points_per_layer]
            for idx, buf in enumerate(self._rgb_W_layers):
                self.register_buffer(f'_rgb_W_{idx}', buf)
        else:
            # self._rgb_W_layers = nn.ParameterList([
            #     nn.Parameter(torch.ones(n, 1)) if i == 0 else nn.Parameter(0.01*torch.rand(n, 1))
            #     for i, n in enumerate(self.num_points_per_layer)
            # ])
            self._rgb_W_layers = nn.ParameterList([
                nn.Parameter(torch.ones(n, 1)) for n in self.num_points_per_layer
            ])
        self._opacity_layers = []
        self._gscenter_cholesky_layers = []
        self._gscenter_features_dc_layers = []
        self._point_ids_layers=[]
        self._next_point_id_layers=[]
        for i, n in enumerate(self.num_points_per_layer):
            self.register_buffer(f'_opacity_{i}', torch.ones(n, 1))
            self.register_buffer(f'_gscenter_cholesky_{i}', torch.ones(n, 3))
            self.register_buffer(f'_gscenter_features_dc_{i}', torch.ones(n, 3))
            self.register_buffer(f'_point_ids_{i}', torch.arange(n, dtype=torch.long))
            self.register_buffer(f'_next_point_id_{i}', torch.tensor(n, dtype=torch.long))
            self._opacity_layers.append(getattr(self, f'_opacity_{i}'))
            self._gscenter_cholesky_layers.append(getattr(self, f'_gscenter_cholesky_{i}'))
            self._gscenter_features_dc_layers.append(getattr(self, f'_gscenter_features_dc_{i}'))
            self._point_ids_layers.append(getattr(self, f'_point_ids_{i}'))
            self._next_point_id_layers.append(getattr(self, f'_next_point_id_{i}'))
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)


    @property
    def get_xyz(self):
        return torch.cat(list(self._xyz_layers), dim=0)
    @property
    def get_features(self):
        if self.is_frozen:
            rgb_w_list = [getattr(self, f'_rgb_W_{i}') for i in range(self.layer_number)]
        else:
            rgb_w_list = list(self._rgb_W_layers)
        return torch.cat([
            f * w for f, w in zip(self._features_dc_layers, rgb_w_list)
        ], dim=0)
    @property
    def get_cholesky_elements(self):
        return torch.cat(list(self._cholesky_layers), dim=0)
    
    def forward_layer(self, idx):
        xyz = torch.cat([torch.tanh(self._xyz_layers[i]) for i in range(idx + 1)], dim=0).to(self.device)
        cholesky = torch.cat([self._cholesky_layers[i] + self.cholesky_bound for i in range(idx + 1)], dim=0).to(self.device)
        features_dc = torch.cat([self._features_dc_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
        opacity = torch.cat([self._opacity_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
        if self.is_frozen:
            rgb_W = torch.cat([getattr(self, f'_rgb_W_{i}') for i in range(idx + 1)], dim=0).to(self.device)
        else:
            rgb_W = torch.cat([self._rgb_W_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
        features = features_dc * rgb_W
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            xyz, cholesky, self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            features, opacity, self.H, self.W,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        # out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
        

    def forward_layer_delta(self, idx):
        xyz = torch.tanh(self._xyz_layers[idx]).to(self.device)
        cholesky = (self._cholesky_layers[idx] + self.cholesky_bound).to(self.device)
        features_dc = self._features_dc_layers[idx].to(self.device)
        opacity = self._opacity_layers[idx].to(self.device)

        if self.is_frozen:
            rgb_W = getattr(self, f'_rgb_W_{idx}').to(self.device)
        else:
            rgb_W = self._rgb_W_layers[idx].to(self.device)

        features = features_dc * rgb_W

        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            xyz, cholesky, self.H, self.W, self.tile_bounds
        )

        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            features, opacity, self.H, self.W,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )

        # out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img}
    

    def forward_layer_delta_gradient(self, idx):
        xyz = torch.tanh(self._xyz_layers[idx]).to(self.device)
        cholesky = (self._cholesky_layers[idx] + self.cholesky_bound).to(self.device)
        features_dc = self._gscenter_features_dc_layers[idx].to(self.device)
        opacity = self._opacity_layers[idx].to(self.device)
        def safe_norm(param, dim=1):
            if param.grad is not None:
                return param.grad.norm(dim=dim).detach()
            else:
                return torch.zeros(param.shape[0], device=param.device)

        grad_xyz = safe_norm(self._xyz_layers[idx], dim=1)
        grad_cholesky = safe_norm(self._cholesky_layers[idx], dim=1)
        grad_features_dc = safe_norm(self._features_dc_layers[idx], dim=1)
        grad_rgb_W = safe_norm(self._rgb_W_layers[idx], dim=1)

        grad_total = grad_xyz + grad_cholesky + grad_features_dc + grad_rgb_W

        grad_min, grad_max = grad_total.min(), grad_total.max()
        if grad_max > grad_min:
            grad_norm_normed = (grad_total - grad_min) / (grad_max - grad_min)
        else:
            grad_norm_normed = torch.zeros_like(grad_total)

        yellow_colors = torch.stack([
            grad_norm_normed,  # R
            grad_norm_normed,  # G
            torch.zeros_like(grad_norm_normed)  # B
        ], dim=1)

        features = features_dc*yellow_colors




        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            xyz, cholesky, self.H, self.W, self.tile_bounds
        )

        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            features, opacity, self.H, self.W,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )

        # out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img}
    
    

    
    
    def forward_layer_gs_center(self, idx):
        xyz = torch.cat([torch.tanh(self._xyz_layers[i]) for i in range(idx + 1)], dim=0).to(self.device)
        cholesky = torch.cat([self._gscenter_cholesky_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
        features = torch.cat([self._gscenter_features_dc_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
        opacity = torch.cat([self._opacity_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
        xyz, cholesky, self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
        xys, depths, radii, conics, num_tiles_hit,
        features, opacity, self.H, self.W,
        self.BLOCK_H, self.BLOCK_W,
        background=self.background, return_alpha=False
        )
        # out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    def forward_layer_delta_gs_center(self, idx):
        xyz = torch.tanh(self._xyz_layers[idx]).to(self.device).to(self.device)
        cholesky = (self._gscenter_cholesky_layers[idx]).to(self.device)
        features = self._gscenter_features_dc_layers[idx].to(self.device).to(self.device)
        opacity = self._opacity_layers[idx].to(self.device)
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            xyz, cholesky, self.H, self.W, self.tile_bounds
        )

        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            features, opacity, self.H, self.W,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )

        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()

        return {"render": out_img}
    
    def update_optimizer(self):
        self.optimizer = Adan(self.parameters(), lr=self.lr)

    def pruning_control_grad(self, iter, layer_idx, removal_rate, start_iter_threshold_remove, end_iter_threshold_remove):
        if iter > end_iter_threshold_remove or iter < start_iter_threshold_remove:
            return

        removal_iteration_num = end_iter_threshold_remove - start_iter_threshold_remove
        num_points = self.num_points_per_layer[layer_idx]
        removal_rate_per_step = removal_rate / int(removal_iteration_num / self.GSP_interval)

        if iter < end_iter_threshold_remove:
            GSP_num = int(removal_rate_per_step * self.initial_num_points_per_layer[layer_idx])
        elif iter == end_iter_threshold_remove:
            GSP_num = num_points - self.initial_num_points_per_layer[layer_idx]
        # print(num_points, self.initial_num_points_per_layer[layer_idx], GSP_num)

        if GSP_num == 0:
            return

        def safe_norm(param, dim=1):
            if param.grad is not None:
                return param.grad.norm(dim=dim).detach()
            else:
                return torch.zeros(param.shape[0], device=param.device)

        grad_xyz = safe_norm(self._xyz_layers[layer_idx], dim=1)
        grad_cholesky = safe_norm(self._cholesky_layers[layer_idx], dim=1)
        grad_features_dc = safe_norm(self._features_dc_layers[layer_idx], dim=1)
        grad_rgb_W = safe_norm(self._rgb_W_layers[layer_idx], dim=1)

        grad_total = grad_xyz + grad_cholesky + grad_features_dc + grad_rgb_W


        # grad = self._xyz_layers[layer_idx].grad
        # if grad is None:
        #     print(f"[Warning] No grad found for layer {layer_idx} at iter {iter}")
        #     return
        # grad_norm = grad.norm(dim=1).detach()
        
        _, sorted_indices = torch.sort(grad_total)
        remove_indices = sorted_indices[:GSP_num]

        keep_indices = torch.ones(num_points, dtype=torch.bool, device=grad_total.device)
        keep_indices[remove_indices] = False
        keep_indices_cpu = keep_indices.cpu()

        with torch.no_grad():
            self._xyz_layers[layer_idx] = nn.Parameter(self._xyz_layers[layer_idx][keep_indices])
            self._cholesky_layers[layer_idx] = nn.Parameter(self._cholesky_layers[layer_idx][keep_indices])
            self._features_dc_layers[layer_idx] = nn.Parameter(self._features_dc_layers[layer_idx][keep_indices])
            self._rgb_W_layers[layer_idx] = nn.Parameter(self._rgb_W_layers[layer_idx][keep_indices])
            self._opacity_layers[layer_idx] = self._opacity_layers[layer_idx][keep_indices_cpu]
            self._gscenter_cholesky_layers[layer_idx] = self._gscenter_cholesky_layers[layer_idx][keep_indices_cpu]
            self._gscenter_features_dc_layers[layer_idx] = self._gscenter_features_dc_layers[layer_idx][keep_indices_cpu]
            self.num_points_per_layer[layer_idx] = self._xyz_layers[layer_idx].shape[0]
            self.cumulative_points = np.cumsum(self.num_points_per_layer).tolist()
            self.total_points = sum(self.num_points_per_layer)

        self.update_optimizer()

    # def pruning_control_grad(self, iter, layer_idx, removal_rate, start_iter_threshold_remove, end_iter_threshold_remove):
    #     if iter > end_iter_threshold_remove or iter < start_iter_threshold_remove:
    #         return

    #     removal_iteration_num = end_iter_threshold_remove - start_iter_threshold_remove
    #     num_points = self.num_points_per_layer[layer_idx]
    #     removal_rate_per_step = removal_rate / int(removal_iteration_num / self.GSP_interval)

    #     if iter < end_iter_threshold_remove:
    #         GSP_num = int(removal_rate_per_step * self.initial_num_points_per_layer[layer_idx])
    #     elif iter == end_iter_threshold_remove:
    #         GSP_num = num_points - self.initial_num_points_per_layer[layer_idx]
    #     print(num_points, self.initial_num_points_per_layer[layer_idx], GSP_num)

    #     if GSP_num == 0:
    #         return

    #     grad = self._xyz_layers[layer_idx].grad
    #     if grad is None:
    #         print(f"[Warning] No grad found for layer {layer_idx} at iter {iter}")
    #         return
    #     grad_norm = grad.norm(dim=1).detach()
    #     _, sorted_indices = torch.sort(grad_norm)
    #     remove_indices = sorted_indices[:GSP_num]

    #     keep_indices = torch.ones(num_points, dtype=torch.bool, device=grad_norm.device)
    #     keep_indices[remove_indices] = False
    #     keep_indices_cpu = keep_indices.cpu()

    #     with torch.no_grad():
    #         self._xyz_layers[layer_idx] = nn.Parameter(self._xyz_layers[layer_idx][keep_indices])
    #         self._cholesky_layers[layer_idx] = nn.Parameter(self._cholesky_layers[layer_idx][keep_indices])
    #         self._features_dc_layers[layer_idx] = nn.Parameter(self._features_dc_layers[layer_idx][keep_indices])
    #         self._rgb_W_layers[layer_idx] = nn.Parameter(self._rgb_W_layers[layer_idx][keep_indices])
    #         self._opacity_layers[layer_idx] = self._opacity_layers[layer_idx][keep_indices_cpu]
    #         self._gscenter_cholesky_layers[layer_idx] = self._gscenter_cholesky_layers[layer_idx][keep_indices_cpu]
    #         self._gscenter_features_dc_layers[layer_idx] = self._gscenter_features_dc_layers[layer_idx][keep_indices_cpu]
    #         self.num_points_per_layer[layer_idx] = self._xyz_layers[layer_idx].shape[0]
    #         self.cumulative_points = np.cumsum(self.num_points_per_layer).tolist()
    #         self.total_points = sum(self.num_points_per_layer)

    #     self.update_optimizer()

    
    
    
    
    def pruning_control(self, iter, layer_idx, removal_rate, start_iter_threshold_remove, end_iter_threshold_remove):
        if iter >= end_iter_threshold_remove or iter <start_iter_threshold_remove:
            return
        removal_iteration_num = end_iter_threshold_remove-start_iter_threshold_remove
        num_points = self.num_points_per_layer[layer_idx]
        removal_rate_per_step = removal_rate / int(removal_iteration_num / self.GSP_interval)
        if iter < end_iter_threshold_remove:
            GSP_num = int(removal_rate_per_step * self.initial_num_points_per_layer[layer_idx])
        rgb_W = self._rgb_W_layers[layer_idx]
        rgb_weight = torch.norm(rgb_W, dim=1)
        _, sorted_indices = torch.sort(rgb_weight)
        remove_indices = sorted_indices[:GSP_num]
        keep_indices = torch.ones(num_points, dtype=torch.bool, device=rgb_W.device)
        keep_indices[remove_indices] = False
        keep_indices_cpu = keep_indices.cpu()
        with torch.no_grad():
            self._xyz_layers[layer_idx] = nn.Parameter(self._xyz_layers[layer_idx][keep_indices])
            self._cholesky_layers[layer_idx] = nn.Parameter(self._cholesky_layers[layer_idx][keep_indices])
            self._features_dc_layers[layer_idx] = nn.Parameter(self._features_dc_layers[layer_idx][keep_indices])
            self._rgb_W_layers[layer_idx] = nn.Parameter(rgb_W[keep_indices])
            self._opacity_layers[layer_idx] = self._opacity_layers[layer_idx][keep_indices_cpu]
            self._point_ids_layers[layer_idx] = self._point_ids_layers[layer_idx][keep_indices_cpu]
            self._gscenter_cholesky_layers[layer_idx] = self._gscenter_cholesky_layers[layer_idx][keep_indices_cpu]
            self._gscenter_features_dc_layers[layer_idx] = self._gscenter_features_dc_layers[layer_idx][keep_indices_cpu]
            self.num_points_per_layer[layer_idx] = self._xyz_layers[layer_idx].shape[0]
            self.cumulative_points = np.cumsum(self.num_points_per_layer).tolist()
            self.total_points = sum(self.num_points_per_layer)
            
        self.update_optimizer()  

    def adding_control(self, layer_idx, removal_rate):
        GSA_num = int(self.initial_num_points_per_layer[layer_idx] * removal_rate)
        if GSA_num == 0:
            return
        new_xyz = torch.atanh(2 * (torch.rand(GSA_num, 2) - 0.5)).to(self._xyz_layers[layer_idx].device)
        new_cholesky = torch.rand(GSA_num, 3).to(self._xyz_layers[layer_idx].device)
        new_features_dc = torch.rand(GSA_num, 3).to(self._xyz_layers[layer_idx].device)
        new_rgb_W = 0.01 * torch.ones(GSA_num, 1).to(self._xyz_layers[layer_idx].device)
        new_ids = torch.arange(self._next_point_id_layers[layer_idx].item(), self._next_point_id_layers[layer_idx].item() + GSA_num, dtype=torch.long, device=self._point_ids_layers[layer_idx].device)
        self._point_ids_layers[layer_idx] = torch.cat((self._point_ids_layers[layer_idx], new_ids), dim=0)
        self._next_point_id_layers[layer_idx] = torch.tensor(self._next_point_id_layers[layer_idx].item() + GSA_num, dtype=torch.long, device=self._point_ids_layers[layer_idx].device)
        self._xyz_layers[layer_idx] = nn.Parameter(torch.cat([self._xyz_layers[layer_idx], new_xyz], dim=0))
        self._cholesky_layers[layer_idx] = nn.Parameter(torch.cat([self._cholesky_layers[layer_idx], new_cholesky], dim=0))
        self._features_dc_layers[layer_idx] = nn.Parameter(torch.cat([self._features_dc_layers[layer_idx], new_features_dc], dim=0))
        self._rgb_W_layers[layer_idx] = nn.Parameter(torch.cat([self._rgb_W_layers[layer_idx], new_rgb_W], dim=0))
        self._opacity_layers[layer_idx] = torch.cat([self._opacity_layers[layer_idx], torch.ones(GSA_num, 1).to(self._opacity_layers[layer_idx].device)], dim=0)
        self._gscenter_cholesky_layers[layer_idx] = torch.cat([self._gscenter_cholesky_layers[layer_idx], torch.ones(GSA_num, 3).to(self._gscenter_cholesky_layers[layer_idx].device)], dim=0)
        self._gscenter_features_dc_layers[layer_idx] = torch.cat([self._gscenter_features_dc_layers[layer_idx], torch.ones(GSA_num, 3).to(self._gscenter_features_dc_layers[layer_idx].device)], dim=0)
        self.num_points_per_layer[layer_idx] = self._xyz_layers[layer_idx].shape[0]
        self.cumulative_points = np.cumsum(self.num_points_per_layer).tolist()
        self.total_points = sum(self.num_points_per_layer)
        self.update_optimizer()
    
    def train_iter(self, gt_images, iter, return_all_grad=False):
        # if self.is_GSA and self.is_GSP:
        #     if iter == 1:
        #         self.adding_control(1, self.removal_rate)
        #     elif 500 <= iter <= 1000 and iter % self.GSP_interval == 0:
        #         self.pruning_control(iter, 1, self.removal_rate, 500, 1000)
        # elif self.is_GSP:
        #     if iter % self.GSP_interval == 0:
        #         self.pruning_control(iter, 1, 0, 4000)
        # if self.is_GR and iter ==1:
        #     for k in range(1, self.layer_number):
                # self.adding_control(k, self.removal_rate)

        if self.is_GSA and self.is_GSP:
            if iter == 1:
                for k in range(0, self.layer_number):
                    if k==0:
                        self.adding_control(k, self.removal_rate)
                    else:
                        self.adding_control(k, self.removal_rate*2)
            elif 500 <= iter <= 1000 and iter % self.GSP_interval == 0:
                for k in range(0, self.layer_number):
                    if k==0:
                        self.pruning_control(iter, k, self.removal_rate, 500, 1000)
                    else:
                        self.pruning_control(iter, k, self.removal_rate*2, 500, 1000)
        elif self.is_GSP:
            if iter % self.GSP_interval == 0:
                for k in range(0, self.layer_number):
                    if k==0:
                        self.pruning_control(iter, k, self.removal_rate, 0, 4000)
                    else:
                        self.pruning_control(iter, k, self.removal_rate*2, 0, 4000)
        if self.is_GR and iter ==1:
            for k in range(1, self.layer_number):
                self.adding_control(k, self.removal_rate)

        if self.layer_number > 1:
            final_idx = self.layer_number - 1
            if self.train_strategy == "sequential":
                layer_id = (iter - 1) % (self.layer_number - 1)
            else:
                layer_id = torch.randint(0, self.layer_number - 1, (1,)).item()
            img_random = self.forward_layer(layer_id)["render"]
            img_final = self.forward_layer(final_idx)["render"]
            loss_random = loss_fn(img_random.squeeze(0), gt_images[layer_id].squeeze(0), self.loss_type, lambda_value=0)
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0)
            loss = loss_random + loss_final
        else:
            final_idx = self.layer_number - 1
            img_final = self.forward_layer(final_idx)["render"]
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0)
            loss = loss_final

        loss.backward()

        if 6500 <= iter <=7000 and self.is_GR:
            if iter % self.GSP_interval == 0:
                for k in range(1, self.layer_number):
                    self.pruning_control_grad(iter, k, self.removal_rate, 6500, 7000)

        all_grad_norms = None
        gradient_image=None
        if return_all_grad:
            all_grad_norms = []
            for layer in self._xyz_layers:
                grad = layer.grad
                if grad is not None:
                    all_grad_norms.append(grad.norm(dim=1).detach().cpu())
                else:
                    all_grad_norms.append(torch.zeros(layer.shape[0]))
            all_grad_norms = torch.cat(all_grad_norms, dim=0).numpy()  # shape: [total_points]
            if iter ==1 or iter %1000==0:
                gradient_image = self.render_delta_gradient()

        with torch.no_grad():
            grads = [p.grad for p in self.parameters() if p.grad is not None]
            avg_grad = sum([g.abs().mean().detach().cpu().item() for g in grads]) / len(grads) if grads else 0.0
            mse_loss = F.mse_loss(img_final, gt_images[-1])
            psnr = 10 * math.log10(1.0 / mse_loss.item())

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        if return_all_grad:
            return loss_final, loss, psnr, img_final, avg_grad, all_grad_norms, gradient_image
        else:
            return loss_final, loss, psnr, img_final, avg_grad

    
    def pre_train_iter(self, gt_images,iter):
        if self.layer_number>1:
            final_idx = self.layer_number - 1
            if self.train_strategy == "sequential":
                layer_id=(iter-1)%(self.layer_number - 1)
            else:
                layer_id = torch.randint(0, self.layer_number - 1, (1,)).item()  # 不选最后一层
            img_random = self.forward_layer(layer_id)["render"]
            img_final = self.forward_layer(final_idx)["render"]
            loss_random = loss_fn(img_random.squeeze(0), gt_images[layer_id].squeeze(0), self.loss_type, lambda_value=0)
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0)
            loss = loss_random + loss_final
        else:
            final_idx = self.layer_number - 1
            img_final = self.forward_layer(final_idx)["render"]
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0)
            loss = loss_final
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss
    
    def render(self,gt_images):
        rendered_images = []
        psnr_list = []
        ms_ssim_list = []
        for idx in range (0,self.layer_number):
            out = self.forward_layer(idx)["render"].float()
            rendered_image = torch.clamp(out, 0, 1)
            ds_rendered_image = torch.nn.functional.interpolate(rendered_image,
                                            scale_factor=self.scaling_factor[idx],
                                            mode='area')
            ds_gt_images = torch.nn.functional.interpolate(gt_images[idx],
                                            scale_factor=self.scaling_factor[idx],
                                            mode='area')
            mse_loss = F.mse_loss(ds_rendered_image, ds_gt_images.float())
            psnr = 10 * math.log10(1.0 / mse_loss.item())
            ms_ssim_value = ms_ssim(ds_rendered_image, ds_gt_images.float(), data_range=1, size_average=True).item()
            rendered_images.append(ds_rendered_image)
            psnr_list.append(psnr)
            ms_ssim_list.append(ms_ssim_value)
        return rendered_images, psnr_list, ms_ssim_list
    
    
    def render_gs_center(self):
        rendered_images = []
        for idx in range (0,self.layer_number):
            out = self.forward_layer_gs_center(idx)["render"].float()
            rendered_image = torch.clamp(out, 0, 1)
            rendered_images.append(rendered_image)
        return rendered_images
    
    def render_delta_gs_center(self):
        rendered_images = []
        for idx in range (0,self.layer_number):
            out = self.forward_layer_delta_gs_center(idx)["render"].float()
            rendered_image = torch.clamp(out, 0, 1)
            rendered_images.append(rendered_image)
        return rendered_images
    
    def render_delta(self):
        rendered_images = []
        for idx in range (0,self.layer_number):
            out = self.forward_layer_delta(idx)["render"].float()
            rendered_image = torch.clamp(out, 0, 1)
            rendered_images.append(rendered_image)
        return rendered_images
    
    def render_delta_gradient(self):
        rendered_images = []
        transform = transforms.ToPILImage()
        for idx in range (0,self.layer_number):
            out = self.forward_layer_delta_gradient(idx)["render"].float()
            rendered_image = torch.clamp(out, 0, 1)
            rendered_image = transform(rendered_image.squeeze(0))
            rendered_images.append(rendered_image)
        return rendered_images


        




    