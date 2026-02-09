from gsplat.project_gaussians_2d import project_gaussians_2d
from gsplat.rasterize_sum import rasterize_gaussians_sum
from utils import *
import torch
import torch.nn as nn
import numpy as np
from quantize import *
import math
from optimizer import Adan
from pytorch_msssim import ms_ssim
import time


class GaussianSplats(nn.Module):
    def __init__(self, num_points_per_layer, scaling_factor, **kwargs):
        super().__init__()
        self.num_points_per_layer = num_points_per_layer #Number of added Gaussians in each layer (Delta Gaussians)
        self.cumulative_points = np.cumsum(num_points_per_layer).tolist() #Number of Gaussians in each layer
        self.total_points = self.cumulative_points[-1] #Total number of Gaussians
        self.scaling_factor = scaling_factor #Downsample scale in each layer
        self.layer_number =  len(self.num_points_per_layer) #Number of layers
        self.H, self.W = kwargs["H"], kwargs["W"] #Height and width
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.device = kwargs["device"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        self._xyz_layers = []
        self._cholesky_layers = []
        self._features_dc_layers = []
        for i, n in enumerate(self.num_points_per_layer):
            self.register_buffer(f'_xyz_layers_{i}', torch.atanh(2 * (torch.rand(n, 2) - 0.5)))
            self.register_buffer(f'_cholesky_layers_{i}', torch.rand(n, 3))
            self.register_buffer(f'_features_dc_layers_{i}', torch.rand(n, 3))
            self._xyz_layers.append(getattr(self, f'_xyz_layers_{i}'))
            self._cholesky_layers.append(getattr(self, f'_cholesky_layers_{i}'))
            self._features_dc_layers.append(getattr(self, f'_features_dc_layers_{i}'))
        self._opacity_layers = []
        self._gscenter_cholesky_layers = []
        self._gscenter_features_dc_layers = []
        for i, n in enumerate(self.num_points_per_layer):
            self.register_buffer(f'_opacity_{i}', torch.ones(n, 1))
            self.register_buffer(f'_gscenter_cholesky_{i}', torch.ones(n, 3))
            self.register_buffer(f'_gscenter_features_dc_{i}', torch.ones(n, 3))
            self._opacity_layers.append(getattr(self, f'_opacity_{i}'))
            self._gscenter_cholesky_layers.append(getattr(self, f'_gscenter_cholesky_{i}'))
            self._gscenter_features_dc_layers.append(getattr(self, f'_gscenter_features_dc_{i}'))
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        self.xyz_quantizers = [FakeQuantizationHalf.apply for _ in range(self.layer_number)]
        self.features_dc_quantizers = nn.ModuleList([
            VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5)
            for _ in range(self.layer_number)
        ])
        self.cholesky_quantizers = nn.ModuleList([
            UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)
            for _ in range(self.layer_number)
        ])

    @property
    def get_xyz(self):
        return torch.cat(list(self._xyz_layers), dim=0)
    @property
    def get_features(self):
        return torch.cat(list(self._features_dc_layers), dim=0)
    @property
    def get_cholesky_elements(self):
        return torch.cat(list(self._cholesky_layers), dim=0)
    
    
    def forward_layer(self, idx):
        xyz = torch.cat([torch.tanh(self._xyz_layers[i]) for i in range(idx + 1)], dim=0).to(self.device)
        cholesky = torch.cat([self._cholesky_layers[i] + self.cholesky_bound for i in range(idx + 1)], dim=0).to(self.device)
        features = torch.cat([self._features_dc_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
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
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def forward(self):
        idx = self.layer_number - 1
        return self.forward_layer(idx)["render"]

    
    def render(self,gt_images):
        repeat = 100
        with torch.no_grad():
            self.eval()
            test_start_time = time.time()
            for _ in range(repeat):
                _ = self()
            avg_time = (time.time() - test_start_time) / repeat
        render_fps = (1.0 / avg_time) if avg_time > 0 else 0.0

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
        return rendered_images, psnr_list, ms_ssim_list, render_fps
    

    

class GaussianSplats_delta(nn.Module):
    def __init__(self, num_points_per_layer, scaling_factor, **kwargs):
        super().__init__()
        self.num_points_per_layer = num_points_per_layer #Number of added Gaussians in each layer (Delta Gaussians)
        self.cumulative_points = np.cumsum(num_points_per_layer).tolist() #Number of Gaussians in each layer
        self.total_points = self.cumulative_points[-1] #Total number of Gaussians
        self.scaling_factor = scaling_factor #Downsample scale in each layer
        self.layer_number =  len(self.num_points_per_layer) #Number of layers
        self.H, self.W = kwargs["H"], kwargs["W"] #Height and width
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.device = kwargs["device"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        self._xyz_layers = []
        self._cholesky_layers = []
        self._features_dc_layers = []
        for i, n in enumerate(self.num_points_per_layer):
            self.register_buffer(f'_xyz_layers_{i}', torch.atanh(2 * (torch.rand(n, 2) - 0.5)))
            self.register_buffer(f'_cholesky_layers_{i}', torch.rand(n, 3))
            self.register_buffer(f'_features_dc_layers_{i}', torch.rand(n, 3))
            self._xyz_layers.append(getattr(self, f'_xyz_layers_{i}'))
            self._cholesky_layers.append(getattr(self, f'_cholesky_layers_{i}'))
            self._features_dc_layers.append(getattr(self, f'_features_dc_layers_{i}'))
        self._opacity_layers = []
        self._gscenter_cholesky_layers = []
        self._gscenter_features_dc_layers = []
        self._p_xyz_layers = []
        self._p_cholesky_layers = []
        self._p_features_dc_layers = []
        self.add_num_Gaussians=[]
        for i, n in enumerate(self.num_points_per_layer):
            self.register_buffer(f'_p_xyz_layers_{i}',torch.atanh(2 * (torch.rand(n, 2) - 0.5)))
            self.register_buffer(f'_p_cholesky_layers_{i}', torch.rand(n, 3))
            self.register_buffer(f'_p_features_dc_layers_{i}', torch.rand(n, 3))
            self.register_buffer(f'_opacity_{i}', torch.ones(n, 1))
            self.register_buffer(f'_gscenter_cholesky_{i}', torch.ones(n, 3))
            self.register_buffer(f'_gscenter_features_dc_{i}', torch.ones(n, 3))
            self._p_xyz_layers.append(getattr(self, f'_p_xyz_layers_{i}'))
            self._p_cholesky_layers.append(getattr(self, f'_p_cholesky_layers_{i}'))
            self._p_features_dc_layers.append(getattr(self, f'_p_features_dc_layers_{i}'))
            self._opacity_layers.append(getattr(self, f'_opacity_{i}'))
            self._gscenter_cholesky_layers.append(getattr(self, f'_gscenter_cholesky_{i}'))
            self._gscenter_features_dc_layers.append(getattr(self, f'_gscenter_features_dc_{i}'))
            self.add_num_Gaussians.append(0)
        self.register_buffer('background', torch.ones(3))
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.register_buffer('cholesky_bound', torch.tensor([0.5, 0, 0.5]).view(1, 3))
        
        # per-layer quantizers for delta model
        self.xyz_quantizers_I = [FakeQuantizationHalf.apply for _ in range(self.layer_number)]
        self.features_dc_quantizers_I = nn.ModuleList([
            VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5)
            for _ in range(self.layer_number)
        ])
        self.cholesky_quantizers_I = nn.ModuleList([
            UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)
            for _ in range(self.layer_number)
        ])

        self.xyz_quantizers = nn.ModuleList([
            FakeQuantizationFP12()
            for _ in range(self.layer_number)
        ])
        self.features_dc_quantizers = nn.ModuleList([
            VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5)
            for _ in range(self.layer_number)
        ])
        self.cholesky_quantizers = nn.ModuleList([
            UniformQuantizer(signed=True, bits=6, learned=True, num_channels=3)
            for _ in range(self.layer_number)
        ])


    @property
    def get_xyz(self):
        return torch.cat(list(self._xyz_layers), dim=0)
    @property
    def get_features(self):
        return torch.cat(list(self._features_dc_layers), dim=0)
    @property
    def get_cholesky_elements(self):
        return torch.cat(list(self._cholesky_layers), dim=0)
    @property
    def num_delta_Gaussians(self):
        # elementwise difference per layer: num_points_per_layer[i] - add_num_Gaussians[i]
        return [n - a for n, a in zip(self.num_points_per_layer, self.add_num_Gaussians)]
    
    

    # def forward_layer(self, idx):
    #     xyz = torch.cat([torch.tanh(self._xyz_layers[i]) for i in range(idx + 1)], dim=0).to(self.device)
    #     cholesky = torch.cat([self._cholesky_layers[i] + self.cholesky_bound for i in range(idx + 1)], dim=0).to(self.device)
    #     features = torch.cat([self._features_dc_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
    #     opacity = torch.cat([self._opacity_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
    #     xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
    #         xyz, cholesky, self.H, self.W, self.tile_bounds
    #     )
    #     out_img = rasterize_gaussians_sum(
    #         xys, depths, radii, conics, num_tiles_hit,
    #         features, opacity, self.H, self.W,
    #         self.BLOCK_H, self.BLOCK_W,
    #         background=self.background, return_alpha=False
    #     )
    #     out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
    #     return {"render": out_img}

    def forward_layer(self, idx):
        means_list = []
        cholesky_list = []
        colors_list = []
        opacity_list = []
        for i in range(idx + 1):
            xyz_i = self._xyz_layers[i].to(self.device)+self._p_xyz_layers[i].to(self.device)
            means_list.append(xyz_i)

            cholesky_i = self._cholesky_layers[i].to(self.device)+self._p_cholesky_layers[i].to(self.device)+self.cholesky_bound
            cholesky_list.append(cholesky_i)

            features_i = self._features_dc_layers[i].to(self.device)+self._p_features_dc_layers[i].to(self.device)
            colors_list.append(features_i)

            opacity_i = self._opacity_layers[i].to(self.device)
            opacity_list.append(opacity_i)

        means = torch.tanh(torch.cat(means_list, dim=0)).to(self.device)
        cholesky_elements = torch.cat(cholesky_list, dim=0).to(self.device)
        colors = torch.cat(colors_list, dim=0).to(self.device)
        opacity = torch.cat(opacity_list, dim=0).to(self.device)

        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            means, cholesky_elements, self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            xys, depths, radii, conics, num_tiles_hit,
            colors, opacity, self.H, self.W,
            self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def forward(self):
        idx = self.layer_number - 1
        return self.forward_layer(idx)["render"]
    

    def render(self,gt_images):
        repeat = 100
        with torch.no_grad():
            self.eval()
            test_start_time = time.time()
            for _ in range(repeat):
                _ = self()
            avg_time = (time.time() - test_start_time) / repeat
        render_fps = (1.0 / avg_time) if avg_time > 0 else 0.0

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
        return rendered_images, psnr_list, ms_ssim_list,render_fps
    



