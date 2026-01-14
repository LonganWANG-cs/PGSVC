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
        self.loss_type = kwargs["loss_type"]
        self.device = kwargs["device"]
        self.lr = kwargs["lr"]
        self.train_strategy = kwargs["training_strategy"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        self._xyz_layers = nn.ParameterList([
            nn.Parameter(torch.atanh(2 * (torch.rand(n, 2) - 0.5))) for n in self.num_points_per_layer
        ])
        self._cholesky_layers = nn.ParameterList([
            nn.Parameter(torch.rand(n, 3)) for n in self.num_points_per_layer
        ])
        self._features_dc_layers = nn.ParameterList([
            nn.Parameter(torch.rand(n, 3)) for n in self.num_points_per_layer
        ])
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

        # per-layer quantizers
        self.xyz_quantizers = [FakeQuantizationHalf.apply for _ in range(self.layer_number)]
        self.features_dc_quantizers = nn.ModuleList([
            VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5)
            for _ in range(self.layer_number)
        ])
        self.cholesky_quantizers = nn.ModuleList([
            UniformQuantizer(signed=False, bits=6, learned=True, num_channels=3)
            for _ in range(self.layer_number)
        ])
        self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

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


    def forward_layer_quantize(self, idx):
        means_list = []
        cholesky_list = []
        colors_list = []
        opacity_list = []

        l_vqm = 0
        l_vqs = 0
        l_vqr = 0
        l_vqc = 0

        for i in range(idx + 1):
            xyz_i = self._xyz_layers[i].to(self.device)
            means_i = self.xyz_quantizers[i](xyz_i)
            means_list.append(means_i)

            cholesky_i = self._cholesky_layers[i].to(self.device)
            cholesky_q_i, l_vqs_i, _ = self.cholesky_quantizers[i](cholesky_i)
            cholesky_q_i = cholesky_q_i + self.cholesky_bound
            cholesky_list.append(cholesky_q_i)
            l_vqs = l_vqs + l_vqs_i

            features_i = self._features_dc_layers[i].to(self.device)
            colors_i, l_vqc_i, _ = self.features_dc_quantizers[i](features_i)
            colors_list.append(colors_i)
            l_vqc = l_vqc + l_vqc_i

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
        # out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss}
    
    # def forward_layer_quantize(self, idx):
    #     xyz = torch.cat([self._xyz_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
    #     cholesky = torch.cat([self._cholesky_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
    #     features = torch.cat([self._features_dc_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
    #     opacity = torch.cat([self._opacity_layers[i] for i in range(idx + 1)], dim=0).to(self.device)
    #     means = torch.tanh(self.xyz_quantizer(xyz))
    #     l_vqm = 0
    #     cholesky_elements, l_vqs, _ = self.cholesky_quantizer(cholesky)
    #     cholesky_elements = cholesky_elements + self.cholesky_bound
    #     l_vqr = 0
    #     colors, l_vqc, _ = self.features_dc_quantizer(features)
    #     xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
    #         means, cholesky_elements, self.H, self.W, self.tile_bounds
    #     )
    #     out_img = rasterize_gaussians_sum(
    #         xys, depths, radii, conics, num_tiles_hit,
    #         colors, opacity, self.H, self.W,
    #         self.BLOCK_H, self.BLOCK_W,
    #         background=self.background, return_alpha=False
    #     )
    #     # out_img = torch.clamp(out_img, 0, 1)
    #     out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
    #     vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
    #     return {"render": out_img, "vq_loss": vq_loss}

    
    
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
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    
    def update_optimizer(self):
        self.optimizer = Adan(self.parameters(), lr=self.lr)

    
    def train_iter(self, gt_images, iter):
        if self.layer_number > 1:
            final_idx = self.layer_number - 1
            if self.train_strategy == "sequential":
                layer_id = (iter - 1) % (self.layer_number - 1)
            else:
                layer_id = torch.randint(0, self.layer_number - 1, (1,)).item()
            render_pkg_randowm = self.forward_layer_quantize(layer_id)
            render_pkg_final = self.forward_layer_quantize(final_idx)
            img_random = render_pkg_randowm["render"]
            img_final = render_pkg_final["render"]
            loss_random = loss_fn(img_random.squeeze(0), gt_images[layer_id].squeeze(0), self.loss_type, lambda_value=0) + render_pkg_randowm["vq_loss"]
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0) + render_pkg_final["vq_loss"]
            loss = loss_random + loss_final
        else:
            final_idx = self.layer_number - 1
            render_pkg_final = self.forward_layer_quantize(final_idx)
            img_final = render_pkg_final["render"]
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0) + render_pkg_final["vq_loss"]
            loss = loss_final
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(img_final, gt_images[-1])
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss_final, loss, psnr, img_final

    def render(self,gt_images):
        rendered_images = []
        psnr_list = []
        ms_ssim_list = []
        for idx in range (0,self.layer_number):
            out = self.forward_layer_quantize(idx)["render"].float()
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
    


    @torch.no_grad()
    def compute_bits_for_layer(self, i: int):
        m_bit=0
        s_bit=0
        c_bit=0

        m_bit = 16 * self.num_points_per_layer[i] * 2

        cholesky_i = self._cholesky_layers[i].to(self.device)
        _, _, s_bit = self.cholesky_quantizers[i](cholesky_i)

        features_i = self._features_dc_layers[i].to(self.device)
        _, _, c_bit = self.features_dc_quantizers[i](features_i)

        total_bit = int(m_bit + s_bit + c_bit)
        return total_bit


    
    # def compute_bits_for_layer(self, i: int):
    #     device = self.device
    #     xyz_i = self._xyz_layers[i].to(device)
    #     num_points = xyz_i.shape[0]
    #     m_bit = 16 * num_points * 2
    #     cholesky_i = self._cholesky_layers[i].to(device)
    #     _, _, s_bit = self.cholesky_quantizer(cholesky_i)
    #     r_bit = 0
    #     features_dc_i = self._features_dc_layers[i].to(device)
    #     _, _, c_bit = self.features_dc_quantizer(features_dc_i)
    #     total_bit = int(m_bit + s_bit + r_bit + c_bit)
    #     return total_bit

    def record_bits_up_to(self, layer_number: int):
        bits_list = []
        for i in range(layer_number):
            total_bit = self.compute_bits_for_layer(i)
            bits_list.append(total_bit)
        return bits_list


    @torch.no_grad()
    def get_dequantized_Gaussians(self):
        means_list = []
        cholesky_list = []
        colors_list = []
        for i in range(self.layer_number):
            xyz_i = self._xyz_layers[i].to(self.device)
            means_i = xyz_i.half().float()
            means_list.append(means_i)

            cholesky_i = self._cholesky_layers[i].to(self.device)
            cholesky_q_i, _, _ = self.cholesky_quantizers[i](cholesky_i)
            # cholesky_q_i = cholesky_q_i + self.cholesky_bound
            cholesky_list.append(cholesky_q_i)

            features_i = self._features_dc_layers[i].to(self.device)
            colors_i, _, _ = self.features_dc_quantizers[i](features_i)
            colors_list.append(colors_i)

        return means_list, cholesky_list, colors_list

    @torch.no_grad()
    def get_quantized_Model(self):
        """Return quantized Gaussians per layer together with quantizer parameters.

        Returns
        -------
        encoded_layers: list[dict]
            Per-layer quantized Gaussian parameters (indices / quantized values).
        quantizers_state: dict
            State dicts of all learnable quantizers for later reconstruction.
        """
        encoded_layers = []

        for i in range(self.layer_number):
            layer_dict = {}


            xyz_i = self._xyz_layers[i].to(self.device)
            features_i = self._features_dc_layers[i].to(self.device)
            cholesky_i = self._cholesky_layers[i].to(self.device)

            # whole layer as delta Gaussians
            q_xyz = xyz_i.half()
            quant_cholesky, _ = self.cholesky_quantizers[i].compress(cholesky_i)
            _, feat_idx = self.features_dc_quantizers[i].compress(features_i)

            layer_dict["xyz"] = q_xyz.cpu()
            layer_dict["cholesky"] = quant_cholesky.cpu()
            layer_dict["features_index"] = feat_idx.cpu()

            encoded_layers.append(layer_dict)

        quantizers_state = {
            # "xyz_quantizers": [q.state_dict() for q in self.xyz_quantizers], no state to save
            # xyz_quantizers_I are functional (FakeQuantizationHalf.apply), no state to save
            "features_dc_quantizers": [q.state_dict() for q in self.features_dc_quantizers],
            "cholesky_quantizers": [q.state_dict() for q in self.cholesky_quantizers],
            "is_k_frame": True,
        }

        return encoded_layers, quantizers_state

    




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
        self.loss_type = kwargs["loss_type"]
        self.device = kwargs["device"]
        self.lr = kwargs["lr"]
        self.train_strategy = kwargs["training_strategy"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) 
        self._xyz_layers = nn.ParameterList([
            nn.Parameter(torch.atanh(2 * (torch.rand(n, 2) - 0.5))) for n in self.num_points_per_layer
        ])
        self._cholesky_layers = nn.ParameterList([
            nn.Parameter(torch.rand(n, 3)) for n in self.num_points_per_layer
        ])
        self._features_dc_layers = nn.ParameterList([
            nn.Parameter(torch.rand(n, 3)) for n in self.num_points_per_layer
        ])
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

        self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

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
    
    def forward_layer_quantize(self, idx):
        means_list = []
        cholesky_list = []
        colors_list = []
        opacity_list = []

        l_vqm = 0
        l_vqs = 0
        l_vqr = 0
        l_vqc = 0

        for i in range(idx + 1):
            if self.add_num_Gaussians[i] > 0:
                num_delta_Gaussians = self.num_delta_Gaussians[i]

                xyz_i = self._xyz_layers[i].to(self.device)
                p_xyz_i = self._p_xyz_layers[i].to(self.device)
                means_i_front = self.xyz_quantizers[i](xyz_i[:num_delta_Gaussians])
                means_i_back = self.xyz_quantizers_I[i](xyz_i[num_delta_Gaussians:])
                means_i = torch.cat([means_i_front, means_i_back], dim=0) + p_xyz_i
                means_list.append(means_i)

                cholesky_i = self._cholesky_layers[i].to(self.device)
                p_cholesky_i = self._p_cholesky_layers[i].to(self.device)
                cholesky_q_i_front, l_vqs_i_front, _ = self.cholesky_quantizers[i](cholesky_i[:num_delta_Gaussians])
                cholesky_q_i_back, l_vqs_i_back, _ = self.cholesky_quantizers_I[i](cholesky_i[num_delta_Gaussians:])
                cholesky_q_i = torch.cat([cholesky_q_i_front, cholesky_q_i_back], dim=0) + p_cholesky_i + self.cholesky_bound
                cholesky_list.append(cholesky_q_i)
                l_vqs = l_vqs + l_vqs_i_front + l_vqs_i_back

                features_i = self._features_dc_layers[i].to(self.device)
                p_features_i = self._p_features_dc_layers[i].to(self.device)
                colors_i_front, l_vqc_i_front, _ = self.features_dc_quantizers[i](features_i[:num_delta_Gaussians])
                colors_i_back, l_vqc_i_back, _ = self.features_dc_quantizers_I[i](features_i[num_delta_Gaussians:])
                colors_i = torch.cat([colors_i_front, colors_i_back], dim=0) + p_features_i
                colors_list.append(colors_i)
                l_vqc = l_vqc + l_vqc_i_front + l_vqc_i_back
            else:
                xyz_i = self._xyz_layers[i].to(self.device)
                p_xyz_i = self._p_xyz_layers[i].to(self.device)
                means_i = self.xyz_quantizers[i](xyz_i) + p_xyz_i
                means_list.append(means_i)

                cholesky_i = self._cholesky_layers[i].to(self.device)
                p_cholesky_i = self._p_cholesky_layers[i].to(self.device)
                cholesky_q_i, l_vqs_i, _ = self.cholesky_quantizers[i](cholesky_i)
                cholesky_q_i = cholesky_q_i + p_cholesky_i + self.cholesky_bound
                cholesky_list.append(cholesky_q_i)
                l_vqs = l_vqs + l_vqs_i

                features_i = self._features_dc_layers[i].to(self.device)
                p_features_i = self._p_features_dc_layers[i].to(self.device)
                colors_i, l_vqc_i, _ = self.features_dc_quantizers[i](features_i)
                colors_i = colors_i + p_features_i
                colors_list.append(colors_i)
                l_vqc = l_vqc + l_vqc_i

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
        # out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc
        return {"render": out_img, "vq_loss": vq_loss}

    
    
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
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}
    
    
    def update_optimizer(self):
        self.optimizer = Adan(self.parameters(), lr=self.lr)

    
    def train_iter(self, gt_images, iter):
        if self.layer_number > 1:
            final_idx = self.layer_number - 1
            if self.train_strategy == "sequential":
                layer_id = (iter - 1) % (self.layer_number - 1)
            else:
                layer_id = torch.randint(0, self.layer_number - 1, (1,)).item()
            render_pkg_randowm = self.forward_layer_quantize(layer_id)
            render_pkg_final = self.forward_layer_quantize(final_idx)
            img_random = render_pkg_randowm["render"]
            img_final = render_pkg_final["render"]
            loss_random = loss_fn(img_random.squeeze(0), gt_images[layer_id].squeeze(0), self.loss_type, lambda_value=0) + render_pkg_randowm["vq_loss"]
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0) + render_pkg_final["vq_loss"]
            loss = loss_random + loss_final
        else:
            final_idx = self.layer_number - 1
            render_pkg_final = self.forward_layer_quantize(final_idx)
            img_final = render_pkg_final["render"]
            loss_final = loss_fn(img_final.squeeze(0), gt_images[-1].squeeze(0), self.loss_type, lambda_value=0) + render_pkg_final["vq_loss"]
            loss = loss_final
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(img_final, gt_images[-1])
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        return loss_final, loss, psnr, img_final

    def render(self,gt_images):
        rendered_images = []
        psnr_list = []
        ms_ssim_list = []
        for idx in range (0,self.layer_number):
            out = self.forward_layer_quantize(idx)["render"].float()
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
    


    @torch.no_grad()
    def compute_bits_for_layer(self, i: int):
        m_bit=0
        s_bit=0
        c_bit=0

        num_delta_Gaussians = self.num_delta_Gaussians[i]
        if self.add_num_Gaussians[i]>0:
            
            m_bit = 12 * num_delta_Gaussians * 2 + 16 * self.add_num_Gaussians[i]*2


            cholesky_i = self._cholesky_layers[i].to(self.device)
            _, _, s_bit_i_front = self.cholesky_quantizers[i](cholesky_i[:num_delta_Gaussians])
            _, _, s_bit_i_back = self.cholesky_quantizers_I[i](cholesky_i[num_delta_Gaussians:])
            s_bit = s_bit_i_front + s_bit_i_back


            features_i = self._features_dc_layers[i].to(self.device)
            _, _, c_bit_i_front = self.features_dc_quantizers[i](features_i[:num_delta_Gaussians])
            _, _, c_bit_i_back = self.features_dc_quantizers_I[i](features_i[num_delta_Gaussians:])
            c_bit = c_bit_i_front+c_bit_i_back

        else:
            m_bit = 12 * num_delta_Gaussians * 2


            cholesky_i = self._cholesky_layers[i].to(self.device)
            _, _, s_bit = self.cholesky_quantizers[i](cholesky_i)


            features_i = self._features_dc_layers[i].to(self.device)
            _, _, c_bit = self.features_dc_quantizers[i](features_i)

        total_bit = int(m_bit + s_bit + c_bit)
        return total_bit

    def record_bits_up_to(self, layer_number: int):
        bits_list = []
        for i in range(layer_number):
            total_bit = self.compute_bits_for_layer(i)
            bits_list.append(total_bit)
        return bits_list

    @torch.no_grad()
    def get_dequantized_Gaussians(self):
        """Dequantize all layers using current per-layer quantizers and residuals."""
        means_list = []
        cholesky_list = []
        colors_list = []
        for i in range(self.layer_number):
            if self.add_num_Gaussians[i] > 0:
                num_delta_Gaussians = self.num_delta_Gaussians[i]

                xyz_i = self._xyz_layers[i].to(self.device)
                p_xyz_i = self._p_xyz_layers[i].to(self.device)
                means_i_front = self.xyz_quantizers[i](xyz_i[:num_delta_Gaussians])
                means_i_back = xyz_i[num_delta_Gaussians:].half().float()
                means_i = torch.cat([means_i_front, means_i_back], dim=0) + p_xyz_i
                means_list.append(means_i)

                cholesky_i = self._cholesky_layers[i].to(self.device)
                p_cholesky_i = self._p_cholesky_layers[i].to(self.device)
                cholesky_q_i_front, _, _ = self.cholesky_quantizers[i](cholesky_i[:num_delta_Gaussians])
                cholesky_q_i_back, _, _ = self.cholesky_quantizers_I[i](cholesky_i[num_delta_Gaussians:])
                # cholesky_q_i = torch.cat([cholesky_q_i_front, cholesky_q_i_back], dim=0) + p_cholesky_i + self.cholesky_bound
                cholesky_q_i = torch.cat([cholesky_q_i_front, cholesky_q_i_back], dim=0) + p_cholesky_i
                cholesky_list.append(cholesky_q_i)

                features_i = self._features_dc_layers[i].to(self.device)
                p_features_i = self._p_features_dc_layers[i].to(self.device)
                colors_i_front, _, _ = self.features_dc_quantizers[i](features_i[:num_delta_Gaussians])
                colors_i_back, _, _ = self.features_dc_quantizers_I[i](features_i[num_delta_Gaussians:])
                colors_i = torch.cat([colors_i_front, colors_i_back], dim=0) + p_features_i
                colors_list.append(colors_i)
            else:
                xyz_i = self._xyz_layers[i].to(self.device)
                p_xyz_i = self._p_xyz_layers[i].to(self.device)
                means_i = self.xyz_quantizers[i](xyz_i) + p_xyz_i
                means_list.append(means_i)

                cholesky_i = self._cholesky_layers[i].to(self.device)
                p_cholesky_i = self._p_cholesky_layers[i].to(self.device)
                cholesky_q_i, _, _ = self.cholesky_quantizers[i](cholesky_i)
                # cholesky_q_i = cholesky_q_i + p_cholesky_i + self.cholesky_bound
                cholesky_q_i = cholesky_q_i + p_cholesky_i
                cholesky_list.append(cholesky_q_i)

                features_i = self._features_dc_layers[i].to(self.device)
                p_features_i = self._p_features_dc_layers[i].to(self.device)
                colors_i, _, _ = self.features_dc_quantizers[i](features_i)
                colors_i = colors_i + p_features_i
                colors_list.append(colors_i)

        return means_list, cholesky_list, colors_list

    @torch.no_grad()
    def get_quantized_Model(self):
        """Return quantized Gaussians per layer together with quantizer parameters.

        Returns
        -------
        encoded_layers: list[dict]
            Per-layer quantized Gaussian parameters (indices / quantized values).
        quantizers_state: dict
            State dicts of all learnable quantizers for later reconstruction.
        """
        encoded_layers = []

        for i in range(self.layer_number):
            layer_dict = {}
            num_delta_Gaussians = self.num_delta_Gaussians[i]

            xyz_i = self._xyz_layers[i].to(self.device)
            features_i = self._features_dc_layers[i].to(self.device)
            cholesky_i = self._cholesky_layers[i].to(self.device)

            if self.add_num_Gaussians[i] > 0:
                # xyz: front uses learned FP12 quantizer (has encode), back uses half-precision fake quantizer
                q_front = self.xyz_quantizers[i].encode(xyz_i[:num_delta_Gaussians])
                q_back = self.xyz_quantizers_I[i](xyz_i[num_delta_Gaussians:]).half()

                # cholesky
                quant_cholesky_front, _ = self.cholesky_quantizers[i].compress(cholesky_i[:num_delta_Gaussians])
                quant_cholesky_back, _ = self.cholesky_quantizers_I[i].compress(cholesky_i[num_delta_Gaussians:])
                quant_cholesky = torch.cat([quant_cholesky_front, quant_cholesky_back], dim=0)

                # features
                _, feat_idx_front = self.features_dc_quantizers[i].compress(features_i[:num_delta_Gaussians])
                _, feat_idx_back = self.features_dc_quantizers_I[i].compress(features_i[num_delta_Gaussians:])
                feat_idx = torch.cat([feat_idx_front, feat_idx_back], dim=0)

                layer_dict["xyz_front"] = q_front.cpu()
                layer_dict["xyz_back"] = q_back.cpu()
                layer_dict["cholesky"] = quant_cholesky.cpu()
                layer_dict["features_index"] = feat_idx.cpu()
            else:
                # whole layer as delta Gaussians
                q_xyz = self.xyz_quantizers[i].encode(xyz_i)
                quant_cholesky, _ = self.cholesky_quantizers[i].compress(cholesky_i)
                _, feat_idx = self.features_dc_quantizers[i].compress(features_i)

                layer_dict["xyz"] = q_xyz.cpu()
                layer_dict["cholesky"] = quant_cholesky.cpu()
                layer_dict["features_index"] = feat_idx.cpu()

            encoded_layers.append(layer_dict)

        quantizers_state = {
            # "xyz_quantizers": [q.state_dict() for q in self.xyz_quantizers], no state to save
            # xyz_quantizers_I are functional (FakeQuantizationHalf.apply), no state to save
            "features_dc_quantizers": [q.state_dict() for q in self.features_dc_quantizers],
            "features_dc_quantizers_I": [q.state_dict() for q in self.features_dc_quantizers_I],
            "cholesky_quantizers": [q.state_dict() for q in self.cholesky_quantizers],
            "cholesky_quantizers_I": [q.state_dict() for q in self.cholesky_quantizers_I],
            "is_k_frame": False,
        }

        return encoded_layers, quantizers_state
