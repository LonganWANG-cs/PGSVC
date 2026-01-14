#Trace Gaussians
import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from sklearn.mixture import GaussianMixture
import pandas as pd
import torch.nn as nn

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        frame, # The input video frame ground truth.
        frame_num, # The frame id in the video sequence.
        iterations, # Number of training iterations.
        num_points_per_layer: list, # Number of Gaussians in each layer.
        result_save_path: Path, # Path to save the results.
        scaling_factor: list, # Scaling factors for downsampling the frame in each layer.
        predicted_model=None, # Pretrained model to initialize the training.
        is_GSP = None, # Whether to use Gaussian Splatting Pruning
        is_GSA = None, # Whether to use Gaussian Splatting Argumentation
        is_GR = None, # Whether to use Gradient Removal
        args = None, # Arguments containing various training configurations.
    ):
        self.frame_num=frame_num
        self.num_points_per_layer = num_points_per_layer
        self.scaling_factor = scaling_factor
        self.result_save_path = result_save_path
        self.iterations = iterations
        self.model_name =args.model_name
        self.show_gaussians = args.show_gaussians # Whether to visualize the Gaussian centers.
        self.trace_gaussians = args.trace_gaussians # Whether to trace the Gaussian training process.
        self.save_grad = args.save_grad # Whether to save the gradient information.
        self.save_imgs = args.save_imgs # Whether to save the rendered images.
        self.save_delta_imgs=args.save_delta_imgs # Whether to save the delta images.
        BLOCK_H, BLOCK_W = 16, 16 
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(frame).to(self.device)
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.gt_images = [] # List to store downsampled ground truth images for each layer.
        for s in self.scaling_factor:
            downsampled = torch.nn.functional.interpolate(self.gt_image, scale_factor=s, mode='area')
            resized_back = torch.nn.functional.interpolate(downsampled, size=(self.H, self.W), mode='bilinear', align_corners=False)
            self.gt_images.append(resized_back.to(self.device))     
        self.result_save_path = Path(result_save_path) / str(self.frame_num)
        if self.save_imgs:
            self.logwriter = LogWriter(self.result_save_path)
        if self.model_name == "PrGS":
            from PGSVC_overfitting import GaussianSplats
            self.gaussian_model = GaussianSplats(
                num_points_per_layer = self.num_points_per_layer,
                scaling_factor = self.scaling_factor,
                H = self.H,
                W = self.W,
                BLOCK_H = BLOCK_H,
                BLOCK_W = BLOCK_W,
                loss_type=args.loss_type,
                device=self.device,
                lr=args.lr,
                removal_rate = args.removal_rate,
                training_strategy = args.training_strategy,
                is_GSP = is_GSP,
                is_GSA = is_GSA,
                is_GR = is_GR,
                is_frozen = args.is_frozen).to(self.device) 
        if predicted_model is not None:
            xyz_layers = predicted_model['_xyz_layers']
            cholesky_layers = predicted_model['_cholesky_layers']
            features_dc_layers = predicted_model['_features_dc_layers']
            point_ids_layers = predicted_model['_point_ids_layers']
            next_point_id_layers = predicted_model['_next_point_id_layers']
            assert len(xyz_layers) == len(self.gaussian_model._xyz_layers), "Layer count mismatch"
            for i, _ in enumerate(self.num_points_per_layer):
                self.gaussian_model._xyz_layers[i] = nn.Parameter(xyz_layers[i].to(self.device))
                self.gaussian_model._cholesky_layers[i] = nn.Parameter(cholesky_layers[i].to(self.device))
                self.gaussian_model._features_dc_layers[i] = nn.Parameter(features_dc_layers[i].to(self.device))
                self.gaussian_model._point_ids_layers[i] = point_ids_layers[i].to(self.device)
                self.gaussian_model._next_point_id_layers[i] = next_point_id_layers[i].to(self.device)
            self.gaussian_model.update_optimizer()

    def train(self): 
        img_list=[]
        loss_list=[]
        grad_list=[] 
        iter_list=[] 
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        self.gaussian_model.train()
        early_stopping = EarlyStopping(patience=100, min_delta=1e-7)
        # early_stopping = EarlyStopping(patience=500, min_delta=1e-9)
        stabel_control=8000
        for iter in range(1, self.iterations+1):
            # if iter==1 or iter %1000==0:
            #     self.Frame_Render(iter)
            return_all_grad=False
            if self.save_grad and (iter==1 or iter %1000==0):
                return_all_grad=True
            result = self.gaussian_model.train_iter(self.gt_images, iter, return_all_grad=return_all_grad)
            if return_all_grad:
                loss_final, loss, psnr, image, avg_grad, all_grad_norms, gradient_image = result
                self.save_gaussian_gradient_info_from_result(
                    all_grad_norms, filename=f"Gaussian_grad_iter{iter}.csv"
                )
                if gradient_image is not None:
                    for idx in range(len(self.num_points_per_layer)):
                        gradient_delta_img = gradient_image[idx]
                        gradient_delta_name = f"{self.frame_num}_gradient_delta_fitting_Layer_{idx+1}_{iter}.png"
                        gradient_delta_img.save(str(self.result_save_path / gradient_delta_name))
            else:
                loss_final, loss, psnr, image, avg_grad = result
            iter_list.append(iter)
            loss_list.append(loss_final.item() if torch.is_tensor(loss_final) else loss_final)
            grad_list.append(avg_grad)
            with torch.no_grad():
                if iter % 25 == 0 or iter==1:
                    if self.trace_gaussians:
                        transform = transforms.ToPILImage()
                        img = transform(image.float().squeeze(0))
                        img_list.append(img)
                    progress_bar.set_postfix({f"Loss":f"{loss_final.item():.{7}f}", "PSNR":f"{psnr:.{4}f}", "Number of Gaussians":len(self.gaussian_model.get_xyz)})
                    progress_bar.update(25) 
                stabel_control=stabel_control-1
                if stabel_control<0 and early_stopping(loss_final.item()):
                    break
        progress_bar.close()
        psnr_value, ms_ssim_value, rendered_images, rendered_images_gs,rendered_images_delta = self.test()
        if self.trace_gaussians:
            generate_video(image_list=img_list, result_save_path=self.result_save_path, fps=48,origin=True,layer_number=len(self.num_points_per_layer))
        if self.save_grad:
            log_df = pd.DataFrame({
                'iter': iter_list,
                'loss': loss_list,
                'avg_grad': grad_list,
            })
            csv_log_path = self.result_save_path / "train_curve.csv"
            log_df.to_csv(csv_log_path, index=False)
        Gmodel = {}
        if self.gaussian_model.is_frozen:
            rgb_W_layers = [getattr(self.gaussian_model, f'_rgb_W_{i}') for i in range(len(self.gaussian_model._features_dc_layers))]
        else:
            rgb_W_layers = list(self.gaussian_model._rgb_W_layers)
        Gmodel = {
            '_xyz_layers': [p.detach().cpu() for p in self.gaussian_model._xyz_layers],
            '_cholesky_layers': [p.detach().cpu() for p in self.gaussian_model._cholesky_layers],
            '_features_dc_layers': [
                (f * w).detach().cpu() for f, w in zip(self.gaussian_model._features_dc_layers, rgb_W_layers)
            ],
            '_point_ids_layers': [p.detach().cpu() for p in self.gaussian_model._point_ids_layers],
            '_next_point_id_layers': [p.detach().cpu() for p in self.gaussian_model._next_point_id_layers],
        }
        total_num = len(self.gaussian_model.get_xyz)
        layer_wise_gaussian_counts = self.gaussian_model.num_points_per_layer
        print(f"Number of Gaussians: {total_num}")
        print(f"Number of Gaussians per layer: {layer_wise_gaussian_counts}")
        return psnr_value, ms_ssim_value, rendered_images, rendered_images_gs, rendered_images_delta, Gmodel, total_num, layer_wise_gaussian_counts

    def pre_train(self): 
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        self.gaussian_model.train()
        for iter in range(1, self.iterations+1):
            loss = self.gaussian_model.pre_train_iter(self.gt_images,iter)
            with torch.no_grad():
                if iter % 25 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}"})
                    progress_bar.update(25) 
        progress_bar.close()
        if self.gaussian_model.is_frozen:
            rgb_W_layers = [getattr(self.gaussian_model, f'_rgb_W_{i}') for i in range(len(self.gaussian_model._features_dc_layers))]
        else:
            rgb_W_layers = list(self.gaussian_model._rgb_W_layers)
        Gmodel = {}
        Gmodel = {
            '_xyz_layers': [p.detach().cpu() for p in self.gaussian_model._xyz_layers],
            '_cholesky_layers': [p.detach().cpu() for p in self.gaussian_model._cholesky_layers],
            '_features_dc_layers': [
                (f * w).detach().cpu() for f, w in zip(self.gaussian_model._features_dc_layers, rgb_W_layers)
            ],
            '_point_ids_layers': [p.detach().cpu() for p in self.gaussian_model._point_ids_layers],
            '_next_point_id_layers': [p.detach().cpu() for p in self.gaussian_model._next_point_id_layers],
        }
        return Gmodel, loss

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            rendered_images, psnr_list, ms_ssim_list = self.gaussian_model.render(self.gt_images)
            rendered_images_gs = rendered_images 
            rendered_images_delta_gs = rendered_images
            rendered_images_delta = rendered_images
            transform = transforms.ToPILImage()
            transformed_rendered_images = []
            transformed_rendered_images_gs = []
            transformed_rendered_images_delta_gs = []
            transformed_rendered_images_delta = []
            if self.save_imgs:
                for idx in range(len(self.num_points_per_layer)):
                    img = transform(rendered_images[idx].squeeze(0))
                    transformed_rendered_images.append(img)
                    name = f"{self.frame_num}_fitting_Layer_{idx+1}.png"
                    img.save(str(self.result_save_path / name))
                if self.show_gaussians:
                    rendered_images_gs = self.gaussian_model.render_gs_center()
                    rendered_images_delta_gs = self.gaussian_model.render_delta_gs_center()
                    for idx in range(len(self.num_points_per_layer)):
                        gscenter_img = transform(rendered_images_gs[idx].squeeze(0))
                        delta_gscenter_img = transform(rendered_images_delta_gs[idx].squeeze(0))
                        transformed_rendered_images_gs.append(gscenter_img)
                        transformed_rendered_images_delta_gs.append(delta_gscenter_img)
                        gscenter_name = f"{self.frame_num}gscenter_fitting_Layer_{idx+1}.png"
                        delta_gscenter_name = f"{self.frame_num}delta_gscenter_fitting_Layer_{idx+1}.png"
                        gscenter_img.save(str(self.result_save_path / gscenter_name))
                        delta_gscenter_img.save(str(self.result_save_path / delta_gscenter_name))
                else:
                    transformed_rendered_images_gs = transformed_rendered_images
                if self.save_delta_imgs:
                    rendered_images_delta = self.gaussian_model.render_delta()
                    for idx in range(len(self.num_points_per_layer)):
                        delta_img = transform(rendered_images_delta[idx].squeeze(0))
                        transformed_rendered_images_delta.append(delta_img)
                        delta_name = f"{self.frame_num}delta_fitting_Layer_{idx+1}.png"
                        delta_img.save(str(self.result_save_path / delta_name))
                else:
                    transformed_rendered_images_delta = transformed_rendered_images
            else:
                for idx in range(len(self.num_points_per_layer)):
                    transformed_rendered_images.append(transform(rendered_images[idx].squeeze(0)))
                if self.show_gaussians:
                    rendered_images_gs = self.gaussian_model.render_gs_center()
                    for idx in range(len(self.num_points_per_layer)):
                        transformed_rendered_images_gs.append(transform(rendered_images_gs[idx].squeeze(0)))
                else:
                    transformed_rendered_images_gs = transformed_rendered_images
                if self.save_delta_imgs:
                    rendered_images_delta = self.gaussian_model.render_delta()
                    for idx in range(len(self.num_points_per_layer)):
                        transformed_rendered_images_delta.append(transform(rendered_images_delta[idx].squeeze(0)))
                else:
                    transformed_rendered_images_delta = transformed_rendered_images
            return psnr_list, ms_ssim_list, transformed_rendered_images, transformed_rendered_images_gs, transformed_rendered_images_delta
        
    def Frame_Render(self, iter):
        self.gaussian_model.eval()
        transform = transforms.ToPILImage()
        with torch.no_grad():
            rendered_images, __, __ = self.gaussian_model.render(self.gt_images)
            rendered_images_delta = rendered_images 
            for idx in range(len(self.num_points_per_layer)):
                img = transform(rendered_images[idx].squeeze(0))
                name = f"{self.frame_num}_fitting_Layer_{idx+1}_{iter}.png"
                img.save(str(self.result_save_path / name))
            rendered_images_delta = self.gaussian_model.render_delta()
            for idx in range(len(self.num_points_per_layer)):
                delta_img = transform(rendered_images_delta[idx].squeeze(0))
                delta_name = f"{self.frame_num}_delta_fitting_Layer_{idx+1}_{iter}.png"
                delta_img.save(str(self.result_save_path / delta_name))
            

    def save_gaussian_gradient_info_from_result(self, grad_array, filename="Gaussian_grad.csv"):
        """Save per-point grad with id and layer info, given flat grad_array from train_iter()."""
        grad_values = grad_array.tolist()
        layer_ids = []
        point_ids = []
        counter = 0
        for i, layer in enumerate(self.gaussian_model._xyz_layers):
            num_points = layer.shape[0]
            layer_ids.extend([i] * num_points)
            point_ids.extend(range(counter, counter + num_points))
            counter += num_points
        df = pd.DataFrame({
            'id': point_ids,
            'layer': layer_ids,
            'grad': grad_values
        })
        grad_csv_path = self.result_save_path / filename
        df.to_csv(grad_csv_path, index=False)





def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-d", "--dataset", type=str, default='/home/e/e1344641/data/kodak/', help="Training dataset path")
    parser.add_argument("--data_name", type=str, default='kodak', help="Training dataset name")
    parser.add_argument("--model_name", type=str, default="PrGS", help="Name of the model")
    parser.add_argument("--savdir", type=str, default="result", help="Path to save results")
    parser.add_argument("--savdir_m", type=str, default="models", help="Path to save models")
    parser.add_argument("--training_strategy", type=str, default="sequential", help="Select random or sequential when taining each layer")
    parser.add_argument("--layer_number", type=int, default=3, help="number of layers (default: %(default)s)") 
    parser.add_argument("--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)")
    parser.add_argument("--num_points", type=int, default=12000, help="2D GS points (default: %(default)s)")
    parser.add_argument("--loss_type", type=str, default='L2', help="Type of Loss")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s)",)
    parser.add_argument("--look_ahead", type=int, default=1)
    parser.add_argument("--removal_rate", type=float, default=0.2, help="Removal rate")
    parser.add_argument("--is_downsample", action="store_true", help="Downsample in the lower layers")
    parser.add_argument("--is_frozen", action="store_true", help="Froze the rgbW")
    parser.add_argument("--is_GSP", action="store_true", help="GSP")
    parser.add_argument("--is_GSA", action="store_true", help="GSA")
    parser.add_argument("--is_GR", action="store_true", help="Gradient removal")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--save_grad", action="store_true", help="Save grad")
    parser.add_argument("--save_video", action="store_true", help="Save video")
    parser.add_argument("--save_delta_imgs", action="store_true", help="Save delta image")
    parser.add_argument("--show_gaussians", action="store_true", help="Show the distribution of gaussian centers")
    parser.add_argument("--trace_gaussians", action="store_true", help="Trace the gaussians' training process")
    parser.add_argument("--fps", type=int, default=120, help="number of frames per second (default: %(default)s)")
    parser.add_argument("--video_length", type=int, default=24, help="number of images to train in the dataset (default: %(default)s)") 
    parser.add_argument("--width", type=int, default=1920, help="width (default: %(default)s)")
    parser.add_argument("--height", type=int, default=1080, help="height (default: %(default)s)")
    args = parser.parse_args(argv)
    return args





def main(argv):
    psnr_list = []
    msssim_list = []
    frames = []
    args = parse_args(argv)
    gmodel_save_path = Path(f"./checkpoints/{args.savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.layer_number}")
    result_save_path = Path(f"./checkpoints/{args.savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.layer_number}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)
    result_save_path.mkdir(parents=True, exist_ok=True)
    logwriter = LogWriter(result_save_path)
    step = args.num_points // args.layer_number
    layers = [step] * (args.layer_number - 1)
    last_layer = args.num_points - step * (args.layer_number - 1)
    layers.append(last_layer)
    cumulative_layers = np.cumsum(layers).tolist()
    if args.is_downsample:
        scaling_factor = [ (layer / cumulative_layers[-1]) ** 0.5 for layer in cumulative_layers ]
    else:
        scaling_factor = [1.0 for _ in cumulative_layers]
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    video_frames = process_yuv_video(args.dataset, args.width, args.height)
    frame_length, start=min(args.video_length, len(video_frames)),0
    rendered_images_list = []
    gmodels_state_dict = {}
    #Dynamic K-frames selection
    output_path_K_frames =  Path(f"./checkpoints/{args.savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.layer_number}/K_frames.txt")
    if output_path_K_frames.exists():
        with open(output_path_K_frames, "r") as f:
            K_frames = [int(line.strip()) for line in f.readlines()]
    else:
        if frame_length>1:
            loss_list=[]
            for i in range(start, start+frame_length):
                frame_num=i+1
                if frame_num ==1:
                    loss_extractor_K = SimpleTrainer2d(
                        frame = video_frames[i+1],
                        frame_num=frame_num + 1,
                        iterations = 500,
                        num_points_per_layer = [5000],
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=None,
                        args=args
                    )
                    Gmodel,_= loss_extractor_K.pre_train()
                    loss=0
                    loss_list.append(0)
                else:
                    loss_extractor_K = SimpleTrainer2d(
                        frame = video_frames[i+1],
                        frame_num=frame_num + 1,
                        iterations = 500,
                        num_points_per_layer = [5000],
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=None,
                        args=args
                    )
                    loss_extractor_P = SimpleTrainer2d(
                        frame = video_frames[i+1],
                        frame_num=frame_num + 1,
                        iterations = 100,
                        num_points_per_layer = [5000],
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=Gmodel,
                        args=args
                    )
                    Gmodel, loss_K = loss_extractor_K.pre_train()
                    _, loss_P = loss_extractor_P.pre_train()
                    loss_list.append(loss_P-loss_K)
            loss_list = np.array([
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in loss_list
            ])
            values_to_normalize = loss_list[1:]
            min_value = np.min(values_to_normalize)
            max_value = np.max(values_to_normalize)
            normalized_loss_list = [loss_list[0]] + [(v - min_value) / (max_value - min_value) for v in values_to_normalize]
            outlier_indices = detect_outliers_mean_diff(normalized_loss_list, window_size=10, threshold=3)
            K_frames = [int(x + 1) for x in outlier_indices]
            K_frames = np.insert(K_frames, 0, 1)
            K_frames = K_frames.astype(int)
            output_path_K_frames = Path(f"./checkpoints/{args.savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.layer_number}/K_frames.txt")
            with open(output_path_K_frames, "w") as f:
                for frame in K_frames:
                    f.write(f"{frame}\n")
            output_path_normalized_loss = Path(f"./checkpoints/{args.savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.layer_number}/loss.txt")
            with open(output_path_normalized_loss, "w") as f:
                for idx, loss in enumerate(normalized_loss_list):
                    f.write(f"{idx+1},{loss:.6f}\n")
        else:
            K_frames=[1]
    print("K-frames:", K_frames)
    # Start training
    Gmodel=None
    for i in range(start, start + frame_length):
        frame_num=i+1
        if args.is_GSP and args.is_GSA:
            if frame_num in K_frames:
                #K-frame inhancement
                #######Look ahead frame limit############
                p_end = None
                for offset in range(1, args.look_ahead + 1):
                    if frame_num + offset in K_frames:
                        p_end = frame_num + offset - 1 
                        break
                if p_end is None:
                    p_end = frame_num + args.look_ahead 
                p_end = min(p_end, start + frame_length)                 
                if p_end != frame_num:
                    #######Training the K-frame first############
                    trainer_K = SimpleTrainer2d(
                            frame = video_frames[i],
                            frame_num=frame_num,
                            iterations = 10000,
                            num_points_per_layer = layers,
                            result_save_path=result_save_path,
                            scaling_factor=scaling_factor,
                            predicted_model=None,
                            is_GSP=True,
                            is_GSA=True,
                            is_GR=args.is_GR,
                            args=args
                        )
                    _, _, _, _, _, Gmodel, _, _ = trainer_K.train()
                    trainer_K = SimpleTrainer2d(
                            frame = video_frames[i],
                            frame_num=frame_num,
                            iterations = args.iterations,
                            num_points_per_layer = layers,
                            result_save_path=result_save_path,
                            scaling_factor=scaling_factor,
                            predicted_model=Gmodel,
                            is_GSP=False,
                            is_GSA=False,
                            is_GR=args.is_GR,
                            args=args
                        )
                    print(f"Look ahead: Training frame {i+1}")
                    _, _, _, _, _, Gmodel, _, _ = trainer_K.train()
                    #######Training the look ahead frames(P-frames)############
                    for p in range(frame_num, p_end):
                        trainer_P= SimpleTrainer2d(
                            frame=video_frames[p],
                            frame_num=p,
                            iterations=10000,
                            num_points_per_layer=layers,
                            result_save_path=result_save_path,
                            scaling_factor=scaling_factor,
                            predicted_model=Gmodel,
                            is_GSP=True,
                            is_GSA=True,
                            is_GR=args.is_GR,
                            args=args
                        )
                        _, _, _, _, _, Gmodel, _, _ = trainer_P.train()
                        trainer_P= SimpleTrainer2d(
                            frame=video_frames[p],
                            frame_num=p,
                            iterations=args.iterations,
                            num_points_per_layer=layers,
                            result_save_path=result_save_path,
                            scaling_factor=scaling_factor,
                            predicted_model=Gmodel,
                            is_GSP=False,
                            is_GSA=False,
                            is_GR=args.is_GR,
                            args=args
                        )
                        print(f"Look ahead: Training frame {p+1}")
                        _, _, _, _, _, Gmodel, _, _ = trainer_P.train()
                    #############Load the look ahead frame pretrained model to retrain the K-frame############
                    trainer = SimpleTrainer2d(
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = 10000,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=Gmodel,
                        is_GSP=True,
                        is_GSA=True,
                        is_GR=args.is_GR,
                        args=args
                    )
                    _, _, _, _, _, Gmodel, _, _ = trainer.train()
                    trainer = SimpleTrainer2d( 
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = args.iterations,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=Gmodel,
                        is_GSP=False,
                        is_GSA=False,
                        is_GR=args.is_GR,
                        args=args
                    )
                else:
                    trainer = SimpleTrainer2d( #No look ahead, just train the K-frame
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = 10000,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=None,
                        is_GSP=True,
                        is_GSA=True,
                        is_GR=args.is_GR,
                        args=args
                    )
                    _, _, _, _, _, Gmodel, _, _ = trainer.train()
                    trainer = SimpleTrainer2d( 
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = args.iterations,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=Gmodel,
                        is_GSP=False,
                        is_GSA=False,
                        is_GR = args.is_GR,
                        args=args
                    )
            else:
                trainer = SimpleTrainer2d( #No look ahead, just train the K-frame
                    frame = video_frames[i],
                    frame_num=frame_num,
                    iterations = 10000,
                    num_points_per_layer = layers,
                    result_save_path=result_save_path,
                    scaling_factor=scaling_factor,
                    predicted_model=Gmodel,
                    is_GSP=True,
                    is_GSA=True,
                    is_GR=args.is_GR,
                    args=args
                )
                _, _, _, _, _, Gmodel, _, _ = trainer.train()
                trainer = SimpleTrainer2d( #Training the P-frame
                    frame = video_frames[i],
                    frame_num=frame_num,
                    iterations = args.iterations,
                    num_points_per_layer = P_layers,
                    result_save_path=result_save_path,
                    scaling_factor=scaling_factor,
                    predicted_model=Gmodel,
                    is_GSP = False,
                    is_GSA = False,
                    is_GR = args.is_GR,
                    args=args
                )
        else:
            if frame_num in K_frames:
                #K-frame inhancement
                p_end = None
                for offset in range(1, args.look_ahead + 1):
                    if frame_num + offset in K_frames:
                        p_end = frame_num + offset - 1 
                        break
                if p_end is None:
                    p_end = frame_num + args.look_ahead 
                p_end = min(p_end, start + frame_length) #Look ahead frame limit
                trainer_K = SimpleTrainer2d(
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = args.iterations,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=None,
                        is_GSP=args.is_GSP,
                        is_GSA=args.is_GSA,
                        is_GR = args.is_GR,
                        args=args
                    )
                if p_end != frame_num:
                    print(f"Look ahead: Training frame {i+1}")
                    _, _, _, _, _, Gmodel, _, _ = trainer_K.train() #Training the K-frame first
                    for p in range(frame_num, p_end):
                        trainer_P= SimpleTrainer2d(
                            frame=video_frames[p],
                            frame_num=p,
                            iterations=args.iterations,
                            num_points_per_layer=layers,
                            result_save_path=result_save_path,
                            scaling_factor=scaling_factor,
                            predicted_model=Gmodel,
                            is_GSP=args.is_GSP,
                            is_GSA=args.is_GSA,
                            is_GR = args.is_GR,
                            args=args
                        )
                        print(f"Look ahead: Training frame {p+1}")
                        _, _, _, _, _, Gmodel, _, _ = trainer_P.train() #Training the look ahead frames(P-frames)
                    trainer = SimpleTrainer2d( #Load the look ahead frame pretrained model to retrain the K-frame
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = args.iterations,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=Gmodel,
                        is_GSP = args.is_GSP,
                        is_GSA = args.is_GSA,
                        is_GR = args.is_GR,
                        args=args
                    )
                else:
                    trainer = SimpleTrainer2d( #No look ahead, just train the K-frame
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = args.iterations,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        predicted_model=None,
                        is_GSP=args.is_GSP,
                        is_GSA=args.is_GSA,
                        is_GR = args.is_GR,
                        args=args
                    )
            else:
                trainer = SimpleTrainer2d( #Training the P-frame
                    frame = video_frames[i],
                    frame_num=frame_num,
                    iterations = args.iterations,
                    num_points_per_layer = P_layers,
                    result_save_path=result_save_path,
                    scaling_factor=scaling_factor,
                    predicted_model=Gmodel,
                    is_GSP = args.is_GSP,
                    is_GSA = args.is_GSA,
                    is_GR = args.is_GR,
                    args=args
                )
        torch.cuda.empty_cache()
        start_time = time.time()
        print(f"Training frame {frame_num}")
        psnr, ms_ssim, rendered_images, _, _, Gmodel, _, P_layers = trainer.train()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training finished in {total_time:.2f} seconds.")
        for idx in range(0,args.layer_number):
            logwriter.write("{}: Layer{}_Image: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Encoding:{:.2f}".format(frame_num, idx+1, trainer.W, trainer.H, psnr[idx], ms_ssim[idx], total_time))
        frames.append(frame_num)
        psnr_list.append(psnr[-1])
        msssim_list.append(ms_ssim[-1])
        rendered_images_list.append(rendered_images)
        gmodels_state_dict[f"frame_{frame_num}"] = Gmodel
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_msssim = sum(msssim_list) / len(msssim_list)
    frames.append("Average")
    psnr_list.append(avg_psnr)
    msssim_list.append(avg_msssim)
    df = pd.DataFrame({
        "Image Name": frames,
        "PSNR": psnr_list,
        "MS-SSIM": msssim_list
    })
    csv_path = result_save_path / "results.csv"
    df.to_csv(csv_path, index=False)
    torch.save(gmodels_state_dict, gmodel_save_path / "gmodels_state_dict.pth")
    print("=== Progressive Training Done ===")
    print(f"Average PSNR: {avg_psnr:.4f}, Average MS-SSIM: {avg_msssim:.6f}")
    if args.save_video:
        generate_progressive_video(result_save_path, rendered_images_list,"rendered_video.mp4")

if __name__ == "__main__":
    main(sys.argv[1:])


