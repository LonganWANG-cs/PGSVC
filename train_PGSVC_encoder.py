import time
from pathlib import Path
import argparse
import numpy as np
import torch
import sys
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import torch.nn as nn
from PGSVC_encoder import GaussianSplats, GaussianSplats_delta
import copy
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
        overfitted_model=None, # Pretrained model to initialize the training.
        previous_model=None, # Previous frame's trained model
        args = None, # Arguments containing various training configurations.
    ):
        self.frame_num=frame_num
        self.num_points_per_layer = num_points_per_layer
        self.scaling_factor = scaling_factor
        self.result_save_path = result_save_path
        self.iterations = iterations
        self.model_name =args.model_name
        self.show_gaussians = args.show_gaussians # Whether to visualize the Gaussian centers.
        self.save_imgs = args.save_imgs # Whether to save the rendered images.
        BLOCK_H, BLOCK_W = 16, 16 
        self.device = torch.device("cuda:0")
        self.gt_image = image_to_tensor(frame).to(self.device)
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.gt_images = [] # List to store downsampled ground truth images for each layer.
        self.previous_model=previous_model
        self.overfitted_model=overfitted_model
        self.current_point_ids_layers = None
        for s in self.scaling_factor:
            # downsampled = torch.nn.functional.interpolate(self.gt_image, scale_factor=s, mode='area')
            # resized_back = torch.nn.functional.interpolate(downsampled, size=(self.H, self.W), mode='bilinear', align_corners=False)
            # self.gt_images.append(resized_back.to(self.device))   
            self.gt_images.append(self.gt_image.to(self.device))  
        self.result_save_path = Path(result_save_path) / str(self.frame_num)
        if self.save_imgs:
            self.logwriter = LogWriter(self.result_save_path)
        if previous_model is not None and overfitted_model is not None:
            self.gaussian_model = GaussianSplats_delta(
                num_points_per_layer = self.num_points_per_layer,
                scaling_factor = self.scaling_factor,
                H = self.H,
                W = self.W,
                BLOCK_H = BLOCK_H,
                BLOCK_W = BLOCK_W,
                loss_type=args.loss_type,
                device=self.device,
                lr=args.lr,
                training_strategy = args.training_strategy).to(self.device) 
            # xyz_layers = overfitted_model['_xyz_layers']
            # cholesky_layers = overfitted_model['_cholesky_layers']
            # features_dc_layers = overfitted_model['_features_dc_layers']
            if '_point_ids_layers' not in overfitted_model:
                raise KeyError("P-frame encoding requires overfitted_model['_point_ids_layers'] for alignment")
            current_point_ids_layers = overfitted_model['_point_ids_layers']
            self.current_point_ids_layers = current_point_ids_layers
            previous_point_ids_layers = previous_model['_point_ids_layers']
            for i, _ in enumerate(self.num_points_per_layer):
                cur_len = overfitted_model['_xyz_layers'][i].shape[0]
                cur_ids = current_point_ids_layers[i].detach().cpu().numpy()
                prev_ids = previous_point_ids_layers[i].detach().cpu().numpy()
                prev_index = {int(k): idx for idx, k in enumerate(prev_ids.tolist())}
                def _count_unmatched(prev_index_map, cur_ids_list):
                    unmatched = 0
                    for pid in cur_ids_list:
                        if prev_index_map.get(int(pid), -1) == -1:
                            unmatched += 1
                    return unmatched
                def align_param(name):
                    cur = overfitted_model[name][i].to(self.device)
                    dtype = cur.dtype
                    shape_tail = cur.shape[1:]
                    aligned_prev = torch.zeros((cur_len,)+shape_tail, dtype=dtype, device=self.device)
                    for ci, pid in enumerate(cur_ids):
                        pi = prev_index.get(int(pid), -1)
                        if pi != -1:
                            aligned_prev[ci] = previous_model[name][i][pi].to(self.device)
                    # Reorder: matched pairs first, unmatched to the end (consistent order for both tensors)
                    matched_idx = []
                    unmatched_idx = []
                    for ci, pid in enumerate(cur_ids):
                        if prev_index.get(int(pid), -1) != -1:
                            matched_idx.append(ci)
                        else:
                            unmatched_idx.append(ci)
                    if len(matched_idx) == 0 or len(unmatched_idx) == 0:
                        return cur, aligned_prev
                    order = torch.as_tensor(matched_idx + unmatched_idx, device=self.device, dtype=torch.long)
                    cur = cur.index_select(0, order)
                    aligned_prev = aligned_prev.index_select(0, order)
                    return cur, aligned_prev
                cur_xyz, prev_xyz = align_param('_xyz_layers')
                cur_ch, prev_ch = align_param('_cholesky_layers')
                cur_feat, prev_feat = align_param('_features_dc_layers')
                self.gaussian_model.add_num_Gaussians[i] = _count_unmatched(prev_index, cur_ids)
                
                self.gaussian_model._xyz_layers[i] = nn.Parameter((cur_xyz-prev_xyz).to(self.device))
                self.gaussian_model._cholesky_layers[i] = nn.Parameter((cur_ch-prev_ch).to(self.device))
                self.gaussian_model._features_dc_layers[i] = nn.Parameter((cur_feat-prev_feat).to(self.device))
                self.gaussian_model._p_xyz_layers[i] = prev_xyz.to(self.device)
                self.gaussian_model._p_cholesky_layers[i] = prev_ch.to(self.device)
                self.gaussian_model._p_features_dc_layers[i] = prev_feat.to(self.device)
            self.gaussian_model.update_optimizer()

        elif overfitted_model is not None:
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
                training_strategy = args.training_strategy).to(self.device) 
            # For K-frames, also persist point ids for decoder alignment. If not provided by the
            # checkpoint, generate deterministic per-layer ids [0..N-1].
            if '_point_ids_layers' in overfitted_model:
                self.current_point_ids_layers = overfitted_model['_point_ids_layers']
            else:
                self.current_point_ids_layers = [
                    torch.arange(t.shape[0], dtype=torch.long)
                    for t in overfitted_model['_xyz_layers']
                ]
            xyz_layers = overfitted_model['_xyz_layers']
            cholesky_layers = overfitted_model['_cholesky_layers']
            features_dc_layers = overfitted_model['_features_dc_layers']
            assert len(xyz_layers) == len(self.gaussian_model._xyz_layers), "Layer count mismatch"
            for i, _ in enumerate(self.num_points_per_layer):
                self.gaussian_model._xyz_layers[i] = nn.Parameter(xyz_layers[i].to(self.device))
                self.gaussian_model._cholesky_layers[i] = nn.Parameter(cholesky_layers[i].to(self.device))
                self.gaussian_model._features_dc_layers[i] = nn.Parameter(features_dc_layers[i].to(self.device))
            self.gaussian_model.update_optimizer()

    def train(self): 
        img_list=[]
        progress_bar = tqdm(range(1, int(self.iterations)+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        for iter in range(1, self.iterations+1):
            loss_final, _, psnr, _ = self.gaussian_model.train_iter(self.gt_images, iter)
            if best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
            with torch.no_grad():
                if iter % 25 == 0 or iter==1:
                    progress_bar.set_postfix({f"Loss":f"{loss_final.item():.{7}f}", "PSNR":f"{psnr:.{4}f}", "Number of Gaussians":len(self.gaussian_model.get_xyz)})
                    progress_bar.update(25) 
        progress_bar.close()
        self.gaussian_model.load_state_dict(best_model_dict)
        psnr_value, ms_ssim_value, bpp, rendered_images, rendered_images_gs, decoded_Gmodel, encoded_Gmodel = self.test()
        return psnr_value, ms_ssim_value, bpp, rendered_images, rendered_images_gs, decoded_Gmodel, encoded_Gmodel

    def test(self):
        self.gaussian_model.eval()
        with torch.no_grad():
            rendered_images, psnr_list, ms_ssim_list = self.gaussian_model.render(self.gt_images)
            rendered_images_gs = rendered_images 
            transform = transforms.ToPILImage()
            transformed_rendered_images = []
            transformed_rendered_images_gs = []
            if self.save_imgs:
                for idx in range(len(self.num_points_per_layer)):
                    img = transform(rendered_images[idx].squeeze(0))
                    transformed_rendered_images.append(img)
                    name = f"{self.frame_num}_fitting_Layer_{idx+1}.png"
                    img.save(str(self.result_save_path / name))
                if self.show_gaussians:
                    rendered_images_gs = self.gaussian_model.render_gs_center()
                    for idx in range(len(self.num_points_per_layer)):
                        gscenter_img = transform(rendered_images_gs[idx].squeeze(0))
                        transformed_rendered_images_gs.append(gscenter_img)
                        gscenter_name = f"{self.frame_num}gscenter_fitting_Layer_{idx+1}.png"
                        gscenter_img.save(str(self.result_save_path / gscenter_name))
                else:
                    transformed_rendered_images_gs = transformed_rendered_images
            else:
                for idx in range(len(self.num_points_per_layer)):
                    transformed_rendered_images.append(transform(rendered_images[idx].squeeze(0)))
                if self.show_gaussians:
                    rendered_images_gs = self.gaussian_model.render_gs_center()
                    for idx in range(len(self.num_points_per_layer)):
                        transformed_rendered_images_gs.append(transform(rendered_images_gs[idx].squeeze(0)))
                else:
                    transformed_rendered_images_gs = transformed_rendered_images
            num_layers = len(self.num_points_per_layer)
            bits_list = self.gaussian_model.record_bits_up_to(num_layers)  # 每层 total_bit
            print(f"Bits per layer: {bits_list}")
            bits_list_cumsum = np.cumsum(bits_list).tolist()
            print(f"bits_list_cumsum: {bits_list_cumsum}")
            denom_per_layer = [
                img.size[0] * img.size[1] for img in transformed_rendered_images
            ]
            print(f"denom_per_layer: {denom_per_layer}")
            bpp = [
                bits_list_cumsum[i] / max(1, denom_per_layer[i]) for i in range(num_layers)
            ]
            decoded_Gmodel = {}
            means_list, cholesky_list, colors_list = self.gaussian_model.get_dequantized_Gaussians()
            decoded_Gmodel = {
                '_xyz_layers': [p.detach().cpu() for p in means_list],
                '_cholesky_layers': [p.detach().cpu() for p in cholesky_list],
                '_features_dc_layers': [p.detach().cpu() for p in colors_list],
            }

            # quantized representation (encoded Gaussians + quantizer params)
            encoded_layers, quantizers_state = self.gaussian_model.get_quantized_Model()
            encoded_Gmodel = {
                'encoded_layers': encoded_layers,
                'quantizers_state': quantizers_state,
            }

            # also save current point ids (for delta / predictive frames)
            if hasattr(self, 'current_point_ids_layers') and self.current_point_ids_layers is not None:
                encoded_Gmodel['_point_ids_layers'] = [p.detach().cpu() for p in self.current_point_ids_layers]

            # if previous_model is not None and overfitted_model is not None:

            return psnr_list, ms_ssim_list, bpp, transformed_rendered_images, transformed_rendered_images_gs, decoded_Gmodel, encoded_Gmodel
        
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-d", "--dataset", type=str, default='/home/e/e1344641/data/kodak/', help="Training dataset path")
    parser.add_argument("--model_path", type=str, default='/home/model/gmodels_state_dict.pth', help="Testing model path")
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: %(default)s)")
    parser.add_argument("--is_downsample", action="store_true", help="Downsample in the lower layers")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument("--save_video", action="store_true", help="Save video")
    parser.add_argument("--show_gaussians", action="store_true", help="Show the distribution of gaussian centers")
    parser.add_argument("--fps", type=int, default=120, help="number of frames per second (default: %(default)s)")
    parser.add_argument("--video_length", type=int, default=24, help="number of images to train in the dataset (default: %(default)s)") 
    parser.add_argument("--width", type=int, default=1920, help="width (default: %(default)s)")
    parser.add_argument("--height", type=int, default=1080, help="height (default: %(default)s)")
    parser.add_argument("--I_interval", type=int, default=5, help="I_frame_interval (default: %(default)s)")
    args = parser.parse_args(argv)
    args = parser.parse_args(argv)
    return args


def main(argv):
    
    psnr_list = []
    msssim_list = []
    bpp_list = []
    frames = []
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
    gmodel_save_path = Path(f"./checkpoints/{args.savdir_m}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.layer_number}")
    result_save_path = Path(f"./checkpoints/{args.savdir}/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.layer_number}")
    gmodel_save_path.mkdir(parents=True, exist_ok=True)
    result_save_path.mkdir(parents=True, exist_ok=True)
    logwriter = LogWriter(result_save_path)
    
    video_frames = process_yuv_video(args.dataset, args.width, args.height)
    frame_length, start=min(args.video_length, len(video_frames)),0
    
    output_path_K_frames = Path(f"./checkpoints/{args.savdir}/{args.data_name}/K_frames.txt")
    if output_path_K_frames.exists():
        with open(output_path_K_frames, "r") as f:
            K_frames = [int(line.strip()) for line in f.readlines()]
    else:
        K_frames = [1]
        k_frame_num = 0
    I_interval = args.I_interval
    periodic_k = list(range(1, frame_length + 1, I_interval))
    K_frames = sorted(set([k for k in K_frames if 1 <= k <= frame_length] + periodic_k))
    print(K_frames)
    rendered_images_list = []
    encoded_gmodels_state_dict = {}
    Gmodel=None
    K_Gmodel=None
    checkpoints = torch.load(args.model_path,map_location=torch.device("cuda:0"))
    for i in range(start, start + frame_length):
        frame_num=i+1
        modelid=f"frame_{frame_num}"
        overfitted_Model = checkpoints[modelid]
        layers = [tensor.shape[0] for tensor in overfitted_Model['_xyz_layers']]
        layer_number = len(layers)
        cumulative_layers = np.cumsum(layers).tolist()
        if args.is_downsample:
            scaling_factor = [ (layer / cumulative_layers[-1]) ** 0.5 for layer in cumulative_layers ]
        else:
            scaling_factor = [1.0 for _ in cumulative_layers]
        if frame_num in K_frames:
            k_frame_num=frame_num
            print(f"modelid:frame_{k_frame_num};")
            trainer = SimpleTrainer2d(
                        frame = video_frames[i],
                        frame_num=frame_num,
                        iterations = args.iterations,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        overfitted_model=overfitted_Model,
                        args=args
                    )
        else:
            # previous_modelid = f"frame_{k_frame_num}"
            previous_modelid = f"frame_{i}"
            # if (K_Gmodel is not None) and ('_point_ids_layers' in checkpoints[previous_modelid]):
            #     K_Gmodel = dict(K_Gmodel)
            #     K_Gmodel['_point_ids_layers'] = checkpoints[previous_modelid]['_point_ids_layers']
            if (Gmodel is not None) and ('_point_ids_layers' in checkpoints[previous_modelid]):
                Gmodel = dict(Gmodel)
                Gmodel['_point_ids_layers'] = checkpoints[previous_modelid]['_point_ids_layers']
            print(f"modelid:{modelid}; p_modelid:{previous_modelid}")
            # trainer = SimpleTrainer2d(
            #             frame = video_frames[frame_num-1],
            #             frame_num=frame_num,
            #             iterations = args.iterations,
            #             num_points_per_layer = layers,
            #             result_save_path=result_save_path,
            #             scaling_factor=scaling_factor,
            #             overfitted_model=overfitted_Model,
            #             previous_model=K_Gmodel,
            #             args=args
            #         )
            trainer = SimpleTrainer2d(
                        frame = video_frames[frame_num-1],
                        frame_num=frame_num,
                        iterations = args.iterations,
                        num_points_per_layer = layers,
                        result_save_path=result_save_path,
                        scaling_factor=scaling_factor,
                        overfitted_model=overfitted_Model,
                        previous_model=Gmodel,
                        args=args
                    )
        torch.cuda.empty_cache()
        start_time = time.time()
        print(f"Training frame {frame_num}")
        psnr, ms_ssim, bpp, rendered_images, _, Gmodel, encoded_Gmodel = trainer.train()
        if frame_num in K_frames:
            K_Gmodel=Gmodel
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training finished in {total_time:.2f} seconds.")
        for idx in range(0,layer_number):
            logwriter.write("{}: Layer{}_Image: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Bpp:{:.4f}, Encoding:{:.2f}".format(
                frame_num, idx+1, trainer.W, trainer.H, psnr[idx], ms_ssim[idx], bpp[idx], total_time))
        frames.append(frame_num)
        psnr_list.append(psnr[-1])
        msssim_list.append(ms_ssim[-1])
        bpp_list.append(bpp[-1])
        rendered_images_list.append(rendered_images)
        encoded_gmodels_state_dict[f"frame_{frame_num}"] = encoded_Gmodel
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_msssim = sum(msssim_list) / len(msssim_list)
    avg_bpp = sum(bpp_list) / len(bpp_list)
    frames.append("Average")
    psnr_list.append(avg_psnr)
    msssim_list.append(avg_msssim)
    bpp_list.append(avg_bpp)
    torch.save(encoded_gmodels_state_dict, gmodel_save_path / "encoded_gmodels_state_dict.pth")
    print("=== Progressive Training Done ===")
    print(f"Average BPP: {avg_bpp:.4f},Average PSNR: {avg_psnr:.4f}, Average MS-SSIM: {avg_msssim:.6f}")
    if args.save_video:
        generate_progressive_video(result_save_path, rendered_images_list,"rendered_video.mp4")

if __name__ == "__main__":
    main(sys.argv[1:])


