import csv
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

from PGSVC_decoder import GaussianSplats, GaussianSplats_delta

import copy

class SimpleTrainer2d:

    """Trains random 2d gaussians to fit an image."""

    @torch.no_grad()
    def __init__(

        self,

        frame, # The input video frame ground truth.

        frame_num, # The frame id in the video sequence.

        num_points_per_layer: list, # Number of Gaussians in each layer.

        result_save_path: Path, # Path to save the results.

        scaling_factor: list, # Scaling factors for downsampling the frame in each layer.

        quantized_model=None, # Pretrained model to initialize the training.

        previous_model=None, # Previous frame's trained model

        args = None, # Arguments containing various training configurations.

    ):

        self.frame_num=frame_num

        self.num_points_per_layer = num_points_per_layer

        self.scaling_factor = scaling_factor

        self.result_save_path = result_save_path

        self.model_name =args.model_name

        self.save_video = args.save_video
        self.save_imgs = args.save_imgs # Whether to save the rendered images.

        BLOCK_H, BLOCK_W = 16, 16 

        self.device = torch.device("cuda:0")

        self.frame = frame
        self.H, self.W = frame.shape[:2]

        self.result_save_path = Path(result_save_path) / str(self.frame_num)

        if self.save_imgs:

            self.logwriter = LogWriter(self.result_save_path)

        if previous_model is not None and quantized_model is not None:

            self.gaussian_model = GaussianSplats_delta(

                num_points_per_layer = self.num_points_per_layer,

                scaling_factor = self.scaling_factor,

                H = self.H,

                W = self.W,

                BLOCK_H = BLOCK_H,

                BLOCK_W = BLOCK_W,

                device=self.device).to(self.device)  

            checkpoint = quantized_model['encoded_layers']

            checkpoint_quantizer=quantized_model['quantizers_state']

            self.current_point_ids_layers = []
            for i, current_ids in enumerate(quantized_model['_point_ids_layers']):
                cur_len = self.num_points_per_layer[i]
                prev_ids = previous_model['_point_ids_layers'][i].tolist()
                prev_index = {point_id: row for row, point_id in enumerate(prev_ids)}
                matched = []
                unmatched = []
                previous_rows = []
                for row, point_id in enumerate(current_ids.tolist()):
                    if point_id in prev_index:
                        matched.append(row)
                        previous_rows.append(prev_index[point_id])
                    else:
                        unmatched.append(row)

                add_num = len(unmatched)
                self.gaussian_model.add_num_Gaussians[i] = add_num
                order = torch.tensor(matched + unmatched, dtype=torch.long)
                self.current_point_ids_layers.append(current_ids.index_select(0, order))
                previous_rows = torch.tensor(previous_rows, dtype=torch.long)

                # load quantizer parameters

                with torch.no_grad():

                    if isinstance(checkpoint_quantizer, dict):

                        ch_qs = checkpoint_quantizer.get('cholesky_quantizers', None)

                        ch_qs_I = checkpoint_quantizer.get('cholesky_quantizers_I', None)

                        feat_qs = checkpoint_quantizer.get('features_dc_quantizers', None)

                        feat_qs_I = checkpoint_quantizer.get('features_dc_quantizers_I', None)

                        if isinstance(ch_qs, (list, tuple)) and i < len(ch_qs):

                            self.gaussian_model.cholesky_quantizers[i].load_state_dict(ch_qs[i], strict=False)

                        if isinstance(feat_qs, (list, tuple)) and i < len(feat_qs):

                            self.gaussian_model.features_dc_quantizers[i].load_state_dict(feat_qs[i], strict=False)

                        # I-branch quantizers are used for newly added Gaussians

                        if add_num > 0:

                            if isinstance(ch_qs_I, (list, tuple)) and i < len(ch_qs_I):

                                self.gaussian_model.cholesky_quantizers_I[i].load_state_dict(ch_qs_I[i], strict=False)

                            if isinstance(feat_qs_I, (list, tuple)) and i < len(feat_qs_I):

                                self.gaussian_model.features_dc_quantizers_I[i].load_state_dict(feat_qs_I[i], strict=False)



                # decode current layer quantized parameters to tensors

                layer_ckpt = checkpoint[i]

                device = self.device
                q_ch = layer_ckpt['cholesky'].to(device, dtype=torch.float32)
                feat_idx = layer_ckpt['features_index'].to(device, dtype=torch.long)

                if add_num > 0:

                    num_delta_Gaussians = cur_len - add_num

                    xyz_front = self.gaussian_model.xyz_quantizers[i].decode(layer_ckpt['xyz_front'].to(device))

                    xyz_back = layer_ckpt['xyz_back'].to(device).float()

                    cur_xyz_full = torch.zeros((cur_len, 2), device=device, dtype=torch.float32)

                    cur_xyz_full[:num_delta_Gaussians] = xyz_front.to(device).float()

                    cur_xyz_full[num_delta_Gaussians:] = xyz_back



                    q_ch_front = q_ch[:num_delta_Gaussians]

                    q_ch_back = q_ch[num_delta_Gaussians:]

                    feat_idx_front = feat_idx[:num_delta_Gaussians]

                    feat_idx_back = feat_idx[num_delta_Gaussians:]



                    cur_ch_front = self.gaussian_model.cholesky_quantizers[i].decompress(q_ch_front)

                    cur_ch_back = self.gaussian_model.cholesky_quantizers_I[i].decompress(q_ch_back)

                    cur_ch_full = torch.cat([cur_ch_front, cur_ch_back], dim=0)



                    color_front = self.gaussian_model.features_dc_quantizers[i].decompress(feat_idx_front)

                    color_back = self.gaussian_model.features_dc_quantizers_I[i].decompress(feat_idx_back)

                    cur_feat_full = torch.cat([color_front, color_back], dim=0)

                else:

                    cur_xyz_full = self.gaussian_model.xyz_quantizers[i].decode(layer_ckpt['xyz'].to(device)).to(device).float()

                    cur_ch_full = self.gaussian_model.cholesky_quantizers[i].decompress(q_ch)

                    cur_feat_full = self.gaussian_model.features_dc_quantizers[i].decompress(feat_idx)



                # Residuals are stored in matched-first order. Gather the predictors in the same order.
                def align_previous(name, residual):
                    source = previous_model[name][i]
                    aligned = torch.zeros_like(residual)
                    aligned[:len(matched)] = source.index_select(0, previous_rows).to(device)
                    return aligned

                self.gaussian_model._xyz_layers[i] = cur_xyz_full
                self.gaussian_model._cholesky_layers[i] = cur_ch_full
                self.gaussian_model._features_dc_layers[i] = cur_feat_full
                self.gaussian_model._p_xyz_layers[i] = align_previous('_xyz_layers', cur_xyz_full)
                self.gaussian_model._p_cholesky_layers[i] = align_previous('_cholesky_layers', cur_ch_full)
                self.gaussian_model._p_features_dc_layers[i] = align_previous('_features_dc_layers', cur_feat_full)

        elif quantized_model is not None:

            # I-frame decoding: reconstruct GaussianSplats buffers from quantized representation

            self.gaussian_model = GaussianSplats(

                num_points_per_layer=self.num_points_per_layer,

                scaling_factor=self.scaling_factor,

                H=self.H,

                W=self.W,

                BLOCK_H=BLOCK_H,

                BLOCK_W=BLOCK_W,

                device=self.device,

            ).to(self.device)



            encoded_layers = quantized_model.get('encoded_layers', None)

            quantizers_state = quantized_model.get('quantizers_state', None)

            if not isinstance(encoded_layers, (list, tuple)):

                raise TypeError(f"quantized_model['encoded_layers'] must be list/tuple, got {type(encoded_layers)}")



            with torch.no_grad():

                if isinstance(quantizers_state, dict):

                    ch_qs = quantizers_state.get('cholesky_quantizers', None)

                    feat_qs = quantizers_state.get('features_dc_quantizers', None)

                    if isinstance(ch_qs, (list, tuple)):

                        for li in range(min(len(ch_qs), len(self.gaussian_model.cholesky_quantizers))):

                            self.gaussian_model.cholesky_quantizers[li].load_state_dict(ch_qs[li], strict=False)

                    if isinstance(feat_qs, (list, tuple)):

                        for li in range(min(len(feat_qs), len(self.gaussian_model.features_dc_quantizers))):

                            self.gaussian_model.features_dc_quantizers[li].load_state_dict(feat_qs[li], strict=False)



            for i, _ in enumerate(self.num_points_per_layer):

                layer_ckpt = encoded_layers[i]

                if not isinstance(layer_ckpt, dict):

                    raise TypeError(f"encoded_layers[{i}] must be dict, got {type(layer_ckpt)}")

                if 'xyz' not in layer_ckpt:

                    raise KeyError(f"I-frame encoded_layers[{i}] missing key 'xyz'")

                if 'cholesky' not in layer_ckpt:

                    raise KeyError(f"I-frame encoded_layers[{i}] missing key 'cholesky'")

                if 'features_index' not in layer_ckpt:

                    raise KeyError(f"I-frame encoded_layers[{i}] missing key 'features_index'")



                xyz = layer_ckpt['xyz'].to(self.device).float()

                q_ch = layer_ckpt['cholesky'].to(self.device, dtype=torch.float32)

                feat_idx = layer_ckpt['features_index'].to(self.device, dtype=torch.long)

                ch = self.gaussian_model.cholesky_quantizers[i].decompress(q_ch)

                feat = self.gaussian_model.features_dc_quantizers[i].decompress(feat_idx)



                # overwrite registered buffers (these are plain tensors in PGSVC_decoder.GaussianSplats)

                self.gaussian_model._xyz_layers[i] = xyz

                self.gaussian_model._cholesky_layers[i] = ch

                self.gaussian_model._features_dc_layers[i] = feat



    def render(self, layer_times=None):

        psnr_value, ms_ssim_value, rendered_images, decoded_Gmodel = self.test(layer_times)

        return psnr_value, ms_ssim_value, rendered_images, decoded_Gmodel



    def test(self, layer_times=None):

        self.gaussian_model.eval()

        with torch.no_grad():

            gt_image = image_to_tensor(self.frame).to(self.device)
            rendered_images, psnr_list, ms_ssim_list = self.gaussian_model.render(
                [gt_image] * len(self.scaling_factor), layer_times
            )

            transform = transforms.ToPILImage()

            transformed_rendered_images = []

            if self.save_imgs:

                for idx in range(len(self.num_points_per_layer)):

                    img = transform(rendered_images[idx].squeeze(0))

                    transformed_rendered_images.append(img)

                    name = f"{self.frame_num}_fitting_Layer_{idx+1}.png"

                    img.save(str(self.result_save_path / name))

            elif self.save_video:

                for idx in range(len(self.num_points_per_layer)):

                    transformed_rendered_images.append(transform(rendered_images[idx].squeeze(0)))

            torch.cuda.synchronize(self.device)
            state_start = time.perf_counter()
            decoded_Gmodel = {}

            if hasattr(self.gaussian_model, '_p_xyz_layers') and hasattr(self.gaussian_model, '_p_cholesky_layers') and hasattr(self.gaussian_model, '_p_features_dc_layers'):

                decoded_Gmodel = {

                    '_xyz_layers': [

                        (self.gaussian_model._xyz_layers[i] + self.gaussian_model._p_xyz_layers[i]).detach().cpu()

                        for i in range(len(self.num_points_per_layer))

                    ],

                    '_cholesky_layers': [

                        (self.gaussian_model._cholesky_layers[i] + self.gaussian_model._p_cholesky_layers[i]).detach().cpu()

                        for i in range(len(self.num_points_per_layer))

                    ],

                    '_features_dc_layers': [

                        (self.gaussian_model._features_dc_layers[i] + self.gaussian_model._p_features_dc_layers[i]).detach().cpu()

                        for i in range(len(self.num_points_per_layer))

                    ],

                }

                if hasattr(self, 'current_point_ids_layers') and self.current_point_ids_layers is not None:

                    decoded_Gmodel['_point_ids_layers'] = [p.detach().cpu() for p in self.current_point_ids_layers]

            else:

                decoded_Gmodel = {

                    '_xyz_layers': [

                        self.gaussian_model._xyz_layers[i].detach().cpu()

                        for i in range(len(self.num_points_per_layer))

                    ],

                    '_cholesky_layers': [

                        (self.gaussian_model._cholesky_layers[i]).detach().cpu()

                        for i in range(len(self.num_points_per_layer))

                    ],

                    '_features_dc_layers': [

                        self.gaussian_model._features_dc_layers[i].detach().cpu()

                        for i in range(len(self.num_points_per_layer))

                    ],

                }



 

            torch.cuda.synchronize(self.device)
            self.state_ms = (time.perf_counter() - state_start) * 1000
            return psnr_list, ms_ssim_list, transformed_rendered_images, decoded_Gmodel

        

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

    return args





def main(argv):

    

    psnr_list = []

    msssim_list = []

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

    

    video_frames = process_yuv_video(args.dataset, args.width, args.height, args.video_length)

    frame_length, start=min(args.video_length, len(video_frames)),0

    

    quantized_gmodels_state_dict = torch.load(args.model_path, map_location="cpu")

    def _get_num_points_per_layer(frame_dict: dict):

        if not isinstance(frame_dict, dict):

            raise TypeError(f"frame_dict must be dict, got {type(frame_dict)}")

        # Preferred: quantized representation saved by train_PGSVC_encoder_newIP.py

        if 'encoded_layers' in frame_dict:

            encoded_layers = frame_dict['encoded_layers']

            if not isinstance(encoded_layers, (list, tuple)):

                raise TypeError(f"encoded_layers must be list/tuple, got {type(encoded_layers)}")

            layers = []

            for ld in encoded_layers:

                if not isinstance(ld, dict):

                    raise TypeError(f"encoded_layers element must be dict, got {type(ld)}")

                # I-frame case

                if 'xyz' in ld:

                    layers.append(int(ld['xyz'].shape[0]))

                    continue

                # delta-frame case (xyz_front/xyz_back); use cholesky length as authoritative

                if 'cholesky' in ld:

                    layers.append(int(ld['cholesky'].shape[0]))

                    continue

                raise KeyError("encoded_layers element missing required keys: 'xyz' or 'cholesky'")

            return layers

        # Fallback: legacy / unquantized checkpoints

        if '_xyz_layers' in frame_dict:

            return [int(t.shape[0]) for t in frame_dict['_xyz_layers']]

        raise KeyError("Cannot infer num_points_per_layer: missing 'encoded_layers' and '_xyz_layers'")



    def _parse_frame_idx(key: str):

        try:

            return int(str(key).split('_')[-1])

        except Exception:

            return None



    def _to_bool(val):

        try:

            return bool(val.item()) if hasattr(val, 'item') else bool(val)

        except Exception:

            return bool(val)



    K_frames = []

    for k, v in quantized_gmodels_state_dict.items():

        idx = _parse_frame_idx(k)

        if idx is None:

            continue



        is_k = False

        if isinstance(v, dict):

            # saved by train_PGSVC_encoder_newIP.py as:

            # frame_i -> { 'encoded_layers': ..., 'quantizers_state': { ..., 'is_k_frame': ... } }

            qs = v.get('quantizers_state', None)

            if isinstance(qs, dict) and ('is_k_frame' in qs):

                is_k = _to_bool(qs['is_k_frame'])

            elif 'is_k_frame' in v:

                is_k = _to_bool(v['is_k_frame'])



        if is_k:

            K_frames.append(idx)



    K_frames = sorted([k for k in K_frames if 1 <= k <= frame_length])

    print(f"K_frames(from model): {K_frames}")



    rendered_images_list = []

    P_Gmodel=None
    timing_rows = []

  

    for i in range(start, start + frame_length):

        frame_num=i+1

        modelid=f"frame_{frame_num}"

        Model = quantized_gmodels_state_dict[modelid]

        layers = _get_num_points_per_layer(Model)

        layer_number = len(layers)

        cumulative_layers = np.cumsum(layers).tolist()

        if args.is_downsample:

            scaling_factor = [ (layer / cumulative_layers[-1]) ** 0.5 for layer in cumulative_layers ]

        else:

            scaling_factor = [1.0 for _ in cumulative_layers]

        print(cumulative_layers)

        torch.cuda.synchronize()
        decode_start = time.perf_counter()
        if frame_num in K_frames:

            k_frame_num=frame_num

            print(f"modelid:frame_{k_frame_num};")

            render = SimpleTrainer2d(

                        frame = video_frames[i],

                        frame_num=frame_num,

                        num_points_per_layer = layers,

                        result_save_path=result_save_path,

                        scaling_factor=scaling_factor,

                        quantized_model=Model,

                        args=args

                    )

        else:

            p_modelid = f"frame_{frame_num - 1}"

            prev_frame_dict = quantized_gmodels_state_dict.get(p_modelid, None)

            if P_Gmodel is not None:

                # delta decoding requires point-id alignment.

                # Prefer point ids already carried by P_Gmodel (keeps consistent ordering with decoded tensors).

                if '_point_ids_layers' not in P_Gmodel:

                    if isinstance(prev_frame_dict, dict) and ('_point_ids_layers' in prev_frame_dict):

                        P_Gmodel = dict(P_Gmodel)

                        P_Gmodel['_point_ids_layers'] = prev_frame_dict['_point_ids_layers']

                    else:

                        raise KeyError(

                            f"Missing '_point_ids_layers' for delta decoding of {modelid}. "

                            f"Expected it in previous frame dict '{p_modelid}' or in P_Gmodel."

                        )



            print(f"modelid:{modelid}; p_modelid:{p_modelid}")

            render = SimpleTrainer2d(

                        frame = video_frames[frame_num-1],

                        frame_num=frame_num,

                        num_points_per_layer = layers,

                        result_save_path=result_save_path,

                        scaling_factor=scaling_factor,

                        quantized_model=Model,

                        previous_model=P_Gmodel,

                        args=args

                    )

        

        torch.cuda.synchronize()
        model_ms = (time.perf_counter() - decode_start) * 1000
        layer_times = []
        psnr, ms_ssim, rendered_images, decoded_Gmodel = render.render(layer_times)
        decode_ms = model_ms + render.state_ms
        render_ms = sum(layer_times)
        total_ms = decode_ms + render_ms
        timing_rows.append([
            args.data_name, frame_num, "I" if frame_num in K_frames else "P",
            decode_ms, render_ms, total_ms, 1000 / total_ms, *layer_times,
        ])

        P_Gmodel = decoded_Gmodel

        for idx in range(0,layer_number):

            logwriter.write("{}: Layer{}_Image: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}".format(

                frame_num, idx+1, render.W, render.H, psnr[idx], ms_ssim[idx]))

        frames.append(frame_num)

        psnr_list.append(psnr[-1])

        msssim_list.append(ms_ssim[-1])

        if args.save_video:
            rendered_images_list.append(rendered_images)

    timing_path = result_save_path / "decode_per_frame.csv"
    with timing_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sequence", "frame", "frame_type", "decode_ms", "render_ms", "total_ms", "fps",
            *(f"render_L{i}_ms" for i in range(args.layer_number)),
        ])
        writer.writerows(timing_rows)
    print(f"Per-frame decode timings: {timing_path}")

    avg_psnr = sum(psnr_list) / len(psnr_list)

    avg_msssim = sum(msssim_list) / len(msssim_list)

    frames.append("Average")

    psnr_list.append(avg_psnr)

    msssim_list.append(avg_msssim)

    print("=== Progressive Decoding Done ===")

    print(f"Average PSNR: {avg_psnr:.4f}, Average MS-SSIM: {avg_msssim:.6f}")

    if args.save_video:

        generate_progressive_video(result_save_path, rendered_images_list,"rendered_video.mp4")



if __name__ == "__main__":

    main(sys.argv[1:])





