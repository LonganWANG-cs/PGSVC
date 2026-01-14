import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision.transforms as transforms

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        print(text)
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    return loss


def strip_lowerdiag(L):
    if L.shape[1] == 3:
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]

    elif L.shape[1] == 2:
        uncertainty = torch.zeros((L.shape[0], 3), dtype=torch.float, device="cuda")
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 1, 1]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_rotation_2d(r):
    '''
    Build rotation matrix in 2D.
    '''
    R = torch.zeros((r.size(0), 2, 2), device='cuda')
    R[:, 0, 0] = torch.cos(r)[:, 0]
    R[:, 0, 1] = -torch.sin(r)[:, 0]
    R[:, 1, 0] = torch.sin(r)[:, 0]
    R[:, 1, 1] = torch.cos(r)[:, 0]
    return R

def build_scaling_rotation_2d(s, r, device):
    L = torch.zeros((s.shape[0], 2, 2), dtype=torch.float, device='cuda')
    R = build_rotation_2d(r, device)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L = R @ L
    return L
    
def build_covariance_from_scaling_rotation_2d(scaling, scaling_modifier, rotation, device):
    '''
    Build covariance metrix from rotation and scale matricies.
    '''
    L = build_scaling_rotation_2d(scaling_modifier * scaling, rotation, device)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def build_triangular(r):
    R = torch.zeros((r.size(0), 2, 2), device=r.device)
    R[:, 0, 0] = r[:, 0]
    R[:, 1, 0] = r[:, 1]
    R[:, 1, 1] = r[:, 2]
    return R



def generate_video(image_list, result_save_path,fps,origin,layer_number):
    video_path = result_save_path / "video"
    video_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    # Define the output video file name
    if origin:
        filename = f"L{layer_number}.mp4"
    else:
        filename = f"L{layer_number}_distribution.mp4"
    # Get the size of the first image dynamically
    first_image = image_list[0]
    width, height = first_image.size  # Extract the size of the first image
    # Create the video writer with the actual image dimensions
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)
    # Add images to the video writer
    for img in tqdm(image_list, desc="Processing images", unit="image"):  # Iterate directly over the image_list      
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video.write(img_cv)
    # Finalize and close the video writer
    video.release()
    if origin:
        print("video.mp4: MP4 video created successfully.")
    else:
        print("combined_video.mp4: MP4 video created successfully.")


# def generate_progressive_video(videodir, image_list, video_name):
#     video_path = Path(videodir) / "video"
#     video_path.mkdir(parents=True, exist_ok=True)
    
#     for i in range (len(image_list)):
#         first_image = image_list[i][0]
#         print(np.shape(first_image))
#         width, height = first_image.size 
#         output_size = (width, height)
#         layer_num = i+1
#         progressive_video_path = Path(video_path)/str(layer_num)
#         video = cv2.VideoWriter(str(progressive_video_path / video_name), cv2.VideoWriter_fourcc(*'mp4v'), 24, output_size)
#         for img in tqdm(image_list, desc=f"Processing layer {layer_num} video", unit="image"): 
#             if img.mode != 'RGB':
#                 img = img.convert('RGB')
#             img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             video.write(img_cv)
#         video.release()
def generate_progressive_video(videodir, image_list, video_name):
    video_path = Path(videodir) / "video"
    video_path.mkdir(parents=True, exist_ok=True)
    num_frames = len(image_list)
    num_layers = len(image_list[0])
    print(f"Generating {num_layers} videos from {num_frames} frames.")
    transposed = [[image_list[f][l] for f in range(num_frames)] for l in range(num_layers)]
    for layer_idx, images in enumerate(transposed):
        first_image = images[0]
        if isinstance(first_image, torch.Tensor):
            first_image = first_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            first_image = Image.fromarray((first_image * 255).astype(np.uint8))
        width, height = first_image.size 
        output_size = (width, height)

        progressive_video_path = video_path / str(layer_idx + 1)
        progressive_video_path.mkdir(parents=True, exist_ok=True)

        video = cv2.VideoWriter(
            str(progressive_video_path / video_name), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            24, 
            output_size
        )

        for img in tqdm(images, desc=f"Processing Layer {layer_idx + 1}", unit="frame"):
            if isinstance(img, torch.Tensor):
                img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            video.write(img_cv)
        video.release()



class EarlyStopping:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience  # Number of tolerated iterations with no improvement
        self.min_delta = min_delta  # Minimum improvement threshold
        self.best_loss = None  # Stores the best loss value
        self.counter = 0  # Tracks the number of iterations without improvement

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False  # Do not stop training

        # If the improvement over the previous best loss is less than min_delta, consider it no improvement
        if self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0  # Reset counter
        else:
            self.counter += 1

        # If the counter exceeds patience, stop training
        if self.counter >= self.patience:
            return True  # Stop training

        return False  # Continue training


def save_image_difference(image_tensor, gt_image_tensor, result_save_path,layer_number):
    result_save_path.mkdir(parents=True, exist_ok=True)
    output_path = result_save_path / f"difference_L{layer_number}.png"
    image_tensor = image_tensor.squeeze(0)
    gt_image_tensor = gt_image_tensor.squeeze(0)
    diff = torch.abs(image_tensor - gt_image_tensor)
    diff = diff / diff.max()
    diff_image = to_pil_image(diff)
    diff_image.save(output_path)
    print(f"Difference image saved to {output_path}")

def image_to_tensor(img: Image.Image):
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return img_tensor

def process_yuv_video(file_path, width, height):
    
    frame_size = width * height * 3 // 2
    
    file_size = os.path.getsize(file_path)
    total_frames = file_size // frame_size
    
    video_frames = []
    
    with open(file_path, 'rb') as f:
        
        for _ in tqdm(range(total_frames), desc="Processing frames"):
            
            yuv_frame = f.read(frame_size)
            if not yuv_frame:
                break  
            
            yuv = np.frombuffer(yuv_frame, dtype=np.uint8).reshape((height * 3 // 2, width))
            
            rgb_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            
            video_frames.append(rgb_frame)
    return video_frames


# def detect_outliers_mean_diff(values, window_size=10, threshold=3):
#     outliers = []
#     for i in range(len(values)):
#         # Define the window range
#         start_idx = max(0, i - window_size)
#         end_idx = min(len(values), i + window_size)
        
#         # Calculate local mean and standard deviation
#         local_mean = np.mean(values[start_idx:end_idx])
#         local_std = np.std(values[start_idx:end_idx])
#         # Check if the value exceeds the threshold
#         if (values[i] - local_mean) > threshold * local_std:
#             outliers.append(i)
#         elif(values[i] > local_mean*threshold):
#             outliers.append(i)
#     return outliers
def detect_outliers_mean_diff(values, window_size=10, threshold=3):
    outliers = []
    for i in range(len(values)):
        # Define the window range
        start_idx = max(0, i - window_size)
        end_idx = min(len(values), i + window_size)
        #end_idx = i
        
        # Calculate local mean and standard deviation
        local_mean = np.mean(values[start_idx:end_idx])
        local_std = np.std(values[start_idx:end_idx])
        # Check if the value exceeds the threshold
        if (values[i] - local_mean) > threshold * local_std:
            outliers.append(i)
        elif(values[i] > local_mean*threshold):
            outliers.append(i)
    return outliers

