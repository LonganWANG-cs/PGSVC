#!/bin/bash

#SBATCH --job-name=PwqLQ    # Job name
#SBATCH --output=output/PGSVC_wq_L_Q_o.txt # Standard output and error log
#SBATCH --error=output/PGSVC_wq_L_Q_e.txt  # Error log
#SBATCH --time=48:00:00                 # Time limit hrs:min:sec
#SBATCH --gres=gpu:h100-47:1
#SBATCH --mail-type=ALL                 # Get email for all status updates
#SBATCH --mail-user=wanglongan@comp.nus.edu.sg # Email for notifications
#SBATCH --mem=24G                    # Request 16GB of memory
# Activate the environment if needed
source ~/.bashrc
conda activate pGS  # Replace 'torch' with the name of your conda environment


python train_PGSVC_overfitting.py --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --num_points 12000 --layer_number 3 --video_length 5 --iterations 50000 --savdir PGSVC_overfit_result --savdir_m PGSVC_overfit_model --training_strategy sequential --is_downsample --is_GSP --is_GSA --save_video

python train_PGSVC_encoder.py --model_path /home/e/e1344641/PGSVC/checkpoints/PGSVC_overfit_model/Beauty/PrGS_50000_12000_3/gmodels_state_dict.pth --num_points 12000 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --layer_number 3 --video_length 5 --iterations 50000 --savdir PGSVC_encode_result --savdir_m PGSVC_encode_model --save_video

python test_PGSVC_decoder_fps.py --model_path /home/e/e1344641/PGSVC/checkpoints/PGSVC_encode_model/Beauty/PrGS_50000_12000_3/encoded_gmodels_state_dict.pth --num_points 12000 --dataset /home/e/e1344641/data/UVG/Beauty/Beauty_1920x1080_120fps_420_8bit_YUV.yuv --data_name Beauty --layer_number 3 --video_length 5 --iterations 50000 --savdir PGSVC_decoder_result --savdir_m PGSVC_decoder_model --save_video