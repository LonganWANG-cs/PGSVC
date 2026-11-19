#!/usr/bin/env bash
#SBATCH --job-name=pgsvc-uvg
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-6%4
#SBATCH --output=output/pgsvc-uvg-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the PGSVC checkout}"

run_name=${1:?Pass a run name for the checkpoint directories}
frame_limit=${2:-0}
iterations=${3:-5000}
sequence_names=(Beauty Bosphorus HoneyBee Jockey ReadySetGo ShakeNDry YachtRide)
sequence=${sequence_names[${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}]}
file_stem=$sequence
if [[ $sequence == ReadySetGo ]]; then
    file_stem=ReadySteadyGo
fi

dataset="${UVG_ROOT:-$HOME/data/UVG}/$sequence/${file_stem}_1920x1080_120fps_420_8bit_YUV.yuv"
frame_bytes=$((1920 * 1080 * 3 / 2))
file_bytes=$(stat --format=%s "$dataset")
if (( file_bytes == 0 || file_bytes % frame_bytes != 0 )); then
    echo "Invalid UVG YUV420 file size: $dataset ($file_bytes bytes)" >&2
    exit 1
fi
frame_count=$((file_bytes / frame_bytes))
if (( frame_limit > 0 && frame_limit < frame_count )); then
    frame_count=$frame_limit
fi

source "${CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh"
conda activate "${PGSVC_ENV:-pGS}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

echo "Revision $(git rev-parse HEAD) | $sequence | $frame_count frames | $iterations iterations per stage"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

model_dir="PrGS_${iterations}_12000_2"
fitted="checkpoints/$run_name/overfit_model/$sequence/$model_dir/gmodels_state_dict.pth"
encoded="checkpoints/$run_name/encoded_model/$sequence/$model_dir/encoded_gmodels_state_dict.pth"
key_frames="checkpoints/$run_name/overfit_result/$sequence/$model_dir/K_frames.txt"
timings="checkpoints/$run_name/decoder_result/$sequence/$model_dir/decode_per_frame.csv"
common=(
    --dataset "$dataset" --data_name "$sequence" --width 1920 --height 1080
    --fps 120 --video_length "$frame_count" --num_points 12000
    --layer_number 2 --iterations "$iterations" --training_strategy sequential --is_downsample
)

echo "Stage 1: overfit"
python train_PGSVC_overfitting.py "${common[@]}" --single_pass --is_GSP --is_GSA \
    --savdir "$run_name/overfit_result" --savdir_m "$run_name/overfit_model"
test -s "$fitted"

echo "Stage 2: quantize and entropy-code"
python train_PGSVC_encoder.py "${common[@]}" --model_path "$fitted" \
    --entropy_coding --key_frames "$key_frames" \
    --savdir "$run_name/encoder_result" --savdir_m "$run_name/encoded_model"
test -s "$encoded"

echo "Stage 3: decode and time each frame"
python test_PGSVC_decoder.py "${common[@]}" --model_path "$encoded" \
    --savdir "$run_name/decoder_result" --savdir_m "$run_name/decoder_model"
test "$(wc -l < "$timings")" -eq "$((frame_count + 1))"
ls -lh "$fitted" "$encoded" "$timings"
