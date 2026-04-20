#!/bin/bash
#SBATCH --job-name=python_training                     # Job name
#SBATCH --output=../results/logs/job_output_%j.txt     # Output file (%j = job ID)
#SBATCH --error=../results/logs/job_error_%j.txt       # Error file
#SBATCH --partition=gpu_extra                          # Partition (queue) name; change to your cluster's GPU partition
#SBATCH --nodes=1                                      # Number of nodes
#SBATCH --ntasks=1                                     # Number of tasks (processes)
#SBATCH --cpus-per-task=8                              # CPUs per task (for multi-threading)
#SBATCH --mem=200G                                     # Memory per node (16 GB)
#SBATCH --gres=gpu:1                                   # Request 1 GPU (generic; or specify type like gpu:a100:1)
#SBATCH --time=2-00:00:00                              # Time limit (d-hh:mm:ss) 2 days
#SBATCH --mail-type=BEGIN,END,FAIL                     # Email notifications
#SBATCH --mail-user=tigerfan7495@gmail.com             # Your email


# Optional: prevent output buffering so logs appear faster
export PYTHONUNBUFFERED=1


PROJECT_ROOT="/vast/home/fwang/image_ai"
OUT_DIR="$PROJECT_ROOT/outputs7"


# ------------------------------------------------
# Load required modules (ADJUST TO YOUR CLUSTER!)
# ------------------------------------------------
module purge  # Clean environment
module load miniconda/3
module load tensorboard-gpu/2.20.0

# ------------------------------------------------
# Activate your environment (choose ONE approach)
# ------------------------------------------------
source ../myenv/bin/activate

# ------------------------------------------------
# Run your TensorFlow Python code
# ------------------------------------------------
python -u ../src/train_resnet50_with_roc_tf5.py \
  --data_dir "$PROJECT_ROOT/data" \
  --output_dir "$OUT_DIR" \
  --epochs_stage1 20 \
  --epochs_stage2 20 \
  --batch_size 32 \
  --base_lr 1e-4 \
  --finetune_lr 1e-5 \
  --val_split 0.2 \
  --img_size 224 \
  --seed 42


# Optional: Post-processing
echo "Job completed at $(date)"
