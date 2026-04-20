#!/bin/bash
#SBATCH --job-name=python_training     # Job name
#SBATCH --output=../logs/job_output_%j.txt     # Output file (%j = job ID)
#SBATCH --error=../logs/job_error_%j.txt       # Error file
#SBATCH --partition=gpu_extra          # Partition (queue) name; change to your cluster's GPU partition
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=8              # CPUs per task (for multi-threading)
#SBATCH --mem=200G                     # Memory per node (16 GB)
#SBATCH --gres=gpu:1                   # Request 1 GPU (generic; or specify type like gpu:a100:1)
#SBATCH --time=2-00:00:00              # Time limit (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL     # Email notifications
#SBATCH --mail-user=tigerfan7495@gmail.com  # Your email


# Optional: prevent output buffering so logs appear faster
export PYTHONUNBUFFERED=1

# ------------------------------------------------
# Load required modules 
# ------------------------------------------------
module load python39
module load tensorboard-gpu/2.20.0

# ------------------------------------------------
# Activate your environment 
# ------------------------------------------------
source ~/image_ai/myenv/bin/activate

# ------------------------------------------------
# Run your TensorFlow Python code
# ------------------------------------------------
python -u ../src/main.py    # -u = unbuffered output


# Optional: Post-processing
echo "Job completed at $(date)"
