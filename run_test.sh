#!/usr/bin/env bash
#
#SBATCH --job-name=alpha-edit-test
#SBATCH --output=output_test_%j.txt
#SBATCH --error=error_test_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --export=ALL

echo "Starting AlphaEdit test run at $(date)"

cd /home/stud/golab/AlphaEdit

# Set required environment variables
export HOME=/home/stud/golab
export XDG_CONFIG_HOME=$HOME/.config
export MPLCONFIGDIR=$HOME/.config/matplotlib
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# activate environment
source .venv/bin/activate

# Create NLTK data directory and set environment variable
mkdir -p /home/stud/golab/nltk_data
export NLTK_DATA=/home/stud/golab/nltk_data

python3 fix_nltk.py
python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# Run small test to validate everything works
python3 -u -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B.json \
    --ds_name=mcf \
    --dataset_size_limit=20 \
    --num_edits=5 \
    --downstream_eval_steps=2 \
    --conserve_memory \
    --use_cache \
    2>&1 | tee "test_terminal_${SLURM_JOB_ID}.txt"

echo "Test completed at $(date)"
