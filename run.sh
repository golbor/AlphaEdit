#!/usr/bin/env bash
#
#SBATCH --job-name=alpha-edit
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --nodelist=worker-4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=golab.borzooei@campus.lmu.de


nvidia-smi

echo changing directory to AlphaEdit

cd /home/stud/golab/AlphaEdit

pwd

# Set required environment variables
export HOME=/home/stud/golab
export XDG_CONFIG_HOME=$HOME/.config
export MPLCONFIGDIR=$HOME/.config/matplotlib

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# activate environment
source .venv/bin/activate

pip list

# Create NLTK data directory and set environment variable
mkdir -p /home/stud/golab/nltk_data
export NLTK_DATA=/home/stud/golab/nltk_data

python3 fix_nltk.py

python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# Function to run experiment with retry mechanism
run_experiment() {
    local attempt=1
    local max_attempts=3
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts"
        echo "Starting at $(date)"
        
        # Run experiment with checkpointing and error handling
        python3 -u -m experiments.evaluate \
            --alg_name=AlphaEdit \
            --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
            --hparams_fname=Llama3-8B.json \
            --ds_name=mcf \
            --dataset_size_limit=2000 \
            --num_edits=100 \
            --downstream_eval_steps=5 \
            --conserve_memory \
            --use_cache \
            2>&1 | tee -a terminal_${SLURM_JOB_ID}.txt
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "Experiment completed successfully at $(date)"
            break
        else
            echo "Experiment failed with exit code $exit_code at $(date)"
            if [ $attempt -eq $max_attempts ]; then
                echo "All attempts failed. Exiting."
                exit $exit_code
            else
                echo "Retrying in 60 seconds..."
                sleep 60
                ((attempt++))
            fi
        fi
    done
}

# Monitor system resources during execution
monitor_resources() {
    while true; do
        echo "=== Resource Monitor $(date) ===" >> resource_monitor.txt
        nvidia-smi >> resource_monitor.txt 2>&1
        free -h >> resource_monitor.txt 2>&1
        echo "" >> resource_monitor.txt
        sleep 300  # Monitor every 5 minutes
    done
}

# Start resource monitoring in background
monitor_resources &
MONITOR_PID=$!

# Trap to cleanup on exit
trap 'kill $MONITOR_PID 2>/dev/null; exit' EXIT INT TERM

# run experiment with retry mechanism
run_experiment

# Cleanup
kill $MONITOR_PID 2>/dev/null

echo "Job completed at $(date)"