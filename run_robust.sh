#!/usr/bin/env bash
#
#SBATCH --job-name=alpha-edit-robust
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --nodelist=worker-6
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=golab.borzooei@campus.lmu.de


# Function to find the latest run directory
find_latest_run() {
    local results_dir="/home/stud/golab/AlphaEdit/results/AlphaEdit"
    if [ -d "$results_dir" ]; then
        ls -1 "$results_dir" | grep "^run_" | sort -V | tail -1
    fi
}

# Function to check if run is incomplete
is_run_incomplete() {
    local run_dir="$1"
    local expected_files=2000  # dataset_size_limit
    
    if [ -z "$run_dir" ] || [ ! -d "/home/stud/golab/AlphaEdit/results/AlphaEdit/$run_dir" ]; then
        return 0  # No run dir, so incomplete
    fi
    
    local actual_files=$(find "/home/stud/golab/AlphaEdit/results/AlphaEdit/$run_dir" -name "*_edits-case_*.json" | wc -l)
    
    if [ "$actual_files" -lt "$expected_files" ]; then
        return 0  # Incomplete
    else
        return 1  # Complete
    fi
}

nvidia-smi

echo "Starting robust AlphaEdit experiment at $(date)"
echo "changing directory to AlphaEdit"

cd /home/stud/golab/AlphaEdit

pwd

# Set required environment variables
export HOME=/home/stud/golab
export XDG_CONFIG_HOME=$HOME/.config
export MPLCONFIGDIR=$HOME/.config/matplotlib

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=1

# Reduce memory fragmentation
export CUDA_MEMORY_FRACTION=0.9

# activate environment
source .venv/bin/activate

pip list

# Create NLTK data directory and set environment variable
mkdir -p /home/stud/golab/nltk_data
export NLTK_DATA=/home/stud/golab/nltk_data

python3 fix_nltk.py

python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# Check for existing incomplete runs
LATEST_RUN=$(find_latest_run)
echo "Latest run found: $LATEST_RUN"

CONTINUE_ARG=""
if [ -n "$LATEST_RUN" ] && is_run_incomplete "$LATEST_RUN"; then
    echo "Resuming from incomplete run: $LATEST_RUN"
    CONTINUE_ARG="--continue_from_run=$LATEST_RUN"
else
    echo "Starting new run"
fi

# Function to run experiment with retry mechanism
run_experiment() {
    local attempt=1
    local max_attempts=3
    
    while [ $attempt -le $max_attempts ]; do
        echo "=== Attempt $attempt of $max_attempts ===" 
        echo "Starting at $(date)"
        
        # Clear GPU memory before starting
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        # Run experiment with all robustness features enabled
        timeout 6d python3 -u -m experiments.evaluate \
            --alg_name=AlphaEdit \
            --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
            --hparams_fname=Llama3-8B.json \
            --ds_name=mcf \
            --dataset_size_limit=2000 \
            --num_edits=100 \
            --downstream_eval_steps=5 \
            --conserve_memory \
            --use_cache \
            $CONTINUE_ARG \
            2>&1 | tee -a "terminal_${SLURM_JOB_ID}_attempt_${attempt}.txt"
        
        exit_code=$?
        
        echo "Process finished with exit code $exit_code at $(date)"
        
        # Check if we completed successfully
        LATEST_RUN_AFTER=$(find_latest_run)
        if [ $exit_code -eq 0 ] || ( [ -n "$LATEST_RUN_AFTER" ] && ! is_run_incomplete "$LATEST_RUN_AFTER" ); then
            echo "Experiment completed successfully!"
            echo "Final run directory: $LATEST_RUN_AFTER"
            break
        else
            echo "Experiment incomplete or failed."
            if [ $attempt -eq $max_attempts ]; then
                echo "All attempts failed. Final exit code: $exit_code"
                exit $exit_code
            else
                echo "Will retry in 120 seconds..."
                
                # Update continue argument for retry
                LATEST_RUN_FOR_RETRY=$(find_latest_run)
                if [ -n "$LATEST_RUN_FOR_RETRY" ]; then
                    CONTINUE_ARG="--continue_from_run=$LATEST_RUN_FOR_RETRY"
                    echo "Next attempt will continue from: $LATEST_RUN_FOR_RETRY"
                fi
                
                sleep 120
                ((attempt++))
            fi
        fi
    done
}

# Monitor system resources during execution
monitor_resources() {
    local log_file="resource_monitor_${SLURM_JOB_ID}.txt"
    while true; do
        {
            echo "=== Resource Monitor $(date) ==="
            nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
            echo "CPU and Memory:"
            top -bn1 | head -20
            echo "Disk Usage:"
            df -h /home/stud/golab
            echo "----------------------------------------"
        } >> "$log_file" 2>&1
        sleep 300  # Monitor every 5 minutes
    done
}

# Start resource monitoring in background
monitor_resources &
MONITOR_PID=$!

# Trap to cleanup on exit
cleanup() {
    echo "Cleaning up processes..."
    kill $MONITOR_PID 2>/dev/null || true
    # Final resource snapshot
    echo "=== Final Resource Snapshot $(date) ===" >> "resource_final_${SLURM_JOB_ID}.txt"
    nvidia-smi >> "resource_final_${SLURM_JOB_ID}.txt" 2>&1
    free -h >> "resource_final_${SLURM_JOB_ID}.txt" 2>&1
}

trap cleanup EXIT INT TERM

# Run experiment with retry mechanism
run_experiment

# Final cleanup
cleanup

echo "Job completed at $(date)"
echo "Check the results directory for output files"