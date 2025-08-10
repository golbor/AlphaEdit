#!/usr/bin/env bash
#
#SBATCH --job-name=Model-Edit
#SBATCH --output=alpha_output.txt
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --nodelist=worker-4

# activate environment
echo "Activating virtual environment"
source .local/bin/activate
sleep 5
# run experiment
echo "Running experiment"

python3 -m experiments.evaluate     --alg_name=AlphaEdit     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B.json --ds_name=mcf --dataset_size_limit=2000    --num_edits=100 --downstream_eval_steps=5