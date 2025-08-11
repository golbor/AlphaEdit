#!/usr/bin/env bash
#
#SBATCH --job-name=AlphaEdit
#SBATCH --output=output.txt
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE

cd /home/stud/golab/AlphaEdit
# activate environment
source .venv/bin/activate

# run experiment
python3 -m experiments.evaluate     --alg_name=AlphaEdit     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B.json --ds_name=mcf --dataset_size_limit=2000    --num_edits=100 --downstream_eval_steps=5