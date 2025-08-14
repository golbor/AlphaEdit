#!/usr/bin/env bash
#
#SBATCH --job-name=alpha-edit
#SBATCH --output=output.txt
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --nodelist=worker-4

nvidia-smi

echo changing directory to AlphaEdit

cd /home/stud/golab/AlphaEdit

pwd

# activate environment
source .venv/bin/activate

pip list

# export NLTK_DATA=~/nltk_data
# python3 fix_nltk.py

python3 -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"

# run experiment
python3 -m experiments.evaluate     --alg_name=AlphaEdit     --model_name=meta-llama/Meta-Llama-3-8B-Instruct     --hparams_fname=Llama3-8B.json --ds_name=mcf --dataset_size_limit=2000    --num_edits=100 --downstream_eval_steps=5 >> terminal.txt