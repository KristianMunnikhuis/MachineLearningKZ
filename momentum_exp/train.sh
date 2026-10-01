#!/bin/bash -l

#$ -P fheating
#$ -N mom_train
#$ -t 1-16
#$ -l h_rt=00:45:00
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

# Momentum experiment: for each input time, train the old U-Net (phi) and the 2-channel U-Net (phi_pi).
# Task k -> time TIMES[k / 2], channels phi (k even) or phi_pi (k odd).
# -t must be 1 to 2 x (number of TIMES).
# Submit from the repo root:  qsub momentum_exp/train.sh
TIMES=(0 0.5 1 2 3 4 5 6 7)          # must match TIMES in momentum_exp/generate.py
CHANNELS=(phi phi_pi)

i=$((SGE_TASK_ID - 1))
T=${TIMES[$((i / 2))]}
CH=${CHANNELS[$((i % 2))]}

# skip if this run already finished (safe to resubmit)
if [ -f "momentum_exp/results/t${T}_${CH}.json" ]; then
  echo "already done: t=$T $CH"; exit 0
fi

module load miniconda
conda activate kz
echo "start: $(date)   t=$T  channels=$CH"
python -u -m momentum_exp.train --t $T --channels $CH
echo "end:   $(date)"