#!/bin/bash -l

#$ -P fheating
#$ -N mom_gen
#$ -t 1-40
#$ -l h_rt=01:00:00
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

# Momentum experiment: 40 chunks x 50 samples = 2000 samples.
# Submit from the repo root:  qsub momentum_exp/gen.sh

module load miniconda
conda activate kz
export OMP_NUM_THREADS=1

python -u -m momentum_exp.generate $((SGE_TASK_ID - 1))