#!/bin/bash -l

#$ -P fheating
#$ -t 1-40
#$ -l h_rt=01:00:00
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

# Momentum experiment data: 40 chunks x 50 samples. Damping passed in:
#   qsub -v ETA=0.3 -N mom_gen_0.3 jobs/momentum_gen.sh

module load miniconda
conda activate kz
export OMP_NUM_THREADS=1

python -u -m scripts.momentum.generate $((SGE_TASK_ID - 1)) --eta $ETA