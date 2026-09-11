#!/bin/bash -l

#$ -P KibbleZurek_MachineLearning        # your SCC project name
#$ -N kz2d                # job name
#$ -t 1-60                # 60 tasks, one per chunk
#$ -l h_rt=04:00:00       # time limit per task
#$ -j y                   # put errors in the same log as output
#$ -o logs/               # where logs go
#$ -cwd                   # run from the folder you submit from

module load miniconda
conda activate your_env

export OMP_NUM_THREADS=1

# SGE numbers tasks 1-60, our chunks are 0-59
echo "start: $(date)"
python -m scripts.kz_chunk_2d $((SGE_TASK_ID - 1))
echo "end:   $(date)"