#!/bin/bash -l

#$ -P fheating
#$ -N kz2d
#$ -t 1-60
#$ -l h_rt=01:00:00
#$ -j y
#$ -o logs/
#$ -cwd

module load miniconda
conda activate kz

export OMP_NUM_THREADS=1

echo "start: $(date)"
which python
python -m scripts.kz_chunk_2d $((SGE_TASK_ID - 1))
echo "end:   $(date)"