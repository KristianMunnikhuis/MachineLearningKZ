#!/bin/bash -l

#$ -P fheating
#$ -N kz2d
#$ -t 1-40
#$ -l h_rt=02:00:00
#$ -j y
#$ -o logs/
#$ -cwd

# 2D KZ data (v2): one chunk of 50 samples per array task, 40 tasks = 2000 samples per tau.
# Submit one array per tau:  qsub -v TAU=256 -N kz2d_256 jobs/gen2d.sh

module load miniconda
conda activate kz

export OMP_NUM_THREADS=1

TAU=${TAU:-128}

echo "start: $(date)   tau=$TAU   chunk=$((SGE_TASK_ID - 1))"
python -u -m scripts.kz2d.generate_data $((SGE_TASK_ID - 1)) --tau $TAU
echo "end:   $(date)"