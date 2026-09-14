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

# tau comes from the command line: qsub -v TAU=256 jobs/gen2d.sh
TAU=${TAU:-128}

echo "start: $(date)   tau=$TAU"
python -u -m scripts.kz_chunk_2d $((SGE_TASK_ID - 1)) --tau $TAU
echo "end:   $(date)"