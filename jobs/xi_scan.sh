#!/bin/bash -l

#$ -P fheating
#$ -N xiscan
#$ -l h_rt=06:00:00
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

module load miniconda
conda activate kz

export OMP_NUM_THREADS=4

python -u -m scripts.xi_scan --taus 10 20 40 80 160 --n 20 --N 256