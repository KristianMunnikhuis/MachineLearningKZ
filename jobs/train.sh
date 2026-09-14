#!/bin/bash -l

#$ -P fheating
#$ -N kztrain
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

module load miniconda
conda activate kz

TAU=${TAU:-128}

# input times to sweep, as fractions of tau
FRACS="-0.5 -0.25 0 0.1 0.2 0.3 0.4 0.5 0.75 1.0"

for f in $FRACS; do
    T=$(python -c "print($f * $TAU)")
    echo "=== tau=$TAU  t=$T ===" 
    python -u -m scripts.train --tau $TAU --t-in $T
done