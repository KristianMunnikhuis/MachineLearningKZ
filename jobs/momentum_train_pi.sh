#!/bin/bash -l

#$ -P fheating
#$ -t 1-8
#$ -l h_rt=00:45:00
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

# Train the U-Net on the momentum pi alone, one task per input time. Damping passed in:
#   qsub -v ETA=0.3 -N mom_pi_0.3 jobs/momentum_train_pi.sh
# -t must be 1 to (number of TIMES).

TIMES=(0 0.5 1 2 3 4 5 6)
T=${TIMES[$((SGE_TASK_ID - 1))]}

if [ -f "results/momentum/eta${ETA}/t${T}_pi.json" ]; then
  echo "already done: eta=$ETA t=$T pi"; exit 0
fi

module load miniconda
conda activate kz
echo "start: $(date)   eta=$ETA  t=$T  input=pi"
python -u -m scripts.momentum.train_pi --t $T --eta $ETA
echo "end:   $(date)"