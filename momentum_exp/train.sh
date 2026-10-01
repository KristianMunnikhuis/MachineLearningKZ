#!/bin/bash -l

#$ -P fheating
#$ -t 1-16
#$ -l h_rt=00:45:00
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

# For each input time: the old U-Net (phi) and the 2-channel U-Net (phi_pi). Damping passed in:
#   qsub -v ETA=0.3 -N mom_train_0.3 -hold_jid mom_gen_0.3 momentum_exp/train.sh
# -t must be 1 to 2 x (number of TIMES).

TIMES=(0 0.5 1 2 3 4 5 6)
CHANNELS=(phi phi_pi)

i=$((SGE_TASK_ID - 1))
T=${TIMES[$((i / 2))]}
CH=${CHANNELS[$((i % 2))]}

if [ -f "momentum_exp/results_eta${ETA}/t${T}_${CH}.json" ]; then
  echo "already done: eta=$ETA t=$T $CH"; exit 0
fi

module load miniconda
conda activate kz
echo "start: $(date)   eta=$ETA  t=$T  channels=$CH"
python -u -m momentum_exp.train --t $T --channels $CH --eta $ETA
echo "end:   $(date)"