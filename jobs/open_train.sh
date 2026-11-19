#!/bin/bash -l

#$ -P fheating
#$ -N kzopen_train
#$ -l h_rt=03:00:00
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

# One RNN per array task (CPU); task i reads line i of GRID ("case tau t").
# Normally submitted by jobs/open_submit.sh. By hand:
#   qsub -t 1-$(wc -l < jobs/open_train_grid.txt) -v GRID=jobs/open_train_grid.txt jobs/open_train.sh

module load miniconda
conda activate kz

export OMP_NUM_THREADS=$NSLOTS
export KERAS_BACKEND=${KERAS_BACKEND:-torch}         # torch already works in the kz env; set tensorflow to override
GRID=${GRID:-jobs/open_train_grid.txt}
read CASE TAU T <<< "$(sed -n "${SGE_TASK_ID}p" $GRID)"

# skip runs that already finished (safe to resubmit)
if [ -f "results/open/${CASE}_tau${TAU}_t${T}.json" ]; then
  echo "already done: $CASE tau=$TAU t=$T"; exit 0
fi

echo "start: $(date)   case=$CASE  tau=$TAU  t=$T"
python -u -m scripts.kz1d.train_open $CASE $TAU $T
echo "end:   $(date)"
