#!/bin/bash -l

#$ -P fheating
#$ -N kztrain
#$ -l h_rt=00:45:00
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 4
#$ -j y
#$ -o logs/
#$ -cwd

# One U-Net per array task; task i reads line i of GRID ("tau t_hat target").
# Submit:  qsub -t 1-$(wc -l < jobs/train_grid_y_end.txt) -v GRID=jobs/train_grid_y_end.txt jobs/train.sh

module load miniconda
conda activate kz

read TAU THAT TARGET <<< "$(sed -n "${SGE_TASK_ID}p" $GRID)"

# skip runs that already finished (safe to resubmit)
if [ -f "results/v2/tau${TAU}_that${THAT}_${TARGET}.json" ]; then
  echo "already done: tau=$TAU t_hat=$THAT $TARGET"; exit 0
fi

echo "start: $(date)   tau=$TAU  t_hat=$THAT  target=$TARGET"
python -u -m scripts.kz2d.train --tau $TAU --t-hat $THAT --target $TARGET
echo "end:   $(date)"