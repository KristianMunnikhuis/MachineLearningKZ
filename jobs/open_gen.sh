#!/bin/bash -l

#$ -P fheating
#$ -N kzopen_gen
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -j y
#$ -o logs/
#$ -cwd

# One data chunk per array task; task i reads line i of GRID ("tag case tau chunk [extra args]").
# Normally submitted by jobs/open_submit.sh. By hand:
#   qsub -t 1-$(wc -l < jobs/open_gen_grid.txt) -v GRID=jobs/open_gen_grid.txt jobs/open_gen.sh

module load miniconda
conda activate kz

export OMP_NUM_THREADS=1
GRID=${GRID:-jobs/open_gen_grid.txt}
read TAG CASE TAU CHUNK EXTRA <<< "$(sed -n "${SGE_TASK_ID}p" $GRID)"

# skip chunks that already exist (safe to resubmit)
if [ -f "data/1D_open/${TAG}/tau${TAU}/chunk_$(printf %03d $CHUNK).npz" ]; then
  echo "already done: $TAG tau=$TAU chunk=$CHUNK"; exit 0
fi

echo "start: $(date)   tag=$TAG  case=$CASE  tau=$TAU  chunk=$CHUNK  extra=$EXTRA"
python -u -m scripts.kz1d.generate_open $CASE $TAU $CHUNK --tag $TAG $EXTRA
echo "end:   $(date)"
