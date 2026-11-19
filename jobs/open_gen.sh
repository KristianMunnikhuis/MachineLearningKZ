#!/bin/bash -l

#$ -P fheating
#$ -N kzopen_gen
#$ -l h_rt=04:00:00
#$ -l mem_per_core=8G
#$ -j y
#$ -o logs/
#$ -cwd

# One data chunk per array task; task i reads line i of GRID ("case tau chunk").
# Normally submitted by jobs/open_submit.sh. By hand:
#   qsub -t 1-$(wc -l < jobs/open_gen_grid.txt) -v GRID=jobs/open_gen_grid.txt jobs/open_gen.sh

module load miniconda
conda activate kz

export OMP_NUM_THREADS=1
GRID=${GRID:-jobs/open_gen_grid.txt}
read CASE TAU CHUNK <<< "$(sed -n "${SGE_TASK_ID}p" $GRID)"

# skip chunks that already exist (safe to resubmit)
if [ -f "data/1D_open/${CASE}/tau${TAU}/chunk_$(printf %03d $CHUNK).npz" ]; then
  echo "already done: $CASE tau=$TAU chunk=$CHUNK"; exit 0
fi

echo "start: $(date)   case=$CASE  tau=$TAU  chunk=$CHUNK"
python -u -m scripts.kz1d.generate_open $CASE $TAU $CHUNK
echo "end:   $(date)"
