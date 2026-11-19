#!/bin/bash
# Build the grids, submit generation, then training (held until generation finishes).
# Run from the repo root on the SCC:   bash jobs/open_submit.sh
# Edit CASES / TAUS / TIMES / N_CHUNKS here; N_CHUNKS must match --n_chunks in generate_open.py.

CASES="classical qlangevin twa"
TAUS="128"
N_CHUNKS=10
TIMES="-185 -150 -100 -60 -40 -20 -10 -5 0 5 10 15 20 25 30 35 40 50 60 70"

mkdir -p logs
> jobs/open_gen_grid.txt
> jobs/open_train_grid.txt
for CASE in $CASES; do
  for TAU in $TAUS; do
    for CHUNK in $(seq 0 $((N_CHUNKS - 1))); do echo "$CASE $TAU $CHUNK" >> jobs/open_gen_grid.txt; done
    for T in $TIMES; do echo "$CASE $TAU $T" >> jobs/open_train_grid.txt; done
  done
done

GEN=$(qsub -terse -t 1-$(wc -l < jobs/open_gen_grid.txt) -v GRID=jobs/open_gen_grid.txt jobs/open_gen.sh | cut -d. -f1)
echo "generation: job $GEN  ($(wc -l < jobs/open_gen_grid.txt) tasks)"

TRAIN=$(qsub -terse -hold_jid $GEN -t 1-$(wc -l < jobs/open_train_grid.txt) -v GRID=jobs/open_train_grid.txt jobs/open_train.sh | cut -d. -f1)
echo "training:   job $TRAIN  ($(wc -l < jobs/open_train_grid.txt) tasks, waits for $GEN)"
