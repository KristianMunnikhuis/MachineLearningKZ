#!/bin/bash
# Build the grids, submit generation, then training (held until generation finishes).
# Run from the repo root on the SCC:   bash jobs/open_submit.sh
# Finished chunks and finished training runs are skipped, so this is safe to rerun after adding RUNS.
# N_CHUNKS must match --n_chunks in generate_open.py.

# each run: "tag case [extra args for generate_open.py]"
RUNS=(
  # baseline three (tau = 128 done Nov 19)
  "classical             classical"
  "qlangevin             qlangevin"
  "twa                   twa"
  # theta scan: does the quantum-bath lag come from the seed size alone?
  "classical_theta1e-9   classical --theta 1e-9"
  "classical_theta1e-10  classical --theta 1e-10"
  "classical_theta1e-11  classical --theta 1e-11"
  # eta scan: quantum bath + Wigner vacuum start (eta = 0 is twa)
  "qlv_wigner_eta0.01    qlangevin --init wigner --eta 0.01"
  "qlv_wigner_eta0.03    qlangevin --init wigner --eta 0.03"
  "qlv_wigner_eta0.1     qlangevin --init wigner --eta 0.1"
  "qlv_wigner_eta0.3     qlangevin --init wigner --eta 0.3"
  "qlv_wigner_eta1       qlangevin --init wigner --eta 1"
)
TAUS="128"
N_CHUNKS=10
TIMES="-185 -150 -100 -60 -40 -20 -10 -5 0 5 10 15 20 25 30 35 40 50 60 70"

mkdir -p logs
> jobs/open_gen_grid.txt
> jobs/open_train_grid.txt
for RUN in "${RUNS[@]}"; do
  read TAG CASE EXTRA <<< "$RUN"
  for TAU in $TAUS; do
    for CHUNK in $(seq 0 $((N_CHUNKS - 1))); do echo "$TAG $CASE $TAU $CHUNK $EXTRA" >> jobs/open_gen_grid.txt; done
    for T in $TIMES; do echo "$TAG $TAU $T" >> jobs/open_train_grid.txt; done
  done
done

GEN=$(qsub -terse -t 1-$(wc -l < jobs/open_gen_grid.txt) -v GRID=jobs/open_gen_grid.txt jobs/open_gen.sh | cut -d. -f1)
echo "generation: job $GEN  ($(wc -l < jobs/open_gen_grid.txt) tasks, finished chunks exit immediately)"

TRAIN=$(qsub -terse -hold_jid $GEN -t 1-$(wc -l < jobs/open_train_grid.txt) -v GRID=jobs/open_train_grid.txt jobs/open_train.sh | cut -d. -f1)
echo "training:   job $TRAIN  ($(wc -l < jobs/open_train_grid.txt) tasks, waits for $GEN)"
