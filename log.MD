# MachineLearningKZ — 2D

## Question
How early is the final domain pattern of a Kibble-Zurek quench determined?
A U-Net is trained to predict $\text{sign}(\phi_\text{final})$ from a single snapshot $\phi(x,y,t)$.
Predictability vs. input time is used as an operational measure of freeze-out.

**2D vs. 1D:** in 1D, defects are fixed once seeded. In 2D, walls keep moving
(curvature-driven coarsening, $L \sim t^{1/2}$), so *seeding* and *final state* are
different things. Open question: how much of the final pattern is seeded at $\hat t$
vs. selected later by coarsening.

## Model
Classical 2D $\phi^4$, damped second-order dynamics, periodic boundaries.
- `dx = 0.5`, `eta = 1.0`, `theta = 1e-8`
- $\epsilon$ swept linearly from $-1.5$ to $1.0$; transition at $t=0$, run ends at $t=\tau_Q$
- `dt = 0.1` (steps $= 25\,\tau_Q$)

## Pipeline
| file | does |
|---|---|
| `scripts/kz_chunk_2d.py` | one chunk (50 samples), 31 snapshots at $t/\tau_Q \in [-0.5, 1]$; `--tau` |
| `jobs/gen2d.sh` | SGE array, 60 chunks → 3000 samples; `qsub -v TAU=16 jobs/gen2d.sh` |
| `scripts/xi_scan.py` | standalone scan, 20 samples/tau, any `--N`, keeps snapshots |
| `src/kz_ml.py` | data loading, U-Net, train/eval |
| `scripts/train.py` | one model: one $\tau_Q$, one input time → `results/tau*_t*.{json,pt}` |
| `jobs/train.sh` | GPU job, sweeps 10 input times; `qsub -v TAU=9 jobs/train.sh` |
| `src/analysis.py` | results loading, wall length, predictions |

Split 80/10/10, input scaled by train-set std, BCE loss, Adam `lr=1e-3`, 30 epochs,
best-on-validation checkpoint. Baseline = "keep current sign."

## Results so far
**Predictability ($\tau_Q=32$, $N=128$):** U-Net beats baseline from $t/\tau_Q \approx -0.25$.
At $t=0$: 0.723 vs 0.561 baseline. At $t/\tau_Q=0.75$: 0.994 vs 0.942.
Wall-length bias of predictions: $-12\%$ at $t=0$, $<1\%$ for $t/\tau_Q \ge 0.5$.

**Finite size:** $\tau_Q = 32$–$256$ at $N=128$ had 0.4–2.2 domains per side — unusable
for scaling. Faster quenches ($\tau_Q = 4$–$20$) regenerated and trained.

**Length scale:** standard definition, $L = 2\pi/\langle k\rangle$ from the first moment
of $S(k)$ of $\text{sign}(\phi)$. $N=256$ scan, final states:

| $\tau_Q$ | 10 | 20 | 40 | 80 | 160 |
|---|---|---|---|---|---|
| $L$ | 8.75 | 22.3 | 36.8 | 48.0 | 59.3 |

$L \sim \tau_Q^{0.46}$ ($\tau_Q \ge 20$), $\tau_Q^{0.34}$ ($\tau_Q \ge 40$) — still drifting.
$\tau_Q=10$ is outside the scaling regime.

**Collapse:** $L(t)$ curves collapse best with $\hat t \sim \tau_Q^{0.33}$ (by eye),
consistent with mean-field $\nu = 1/2$, $z = 1$.

## Caveats
- Final-state $L$ includes coarsening; late-time exponents are contaminated
- At early times ($t/\tau_Q \lesssim 0.1$) the field is unordered; $L$ hits the lattice floor
- $\tau_Q=160$ has $L \approx 59$, a quarter of the $N=256$ box
- Collapse exponent not yet quantified with an uncertainty
- Predictability is a lower bound (architecture-dependent)

## Next
1. $N=512$ at $\tau_Q = 160, 320$ — settles whether the exponent drift is finite size
2. Quantify the collapse (minimize spread vs. $\alpha$)
3. Train on input times at fixed $t/\hat t$, not fixed $t/\tau_Q$
4. Seeding vs. selection: predict $\phi(\hat t)$ from earlier, and $\phi_\text{final}$ from $\hat t$
------

