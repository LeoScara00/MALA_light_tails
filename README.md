# MALA_light_tails

Research code for studying the convergence behavior of the Metropolis-Adjusted Langevin Algorithm (MALA) when the target measure has lighter than Gaussian tails.

## Main Idea
Given a Lyapunov function $V: \mathbb{R}^d \rightarrow [1,\infty)$, we study the behavior of the drift quantity \
                $$PV(x)-V(x)=\displaystyle\int_{\mathbb{R}^d}[V(y)-V(x)]\alpha(x,y)Q(x,dy)$$,
where $Y\sim Q(x,\cdot)$ is the MALA proposal and $\alpha(x,y)$ is the Metropolis-Hastings acceptance probability. 

The drift integral is decomposed into three regions:

* $A_1 := [ y \in \mathbb{R}^d: \lVert y \rVert < \lVert x \rVert -\epsilon ]$ represents inward moves;
* $A_2 := [y \in \mathbb{R}^d: \lVert x \rVert - \epsilon \leq \lVert y \rVert \leq \lVert x \rVert + \epsilon]$ represents local moves;
* $A_3 := [y \in \mathbb{R}^d: \lVert y \rVert > \lVert x \rVert + \epsilon]$ represents outward moves.

The objective is to understand which regions dominate the drift and how this behavior changes in the tails.

## Repository Structure

```text
MALA_light_tails/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
├── notebooks/
├── results/
└── notes/
```

## Main pieces

- `src/mala_1d.py`: main CLI entry point and compact import surface for the project
- `src/lyapunov.py`: target and Lyapunov functions
- `src/diagnostics.py`: MALA proposal, acceptance, and drift estimation
- `src/partitions.py`: signed one-dimensional region definitions
- `src/plotting.py`: research plots for drift diagnostics
- `src/utils.py`: small numerical helpers
- `notebooks/01_drift_diagnostics.ipynb`: sequential drift report
- `notebooks/02_exponential_lyapunov.ipynb`: exponential Lyapunov experiments

## Main Diagnostics 

The generated diagnostic plots display:

1. Total drift:
    $PV(x)-V(x)$.
2. Regional drift contributions:
    A_1,\ A_2,\ A_3.
3. Proposal probabilities:
    $Q(x,A_k)$.
4. Accepted transition mass:
    $\mathbb E[\alpha(x,Y)\mathbf 1_{A_k}]$.
5. Proposal overshoot diagnostic:
    $R(x)=\frac{h^2\|\nabla U(x)\|}{\|x\|}$.

## Quick start

Install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

Run the CLI from the repository root:

```bash
python MALA_1D.py --lyapunov poly --m 1 --p 2 --h 0.05 --plot
```

Plots are saved to `results/figures/` by default when `--plot` is used.

