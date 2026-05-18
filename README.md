# MALA_light_tails

Lightweight computational research code for studying drift diagnostics of the Metropolis-adjusted Langevin algorithm under light-tailed targets.

## Layout

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

## Notes

This repository is intentionally research-oriented: small modules, notebook-driven experiments, and minimal packaging overhead.
