# SelectiveML

**A model should know when to ask for help. How do you measure whether it does?**

A compact, inspectable ML engineering project: train a neural classifier from scratch, calibrate its confidence, choose a review threshold without touching the test set, and expose where that policy breaks. NumPy is the only dependency. No API keys, downloaded model weights, or private data.

![Measured experiment results](results/reference/dashboard.svg)

## Results, not just a demo

The checked-in experiment uses the public 1,797-image digits dataset. Each seed has 984 training, 268 calibration, 270 policy-selection, and 275 test examples.

| Seed | Linear accuracy | Neural accuracy | Test coverage | Accepted accuracy | Noisy-input accuracy |
|---|---:|---:|---:|---:|---:|
| 42 | 96.36% | 97.45% | 100.00% | 97.45% | 90.18% |
| 43 | 96.00% | 97.45% | 99.64% | 97.81% | 93.82% |
| 44 | 95.64% | 97.82% | 97.45% | 98.51% | 94.18% |

These are **small benchmark results**, not production performance. The three test sets overlap: do not pool them as 825 independent observations. The 98% policy target is empirical on the policy split, **not a test-set or deployment guarantee**. Temperature scaling worsened held-out negative log-likelihood in two of three seeds. That negative result is retained.

## Run it

Python 3.12+; reference run: Python 3.12.14, NumPy 2.5.2. A standard CPU is sufficient.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
OPENBLAS_NUM_THREADS=1 python selective_ml.py experiment --output results/local
python report.py results/local
python selective_ml.py predict --model results/local/model.json --input results/local/example.json
```

On Windows use `.venv\Scripts\activate` and omit the environment-variable prefix. Inspect `results/local/dashboard.svg` and `metrics.json`. The checked-in model also runs immediately using the same predict command with `results/reference`.

## Architecture and protocol

```text
Checksum-verified pixels, divided by the fixed constant 16
  ├─ train:       Adam → linear softmax baseline + 64→64 ReLU→10 MLP
  ├─ calibration: fixed temperature grid → minimum calibration NLL
  ├─ policy:      maximum coverage with ≥98% empirical accuracy, ≥50% coverage
  └─ test:        accuracy / macro-F1 / NLL / Brier / ECE / errors / noise stress
                              ↓
                   JSON model → validated batch inference → accept or review
```

- Both models use fixed 120-epoch training, batch size 64, learning rate 0.003, L2 0.0001. No test-driven tuning or early stopping.
- The MLP has **4,810 trainable parameters**. Backpropagation and Adam are implemented directly and checked against finite-difference gradients.
- Temperature is selected from a fixed logarithmic grid [0.25, 4]. It changes confidence, not predicted labels.
- If no policy threshold satisfies the development constraint, every prediction is deferred. “Review” is a signal, not an implemented human-review service.
- The corruption stress test adds fixed-seed Gaussian noise, sigma 0.2 on normalized pixels, then clips to [0,1]. It is not a realistic out-of-distribution benchmark.
- Export uses JSON rather than executable pickle; inference validates input range, shapes, finite weights and outputs.

## Evidence you can inspect

- [`metrics.json`](results/reference/metrics.json): all seeds, majority/linear/neural baselines, calibration bins, confusion matrices, per-class recall, risk/coverage points, every wrong prediction, and bootstrap accuracy intervals.
- [`splits.json`](results/reference/splits.json): exact row membership for seed 42; other seeds reproduce deterministically.
- [`model.json`](results/reference/model.json): seed-42 weights, temperature and policy; no “best-seed” selection.
- [`test_selective_ml.py`](test_selective_ml.py): gradient checks, data integrity, split independence, policy failure, calibration objective, reproducibility, tiny-set overfit, safe input handling and artifact round trip.
- [`MODEL_CARD.md`](MODEL_CARD.md): limitations and intended use.

## What I would improve next

Use a harder task with a writer- or source-independent split, compare PyTorch and classical baselines, add conformal prediction with explicit assumptions, and evaluate human-review cost. Do not infer that digit-recognition accuracy demonstrates LLM, production computer-vision, or employer-data performance.

## Attribution

Dataset: E. Alpaydin and C. Kaynak, [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49), UCI (1998), CC BY 4.0. This repository vendors the 1,797-row subset distributed by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html); it does **not** reproduce the original UCI writer split. See [data provenance](data/README.md).

Temperature-scaling reference: Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html), ICML 2017. This is an educational implementation, not a reproduction of that paper's experiments. Developed with AI assistance; results come from the executable experiment, not generated claims.
