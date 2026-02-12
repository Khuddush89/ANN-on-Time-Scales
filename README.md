# ANN-on-Time-Scales

Residual-based artificial neural network (ANN) framework for solving dynamic equations on time scales, including hybrid continuous–discrete population dynamics models.

---

## Overview

This repository accompanies the research work on artificial neural networks for dynamic equations on time scales. The implementation integrates:

- Time scale calculus (delta derivatives)
- Residual minimization for initial value problems (IVPs)
- Fixed-step batch gradient descent training
- Hybrid continuous–discrete modeling
- Chebyshev–Gauss collocation

The main benchmark experiment reproduces a population dynamics model defined on the hybrid time scale

\[
\mathbb{T} = \bigcup_{k=0}^{5} [2k, 2k+1],
\]

capturing continuous decay phases and discrete reproduction events.

---

## Repository Structure

ANN-on-Time-Scales/
│
├── training/
│ └── first_training_sigmoid.py # ANN training script
│
├── figures/
│ ├── architecture_plot.py # ANN architecture visualization
│ ├── sigmoid_time_scale.py # Continuous/discrete sigmoid plots
│ ├── bipolar_sigmoid.py # Bipolar activation functions
│ └── tanh_time_scale.py # Hyperbolic tangent on time scales
│
├── requirements.txt # Python dependencies
└── README.md
