# Koopman:
A deep learning library for global and exact linearisation of non-linear systems.

## Why Koopman?

1. Future-state prediction:

The problem of future-state prediction for non-linear systems is non-trivial.

2. Automated Model Discovery:

By approximating Koopman Operators with machine learning, we discover parsimonious
models that generalise. This allows us to test important hypotheses that could
not be formulated otherwise using less principled machine learning methods.

3. Robust control of complex systems:

The exact and global linearisation of non-linear systems allows us to leverage
powerful and principled methods for controlling linear systems.

## Examples:

1. Global linearisation of the Lorenz system:

  a. Approximate Koopman operator for future-state prediction, 2-5 time increments into the future. On average, ~0.10 Mean Squared Error on test set.

  b. Code: (1) [Simulated data for the Lorenz system](https://github.com/AidanRocke/Koopman/blob/main/Lorenz_system/simulated_data.py), (2) [Model training and evaluation](https://github.com/AidanRocke/Koopman/blob/main/Lorenz_system/lorenz_koopman.py).

![Exact Lorenz System](https://raw.githubusercontent.com/AidanRocke/Koopman/main/Lorenz_system/images/exact_lorenz.png)
*Exact Lorenz System*

![Interpolated Lorenz System](https://raw.githubusercontent.com/AidanRocke/Koopman/main/Lorenz_system/images/approximate_lorenz.png)
*Interpolated Lorenz System*

## References:

1. Bernard Koopman. Hamiltonian systems and Transformations in Hilbert Space. 1931.

2. Hassan Arbabi. Introduction to Koopman operator theory for dynamical systems. MIT. 2020.

3. Steven L. Brunton. Notes on Koopman operator theory. 2019.

4. Bethany Lusch et al. Deep learning for universal linear embeddings of non-linear dynamics. nature. 2018.

5. Hassan Arbabi, Igor Mezić. Ergodic theory, Dynamic Mode Decomposition and Computation of Spectral properties of the Koopman operator. 2017.  
