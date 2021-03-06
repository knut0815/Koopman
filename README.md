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

## The theoretical minimum:

1. [Dynamical systems and the Koopman Operator](https://keplerlounge.com/physics/2021/09/30/koopman-operator.html)

2. [The Buckingham-Pi theorem, or the fundamental theorem of Dimensionless Analysis](https://keplerlounge.com/physics/2021/09/30/buckingham-pi.html)

3. [Koopman Modes via Singular Value Decomposition](https://keplerlounge.com/physics/2021/10/01/SVD-DMD.html)

4. [The Koopman Operator as a Discrete Fourier Transform for dynamical systems](https://keplerlounge.com/physics/2021/10/01/koopman-DFT.html)

5. [Deep Koopman Operators, the Identity Operator, and Euler's method](https://keplerlounge.com/physics/2021/10/03/koopman-euler.html)

## Examples:

1. Global linearisation of the Lorenz system:

    1. Approximate Koopman operator for future-state prediction, 2-5 time increments into the future. On average, ~0.10 Mean Squared Error on test set. For the training data, only five trajectories were used with 4000 observations per trajectory.

    2. Code: (1) [Simulated data for the Lorenz system](https://github.com/AidanRocke/Koopman/blob/main/Lorenz_system/simulated_data.py), (2) [Koopman approximation](https://github.com/AidanRocke/Koopman/blob/main/Lorenz_system/koopman_approximation.py), (3) [Model evaluation](https://github.com/AidanRocke/Koopman/blob/main/Lorenz_system/model_evaluation.py).

![Exact Lorenz System](https://raw.githubusercontent.com/AidanRocke/Koopman/main/Lorenz_system/images/exact_lorenz.png)
<figcaption align = "center"><b>Exact Lorenz System</b></figcaption>

![Interpolated Lorenz System](https://raw.githubusercontent.com/AidanRocke/Koopman/main/Lorenz_system/images/approximate_lorenz.png)
<figcaption align = "center"><b>Interpolated Lorenz System</b></figcaption>

2. Global linearisation of von Karman vortex shedding:

    1. Approximate Koopman operator for future-state prediction, 5-10 time increments into the future. On average, ~0.01 Mean Squared Error on test set. For the training data, only five trajectories were used with 4000 observations per trajectory which is less than 0.1% of the data used in [4].

    2. Code: (1) [Simulated data for the system](https://github.com/AidanRocke/Koopman/blob/main/von_karman/simulated_data.py), (2) [Koopman approximation](https://github.com/AidanRocke/Koopman/blob/main/von_karman/koopman_approximation.py), (3) [Model evaluation](https://github.com/AidanRocke/Koopman/blob/main/von_karman/model_evaluation.py). Finally, with the [Von Neumann entropy]((https://github.com/AidanRocke/Koopman/blob/main/von_karman/von_neumann_entropy.py)) we find that (4) three dimensions contain 95% of the information in the dynamical system.  

![Exact limit cycle](https://raw.githubusercontent.com/AidanRocke/Koopman/main/von_karman/images/exact_von_karman.png)
<figcaption align = "center"><b>Exact limit cycle</b></figcaption>

![Interpolated limit cycle](https://raw.githubusercontent.com/AidanRocke/Koopman/main/von_karman/images/interpolated_von_karman.png)
<figcaption align = "center"><b>Interpolated limit cycle</b></figcaption>

## References:

1. Bernard Koopman. Hamiltonian systems and Transformations in Hilbert Space. 1931.

2. Hassan Arbabi. Introduction to Koopman operator theory for dynamical systems. MIT. 2020.

3. Steven L. Brunton. Notes on Koopman operator theory. 2019.

4. Bethany Lusch et al. Deep learning for universal linear embeddings of non-linear dynamics. nature. 2018.

5. Hassan Arbabi, Igor Mezi??. Ergodic theory, Dynamic Mode Decomposition and Computation of Spectral properties of the Koopman operator. 2017.  
