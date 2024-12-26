# `fluc_stokes`: A simple Fourier solver for fluctuating Stokes equation 

## Introduction

This is a simple lightweight solver solving for fluctuating Stokes equation in Fourier space. The code also supports Lagrangian particle tracking (LPT) for passive point particles. It is written mostly based on CuPy, and so designated for running on GPUs. However, for now it is designed only for single GPU and multi-GPU parallelization is not supported. Consequently, this code is only designed for medium-size problems. 

## Algorithms

1. Fluctuating Stokes equation solved in Fourier space
2. The 2nd-order _Crank-Nicolson_ for time integration 
3. A _midpoint predictor-corrector_ time integration scheme for particle tracking 