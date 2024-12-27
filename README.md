# `fluc_stokes`: A simple Fourier solver for fluctuating Stokes equation 

## Introduction

This is a simple lightweight solver solving for fluctuating Stokes equation in Fourier space. The code also supports Lagrangian particle tracking (LPT) for passive point particles. It is written mostly based on CuPy, and so designated for running on GPUs. However, for now it is designed only for single GPU and multi-GPU parallelization is not supported. Consequently, this code is only designed for medium-size problems. 

## Algorithms

#### Fluctuating Stokes equation: 
1. solved in Fourier space
2. time integrated using a _Crank-Nicolson_ method ([Usabiaga _et al._, 2012](#UBB+2012) & [Delong _et al._, 2013](#DGED2013))

#### Lagrangian particle tracking:
1. local advecting velocity is interpolated using a _non-uniform FFT (NUFFT)_ provided by [`cufinufft`](https://github.com/flatironinstitute/finufft) ([Shih _et al._, 2021](#cufinufft))
2. the equation of motion is integrated in time using a _midpoint predictor-corrector_ scheme ([Delong _et al._, 2014](DUB+2014))



## References

1. <a id="UBB+2012"></a> Usabiaga, F. B., Bell, J. B., Delgado-Buscalioni, R., Donev, A., Fai, T. G., Griffith, B. E. and Peskin, C. S. 2012 Staggered schemes for fluctuating hydrodynamics. [_Multiscale Model. Simul._ 10(4), 1369-1408](https://doi.org/10.1137/120864520).
2. <a id="DGED2013"></a> Delong, S., Griffith, B. E., Vanden-Eijnden, E. and Donev, A. 2013 Temporal integrators for fluctuating hydrodynamics. [_Phys. Rev. E_ 87, 033302](https://doi.org/10.1103/PhysRevE.87.033302).
3. <a id="cufinufft"></a> Shih, Y.-h, Wright, G., Andén, J., Blaschke, J. & Barnett, A. H. 2021 cuFINUFFT: a load-balanced GPU library for general-purpose nonuniform FFTs. PDSEC2021 conference (best paper prize). [arXiv:2102.08463v2](https://arxiv.org/abs/2102.08463)
4. <a id="DUB+2014"></a> Delong, S., Usabiaga, F. B., Delgado-Buscalioni, R., Griffith, B. E., Donev, A. 2014 Brownian dynamics without Green's functions [J. Chem. Phys. 140, 134110](https://doi.org/10.1063/1.4869866)