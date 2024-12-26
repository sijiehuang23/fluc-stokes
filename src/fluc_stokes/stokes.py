import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
from .base import FourierSpace
from .io import Params
from .math import leray_projection


class Stokes(FourierSpace):
    def __init__(self, params: Params):
        super().__init__(params.N, params.domain, params.rtype)

        self.params = params

        self._allocate()
        self.init_fft_plan(self.u_gpu, self.u_hat)

    def _allocate(self):
        self.u_gpu = cp.zeros((self.ndim, *self.rshape), dtype=self.params.rtype)
        self.u_cpu = np.zeros((self.ndim, *self.rshape), dtype=self.params.rtype)
        self.u_hat = cp.zeros((self.ndim, *self.cshape), dtype=self.params.ctype)
        self.u_hat_full = cp.zeros((self.ndim, *self.rshape), dtype=self.params.ctype)

        self.filter_kernel = cp.ones(self.cshape, dtype=self.params.rtype)

    def initialize(self, u: cp.ndarray, space: str = 'fourier'):
        if u.shape != self.u_gpu.shape:
            raise ValueError(
                f"Input array shape {u.shape} does not match expected shape {self.u_gpu.shape}."
            )
        if space.casefold() == 'fourier':
            self.u_hat[:] = u
            self.project()
            self.backward()
        elif space.casefold() == 'physical':
            self.u_gpu[:] = u
            self.forward()
            self.project()
        else:
            raise ValueError("Invalid space. Use 'fourier' or 'physical'.")
        self.update_full_velocity()

    def update_filter_kernel(self, G: cp.ndarray):
        self.filter_kernel[:] = G

    def project(self):
        leray_projection(self.u_hat, self.k, self.k_over_k2)

    def forward(self):
        self.r2c(self.u_hat, self.u_gpu)

    def backward(self):
        self.c2r(self.u_gpu, self.u_hat)

    def update_full_velocity(self):
        self.c2c(self.u_hat_full, self.u_gpu)
        self.u_hat_full[:] = cufft.fftshift(self.u_hat_full, axes=self._fft_axes)

    def filter(self):
        self.u_hat *= self.filter_kernel

    def gpu_to_cpu(self):
        self.u_cpu[:] = self.u_gpu.get()
