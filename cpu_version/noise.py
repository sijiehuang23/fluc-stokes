import numpy as np
import numba as nb
from abc import ABC, abstractmethod
from periodicflow.math import leray_projection, apply_linear_operator


@nb.njit
def _symmetrize(W, Z):
    dim = W.ndim
    shape = W.shape

    factor = np.sqrt(0.5)

    if dim == 3:
        for j in range(shape[1]):
            for k in range(shape[2]):
                W[0, j, k] = (Z[0, j, k] + Z[0, j, k]) * factor
                W[1, j, k] = (Z[1, j, k] + Z[2, j, k]) * factor
                W[2, j, k] = (Z[2, j, k] + Z[1, j, k]) * factor
                W[3, j, k] = (Z[3, j, k] + Z[3, j, k]) * factor

    elif dim == 4:
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    W[0, j, k, l] = (Z[0, j, k, l] + Z[0, j, k, l]) * factor
                    W[1, j, k, l] = (Z[1, j, k, l] + Z[3, j, k, l]) * factor
                    W[2, j, k, l] = (Z[2, j, k, l] + Z[6, j, k, l]) * factor
                    W[3, j, k, l] = (Z[3, j, k, l] + Z[1, j, k, l]) * factor
                    W[4, j, k, l] = (Z[4, j, k, l] + Z[4, j, k, l]) * factor
                    W[5, j, k, l] = (Z[5, j, k, l] + Z[7, j, k, l]) * factor
                    W[6, j, k, l] = (Z[6, j, k, l] + Z[2, j, k, l]) * factor
                    W[7, j, k, l] = (Z[7, j, k, l] + Z[5, j, k, l]) * factor
                    W[8, j, k, l] = (Z[8, j, k, l] + Z[8, j, k, l]) * factor


@nb.njit
def _noise_divergence(noise, W, k):
    dim = noise.ndim
    shape = noise.shape

    if dim == 3:
        for i in range(shape[1]):
            for l in range(shape[2]):
                kx, ky = k[0, i, l], k[1, i, l]
                noise[0, i, l] = 1j * (kx * W[0, i, l] + ky * W[1, i, l])
                noise[1, i, l] = 1j * (kx * W[2, i, l] + ky * W[3, i, l])

    elif dim == 4:
        for i in range(shape[1]):
            for j in range(shape[2]):
                for l in range(shape[3]):
                    kx, ky, kz = k[0, i, j, l], k[1, i, j, l], k[2, i, j, l]
                    noise[0, i, j, l] = 1j * (kx * W[0, i, j, l] + ky * W[1, i, j, l] + kz * W[2, i, j, l])
                    noise[1, i, j, l] = 1j * (kx * W[3, i, j, l] + ky * W[4, i, j, l] + kz * W[5, i, j, l])
                    noise[2, i, j, l] = 1j * (kx * W[6, i, j, l] + ky * W[7, i, j, l] + kz * W[8, i, j, l])


class _Noise(ABC):
    def __init__(self, full_shape, real_shape, k, k_over_k2, k0_mask_0, noise_mag):
        self._shape = real_shape
        self._dim = len(full_shape)
        self._k = k
        self._k0_mask_0 = k0_mask_0
        self._k_over_k2 = k_over_k2
        self._noise_mag = noise_mag

        self._prod_n_sqrt = np.sqrt(np.prod(full_shape))
        self._noise_normal_factor = self._k0_mask_0 / self._prod_n_sqrt / np.sqrt(2)

        self.noise = np.zeros((self._dim, *self._shape), dtype=np.complex128)
        self._p_hat = np.zeros(self._shape, dtype=np.complex128)

    @abstractmethod
    def _random_fields(self):
        raise NotImplementedError

    @abstractmethod
    def _generate_noise(self):
        raise NotImplementedError

    def update(self):
        self._generate_noise()
        leray_projection(self.noise, self._k, self._k_over_k2, self._p_hat)
        self.noise *= self._noise_mag


class ThermalNoise(_Noise):
    def __init__(self, full_shape, real_shape, k, k_over_k2, k0_mask_0, noise_mag):
        super().__init__(full_shape, real_shape, k, k_over_k2, k0_mask_0, noise_mag)

        self.Z = np.zeros((self._dim**2, *self._shape), dtype=np.complex128)
        self.W = np.zeros_like(self.Z)

    def _random_fields(self):
        shape = self.Z.shape
        normal_factor = self._noise_normal_factor

        self.Z[:] = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        self.Z *= normal_factor

    def _generate_noise(self):
        self._random_fields()
        _symmetrize(self.W, self.Z)
        _noise_divergence(self.noise, self.W, self._k)


class CorrelatedNoise(_Noise):
    def __init__(self, full_shape, real_shape, k, k_over_k2, k0_mask_0, noise_mag):
        super().__init__(full_shape, real_shape, k, k_over_k2, k0_mask_0, noise_mag)

        self.eta = np.zeros((self._dim**2, *self._shape), dtype=np.complex128)
        self.corr_func = np.ones(self._shape)

    def _set_noise_correlation(self, C):
        self.corr_func[:] = C

    def _random_fields(self):
        shape = self.eta.shape
        normal_factor = self._noise_normal_factor

        self.eta[:] = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        self.eta *= normal_factor

    def _generate_noise(self):
        self._random_fields()
        apply_linear_operator(self.noise, self.corr_func, self.eta)
