import cupy as cp
from abc import ABC, abstractmethod


__all__ = ['ThermalNoise', 'CorrelatedNoise']


class _Noise(ABC):
    def __init__(
        self,
        rshape: list,
        cshape: list,
        noise_mag: float,
        mask_zero_mode: cp.ndarray,
        ctype: cp.dtype,
        k: cp.ndarray = None
    ):
        self._dim = len(rshape)
        self._noise_mag = noise_mag
        self._noise_normal_factor = mask_zero_mode / cp.sqrt(cp.prod(cp.array(rshape)) * 2)

        self._raw_noise = cp.zeros((self._dim, *cshape), dtype=ctype)
        self.noise = cp.zeros((self._dim, *cshape), dtype=ctype)

    def _generate_random_fields(self):
        shape = self._raw_noise.shape
        norm_fac = self._noise_normal_factor
        self._raw_noise[:] = cp.random.randn(*shape) + 1j * cp.random.randn(*shape)
        self._raw_noise *= norm_fac

    @abstractmethod
    def _assemble_noise(self):
        raise NotImplementedError

    def update(self):
        self._assemble_noise()
        self.noise *= self._noise_mag


class ThermalNoise(_Noise):
    def __init__(
        self,
        rshape: list,
        cshape: list,
        noise_mag: float,
        mask_zero_mode: cp.ndarray,
        ctype: cp.dtype,
        k: cp.ndarray
    ):
        super().__init__(rshape, cshape, noise_mag, mask_zero_mode, ctype)

        self._k = k

    def _assemble_noise(self):
        for i in range(self._dim):
            self._generate_random_fields()
            self.noise[i] = cp.sum(1j * self._k * self._raw_noise, axis=0)


class CorrelatedNoise(_Noise):
    def __init__(
        self,
        rshape: list,
        cshape: list,
        noise_mag: float,
        mask_zero_mode: cp.ndarray,
        ctype: cp.dtype,
        k: cp.ndarray = None
    ):
        super().__init__(rshape, cshape, noise_mag, mask_zero_mode, ctype)

        self.corr_func = cp.ones(cshape)

    def set_noise_correlation(self, C):
        self.corr_func[:] = C

    def _assemble_noise(self):
        self._generate_random_fields()
        self.noise[:] = self.corr_func * self._raw_noise
