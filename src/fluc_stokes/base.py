import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft


__all__ = ['FourierSpace']


def _wavevector(shape: list, dtype: cp.dtype):
    k = [cp.fft.fftfreq(n, 1 / n) for n in shape]
    k[-1] = k[-1][:shape[-1] // 2 + 1]
    k = cp.array(cp.meshgrid(*k, indexing='ij'), dtype=dtype)
    k2 = cp.sum(k**2, axis=0, dtype=dtype)

    return k, k2


class FourierSpace:
    def __init__(self, N: list, domain: list[list[float]], rtype: cp.dtype):
        if len(domain) != len(N):
            raise ValueError("Length of domain must match the number of dimensions.")

        self.rtype = rtype
        self.N = N
        self.domain = domain
        self.ndim = len(N)
        if self.ndim not in [2, 3]:
            raise ValueError("Only 2D and 3D are supported")

        self._generate_grid(domain, rtype)
        self._set_fft_params()

        self.k, self.k2 = _wavevector(self.rshape, rtype)
        self.k_over_k2 = self.k / cp.where(self.k2 == 0, 1, self.k2)
        self.mask_zero_mode = cp.where(self.k2 == 0, 0, 1)

    def _generate_grid(self, domain: list[list[float]], rtype: cp.dtype):
        d_start, d_end = zip(*domain)
        self.x_gpu = [
            cp.linspace(ds, de, n, endpoint=False, dtype=rtype)
            for ds, de, n in zip(d_start, d_end, self.N)
        ]
        self.x_cpu = [x.get().astype(np.float64) for x in self.x_gpu]

    def _set_fft_params(self):
        self.rshape = list(self.N)
        self.cshape = list(self.N)
        self.cshape[-1] = self.N[-1] // 2 + 1
        self._fft_axes = tuple(range(1, self.ndim + 1))

    def init_fft_plan(self, u: cp.ndarray, u_hat: cp.ndarray):
        self.forward_plan = cufft.get_fft_plan(u, axes=self._fft_axes, value_type='R2C')
        self.backward_plan = cufft.get_fft_plan(u_hat, axes=self._fft_axes, value_type='C2R')
        self.full_plan = cufft.get_fft_plan(u, axes=self._fft_axes)

    def r2c(self, u_hat: cp.ndarray, u: cp.ndarray):
        u_hat[:] = cufft.rfftn(u, norm='forward', axes=self._fft_axes, plan=self.forward_plan)

    def c2r(self, u: cp.ndarray, u_hat: cp.ndarray):
        u[:] = cufft.irfftn(u_hat, norm='forward', axes=self._fft_axes, plan=self.backward_plan)

    def c2c(self, u_hat: cp.ndarray, u: cp.ndarray):
        u_hat[:] = cufft.fftn(u, norm='forward', axes=self._fft_axes, plan=self.full_plan)
