import numpy as np
import numba as nb


class CrankNicolson:
    def __init__(self, L: np.ndarray, dt: float):
        self.L = L
        self.dt = dt
        self.dtsqrt = np.sqrt(dt)
        self.L_m = 1 - dt / 2 * L
        self.L_p = 1 + dt / 2 * L
        self.L_m_inv = 1 / self.L_m

    def stepping(self, u_hat: np.ndarray, noise: np.ndarray):
        self._stepping_jit(u_hat, noise, self.dtsqrt, self.L_p, self.L_m_inv)

    @staticmethod
    @nb.njit
    def _stepping_jit(u_hat, noise, dtsqrt, L_p, L_m_inv):
        ndim = u_hat.ndim
        shape = u_hat.shape

        if ndim == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        u_hat[i, j, k] = L_m_inv[j, k] * (L_p[j, k] * u_hat[i, j, k] + dtsqrt * noise[i, j, k])

        elif ndim == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            u_hat[i, j, k, l] = L_m_inv[j, k] * (
                                L_p[j, k, l] * u_hat[i, j, k, l] + dtsqrt * noise[i, j, k, l]
                            )
