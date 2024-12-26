import cupy as cp
from abc import ABC, abstractmethod


class TimeIntegrator(ABC):
    def __init__(self, dt: float, linear_operator: cp.ndarray):
        self._dt = dt
        self._dt_sqrt = cp.sqrt(dt)
        self._L = linear_operator

    @abstractmethod
    def update_linear_operator(self, L: cp.ndarray):
        raise NotImplementedError

    @abstractmethod
    def integrate(self, u_hat, noise):
        raise NotImplementedError


class CrankNicolson(TimeIntegrator):
    def __init__(self, dt: float, linear_operator: cp.ndarray):
        super().__init__(dt, linear_operator)

        self._L_m_inv = 1 / (1 - dt / 2 * self._L)
        self._L_p = 1 + dt / 2 * self._L

    def update_linear_operator(self, L: cp.ndarray):
        self._L[:] = L
        self._L_m_inv = 1 / (1 - self._dt / 2 * self._L)
        self._L_p = 1 + self._dt / 2 * self._L

    def integrate(self, u_hat: cp.ndarray, noise: cp.ndarray):
        u_hat[:] = self._L_m_inv * (self._L_p * u_hat + self._dt_sqrt * noise)
