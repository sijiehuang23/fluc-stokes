import numpy as np
import scipy.fft as fft
from .time_integrator import CrankNicolson
from .io import Params, SolutionWriter
from .kspace import kspace, fftn_jit, rfftn_jit, irfftn_jit
from .noise import ThermalNoise, CorrelatedNoise
from .lagrangian_particles import LagrangianParticles
from .utils import Timer
from periodicflow.math import leray_projection, apply_linear_operator
from . import logger


class StokesSolver:
    def __init__(self, params: Params):
        if params.verbose:
            logger.info("Instantiating Steady Stokes solver")

        self.params = params
        self._full_shape = params.N.copy()
        self._real_shape = params.N.copy()
        self._real_shape[-1] = self._real_shape[-1] // 2 + 1

        self._generate_grid()
        self._initialize()
        self._init_time_integrator()
        self._init_noise()
        self._init_h5writer()
        self._init_lagrangian_particles()

        self.t = 0.0
        self.step = 0

        self._timer = Timer(params.verbose, params.check_interval)

    def _generate_grid(self):
        self._dim = len(self.params.N)
        if self._dim not in [2, 3]:
            raise ValueError("Only 2D and 3D are supported")

        self._x = [
            np.linspace(d[0], d[1], n, endpoint=False)
            for d, n in zip(self.params.domain, self.params.N)
        ]
        self.X = np.meshgrid(*self._x, indexing='ij')

        lengths = [d[1] - d[0] for d in self.params.domain]
        k, self.ksqrt, self._nyquist_mask = kspace(self.params.N, lengths)
        self.k = np.array(k, dtype=np.float64)
        self.k_over_k2 = self.k / np.where(self.ksqrt == 0, 1, self.ksqrt**2)
        self._k0_mask_0 = np.where(self.ksqrt == 0, 0, 1)

    def _initialize(self):
        self.u = np.zeros((self._dim, *self._full_shape), dtype=np.float64)
        self.u_hat = np.zeros((self._dim, *self._real_shape), dtype=np.complex128)
        self.p_hat = np.zeros(self._real_shape, dtype=np.complex128)

        self.u_hat_full = np.zeros((self._dim, *self._full_shape), dtype=np.complex128)

        self.fft_axes = tuple(range(1, self._dim + 1))

        if self.params.filter:
            self.u_bar_hat = np.zeros_like(self.u_hat)

        self.L = -self.params.viscosity * self.ksqrt**2
        self.G = np.ones(self._real_shape, dtype=np.float64)

    def _init_time_integrator(self):
        self.time_integrator = CrankNicolson(self.L, self.params.dt)

    def _init_noise(self):
        if self.params.verbose:
            logger.info(f"Initializing noise of type {self.params.noise_type}")

        if self.params.noise_type == 'thermal':
            self.noise = ThermalNoise(
                self._full_shape,
                self._real_shape,
                self.k, self.k_over_k2, self._k0_mask_0,
                self.params.noise_mag
            )
        elif self.params.noise_type == 'correlated':
            self.noise = CorrelatedNoise(
                self._full_shape,
                self._real_shape,
                self.k, self.k_over_k2, self._k0_mask_0,
                self.params.noise_mag
            )
        else:
            raise ValueError(f"Unknown noise type: {self.params.noise_type}")

    def _init_lagrangian_particles(self):
        if self.params.enable_particles:
            self.lagrangian_particles = LagrangianParticles(
                self._x,
                self.params.n_particles,
                self.params.dt,
                self.params.write_particle,
                self.params.particle_file_name,
                self.params.particle_write_mode
            )

    def _init_h5writer(self):
        if self.params.write_velocity:
            self.data_writer = SolutionWriter(self.params.velocity_file_name, self._dim)
            self.data_writer.initialize(self._x)

    def initialize_velocity(self, u_hat0: np.ndarray):
        self.u_hat[:] = u_hat0
        leray_projection(
            self.u_hat,
            self.k,
            self.k_over_k2,
            self.p_hat
        )
        self.backward()
        self.reconstruct_full_spectrum()

    def initialize_particles(self, x):
        self.lagrangian_particles.initialize(x, self.u_hat_full)

    def set_linear_operator(self, L: np.ndarray):
        self.L[:] = L

    def set_filter_kernel(self, G):
        self.G[:] = G

    def set_noise_correlation(self, C):
        if hasattr(self.noise, '_set_noise_correlation'):
            self.noise._set_noise_correlation(C)
        else:
            raise ValueError("Noise type does not support setting correlation")

    def forward(self):
        self.u_hat[:] = rfftn_jit(self.u, axes=self.fft_axes)

    def backward(self):
        self.u[:] = irfftn_jit(self.u_hat, axes=self.fft_axes)

    def reconstruct_full_spectrum(self):
        self.backward()
        self.u_hat_full[:] = fft.fftshift(fftn_jit(self.u, axes=self.fft_axes), axes=self.fft_axes)

    def solve(self):
        self._timer.start()

        dt = self.params.dt
        # Linv = -1 / np.where(self.L == 0, 1, self.L) * self._k0_mask_0 / np.sqrt(dt)

        if self.params.write_velocity and self.params.velocity_write_first_step:
            self.data_writer.write(self.u, self.t, self.step)

        while not np.isclose(self.t, self.params.end_time, atol=1e-8):
            self.t += dt
            self.step += 1

            self._timer(self.t, self.step)

            if self.params.enable_particles:
                self.reconstruct_full_spectrum()
                self.lagrangian_particles.stepping(self.u_hat_full)

            self.noise.update()
            self.time_integrator.stepping(self.u_hat, self.noise.noise)
            # apply_linear_operator(self.u_hat, Linv, self.noise.noise)

            if self.params.filter:
                apply_linear_operator(self.u_bar_hat, self.G, self.u_hat)

            if self.params.write_velocity and self.step % self.params.velocity_write_interval == 0:
                self.backward()
                self.data_writer.write(self.u, self.t, self.step)

            if self.params.enable_particles:
                if self.params.write_particle and self.step % self.params.particle_write_interval == 0:
                    self.lagrangian_particles.write(self.t, self.step)

        self._timer.stop()
