import numpy as np
import cupy as cp
from .io import Params, ParticleWriter


try:
    import cufinufft as cnft
except ImportError:
    raise ImportError("finufft is needed for lagrangian_particles")


class LagrangianParticles:
    def __init__(self, params: Params):
        self._set_coordinates(params.domain)
        self._N = params.N
        self._rtype = params.rtype
        self._ctype = params.ctype
        self._n_particles = params.n_particles
        self._dt = params.dt
        self._write_particle = params.write_particle
        self._file_name = params.particle_file_name
        self._write_mode = params.particle_write_mode

        self._n_stages = 2
        self._mpc_factor = cp.array([0.5, 1.0], dtype=self._rtype)

        self._initialize()
        self._init_nufft_plan()
        self._init_h5writer()

    def _set_coordinates(self, domain: list[list]):
        self._ndim = len(domain)

        self._l_boundary = cp.array([d[0] for d in domain])
        self._l_boundary = self._l_boundary[:, cp.newaxis]

        self._L = cp.array([d[1] - d[0] for d in domain])
        self._L = self._L[:, cp.newaxis]

    def _initialize(self):
        self._p_pos_prev = cp.zeros((self._ndim, self._n_particles), dtype=self._rtype)
        self.p_pos_gpu = cp.zeros((self._ndim, self._n_particles), dtype=self._rtype)
        self.p_traj_gpu = cp.zeros((self._ndim, self._n_particles), dtype=self._rtype)
        self.p_vel_gpu = cp.zeros((self._ndim, self._n_particles), dtype=self._rtype)

        self.p_pos_gpu[:] = cp.random.uniform(0, 2 * cp.pi, self.p_pos_gpu.shape, dtype=self._rtype)

        self.p_pos_cpu = np.zeros((self._ndim, self._n_particles), dtype=self._rtype)
        self.p_traj_cpu = np.zeros((self._ndim, self._n_particles), dtype=self._rtype)
        self.p_vel_cpu = np.zeros((self._ndim, self._n_particles), dtype=self._rtype)

    def _init_h5writer(self):
        if self._write_particle:
            self.data_writer = ParticleWriter(self._file_name, self._write_mode)

    def _init_nufft_plan(self):
        self._nufft_plan = cnft.Plan(2, self._N, eps=1e-12, isign=1, dtype=self._ctype)

    def _update_particle_velocity(self, u_hat: cp.ndarray):
        if self._ndim == 2:
            self._nufft_plan.setpts(self.p_pos_gpu[0], self.p_pos_gpu[1])
        elif self._ndim == 3:
            self._nufft_plan.setpts(self.p_pos_gpu[0], self.p_pos_gpu[1], self.p_pos_gpu[2])

        for i in range(self._ndim):
            self.p_vel_gpu[i] = self._nufft_plan.execute(u_hat[i]).real

    def _update_previous_position(self):
        self._p_pos_prev[:] = self.p_pos_gpu

    def initialize(self, xp: cp.ndarray):
        if xp.shape != (self._ndim, self._n_particles):
            raise ValueError(f"Position array shape {xp.shape} must match ({self._ndim}, {self._n_particles}).")
        self.p_pos_gpu[:] = xp

    def gpu_to_cpu(self):
        self.p_pos_cpu[:] = self.p_pos_gpu.get()
        self.p_traj_cpu[:] = self.p_traj_gpu.get()
        self.p_vel_cpu[:] = self.p_vel_gpu.get()

    def write(self, t: float, step: int):
        self.gpu_to_cpu()
        self.data_writer.write(t, step, self.p_pos_cpu, self.p_traj_cpu, self.p_vel_cpu)

    def _midpoint_predictor_corrector(self, stage):
        a = self._mpc_factor[stage]
        self.p_pos_gpu[:] = self._p_pos_prev + a * self._dt * self.p_vel_gpu

        if stage == 1:
            self.p_traj_gpu[:] = self._p_pos_prev + a * self._dt * self.p_vel_gpu

    def _periodic_bc(self):
        self.p_pos_gpu[:] = (self.p_pos_gpu - self._l_boundary) % self._L + self._l_boundary

    def stepping(self, u_hat: cp.ndarray):
        """
        Time stepping loop for the Lagrangian Particle Tracking (LPT). In each step, the motion of the particles is integrated using a midpoint predictor-corrector scheme.
        """

        self._update_previous_position()
        for stage in range(self._n_stages):
            self._update_particle_velocity(u_hat)
            self._midpoint_predictor_corrector(stage)
            self._periodic_bc()
