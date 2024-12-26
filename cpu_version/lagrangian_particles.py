import numpy as np
import numba as nb
from .io import ParticleWriter


try:
    import finufft
except ImportError:
    raise ImportError("finufft is needed for lagrangian_particles")


class LagrangianParticles:
    def __init__(
        self,
        grid: list[np.ndarray],
        n_particles: int = 100,
        dt: np.float64 = 0.01,
        write_particle=False,
        file_name='particles',
        write_mode='w'
    ):
        self._set_coordinates(grid)
        self._n_particles = n_particles
        self._dt = dt
        self._write_particle = write_particle
        self._file_name = file_name
        self._write_mode = write_mode

        self._n_stages = 2
        self._mpc_factor = [0.5, 1.0]

        self._initialize()
        self._init_nufft_plan()
        self._init_h5writer()

    def _set_coordinates(self, grid: list[np.ndarray]):
        self._grid = grid
        self._ndim = len(grid)
        self._N = [len(g) for g in grid]
        self._domain = np.array([[g.min(), g.max()] for g in grid])
        self._L = np.array([g.max() - g.min() for g in grid])

    def _initialize(self):
        self._p_pos_prev = np.zeros((self._ndim, self._n_particles), dtype=np.float64)
        self.p_position = np.zeros((self._ndim, self._n_particles), dtype=np.float64)
        self.p_trajectory = np.zeros((self._ndim, self._n_particles), dtype=np.float64)
        self.p_velocity = np.zeros((self._ndim, self._n_particles), dtype=np.float64)

        self.p_position[:] = np.random.uniform(0, 2 * np.pi, self.p_position.shape)

    def _init_h5writer(self):
        if self._write_particle:
            self.data_writer = ParticleWriter(self._file_name, self._write_mode)

    def _init_nufft_plan(self):
        self.nufft_plan = finufft.Plan(2, self._N, eps=1e-12, isign=1)

    def _update_particle_velocity(self, u_hat: np.ndarray):
        if self._ndim == 2:
            self.nufft_plan.setpts(self.p_position[0], self.p_position[1])
        elif self._ndim == 3:
            self.nufft_plan.setpts(self.p_position[0], self.p_position[1], self.p_position[2])

        for i in range(self._ndim):
            self.p_velocity[i] = self.nufft_plan.execute(u_hat[i]).real

    def _update_previous_position(self):
        self._p_pos_prev[:] = self.p_position

    def initialize(self, xp: np.ndarray, u_hat: np.ndarray):
        if xp.shape != (self._ndim, self._n_particles):
            raise ValueError(f"Position array shape {xp.shape} must match ({self._ndim}, {self._n_particles}).")
        self.p_position[:] = xp
        self._update_particle_velocity(u_hat)

    def write(self, t: np.float64, step: int):
        self.data_writer.write(t, step, self.p_position, self.p_trajectory, self.p_velocity)

    @staticmethod
    @nb.njit
    def _midpoint_predictor_corrector(
        pos_curr: np.ndarray,
        pos_prev: np.ndarray,
        vel: np.ndarray,
        traj: np.ndarray,
        a: float, dt: float, stage: int
    ):
        shape = pos_curr.shape

        for i in range(shape[0]):
            for j in range(shape[1]):
                dx = a * dt * vel[i, j]
                pos_curr[i, j] = pos_prev[i, j] + dx

                if stage == 1:
                    traj[i, j] += dx

    @staticmethod
    @nb.njit
    def _periodic_bc(pos: np.ndarray, domain: list[list], L: list):
        shape = pos.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                pos[i, j] = (pos[i, j] - domain[i][0]) % L[i] + domain[i][0]

    def stepping(self, u_hat: np.ndarray):
        """
        Time stepping loop for the Lagrangian Particle Tracking (LPT). In each step, the motion of the particles is integrated using a midpoint predictor-corrector scheme.
        """

        self._update_previous_position()
        for stage in range(self._n_stages):
            a = self._mpc_factor[stage]
            self._update_particle_velocity(u_hat)
            self._midpoint_predictor_corrector(
                self.p_position,
                self._p_pos_prev,
                self.p_velocity,
                self.p_trajectory,
                a, self._dt, stage
            )
            self._periodic_bc(self.p_position, self._domain, self._L)
