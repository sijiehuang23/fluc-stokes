import cupy as cp
from .io import Params, SolutionWriter
from .stokes import Stokes
from .time_integrator import CrankNicolson
from .noise import ThermalNoise, CorrelatedNoise
from .lagrangian_particles import LagrangianParticles
from .utils import Timer


class Solver:
    def __init__(self, params: Params):
        self.params = params

        self._init_stokes()
        self._init_time_integrator()
        self._init_noise()
        self._init_lagrangian_particles()
        self._init_data_writer()

        self.t = 0.0
        self.step = 0

        self.timer = Timer(params.verbose, params.check_interval)

    def _init_stokes(self):
        self.stokes = Stokes(self.params)

    def _init_time_integrator(self):
        L = -self.params.viscosity * self.stokes.k2
        self.time_integrator = CrankNicolson(self.params.dt, L)

    def _init_data_writer(self):
        if self.params.write_velocity:
            self.velocity_writer = SolutionWriter(
                self.params.velocity_file_name,
                self.stokes.x_cpu,
                self.params.velocity_write_mode
            )

    def _init_noise(self):
        noise_types = {
            'thermal': ThermalNoise,
            'correlated': CorrelatedNoise
        }
        if self.params.noise_type not in noise_types:
            err_msg = f"Invalid noise type '{self.params.noise_type}'. Must be one of {list(noise_types.keys())}."
            raise ValueError(err_msg)
        self.noise = noise_types[self.params.noise_type](
            self.stokes.rshape,
            self.stokes.cshape,
            self.params.noise_mag,
            self.stokes.mask_zero_mode,
            self.params.ctype,
            self.stokes.k
        )

    def _init_lagrangian_particles(self):
        if self.params.enable_particles:
            self.particles = LagrangianParticles(self.params)

    def set_linear_operator(self, L: cp.ndarray):
        self.time_integrator.update_linear_operator(L)

    def set_noise_correlation(self, C: cp.ndarray):
        self.noise.set_noise_correlation(C)

    def set_filter_kernel(self, G: cp.ndarray):
        self.stokes.filter_kernel[:] = G

    def initialize_velocity(self, u0: cp.ndarray, space: str = 'fourier'):
        self.stokes.initialize(u0, space)

    def initialize_particles(self, xp: cp.ndarray):
        self.particles.initialize(xp)

    def solve(self):
        self.timer.start()

        dt = self.params.dt
        end_time = self.params.end_time
        u_hat = self.stokes.u_hat
        noise = self.noise.noise

        enable_filter = self.params.enable_filter
        write_velocity = self.params.write_velocity
        velocity_write_interval = self.params.velocity_write_interval

        enable_particles = self.params.enable_particles
        write_particle = self.params.write_particle
        particle_write_interval = self.params.particle_write_interval

        if write_velocity and self.params.velocity_write_first_step:
            self.stokes.backward()
            self.stokes.gpu_to_cpu()
            self.velocity_writer.write(self.stokes.u_cpu, self.t, self.step)

        while not cp.isclose(self.t, end_time, atol=1e-10, rtol=1e-10):
            self.t += dt
            self.step += 1

            self.timer(self.t, self.step)

            if enable_particles:
                self.stokes.update_full_velocity()
                self.particles.stepping(self.stokes.u_hat_full)

            self.noise.update()
            self.time_integrator.integrate(u_hat, noise)
            self.stokes.project()
            if enable_filter:
                self.stokes.filter()

            write_velocity_ = write_velocity and self.step % velocity_write_interval == 0
            if write_velocity_:
                self.stokes.backward()
                self.stokes.gpu_to_cpu()
                self.velocity_writer.write(self.stokes.u_cpu, self.t, self.step)

            write_particle_ = (
                enable_particles and write_particle and self.step % particle_write_interval == 0
            )
            if write_particle_:
                self.particles.write(self.t, self.step)

        cp.cuda.Device().synchronize()
        self.timer.final()
