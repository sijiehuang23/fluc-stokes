import numpy as np
import cupy as cp
import h5py
import json
import pprint
from abc import ABC, abstractmethod
from . import logger


__all__ = ['Params', 'SolutionWriter', 'ParticleWriter']


class Params:
    def __init__(self, json_file='input.json'):
        self._load_json(json_file)
        self._check_compatibility()
        self._set_data_type()

    def __repr__(self):
        return f"<Params {self.__dict__}>"

    def _load_json(self, json_file):
        """Load JSON file and set attributes."""
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            self._read_params(data)
        except Exception as e:
            raise ValueError(f"Error reading input JSON file {json_file}: {e}")

    def _read_params(self, data):
        defaults = {
            'N': [256, 256],
            'domain': [[0, 2 * cp.pi], [0, 2 * cp.pi]],
            'dtype': 'single',
            'dt': 0.01,
            'end_time': 1.0,
            'viscosity': 1e-2,
            'noise_type': 'thermal',
            'noise_mag': 0.0,
            'enable_filter': False,
            'enable_particles': False,
            'n_particles': 100,
            'verbose': False,
            'check_interval': cp.iinfo(cp.int64).max,
            'write_velocity': False,
            'velocity_file_name': 'data',
            'velocity_write_interval': cp.iinfo(cp.int64).max,
            'velocity_write_first_step': True,
            'velocity_write_mode': 'w',
            'enforce_periodic': True,
            'velocity_write_restart': False,
            'velocity_restart_name': 'restart',
            'velocity_restart_interval': cp.iinfo(cp.int64).max,
            'velocity_restart_mode': 'w',
            'write_particle': False,
            'particle_file_name': 'data',
            'particle_write_interval': cp.iinfo(cp.int64).max,
            'particle_write_first_step': True,
            'particle_write_mode': 'w'
        }

        for key, default in defaults.items():
            setattr(self, key, data.get(key, default))

    def _check_compatibility(self):
        """Check compatibility between related parameters."""
        if len(self.N) != len(self.domain):
            raise ValueError("The length of N must match the number of domain dimensions.")

    def _set_data_type(self):
        if self.dtype.casefold() == 'single':
            self.rtype = cp.float32
            self.ctype = cp.complex64
        elif self.dtype.casefold() == 'double':
            self.rtype = cp.float64
            self.ctype = cp.complex128
        else:
            raise ValueError(f"Invalid data type: {self.dtype}")

    def print(self):
        """Print the parameters in a human-readable format."""
        pprint.pprint(self.__dict__)

    def update(self, params):
        """Update parameters from a dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

    def docs(self):
        """Print documentation for all parameters."""
        docs = {
            "N": "Number of grid points in each dimension, e.g., [Nx, Ny].",
            "domain": "Domain size for each dimension, e.g., [[xmin, xmax], [ymin, ymax]].",
            "dtype": "Data type for the simulation. Options: 'single', 'double'.",
            "dt": "Time step for integration.",
            "end_time": "Simulation end time.",
            "viscosity": "Viscosity of the fluid.",
            "noise_type": "Type of noise to add to the system. Options: 'thermal', 'correlated'.",
            "noise_mag": "Magnitude of the noise.",
            "verbose": "Enable or disable verbose output.",
            "check_interval": "Interval for checking the solution.",
            "write_velocity": "Enable or disable writing the velocity field.",
            "velocity_file_name": "Name of the velocity field file.",
            "velocity_write_interval": "Interval for writing the velocity field.",
            "velocity_write_first_step": "Write the velocity field at the first step.",
            "velocity_write_mode": "Write mode for the velocity field.",
            "enforce_periodic": "Enforce periodic boundary conditions.",
            "velocity_write_restart": "Enable or disable writing the velocity field for restart.",
            "velocity_restart_name": "Name of the velocity field restart file.",
            "velocity_restart_interval": "Interval for writing the velocity field for restart.",
            "velocity_restart_mode": "Write mode for the velocity field for restart.",
            "write_particle": "Enable or disable writing particle data.",
            "particle_file_name": "Name of the particle file.",
            "particle_write_interval": "Interval for writing particle data.",
            "particle_write_first_step": "Write particle data at the first step.",
            "particle_write_mode": "Write mode for particle data."
        }
        for key, doc in docs.items():
            print(f"{key}: {doc}")


class _HDF5Writer(ABC):
    def __init__(self, file_name: str, mode: str = 'w'):
        self.file_name = file_name

        if not self.file_name.endswith('.h5'):
            self.file_name += '.h5'

        self.open(mode)
        self.f.attrs['t'] = 0.0
        self.f.attrs['step'] = 0
        self.close()

    @abstractmethod
    def write(self):
        raise NotImplementedError

    def open(self, mode: str = 'r'):
        self.f = h5py.File(self.file_name, mode)

    def close(self):
        if self.f:
            self.f.close()


class SolutionWriter(_HDF5Writer):
    def __init__(self, file_name: str, x_cpu: list[np.ndarray], mode='w'):
        super().__init__(file_name, mode)

        self.ndim = len(x_cpu)
        self._str_vel = ['u', 'v', 'w'][:self.ndim]
        self._str_x = ['x', 'y', 'z'][:self.ndim]

        self.open('r+')
        for i in range(self.ndim):
            self.f.create_dataset(self._str_x[i], data=x_cpu[i])
        self.close()

    def write(self, u: np.ndarray, t: float, step: int):
        self.open('r+')
        for i in range(self.ndim):
            grp = self.f.require_group(self._str_vel[i])
            grp.create_dataset(f'{step}', data=u[i])
        self.f.attrs['t'] = t
        self.f.attrs['step'] = step
        self.close()


class ParticleWriter(_HDF5Writer):
    def __init__(self, file_name: str, mode: str = 'w'):
        super().__init__(file_name)

        self._str_grps = ['position', 'velocity', 'trajectory']

        self.open(mode)
        for grp in self._str_grps:
            self.f.create_group(grp)
        self.f.attrs['t'] = 0.0
        self.f.attrs['step'] = 0
        self.close()

    def write(self, t: float, step: int, pos: np.ndarray, vel: np.ndarray, traj: np.ndarray):
        self.open('r+')
        grp_pos = self.f.require_group('position')
        grp_pos.create_dataset(f'{step}', data=pos)

        grp_vel = self.f.require_group('velocity')
        grp_vel.create_dataset(f'{step}', data=vel)

        grp_traj = self.f.require_group('trajectory')
        grp_traj.create_dataset(f'{step}', data=traj)

        self.f.attrs['t'] = t
        self.f.attrs['step'] = step
        self.close()
