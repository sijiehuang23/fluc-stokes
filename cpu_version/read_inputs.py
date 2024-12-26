import json
import numpy as np
import pprint


class Params:
    def __init__(self, json_file='input.json'):
        self._load_json(json_file)
        self._check_compatibility()
        self._set_precision()

    def __repr__(self):
        return f"<Params {self.__dict__}>"

    def _load_json(self, json_file):
        """Load JSON file and set attributes."""
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            self._read_params(data)
        except Exception as e:
            raise ValueError(f"Error reading JSON file {json_file}: {e}")

    def _set_precision(self):
        if self.precision.casefold() == 'single':
            self.rtype = np.float32
            self.ctype = np.complex64
        elif self.precision.casefold() == 'double':
            self.rtype = np.float64
            self.ctype = np.complex128

    def _read_params(self, data):
        defaults = {
            'N': [256, 256],
            'domain': [[0, 2 * np.pi], [0, 2 * np.pi]],
            'dt': 0.01,
            'end_time': 1.0,
            'filter': False,
            'viscosity': 1e-2,
            'noise_type': 'thermal',
            'noise_mag': 0.0,
            "enable_particles": True,
            'n_particles': 100,
            'verbose': False,
            'check_interval': np.iinfo(np.int64).max,
            'write_velocity': False,
            'velocity_file_name': 'data',
            'velocity_write_interval': np.iinfo(np.int64).max,
            'velocity_write_first_step': True,
            'velocity_write_mode': 'w',
            'enforce_periodic': True,
            'velocity_write_restart': False,
            'velocity_restart_name': 'restart',
            'velocity_restart_interval': np.iinfo(np.int64).max,
            'velocity_restart_mode': 'w',
            'write_particle': False,
            'particle_file_name': 'data',
            'particle_write_interval': np.iinfo(np.int64).max,
            'particle_write_first_step': True,
            'particle_write_mode': 'w'
        }

        for key, default in defaults.items():
            setattr(self, key, data.get(key, default))

    def _check_compatibility(self):
        """Check compatibility between related parameters."""
        if len(self.N) != len(self.domain):
            raise ValueError("The length of N must match the number of domain dimensions.")

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
            "dt": "Time step for integration.",
            "end_time": "Simulation end time.",
            "filter": "Enable or disable filtering of the velocity field.",
            "viscosity": "Viscosity of the fluid.",
            "noise_type": "Type of noise to add to the system. Options: 'thermal', 'correlated'.",
            "noise_mag": "Magnitude of the noise.",
            "verbose": "Enable or disable verbose output.",
            "check_interval": "Interval for checking the solution.",
            "write_data": "Write velocity data to file.",
            "file_name": "Name of the output file.",
            "write_interval": "Interval for writing data to file.",
            "write_first_step": "Write the initial data to file.",
            "write_mode": "Write mode for the output file.",
            "enforce_periodic": "Enforce periodic boundary conditions.",
            "write_restart": "Write restart data to file.",
            "restart_name": "Name of the restart file.",
            "restart_interval": "Interval for writing restart data to file.",
            "restart_mode": "Write mode for the restart file."
        }
        for key, doc in docs.items():
            print(f"{key}: {doc}")
