from pathlib import Path
import numpy as np
import h5py
import json
import traceback
import pprint
from abc import ABC, abstractmethod
from . import logger
from periodicflow.utils import periodic_bc


__all__ = ['Params', 'HDF5Writer']


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
    def __init__(self, file_name: str, ndim: int):
        super().__init__(file_name)

        self.ndim = ndim
        self._str_vel = ['u', 'v', 'w'][:ndim]
        self._str_x = ['x', 'y', 'z'][:ndim]

    def initialize(self, x):
        self.open('r+')
        for i in range(self.ndim):
            self.f.create_dataset(self._str_x[i], data=x[i])
        self.close()

    def write(self, u, t, step):
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

        _str_grps = ['position', 'velocity', 'trajectory']

        self.open(mode)
        for grp in _str_grps:
            self.f.create_group(grp)
        self.f.attrs['t'] = 0.0
        self.f.attrs['step'] = 0
        self.close()

    def write(self, t, step, pos, vel, traj):
        self.open('r+')
        self.f.create_dataset(f'position/{step}', data=pos)
        self.f.create_dataset(f'velocity/{step}', data=vel)
        self.f.create_dataset(f'trajectory/{step}', data=traj)
        self.f.attrs['t'] = t
        self.f.attrs['step'] = step
        self.close()
