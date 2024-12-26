import importlib
import numpy as np
import scipy.fft as fft
import numba as nb


def __check_package(pkg_name: str):
    if importlib.util.find_spec(pkg_name) is None:
        raise ImportError(f"Package '{pkg_name}' is not installed.")


__check_package('rocket_fft')


@nb.njit
def fftn_jit(x, norm='forward', axes=None):
    return fft.fftn(x, norm=norm, axes=axes)


@nb.njit
def ifftn_jit(x, norm='forward', axes=None):
    return fft.ifftn(x, norm=norm, axes=axes)


@nb.njit
def rfftn_jit(x, norm='forward', axes=None):
    return fft.rfftn(x, norm=norm, axes=axes)


@nb.njit
def irfftn_jit(x, norm='forward', axes=None):
    return fft.irfftn(x, norm=norm, axes=axes)


def kspace(shape: list, length: list = [2 * np.pi] * 2):

    if not isinstance(shape, (list, tuple)):
        raise TypeError("shape must be an int, list, tuple")
    if not isinstance(length, (list, tuple)):
        raise TypeError("length must be an float, list, tuple")

    k = [
        fft.fftfreq(n, d=1 / n) * (2 * np.pi / l)
        for n, l in zip(shape, length)
    ]
    k[-1] = k[-1][:shape[-1] // 2 + 1]

    wavevector = np.meshgrid(*k, indexing='ij')
    wavenumber = np.sqrt(sum(ki**2 for ki in wavevector))

    mask_nyquist = np.ones_like(wavenumber)
    for dim, n in enumerate(shape):
        if n % 2 == 0:
            nyquist_index = n // 2
            slicing = [slice(None)] * len(shape)
            slicing[dim] = nyquist_index
            mask_nyquist[tuple(slicing)] = 0.0

    return wavevector, wavenumber, mask_nyquist
