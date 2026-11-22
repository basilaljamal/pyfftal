"""Unified FFT operations with configurable compute backends.

This package exposes the :class:`FFT` abstraction and module-level convenience
functions.  SciPy is the default backend; optional backends are loaded only
when selected.
"""

from .api import FFT, FFTAL
from .backends import available_backends, system_check

_DEFAULT_FFT = FFT("scipy")


def fft(x, n=None, axis=-1, norm="backward", *, workers=None, overwrite_x=None):
    """Compute a one-dimensional discrete Fourier Transform.

    Parameters
    ----------
    x : array_like
        Input sequence.
    n : int, optional
        Length of the transformed axis.
    axis : int, optional
        Axis over which to compute the transform.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode.
    workers : int, optional
        Maximum worker count for the SciPy backend.
    overwrite_x : bool, optional
        Allow overwrite of the input when supported.

    Returns
    -------
    ndarray
        The transformed array.
    """
    return _DEFAULT_FFT.fft(
        x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
    )


def ifft(x, n=None, axis=-1, norm="backward", *, workers=None, overwrite_x=None):
    """Compute a one-dimensional inverse discrete Fourier Transform."""
    return _DEFAULT_FFT.ifft(
        x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
    )


def rfft(x, n=None, axis=-1, norm="backward", *, workers=None, overwrite_x=None):
    """Compute a one-dimensional FFT of a real-valued input."""
    return _DEFAULT_FFT.rfft(
        x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
    )


def irfft(x, n=None, axis=-1, norm="backward", *, workers=None, overwrite_x=None):
    """Compute an inverse FFT for real-valued output."""
    return _DEFAULT_FFT.irfft(
        x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
    )


__all__ = [
    "FFT",
    "FFTAL",
    "available_backends",
    "system_check",
    "fft",
    "ifft",
    "rfft",
    "irfft",
]
