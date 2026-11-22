"""Public FFT abstraction."""

from __future__ import annotations

from typing import Any

from .backends import create_backend


class FFT:
    """A unified interface for configurable FFT backends.

    Parameters
    ----------
    backend : {"scipy", "fftw", "cuda", "cupy", "pycufft", "opencl"}, optional
        Backend implementation to use.  The default is ``"scipy"``.
    **backend_options : dict
        Backend-specific constructor options.

    Attributes
    ----------
    backend : str
        Canonical name of the selected backend.

    Notes
    -----
    Optional backends raise an informative :class:`ImportError` if their
    dependencies are unavailable.  SciPy remains the portable fallback.

    Examples
    --------
    >>> import numpy as np
    >>> fft = FFT("scipy")
    >>> spectrum = fft.fft(np.array([1.0, 2.0, 1.0, -1.0]))
    >>> abs(fft.ifft(spectrum)[0])
    1.0
    """

    def __init__(self, backend: str = "scipy", **backend_options: Any) -> None:
        self._backend = create_backend(backend, **backend_options)

    @property
    def backend(self) -> str:
        """str: Canonical name of the active backend."""
        return self._backend.name

    @property
    def configuration(self) -> dict[str, Any]:
        """dict: Backend-specific configuration options."""
        return self._backend.configuration

    def configure(self, **options: Any) -> None:
        """Update backend-specific configuration.

        Parameters
        ----------
        **options : dict
            Backend configuration values to update.  Each backend validates
            and stores only options it supports.

        Returns
        -------
        None

        Examples
        --------
        >>> fft = FFT("scipy")
        >>> fft.configure(workers=4)
        """
        self._backend.configure(**options)

    def fft(self, x: Any, n: int | None = None, axis: int = -1,
            norm: str = "backward", *, workers: int | None = None,
            overwrite_x: bool | None = None) -> Any:
        """Compute a one-dimensional discrete Fourier Transform.

        Parameters
        ----------
        x : array_like
            Input sequence.
        n : int, optional
            Transform length.
        axis : int, optional
            Axis over which to transform.
        norm : {"backward", "ortho", "forward"}, optional
            Normalization mode.
        workers : int, optional
            Maximum number of workers.
        overwrite_x : bool, optional
            Allow the input buffer to be overwritten.

        Returns
        -------
        ndarray
            Complex Fourier coefficients.
        """
        return self._backend.fft(
            x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
        )

    def ifft(self, x: Any, n: int | None = None, axis: int = -1,
             norm: str = "backward", *, workers: int | None = None,
             overwrite_x: bool | None = None) -> Any:
        """Compute a one-dimensional inverse discrete Fourier Transform."""
        return self._backend.ifft(
            x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
        )

    def rfft(self, x: Any, n: int | None = None, axis: int = -1,
             norm: str = "backward", *, workers: int | None = None,
             overwrite_x: bool | None = None) -> Any:
        """Compute a one-dimensional FFT of a real-valued input."""
        return self._backend.rfft(
            x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
        )

    def irfft(self, x: Any, n: int | None = None, axis: int = -1,
              norm: str = "backward", *, workers: int | None = None,
              overwrite_x: bool | None = None) -> Any:
        """Compute an inverse FFT expected to produce real-valued output."""
        return self._backend.irfft(
            x, n=n, axis=axis, norm=norm, workers=workers, overwrite_x=overwrite_x
        )

    def fft2(self, x: Any, s: tuple[int, ...] | None = None,
             axes: tuple[int, int] = (-2, -1), norm: str = "backward",
             *, workers: int | None = None, overwrite_x: bool | None = None) -> Any:
        """Compute a two-dimensional discrete Fourier Transform."""
        return self._backend.fft2(
            x, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x
        )

    def ifft2(self, x: Any, s: tuple[int, ...] | None = None,
              axes: tuple[int, int] = (-2, -1), norm: str = "backward",
              *, workers: int | None = None, overwrite_x: bool | None = None) -> Any:
        """Compute a two-dimensional inverse discrete Fourier Transform."""
        return self._backend.ifft2(
            x, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x
        )

    def fftn(self, x: Any, s: tuple[int, ...] | None = None,
             axes: tuple[int, ...] | None = None, norm: str = "backward",
             *, workers: int | None = None, overwrite_x: bool | None = None) -> Any:
        """Compute an N-dimensional discrete Fourier Transform."""
        return self._backend.fftn(
            x, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x
        )

    def ifftn(self, x: Any, s: tuple[int, ...] | None = None,
              axes: tuple[int, ...] | None = None, norm: str = "backward",
              *, workers: int | None = None, overwrite_x: bool | None = None) -> Any:
        """Compute an N-dimensional inverse discrete Fourier Transform."""
        return self._backend.ifftn(
            x, s=s, axes=axes, norm=norm, workers=workers, overwrite_x=overwrite_x
        )


FFTAL = FFT
