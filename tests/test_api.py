import numpy as np
import pytest

from pyfftal import (
    FFT,
    FFTAL,
    available_backends,
    fft,
    ifft,
    system_check,
)


def test_fft_round_trip():
    x = np.array([1.0, 2.0, 1.0, -1.0])
    transform = FFT("scipy")
    y = transform.fft(x)
    np.testing.assert_allclose(transform.ifft(y), x, atol=1e-12)


def test_real_fft_round_trip():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    transform = FFT("scipy")
    np.testing.assert_allclose(transform.irfft(transform.rfft(x)), x, atol=1e-12)


def test_multidimensional_fft():
    x = np.arange(12).reshape(3, 4)
    transform = FFT("scipy")
    np.testing.assert_allclose(transform.ifftn(transform.fftn(x)), x, atol=1e-12)
    np.testing.assert_allclose(transform.ifft2(transform.fft2(x)), x, atol=1e-12)


def test_module_convenience_functions():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(ifft(fft(x)), x, atol=1e-12)


def test_alias_and_availability():
    assert FFTAL is FFT
    assert available_backends()["scipy"] is True


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown FFT backend"):
        FFT("not-a-backend")


def test_backend_configuration():
    transform = FFT("scipy", workers=2)
    assert transform.configuration["workers"] == 2
    transform.configure(workers=3, overwrite_x=True)
    assert transform.configuration["workers"] == 3
    assert transform.configuration["overwrite_x"] is True


def test_invalid_backend_configuration():
    transform = FFT("scipy")
    with pytest.raises(ValueError, match="workers"):
        transform.configure(workers=0)


def test_system_check_shape():
    report = system_check()
    assert "system" in report
    assert "cuda" in report
    assert "opencl" in report
    assert isinstance(report["cuda"]["devices"], list)
    assert isinstance(report["opencl"]["platforms"], list)
