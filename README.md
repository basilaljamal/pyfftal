# pyfftal

`pyfftal` is a small, backend-configurable abstraction layer for Fast Fourier transforms.
It provides one NumPy/SciPy-style API for FFT operations and supports SciPy,
FFTW, CUDA, and OpenCL backends where their optional dependencies are installed.

## Installation

Install the core package:

```bash
pip install .
```

Optional backends can be installed with:

```bash
pip install ".[fftw]"
pip install ".[cuda]"
pip install ".[opencl]"
pip install ".[all]"
```

The `scipy` backend is always available when the core dependencies are
installed.

## Quick start

```python
import numpy as np
from pyfftal import FFT

x = np.asarray([1.0, 2.0, 1.0, -1.0])
fft = FFT(backend="scipy")

spectrum = fft.fft(x)
reconstructed = fft.ifft(spectrum)

print(fft.backend)
# scipy
```

Module-level convenience functions are also available:

```python
from pyfftal import fft, ifft

spectrum = fft(x)
reconstruction = ifft(spectrum)
```

The convenience functions use the SciPy backend by default.

## Supported backends

| Name | Backend | Required packages |
|---|---|---|
| `scipy` | SciPy FFT | `scipy`, `numpy` |
| `fftw` | pyFFTW interface | `pyfftw` |
| `cuda`, `cupy`, `pycufft` | CuPy CUDA FFT | `cupy` |
| `opencl` | Reikna FFT | `pyopencl`, `reikna` |

Use `pyfftal.available_backends()` to check which backends can be loaded in
the current environment.

## System checks

Use `pyfftal.system_check()` to inspect operating-system information, installed
accelerated libraries, CUDA devices, and OpenCL platforms/devices:

```python
from pyfftal import system_check

report = system_check()
print(report["cuda"]["available"])
print(report["opencl"]["available"])
```

The report is diagnostic and does not raise when optional hardware or
dependencies are absent.

## Backend configuration

Pass backend options when constructing an instance:

```python
fft = FFT("scipy", workers=4)
```

Configuration can also be changed at runtime:

```python
fft.configure(workers=8)
print(fft.configuration)
```

Common and backend-specific options:

| Backend | Options |
|---|---|
| `scipy` | `workers`, `overwrite_x` |
| `fftw` | `threads`, `planning_effort`, `planning_timelimit`, `cache_enabled`, `cache_keepalive`, `overwrite_x` |
| `cuda` / `cupy` / `pycufft` | `device`, `return_numpy` |
| `opencl` | `platform`, `device`, `async_execution` |

## API

The `FFT` class exposes `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`,
`fftn`, and `ifftn`. Parameters follow the corresponding SciPy and NumPy
names where practical. Full API documentation is included in Python
docstrings using the NumPy documentation style.
