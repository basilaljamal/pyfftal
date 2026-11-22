"""Backend implementations and discovery."""

from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
from typing import Any

import numpy as np


class BackendError(RuntimeError):
    """Base error for backend operations."""


class BaseBackend:
    name = "base"

    def __init__(self, **options: Any) -> None:
        self._configuration: dict[str, Any] = dict(options)

    @property
    def configuration(self) -> dict[str, Any]:
        """dict: A copy of the active backend configuration."""
        return dict(self._configuration)

    def configure(self, **options: Any) -> None:
        """Update this backend's configuration."""
        self._configuration.update(options)

    def fft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        raise NotImplementedError

    def ifft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        raise NotImplementedError


class SciPyBackend(BaseBackend):
    """SciPy FFT backend."""

    name = "scipy"

    def __init__(self, **options: Any) -> None:
        try:
            from scipy import fft as scipy_fft
        except ImportError as exc:
            raise ImportError("The scipy backend requires scipy and numpy.") from exc
        self._fft = scipy_fft
        super().__init__(
            workers=int(options.get("workers", 1)),
            overwrite_x=bool(options.get("overwrite_x", False)),
            **{key: value for key, value in options.items()
               if key not in ("workers", "overwrite_x")},
        )

    def configure(self, **options: Any) -> None:
        if "workers" in options:
            workers = int(options["workers"])
            if workers < 1:
                raise ValueError("workers must be at least 1")
            options["workers"] = workers
        if "overwrite_x" in options:
            options["overwrite_x"] = bool(options["overwrite_x"])
        super().configure(**options)

    def _call_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "overwrite_x": (
                kwargs["overwrite_x"]
                if kwargs.get("overwrite_x") is not None
                else self._configuration["overwrite_x"]
            ),
            "workers": (
                kwargs["workers"]
                if kwargs.get("workers") is not None
                else self._configuration["workers"]
            ),
        }

    def fft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.fft(
            x, n=n, axis=axis, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )

    def ifft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.ifft(
            x, n=n, axis=axis, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )

    def rfft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.rfft(
            x, n=n, axis=axis, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )

    def irfft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.irfft(
            x, n=n, axis=axis, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )

    def fft2(self, x, s=None, axes=(-2, -1), norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.fft2(
            x, s=s, axes=axes, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )

    def ifft2(self, x, s=None, axes=(-2, -1), norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.ifft2(
            x, s=s, axes=axes, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )

    def fftn(self, x, s=None, axes=None, norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.fftn(
            x, s=s, axes=axes, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )

    def ifftn(self, x, s=None, axes=None, norm="backward", **kwargs):
        options = self._call_options(kwargs)
        return self._fft.ifftn(
            x, s=s, axes=axes, norm=norm,
            overwrite_x=options["overwrite_x"],
            workers=options["workers"],
        )


class FFTWBackend(SciPyBackend):
    """pyFFTW-backed implementation."""

    name = "fftw"

    def __init__(self, **options: Any) -> None:
        try:
            import pyfftw
            from pyfftw.interfaces import numpy_fft
        except ImportError as exc:
            raise ImportError(
                "The fftw backend requires pyfftw. Install it with "
                "'pip install pyfftal[fftw]'."
            ) from exc
        self._fft = numpy_fft
        pyfftw.interfaces.cache.enable()
        threads = int(options.get("threads", 1))
        planning_effort = str(options.get("planning_effort", "FFTW_ESTIMATE"))
        valid_efforts = {
            "FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"
        }
        if planning_effort not in valid_efforts:
            raise ValueError(
                "planning_effort must be one of " + ", ".join(sorted(valid_efforts))
            )
        if threads < 1:
            raise ValueError("threads must be at least 1")
        BaseBackend.__init__(
            self,
            threads=threads,
            planning_effort=planning_effort,
            planning_timelimit=options.get("planning_timelimit"),
            overwrite_x=bool(options.get("overwrite_x", False)),
            cache_enabled=bool(options.get("cache_enabled", True)),
            cache_keepalive=float(options.get("cache_keepalive", 10.0)),
        )
        if options.get("cache_enabled", True):
            pyfftw.interfaces.cache.enable()
        else:
            pyfftw.interfaces.cache.disable()
        pyfftw.interfaces.cache.set_keepalive_time(
            float(options.get("cache_keepalive", 10.0))
        )

    def configure(self, **options: Any) -> None:
        import pyfftw

        if "threads" in options:
            options["threads"] = int(options["threads"])
            if options["threads"] < 1:
                raise ValueError("threads must be at least 1")
        if "planning_effort" in options:
            options["planning_effort"] = str(options["planning_effort"])
            valid = {"FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", "FFTW_EXHAUSTIVE"}
            if options["planning_effort"] not in valid:
                raise ValueError(
                    "planning_effort must be one of " + ", ".join(sorted(valid))
                )
        if "cache_enabled" in options:
            options["cache_enabled"] = bool(options["cache_enabled"])
            if options["cache_enabled"]:
                pyfftw.interfaces.cache.enable()
            else:
                pyfftw.interfaces.cache.disable()
        if "cache_keepalive" in options:
            options["cache_keepalive"] = float(options["cache_keepalive"])
            pyfftw.interfaces.cache.set_keepalive_time(options["cache_keepalive"])
        if "overwrite_x" in options:
            options["overwrite_x"] = bool(options["overwrite_x"])
        BaseBackend.configure(self, **options)

    def _interface_options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "overwrite_input": (
                kwargs["overwrite_x"]
                if kwargs.get("overwrite_x") is not None
                else self._configuration["overwrite_x"]
            ),
            "threads": (
                kwargs["threads"]
                if kwargs.get("threads") is not None
                else self._configuration["threads"]
            ),
            "planning_effort": self._configuration["planning_effort"],
            "planning_timelimit": self._configuration["planning_timelimit"],
        }

    def fft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.fft(
            x, n=n, axis=axis, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )

    def ifft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.ifft(
            x, n=n, axis=axis, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )

    def rfft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.rfft(
            x, n=n, axis=axis, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )

    def irfft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.irfft(
            x, n=n, axis=axis, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )

    def fft2(self, x, s=None, axes=(-2, -1), norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.fft2(
            x, s=s, axes=axes, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )

    def ifft2(self, x, s=None, axes=(-2, -1), norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.ifft2(
            x, s=s, axes=axes, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )

    def fftn(self, x, s=None, axes=None, norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.fftn(
            x, s=s, axes=axes, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )

    def ifftn(self, x, s=None, axes=None, norm="backward", **kwargs):
        options = self._interface_options(kwargs)
        return self._fft.ifftn(
            x, s=s, axes=axes, norm=norm,
            overwrite_input=options["overwrite_input"],
            threads=options["threads"],
            planning_effort=options["planning_effort"],
            planning_timelimit=options["planning_timelimit"],
        )


class CuPyBackend(BaseBackend):
    """CuPy CUDA FFT backend (also usable where pycufft is exposed as cupy)."""

    name = "cuda"

    def __init__(self, **options: Any) -> None:
        try:
            self._cp = importlib.import_module("cupy")
            self._fft = importlib.import_module("cupy.fft")
        except ImportError as exc:
            raise ImportError(
                "The CUDA backend requires cupy. Install it with "
                "'pip install pyfftal[cuda]'."
            ) from exc
        super().__init__(
            device=options.get("device"),
            return_numpy=bool(options.get("return_numpy", True)),
        )
        if self._configuration["device"] is not None:
            self._cp.cuda.Device(int(self._configuration["device"])).use()

    def configure(self, **options: Any) -> None:
        if "device" in options:
            device = options["device"]
            if device is not None:
                device = int(device)
                self._cp.cuda.Device(device).use()
            options["device"] = device
        if "return_numpy" in options:
            options["return_numpy"] = bool(options["return_numpy"])
        super().configure(**options)

    def _result(self, value):
        if hasattr(value, "get") and self._configuration["return_numpy"]:
            return value.get()
        return value

    def _array(self, x):
        if hasattr(x, "__cuda_array_interface__"):
            return x
        return self._cp.asarray(x)

    def fft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        return self._result(self._fft.fft(self._array(x), n=n, axis=axis, norm=norm))

    def ifft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        return self._result(self._fft.ifft(self._array(x), n=n, axis=axis, norm=norm))

    def rfft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        return self._result(self._fft.rfft(self._array(x), n=n, axis=axis, norm=norm))

    def irfft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        return self._result(self._fft.irfft(self._array(x), n=n, axis=axis, norm=norm))

    def fft2(self, x, s=None, axes=(-2, -1), norm="backward", **kwargs):
        return self._result(self._fft.fft2(self._array(x), s=s, axes=axes, norm=norm))

    def ifft2(self, x, s=None, axes=(-2, -1), norm="backward", **kwargs):
        return self._result(self._fft.ifft2(self._array(x), s=s, axes=axes, norm=norm))

    def fftn(self, x, s=None, axes=None, norm="backward", **kwargs):
        return self._result(self._fft.fftn(self._array(x), s=s, axes=axes, norm=norm))

    def ifftn(self, x, s=None, axes=None, norm="backward", **kwargs):
        return self._result(self._fft.ifftn(self._array(x), s=s, axes=axes, norm=norm))


class OpenCLBackend(BaseBackend):
    """OpenCL FFT backend using Reikna/Reikna FFT."""

    name = "opencl"

    def __init__(self, **options: Any) -> None:
        try:
            import pyopencl as cl
            from reikna import cluda
            from reikna.fft import FFT as ReiknaFFT
        except ImportError as exc:
            raise ImportError(
                "The opencl backend requires pyopencl and reikna. Install it with "
                "'pip install pyfftal[opencl]'."
            ) from exc
        platform_index = int(options.get("platform", 0))
        device_index = int(options.get("device", 0))
        platforms = cl.get_platforms()
        if platform_index >= len(platforms):
            raise BackendError(
                f"OpenCL platform index {platform_index} is outside the "
                f"available range (0-{len(platforms) - 1})"
            )
        devices = platforms[platform_index].get_devices()
        if device_index >= len(devices):
            raise BackendError(
                f"OpenCL device index {device_index} is outside the "
                f"available range (0-{len(devices) - 1})"
            )
        api = cluda.ocl_api()
        selected_device = devices[device_index]
        self._thr = api.Thread(cl.CommandQueue(selected_device))
        self._reikna_fft = ReiknaFFT
        super().__init__(
            platform=platform_index,
            device=device_index,
            async_execution=bool(options.get("async_execution", False)),
        )

    def configure(self, **options: Any) -> None:
        if "async_execution" in options:
            options["async_execution"] = bool(options["async_execution"])
        if "platform" in options or "device" in options:
            raise BackendError(
                "OpenCL platform and device selections cannot be changed "
                "after initialization; construct a new FFT instance."
            )
        super().configure(**options)

    def _transform(self, x, inverse=False, n=None, axis=-1):
        arr = np.asarray(x)
        shape = list(arr.shape)
        if n is not None:
            shape[axis] = n
        if inverse:
            dtype = np.complex128 if arr.dtype in (np.complex64, np.complex128) else np.complex128
        else:
            dtype = np.complex128
        arr = np.asarray(arr, dtype=dtype)
        gpu_arr = self._thr.to_device(arr)
        transform = self._reikna_fft(gpu_arr).prepare(self._thr)
        gpu_result = self._thr.empty_like(gpu_arr)
        transform(gpu_result, gpu_arr, inverse=inverse)
        return gpu_result.get()

    def fft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        return self._transform(x, False, n=n, axis=axis)

    def ifft(self, x, n=None, axis=-1, norm="backward", **kwargs):
        return self._transform(x, True, n=n, axis=axis)


def available_backends() -> dict[str, bool]:
    """Return availability for each known backend.

    Returns
    -------
    dict of str to bool
        Mapping from backend names to availability.
    """
    availability = {
        "scipy": True,
        "fftw": False,
        "cuda": False,
        "pycufft": False,
        "opencl": False,
    }
    for module in ("pyfftw",):
        try:
            importlib.import_module(module)
            availability["fftw"] = True
        except ImportError:
            pass
    for module in ("cupy", "cupyx"):
        try:
            module_object = importlib.import_module(module)
            runtime = getattr(getattr(module_object, "cuda", None), "runtime", None)
            if runtime is not None:
                try:
                    availability["cuda"] = runtime.getDeviceCount() > 0
                except Exception:
                    availability["cuda"] = False
            availability["pycufft"] = availability["cuda"]
            break
        except ImportError:
            pass
    try:
        importlib.import_module("pyopencl")
        importlib.import_module("reikna")
        import pyopencl as cl
        availability["opencl"] = any(
            platform.get_devices()
            for platform in cl.get_platforms()
        )
    except ImportError:
        pass
    return availability


def system_check() -> dict[str, Any]:
    """Check software and hardware support for accelerated FFT backends.

    The function is safe to call on systems without CUDA or OpenCL.  It never
    raises for missing accelerated dependencies; availability and diagnostic
    information are returned instead.

    Returns
    -------
    dict
        Nested report containing system, CUDA, and OpenCL information.

    Examples
    --------
    >>> report = system_check()
    >>> "cuda" in report
    True
    """
    report: dict[str, Any] = {
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cuda": {
            "available": False,
            "cupy_installed": False,
            "nvidia_smi": shutil.which("nvidia-smi") is not None,
            "nvcc": shutil.which("nvcc") is not None,
            "devices": [],
            "errors": [],
        },
        "opencl": {
            "available": False,
            "pyopencl_installed": False,
            "reikna_installed": False,
            "platforms": [],
            "errors": [],
        },
    }

    try:
        cupy = importlib.import_module("cupy")
        report["cuda"]["cupy_installed"] = True
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count > 0:
            report["cuda"]["available"] = True
        for index in range(device_count):
            properties = cupy.cuda.runtime.getDeviceProperties(index)
            report["cuda"]["devices"].append({
                "index": index,
                "name": (
                    properties.get("name", "").decode()
                    if isinstance(properties.get("name"), bytes)
                    else str(properties.get("name", "Unknown"))
                ),
                "total_memory_bytes": int(properties.get("totalGlobalMem", 0)),
                "multiprocessor_count": int(properties.get("multiProcessorCount", 0)),
                "compute_capability": (
                    f"{properties.get('major', 0)}.{properties.get('minor', 0)}"
                ),
            })
    except ImportError:
        pass
    except Exception as exc:
        report["cuda"]["errors"].append(str(exc))

    if report["cuda"]["nvidia_smi"]:
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if output.returncode == 0:
                report["cuda"]["nvidia_smi_output"] = [
                    line.strip() for line in output.stdout.splitlines() if line.strip()
                ]
            elif output.stderr:
                report["cuda"]["errors"].append(output.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["cuda"]["errors"].append(str(exc))

    try:
        importlib.import_module("pyopencl")
        report["opencl"]["pyopencl_installed"] = True
    except ImportError:
        pass
    try:
        importlib.import_module("reikna")
        report["opencl"]["reikna_installed"] = True
    except ImportError:
        pass

    if report["opencl"]["pyopencl_installed"]:
        try:
            import pyopencl as cl
            platform_reports = []
            for platform_index, opencl_platform in enumerate(cl.get_platforms()):
                platform_report = {
                    "index": platform_index,
                    "name": opencl_platform.name,
                    "vendor": opencl_platform.vendor,
                    "version": opencl_platform.version,
                    "devices": [],
                }
                for device_index, device in enumerate(opencl_platform.get_devices()):
                    platform_report["devices"].append({
                        "index": device_index,
                        "name": device.name,
                        "vendor": device.vendor,
                        "type": cl.device_type.to_string(device.type),
                        "global_memory_bytes": int(device.global_mem_size),
                        "compute_units": int(device.max_compute_units),
                        "driver_version": device.driver_version,
                        "opencl_version": device.opencl_c_version,
                    })
                platform_reports.append(platform_report)
            report["opencl"]["platforms"] = platform_reports
            report["opencl"]["available"] = any(
                platform_report["devices"] for platform_report in platform_reports
            )
        except Exception as exc:
            report["opencl"]["errors"].append(str(exc))

    return report


def create_backend(name: str = "scipy", **options: Any) -> BaseBackend:
    """Create a backend implementation.

    Parameters
    ----------
    name : str
        Backend name.
    **options : dict
        Backend-specific options.

    Returns
    -------
    BaseBackend
        Initialized backend.
    """
    canonical = str(name).lower().strip()
    aliases = {
        "numpy": "scipy",
        "cupy": "cuda",
        "cufft": "cuda",
        "pycufft": "cuda",
        "pyopecl": "opencl",
    }
    canonical = aliases.get(canonical, canonical)
    constructors = {
        "scipy": SciPyBackend,
        "fftw": FFTWBackend,
        "cuda": CuPyBackend,
        "opencl": OpenCLBackend,
    }
    if canonical not in constructors:
        choices = ", ".join(sorted(set(constructors) | {"cupy", "pycufft"}))
        raise ValueError(f"Unknown FFT backend {name!r}. Available choices: {choices}")
    return constructors[canonical](**options)
