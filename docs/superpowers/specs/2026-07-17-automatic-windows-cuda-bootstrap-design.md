# Automatic Windows CUDA Bootstrap Design

## Context

Thymos can request a DJL PyTorch GPU device, but on Windows DJL's initial CUDA
detection searches `CUDA_PATH` and the process `PATH` before its downloaded
runtime cache is available to the native loader. A machine can therefore have a
working NVIDIA GPU and a complete DJL `cu128` cache while DJL still selects CPU,
or fail loading `cublas64_12.dll` because a dependent DLL in the same cache
directory is outside the Windows DLL search path.

The verified manual workaround sets `PYTORCH_FLAVOR` and prepends the resolved
DJL cache directory to `PATH`. Requiring operators to enter a user-specific,
version-specific cache path is not suitable for deployment.

## Requirements

- `ThymosDevicePolicy.CPU` must not perform CUDA detection or change native
  library configuration.
- `AUTO` and `GPU` must prepare a compatible Windows CUDA runtime before the
  first call that initializes the DJL PyTorch engine.
- Explicit `PYTORCH_FLAVOR` configuration must take precedence over automatic
  selection.
- Automatic selection must query the NVIDIA driver and choose the highest DJL
  CUDA flavor supported by both that driver and the pinned PyTorch release.
- The DJL cache root, PyTorch version, platform classifier, and CUDA flavor must
  be discovered at runtime. User names and absolute cache paths must never be
  hard-coded.
- A missing cache directory must be created before DJL downloads native files so
  it can already be registered with the Windows loader.
- Bootstrap must be process-scoped and idempotent. It must not run in the
  per-message prediction path.
- `GPU` must fail with an actionable error if CUDA preparation is impossible.
  `AUTO` may continue to the existing CPU fallback.
- Non-Windows behavior must remain unchanged.

## Design

Add a focused internal bootstrap component in Thymos. Before
`ThymosDevicePolicy.candidates()` initializes `Engine`, it will:

1. Return immediately for CPU policy or non-Windows hosts.
2. Read an explicit `PYTORCH_FLAVOR` from system properties or the environment.
3. Otherwise query `nvcuda.dll` through JNA for the maximum CUDA version exposed
   by the installed NVIDIA driver.
4. Read DJL's cached PyTorch `files.txt` index, downloading only that small index
   from DJL's official repository when absent.
5. Select the highest indexed `cuNNN` flavor no newer than the requested or
   driver-supported version.
6. Derive the native directory as
   `<DJL engine cache>/pytorch/<pytorch-version>-<flavor>-<classifier>` and create
   it if necessary.
7. Set the effective `PYTORCH_FLAVOR` system property and prepend the derived
   directory to the native Windows process environment using `kernel32` through
   JNA before DJL initializes PyTorch.
8. Log the selected flavor and normalized native directory without exposing
   secrets.

The selection algorithm will be implemented as pure Kotlin logic and tested
independently from Windows and GPU hardware. JNA access, index loading, and
system-property mutation remain in a thin process bootstrap boundary.

## Integration

`DjlQwenTextAffectPredictor.fromBundle()` remains synchronous because model
construction is already a startup operation owned by its caller. It invokes the
bootstrap once immediately before resolving device candidates. Prediction,
tokenization, output mapping, model metadata validation, and CPU fallback remain
unchanged.

Thymos also exposes the same idempotent bootstrap through
`ThymosRuntime.prepare()`. Applications which load another DJL PyTorch model
before the affect bundle must call this entry point first; DJL's PyTorch engine
is process-global and cannot change from CPU to CUDA after initialization.

The Qwen TorchScript export replaces trace-time `Device("cpu")` constants with
the device of each graph's first tensor input. This keeps one artifact portable
across CPU fallback and CUDA inference instead of publishing device-specific
model variants.

OpenEden will consume the resulting Thymos commit. No Persona data, 8D vector,
derived D, VQ-VAE quantization, prompt construction, or runtime state logic is
modified.

## Verification

- Unit tests cover CUDA flavor parsing and highest-compatible selection.
- Unit tests prove CPU and non-Windows paths produce no side effects.
- Unit tests prove explicit flavor configuration wins over driver detection.
- The complete Thymos test suite must pass on JDK 21.
- A fresh JVM with `OPENEDEN_THYMOS_DEVICE=gpu`, no manually amended `PATH`, and
  the real model bundle must report `device=gpu(0)`.
- OpenEden's DJL affect analyzer and message-pipeline tests must pass after the
  dependency update.
