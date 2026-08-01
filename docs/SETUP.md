# Setup

## Fedora

```bash
sudo dnf install cmake ninja-build gcc-c++ opencl-headers ocl-icd-devel \
  opencv-devel boost-devel pocl blender nodejs npm ffmpeg
```

Mesa Rusticl supplies supported AMD OpenCL GPUs on current Fedora releases. Confirm
device availability with `clinfo -l`; package installation alone does not guarantee
that a driver exposes a particular device.

## Ubuntu

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build ocl-icd-opencl-dev \
  opencl-headers pocl-opencl-icd libopencv-dev libboost-dev blender nodejs npm ffmpeg
```

PoCL gives CI and local machines a CPU OpenCL implementation. GPU support requires
the vendor runtime appropriate for that machine.

## Reproducible release build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DFLOWGUARD_WARNINGS_AS_ERRORS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure

cd web
npm ci
npm run lint
npm run build
```

Blender is needed for closed-loop scenarios, but not for CTest's deterministic
synthetic replay. Formal local verification should additionally run Blender and all
locally available OpenCL device modes.
