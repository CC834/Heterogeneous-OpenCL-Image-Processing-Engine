# FlowGuard OpenCL — Exploring Real-Time Visual Drone Collision Avoidance Across CPU and GPU

[![CI](https://github.com/CC834/flowguard-opencl/actions/workflows/ci.yml/badge.svg)](https://github.com/CC834/flowguard-opencl/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CC834/flowguard-opencl)](https://github.com/CC834/flowguard-opencl/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-5ee5c1.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-4b8bbe.svg)](CMakeLists.txt)

**Can a drone infer an approaching collision from ordinary camera motion, react in
time, and schedule the work effectively across the CPU and GPU of an edge-class
computer?**

FlowGuard explores that question as a closed-loop system you can see and measure.
Blender renders deterministic first-person flights, a C++/OpenCL engine estimates
motion without access to simulator ground truth, and a controller steers or brakes
from visual risk alone. The same workload can run on an OpenCL CPU, GPU, fixed
CPU/GPU split, or adaptive split so their real latency trade-offs can be compared.

> FlowGuard is a research prototype with simulated avoidance. It is not a
> safety-certified flight controller.

## See it in action

The 24-second montage below uses the final accepted Blender runs—not hand-authored
UI animation. It shows the frontal wall, offset pillar, doorway, corridor, safe
lateral pass, and moving crossing obstacle through FlowGuard's annotated native
dashboard.

![FlowGuard six-scenario collision-avoidance montage](docs/assets/flowguard-demo.gif)

[Watch the higher-quality MP4](docs/assets/flowguard-scenarios.mp4) ·
[Open the static poster](docs/assets/flowguard-poster.jpg) ·
[Read how the demo was produced](docs/DEMO.md)

## Why explore this area?

- **Collision decisions are deadline-sensitive.** A useful warning must arrive
  before the vehicle has already consumed its stopping or turning distance.
- **On-device perception matters.** A drone cannot assume a fast or reliable network
  connection, so camera processing and avoidance need to work locally.
- **Monocular cameras are widely available.** Optical expansion and motion fields can
  provide time-to-collision cues without pretending to provide LiDAR-quality depth.
- **Edge hardware is heterogeneous.** CPUs and GPUs have different strengths, while
  transfers and synchronization can erase the expected speedup. Measuring all four
  execution modes is more useful than assuming the GPU or a mixed mode must win.
- **Simulation makes the experiment repeatable.** Fixed seeds and evaluation-only
  ground truth allow risky situations to be tested consistently without presenting
  simulation as real-flight evidence.

The project therefore investigates three connected problems: visual motion
estimation, stable avoidance decisions, and truthful heterogeneous-compute
benchmarking.

![FlowGuard benchmark comparison](docs/assets/benchmark-chart.svg)

## What is implemented

- Blender 4.5 LTS kinematic simulation at 30 Hz and 640×360, with six seeded scenarios.
- Length-prefixed, loopback-only TCP frames and versioned command messages.
- OpenCL grayscale, 3×3 Gaussian filter, three-level pyramid, and coarse-to-fine
  16×16 block matching with a ±6-pixel search at every level.
- Best/second-best confidence rejection, weighted affine motion, focus of
  expansion, radial TTC proxies, spatial clusters, and left/centre/right risk.
- Smoothed 2 m/s, ±60°/s avoidance with 3 m/s² braking and turn hysteresis.
- CPU-only, GPU-only, fixed heterogeneous, and adaptive heterogeneous modes.
- Declared virtual onboard-compute profiles for camera degradation, lower frame
  rates, thermal throttling, GPU failure, actuator delay, and CPU failover.
- OpenCV live dashboard and recording, plus a React/TypeScript live/replay dashboard.
- JSONL telemetry, annotated MP4, CSV benchmark results, metadata, and static reports.

Ground truth lives in `Telemetry::evaluation` and is used only when recording
outcomes. Perception and control APIs cannot accept it.

## Simulate onboard drone hardware

The real OpenCL pipeline can run inside deterministic virtual hardware constraints,
making failure and deadline behavior visible before a physical edge board is
available:

```bash
build/flowguard hardware-profiles
build/flowguard simulate --scenario crossing-obstacle --mode adaptive \
  --hardware-profile thermal-throttle-sim --dashboard native,web --record
build/flowguard simulate --scenario corridor --mode adaptive \
  --hardware-profile gpu-failure-sim --dashboard native --record
```

![FlowGuard thermal-throttle virtual hardware run](docs/assets/virtual-hardware-throttle.png)

The frame above is recorded engine output: the declared thermal profile has crossed
its two-second threshold, the added penalty causes a 30 Hz deadline miss, and the
controller is reacting to the expanding synthetic obstacle.

The dashboards identify these runs as **VIRTUAL HARDWARE** and show camera rate,
temperature model, throttling, reused frames, GPU state, CPU failover, and applied
command delay. OpenCL perception is still measured on this host; camera/thermal/
actuator penalties are injected and separately labelled. Formal benchmarks reject
virtual profiles so they cannot be mistaken for physical-device results.

An optional loopback command bridge can drive PX4 SITL with BODY_NED velocity and
yaw-rate setpoints. It is disarmed by default and is documented in
[the PX4 SITL guide](docs/PX4_SITL.md). This is an experimental integration, not a
claim of real flight-controller validation. See [the virtual-hardware model](docs/VIRTUAL_HARDWARE.md)
for exact profiles and measurement boundaries.

## Quick start

Install the platform packages from [the setup guide](docs/SETUP.md), then build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure

cd web
npm ci
npm run build
cd ..
```

Inspect exactly which OpenCL devices FlowGuard can use:

```bash
build/flowguard devices
```

Run a synthetic replay first:

```bash
build/flowguard replay --synthetic expanding --frames 120 --mode cpu --dashboard native --record
```

Then launch the closed-loop Blender corridor scenario. Blender starts headlessly by
default; add `--visible` to see its window.

```bash
build/flowguard simulate --scenario corridor --mode adaptive --dashboard native,web --record
```

The local web dashboard is available at `http://127.0.0.1:8080` when enabled.

## CLI

```text
flowguard devices
flowguard hardware-profiles
flowguard simulate --scenario corridor --mode adaptive --dashboard native,web --record
flowguard simulate --scenario corridor --mode adaptive --hardware-profile edge-balanced-sim
flowguard replay --input flight.mp4 --mode fixed --gpu-ratio 0.70
flowguard benchmark --suite default --modes cpu,gpu,fixed,adaptive --repeats 5
flowguard report --run artifacts/<run-id>
```

Simulation scenarios are `frontal-wall`, `offset-pillar`, `doorway`, `corridor`,
`safe-lateral-pass`, and `crossing-obstacle`. Warning thresholds default to 3.0
seconds (yellow) and 1.5 seconds (red); use `--yellow-ttc` and `--red-ttc` to change
them.

## Honest benchmarking

`flowguard benchmark` creates the deterministic frame sequence once, runs each mode
sequentially, excludes one warm-up frame, and disables visualization. Its CSV
contains throughput, p50/p95/p99 latency, and 33.3 ms deadline misses. Metadata
states the input and measurement boundary.

The included current-host chart is a ten-measured-frame smoke comparison, not a
stable performance study or edge-device result. The retained CSV is in
[`benchmarks/current`](benchmarks/current/ryzen-ai-7-350-radeon-860m/README.md).
The only historical Intel logs are preserved in
[`benchmarks/legacy`](benchmarks/legacy/README.md). They are labelled as legacy raw
evidence, and the old unsupported speedup headline has been removed.

Performance acceptance means reporting every mode truthfully. Adaptive mode is not
assumed to win, especially on an integrated GPU sharing memory with the CPU.

## Architecture

```text
CLI / OpenCV UI / React UI / Blender transport
                         ↓
                  frame orchestration
                         ↓
           perception → risk → control
                  ↕ scheduling policy
                         ↓
           OpenCL / OpenCV / TCP / artifacts
```

See [architecture](docs/ARCHITECTURE.md), [simulator protocol](docs/PROTOCOL.md),
[virtual onboard hardware](docs/VIRTUAL_HARDWARE.md), and the
[versioned telemetry schema](docs/telemetry.schema.json) for the precise contracts.

## Limits and edge-device roadmap

Version 1 uses visual expansion and block motion, not depth sensors or aerodynamic
flight physics. Blender render time and perception time are separate. Results from
the current Ryzen AI 7 350/Radeon 860M machine must never be relabelled as Jetson,
Raspberry Pi, embedded, or real-drone performance.

The post-v1 deployment milestone is documented in [ROADMAP.md](ROADMAP.md). It
requires real-device selection, lower-resolution presets, power measurements where
available, and new device-labelled reports before making any edge claim.

## License

MIT © 2026 Abbe. See [LICENSE](LICENSE).
