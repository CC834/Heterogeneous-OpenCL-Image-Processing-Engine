# FlowGuard Engineering Guide

## Context

FlowGuard OpenCL is a C++17 visual collision-awareness research prototype. Blender
streams deterministic drone-camera frames to a C++ engine; OpenCL perception feeds
risk and simulated control; OpenCV and React dashboards consume one telemetry
contract. Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing a
boundary.

Important paths:

- `include/flowguard/` and `src/`: project-owned C++ APIs and implementations.
- `kernels/`: measured OpenCL perception code.
- `simulator/`: Blender client and evaluation-only ground truth.
- `web/`: React/TypeScript live dashboard and saved-report UI.
- `tests/`: dependency-free behavior tests and OpenCL synthetic replay.
- `benchmarks/legacy/`: immutable, clearly labelled pre-FlowGuard evidence.

## Before editing

- Inspect Git status, relevant callers, tests, and external boundaries first.
- Preserve unrelated changes and existing conventions. Keep the diff tied to the
  requested outcome.
- Reproduce bugs where practical. Record a baseline before performance changes.
- Do not invent device support, numerical equivalence, scenario outcomes, benchmark
  numbers, or edge-hardware results. Resolve them by running the code.

## Verified commands

Native configure, build, and full tests:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Warning-clean release check:

```bash
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DFLOWGUARD_WARNINGS_AS_ERRORS=ON
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

Frontend:

```bash
cd web
npm ci
npm run lint
npm run build
```

Focused runtime checks:

```bash
build/flowguard devices
build/flowguard replay --synthetic expanding --frames 12 --mode cpu --no-dashboard
build/flowguard replay --synthetic expanding --frames 12 --mode gpu --no-dashboard
build/flowguard replay --synthetic expanding --frames 12 --mode fixed --gpu-ratio 0.70 --no-dashboard
build/flowguard replay --synthetic expanding --frames 32 --mode adaptive --no-dashboard
git diff --check
```

Blender and GPU checks are local release verification. CI uses PoCL CPU OpenCL and
synthetic replay because hosted runners do not provide the local Radeon GPU.

## Structure and dependency direction

- Prefer a small number of modules organized by real responsibility. Split when it
  improves readability, focused testing, actual reuse, or dependency isolation—not
  mechanically to reduce line count.
- Do not create a monolith containing CLI, transport, kernels, policy, UI, and
  persistence. Also do not create one file per function, single-implementation
  interfaces, trivial wrappers, speculative factories, or generic `Manager`,
  `Helper`, `Processor`, and `BaseService` classes.
- Keep `main.cpp` and command handling thin. Application code sequences frames;
  perception, scheduling, risk, and control own calculations; infrastructure owns
  OpenCL, sockets, OpenCV, Blender, WebSocket/HTTP, and artifacts.
- Domain APIs may use project values and OpenCV's basic `Mat`/geometry containers.
  They must not depend on Blender objects, HTTP messages, filesystem paths, window
  state, or raw OpenCL handles.
- Ground truth is evaluation-only. Never add it to a perception, risk, scheduler, or
  controller signature. Dashboards render engine telemetry and must not recalculate
  decisions independently.
- Add shared frontend code only after real reuse. Keep feature code colocated and do
  not create empty feature-folder architecture.

## C++ and OpenCL

- Use C++17, RAII, value semantics, standard containers, explicit ownership, and
  checked conversions for dimensions/buffer sizes.
- Check every OpenCL result affecting correctness. Errors must name the operation and
  device; kernel compilation failures must include the build log.
- `cpu` discovers and initializes only CPU OpenCL. `gpu` does the same only for GPU.
  Heterogeneous modes fail clearly unless both exist.
- Keep initialization/warm-up, measured perception, visualization/encoding, Blender
  rendering, and cleanup as separate timing boundaries.
- Establish scalar/reference behavior before optimizing kernels. Cover zero motion,
  translation, expansion, low texture, noise, boundaries, and non-divisible sizes.
  Compare all execution modes within a documented tolerance on identical frames.

## Testing and evidence

- Add behavior-focused tests for calculations, scheduler rules, controller decisions,
  protocol framing, and regressions. Cover failure paths and skip unavailable
  hardware with an explicit reason.
- Protocol changes must test partial reads, oversize/truncation, stale IDs, dimension
  changes, disconnect, and restart semantics.
- Formal modes run sequentially over the same prerecorded frames, exclude documented
  warm-up, and disable dashboards. Retain raw CSV/JSONL plus hardware, runtime,
  resolution, repetitions, and measurement boundary.
- Report throughput, p50/p95/p99, 33.3 ms misses, scheduler allocation, collision,
  minimum clearance, and unnecessary braking where the input provides ground truth.
- Never claim a forced speedup or present this laptop's result as Jetson, Raspberry
  Pi, edge-device, LiDAR, or real-flight evidence.

## Definition of done

- The requested behavior works without unrelated restructuring or new dependencies.
- Architecture boundaries and the ground-truth invariant remain intact.
- Focused checks pass, followed by the broadest justified native and web checks.
- Visual/performance changes include truthful retained evidence.
- Commands, schema, setup, and README match the repository after the change.
- Final reporting names checks actually run and anything not verified.
