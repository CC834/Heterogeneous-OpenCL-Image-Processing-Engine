# FlowGuard Architecture

## System flow

FlowGuard is deliberately a compact native application, not an enterprise layer
tree:

```text
Blender frames / video replay
            ↓
application frame loop ─────→ telemetry ─────→ native UI / web / artifacts
            ↓
OpenCL perception → risk estimation → avoidance controller
            ↕
      tile scheduler
```

`src/main.cpp` maps exceptions to an exit status. `src/application.cpp` is the
composition and frame-sequencing boundary. It can know concrete implementations;
calculation modules cannot know delivery or storage mechanics.

## Responsibilities

### Perception

`opencl_pipeline.cpp` owns OpenCL device selection, contexts, programs, buffers,
events, and the measured pipeline. `kernels/perception.cl` implements BGR-to-gray,
Gaussian filtering, two pyramid reductions, and coarse-to-fine block matching.
Host-side post-processing fits a confidence-weighted affine field and derives focus
of expansion and radial TTC proxies.

CPU-only/GPU-only instantiate only their requested device class. Fixed/adaptive
modes run CPU and GPU tile rows concurrently and assemble a grid in stable order.

### Scheduling, risk, and control

`scheduler.cpp` is pure allocation policy. Adaptive scheduling updates every 30
observations from EWMA tile rates, clamps GPU allocation to 5–95%, ignores changes
below 3%, and moves by no more than ten percentage points.

`risk_control.cpp` converts confident radial vectors to three sector risks and
spatial clusters. The controller chooses the lower-risk side, brakes when neither
side is safe, and smooths speed/yaw/brake with turn hysteresis.

These APIs do not accept `GroundTruth`. That is the principal safety boundary of
the research design.

### Simulator and external I/O

`protocol.cpp` owns partial-read-safe, length-prefixed loopback TCP. The Blender
script owns scene construction, image rendering, kinematics, and true collision/
clearance calculation. Ground truth crosses the protocol in the evaluation field
but application code only copies it into telemetry and artifacts.

`dashboard.cpp` renders authoritative engine telemetry. `web_server.cpp` binds to
127.0.0.1 and serves the React build plus 10 FPS WebSocket telemetry. `artifacts.cpp`
owns run directories, metadata, JSONL, MP4, and benchmark CSV.

## Invariants

- Simulator ground truth never influences perception or control.
- Identical inputs produce stable ordering and equivalent outputs across modes
  within the tolerance established by tests.
- CPU/GPU single-device modes do not require or initialize the other class.
- Formal benchmarks run one mode at a time on the same immutable replay and exclude
  warm-up.
- Rendering, encoding, web delivery, and Blender rendering are outside perception
  timing unless a report explicitly says end-to-end.
- Telemetry schema version 1 is shared by native UI, live web, JSONL, and reports.
- Simulator and web listeners bind to loopback.
- Every performance claim names the hardware/runtime that produced it.

## Adding or splitting code

Create a module when it owns a concrete calculation or independently failing
external boundary and separation improves reading or testing now. Keep adjacent
small responsibilities together otherwise. Do not add abstractions for imagined
devices, future flight stacks, or hypothetical databases.

The React app remains one dashboard feature while it is small. Split live transport,
charts, or replay into feature-local modules only when their independent behavior
and tests justify it.

## Post-v1 evolution

Real edge deployment is a separate evidence milestone. It may add portable device
selection and resolution presets without changing domain policy. A real flight
stack, depth sensor, or LiDAR source would be a new external input boundary; it must
not be simulated and labelled as hardware evidence.
