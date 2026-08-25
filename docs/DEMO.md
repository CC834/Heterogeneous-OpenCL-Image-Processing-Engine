# FlowGuard visual demo

The README montage is 23.8 seconds of recorded FlowGuard output from six separate
Blender 4.5.9 scenarios:

1. Frontal wall — recognise rapid visual expansion and avoid the wall.
2. Offset pillar — steer around a partially blocked path.
3. Doorway — preserve the clear centre and pass through the opening.
4. Corridor — continue between bilateral boundaries without oscillation.
5. Safe lateral pass — avoid treating harmless side motion as a centre collision.
6. Crossing obstacle — slow and steer as an obstacle enters the route.

Each source run used a fixed seed, 640×360 camera frames, a 30 Hz kinematic step,
GPU-mode perception, and the native annotated dashboard. The montage adds only the
scenario-name labels. Flow vectors, focus of expansion, risk bars, warning state,
control values, FPS, latency, and allocation shown in the footage come from the
engine's recorded telemetry.

Two formats are retained:

- [`flowguard-demo.gif`](assets/flowguard-demo.gif) is an optimized 560-pixel-wide,
  8 FPS preview for inline GitHub display.
- [`flowguard-scenarios.mp4`](assets/flowguard-scenarios.mp4) is the full 640×360,
  30 FPS H.264 version for closer inspection or portfolio embedding.

The still [`flowguard-poster.jpg`](assets/flowguard-poster.jpg) is suitable as a
video poster, project-card image, or social preview.

## Recording a new source run

```bash
build/flowguard simulate \
  --scenario crossing-obstacle \
  --mode gpu \
  --frames 120 \
  --no-dashboard \
  --record
```

The display is disabled during the run, but `--record` still produces the annotated
MP4 and synchronized JSONL telemetry under `artifacts/<run-id>/`. Formal performance
benchmarks remain separate and disable visualization and encoding entirely.
