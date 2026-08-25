# Virtual onboard hardware

FlowGuard can now place the real host perception pipeline inside a declared,
deterministic constraint model. This answers questions such as “what happens when
the onboard camera is slower, frames are noisy, the compute module throttles, or
the GPU disappears?” without pretending this laptop has become a Jetson or
Raspberry Pi.

```bash
build/flowguard hardware-profiles
build/flowguard replay --synthetic expanding --frames 120 --mode adaptive \
  --hardware-profile thermal-throttle-sim --dashboard native --record
build/flowguard simulate --scenario crossing-obstacle --mode adaptive \
  --hardware-profile gpu-failure-sim --dashboard native,web --record
```

## Declared profiles

| Profile | Camera | Main experiment |
| --- | --- | --- |
| `desktop-native` | 640×360 at 30 Hz | No synthetic delay; host measurement baseline |
| `edge-balanced-sim` | 480×270 at 30 Hz | Moderate camera noise, blur, and I/O latency |
| `edge-constrained-sim` | 320×180 at 20 Hz | CPU-only, noisy imagery, reused camera frames |
| `thermal-throttle-sim` | 480×270 at 30 Hz | 14 ms penalty after two simulated seconds |
| `gpu-failure-sim` | 480×270 at 30 Hz | GPU loss after two seconds and CPU failover |

The exact values live in `src/hardware_profile.cpp`, are shown by the CLI, and are
recorded per frame. Names are intentionally generic because no named physical
board produced these results.

The simulation and control tick remains 30 Hz. A lower-rate camera profile exposes
the latest camera sample to that loop between sensor updates, marks those frames as
`frame_reused`, and keeps controller integration tied to simulator time. This models
a common onboard producer/consumer arrangement without changing Blender's physics
step.

## Measurement boundary

- OpenCL perception latency is measured on the real host and stays in
  `latency_ms.perception`.
- Camera, actuator, and thermal delays are injected and separately labelled in
  telemetry.
- Telemetry schema v2 adds `virtual_hardware`; saved-report playback remains
  compatible with historical v1 telemetry.
- `latency_ms.total` is the combined experiment budget used for the profile's
  deadline indicator.
- `flowguard benchmark` accepts only `desktop-native`. Virtual-profile experiments
  belong to replay/simulation reports, not hardware benchmark tables.
- Simulator ground truth remains evaluation-only and never enters the hardware
  emulator, perception, scheduler, risk estimator, or controller.

This is software-in-the-loop constraint and fault simulation. Physical-board
performance, power, thermal behavior, and flight behavior still require that board.
