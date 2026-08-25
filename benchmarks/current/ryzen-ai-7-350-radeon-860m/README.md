# Current-host smoke benchmark

Recorded 2026-08-25 on the repository owner's Ryzen AI 7 350/Radeon 860M machine.
The GPU runtime was Mesa Rusticl 25.1.9; the CPU runtime was PoCL 6.0.

The command used one deterministic 640×360 synthetic expansion replay, ran modes
sequentially, disabled dashboards, and excluded one warm-up frame:

```bash
build/flowguard benchmark --suite default \
  --modes cpu,gpu,fixed,adaptive --repeats 1 --frames 12
```

Only ten measured frames remain after pipeline priming and the excluded warm-up.
This compact run supports the README visual and verifies the benchmark workflow; it
is too short to serve as a stable performance claim. Run five repeats over the
default suite before drawing performance conclusions. It is not Jetson, Raspberry
Pi, embedded, real-flight, or power-consumption evidence.

On this smoke workload GPU-only was fastest. Fixed and adaptive heterogeneous modes
did not win. Adaptive stayed at its initial 50% allocation because ten measured
frames do not reach its 30-frame update interval.

`scenario-acceptance.csv` contains the final Blender 4.5.9 headless release checks
using seed 42 and GPU perception. Each scenario ran for 120 frames on the same host.
All six had zero collision and more than 0.5 m clearance. The doorway, corridor,
and safe lateral pass completed 7.93 m of forward travel; the safe lateral pass
held 2.0 m/s and never braked. These are simulation outcomes, not real-flight
safety evidence.
