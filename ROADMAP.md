# Post-v1 roadmap

## Real edge-device validation

The generic virtual profiles now provide reproducible constraint and fault
experiments, but they do not satisfy this milestone. Completion still requires:

- Select and document one real target: Jetson, Raspberry Pi with a supported
  accelerator, or another OpenCL-capable edge computer.
- Add 320×180 and 480×270 presets without changing default desktop evidence.
- Make device selection portable across vendor runtimes and record missing modes as
  unavailable rather than substituting another device.
- Run identical replay inputs sequentially in all supported modes.
- Record OS, runtime, driver, clock/power mode, cooling, and power measurements when
  the platform exposes them.
- Publish a separate report whose title names the actual device.

No task in this milestone may relabel the Ryzen/Radeon v1 results as edge-hardware
or real-flight evidence.
