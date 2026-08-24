# Optional PX4 SITL command bridge

FlowGuard's Blender simulator remains the supported, deterministic closed-loop
demo. An experimental bridge also lets a PX4 software-in-the-loop instance receive
FlowGuard's applied speed and yaw-rate commands through MAVLink:

```text
FlowGuard perception/control
          ↓ loopback UDP JSON
integrations/px4_bridge.py
          ↓ MAVLink BODY_NED velocity setpoints
        PX4 SITL
```

The bridge is an integration boundary, not flight-control firmware. It is loopback
only, validates command bounds and increasing frame IDs, and never arms PX4 unless
`--arm` is supplied explicitly.

## Inspect commands without PX4

Terminal one:

```bash
python3 integrations/px4_bridge.py --print-only
```

Terminal two:

```bash
build/flowguard simulate --scenario corridor --mode cpu \
  --hardware-profile edge-balanced-sim --control-output 127.0.0.1:9002
```

## Connect a PX4 SITL instance

Install PX4 SITL using the [official PX4 development setup](https://docs.px4.io/main/en/dev_setup/dev_env.html),
start a supported multicopter SITL vehicle, and install the optional Python
dependency in a virtual environment:

```bash
python3 -m venv .venv-px4
. .venv-px4/bin/activate
python -m pip install pymavlink
python integrations/px4_bridge.py --mavlink udpin:127.0.0.1:14540
```

Start FlowGuard with `--control-output 127.0.0.1:9002`. First verify the connection
while PX4 remains disarmed. Use `--arm` only in SITL after checking the port and
vehicle state. Port direction varies with the PX4 launch configuration, so override
`--mavlink` when necessary.

PX4 Offboard mode also requires a working position/pose estimate and a continuing
setpoint stream. FlowGuard sends at the selected 20 or 30 Hz camera rate; PX4's
[current Offboard documentation](https://docs.px4.io/main/en/flight_modes/offboard.html)
describes the estimator, pre-stream, timeout, and failsafe requirements.

This repository does not claim PX4 validation from CI or from the current machine:
PX4 and `pymavlink` are optional and were not available in the standard build
environment. A future integration test should synchronize PX4 camera input with the
same frame IDs before collision outcomes are compared.
