# Simulator protocol v1

Blender connects to `127.0.0.1:8765`. FlowGuard never binds this protocol to a
non-loopback address.

Every message starts with a four-byte unsigned big-endian payload length. A frame
payload then contains:

```text
uint32 big-endian metadata_length
metadata_length bytes UTF-8 JSON
width × height × 3 bytes continuous BGR pixels
```

Metadata fields are `schema_version`, `frame_id`, `simulation_time_s`, `seed`,
`width`, `height`, `channels`, pose (`x`, `y`, `z`, `yaw`), and evaluation-only
`nearest_obstacle_m`, `true_ttc_s`, and `collision`.

The reply is a length-prefixed UTF-8 JSON object containing `schema_version`, the
matching `frame_id`, `speed`, `yaw_rate`, and `brake`.

Receivers accumulate partial TCP reads. Oversized, truncated, stale, or
dimension-changing streams fail with a specific error. A simulator restart uses a
new connection and starts frame IDs again from zero.
