import argparse
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "integrations"))

import px4_bridge  # noqa: E402


class Px4BridgeTest(unittest.TestCase):
    def test_valid_command(self):
        payload = json.dumps(
            {
                "schema_version": 1,
                "source": "flowguard-opencl",
                "frame_id": 8,
                "speed_mps": 1.5,
                "yaw_rate_deg_s": -20.0,
                "brake": 0.25,
            }
        ).encode()
        command = px4_bridge.decode_command(payload)
        self.assertEqual(command.frame_id, 8)
        self.assertEqual(command.yaw_rate_deg_s, -20.0)

    def test_invalid_or_nonfinite_command_is_rejected(self):
        with self.assertRaises(ValueError):
            px4_bridge.decode_command(b'{"schema_version":2}')
        payload = b'{"schema_version":1,"source":"flowguard-opencl","frame_id":1,"speed_mps":1,"yaw_rate_deg_s":NaN,"brake":0}'
        with self.assertRaises(ValueError):
            px4_bridge.decode_command(payload)

    def test_listener_must_be_loopback(self):
        self.assertEqual(px4_bridge.loopback_address("localhost:9002"), ("127.0.0.1", 9002))
        with self.assertRaises(argparse.ArgumentTypeError):
            px4_bridge.loopback_address("0.0.0.0:9002")


if __name__ == "__main__":
    unittest.main()
