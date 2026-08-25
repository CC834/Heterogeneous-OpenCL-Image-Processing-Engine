#!/usr/bin/env python3
"""Translate loopback FlowGuard commands into PX4 SITL MAVLink setpoints.

The bridge is optional and intentionally separate from the perception engine. It
does not make FlowGuard a flight controller and does not arm PX4 unless --arm is
passed explicitly.
"""

import argparse
import json
import math
import socket
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    frame_id: int
    speed_mps: float
    yaw_rate_deg_s: float
    brake: float


def loopback_address(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or host not in {"127.0.0.1", "localhost"}:
        raise argparse.ArgumentTypeError("address must be 127.0.0.1:PORT or localhost:PORT")
    try:
        port = int(port_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return "127.0.0.1", port


def decode_command(payload: bytes) -> Command:
    message = json.loads(payload)
    if message.get("schema_version") != 1 or message.get("source") != "flowguard-opencl":
        raise ValueError("unsupported FlowGuard command schema")
    command = Command(
        frame_id=int(message["frame_id"]),
        speed_mps=float(message["speed_mps"]),
        yaw_rate_deg_s=float(message["yaw_rate_deg_s"]),
        brake=float(message["brake"]),
    )
    values = (command.speed_mps, command.yaw_rate_deg_s, command.brake)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("command contains a non-finite value")
    if command.frame_id < 0 or not 0.0 <= command.speed_mps <= 2.0:
        raise ValueError("command is outside FlowGuard speed bounds")
    if abs(command.yaw_rate_deg_s) > 60.0 or not 0.0 <= command.brake <= 1.0:
        raise ValueError("command is outside FlowGuard yaw/brake bounds")
    return command


def send_setpoint(connection, mavutil, command: Command) -> None:
    # BODY_NED velocity: forward speed and yaw-rate only. Position, acceleration,
    # and absolute yaw are deliberately ignored.
    type_mask = 7 | (7 << 6) | (1 << 10)
    connection.mav.set_position_target_local_ned_send(
        int(time.monotonic() * 1000) & 0xFFFFFFFF,
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0.0,
        0.0,
        0.0,
        command.speed_mps,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        math.radians(command.yaw_rate_deg_s),
    )


def enable_offboard(connection, mavutil, command: Command, arm: bool) -> None:
    # PX4 requires a stream of setpoints before it accepts Offboard mode.
    for _ in range(20):
        send_setpoint(connection, mavutil, command)
        time.sleep(0.05)
    modes = connection.mode_mapping()
    if not modes or "OFFBOARD" not in modes:
        raise RuntimeError("PX4 heartbeat did not provide an OFFBOARD mode mapping")
    connection.set_mode("OFFBOARD")
    if arm:
        connection.arducopter_arm()
        print("PX4 arm command sent (--arm was explicitly supplied)", flush=True)
    else:
        print("PX4 remains disarmed; pass --arm only for a safe SITL session", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Bridge FlowGuard loopback commands to PX4 SITL over MAVLink"
    )
    result.add_argument("--listen", type=loopback_address, default=("127.0.0.1", 9002))
    result.add_argument("--mavlink", default="udpin:127.0.0.1:14540")
    result.add_argument("--arm", action="store_true", help="explicitly arm after entering Offboard")
    result.add_argument(
        "--print-only", action="store_true", help="validate and print commands without pymavlink/PX4"
    )
    return result


def main() -> int:
    args = parser().parse_args()
    connection = None
    mavutil = None
    if not args.print_only:
        try:
            from pymavlink import mavutil as imported_mavutil
        except ImportError as error:
            raise SystemExit("pymavlink is required unless --print-only is used") from error
        mavutil = imported_mavutil
        connection = mavutil.mavlink_connection(args.mavlink)
        print(f"Waiting for PX4 heartbeat on {args.mavlink}...", flush=True)
        heartbeat = connection.wait_heartbeat(timeout=30)
        if heartbeat is None:
            raise SystemExit("timed out waiting for a PX4 heartbeat")
        print(
            f"PX4 heartbeat: system {connection.target_system}, component {connection.target_component}",
            flush=True,
        )

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(args.listen)
    receiver.settimeout(2.0)
    print(f"Listening for FlowGuard on udp://{args.listen[0]}:{args.listen[1]}", flush=True)
    previous_frame = -1
    offboard_enabled = False
    while True:
        try:
            payload, _ = receiver.recvfrom(4096)
        except socket.timeout:
            continue
        try:
            command = decode_command(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"Rejected command: {error}", flush=True)
            continue
        if command.frame_id <= previous_frame:
            print(f"Rejected stale frame {command.frame_id}", flush=True)
            continue
        previous_frame = command.frame_id
        if args.print_only:
            print(command, flush=True)
            continue
        if not offboard_enabled:
            enable_offboard(connection, mavutil, command, args.arm)
            offboard_enabled = True
        send_setpoint(connection, mavutil, command)


if __name__ == "__main__":
    raise SystemExit(main())
