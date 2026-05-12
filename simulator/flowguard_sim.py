"""Deterministic Blender 4.5 client for FlowGuard's loopback protocol.

Ground truth is computed here and sent only in evaluation metadata. The engine's
perception and controller receive camera pixels as their decision input.
"""

import argparse
import json
import math
import os
import socket
import struct
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

DT = 1.0 / 30.0
WIDTH = 640
HEIGHT = 360
DRONE_RADIUS = 0.25


def arguments():
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="corridor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=180)
    return parser.parse_args(values)


def material(name, color):
    value = bpy.data.materials.new(name)
    value.diffuse_color = (*color, 1.0)
    value.metallic = 0.15
    value.roughness = 0.55
    return value


def cube(name, location, scale, color, obstacles):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material(name + "Material", color))
    obstacles.append({"object": obj, "half": np.array(scale, dtype=float)})
    return obj


def visual_marker(name, location, scale, color):
    """Add visible texture without adding an evaluation obstacle."""
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(material(name + "Material", color))
    return obj


def wall_fiducials(front_x, width=4.5):
    index = 0
    for y in np.linspace(-width, width, 9):
        for z in (0.5, 1.25, 2.0, 2.75, 3.5):
            color = (0.92, 0.78, 0.12) if index % 2 else (0.05, 0.08, 0.12)
            visual_marker(f"WallMarker{index}", (front_x, float(y), z), (0.025, 0.22, 0.22), color)
            index += 1


def setup_scene(scenario):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.025, 0.035, 0.055)

    bpy.ops.mesh.primitive_plane_add(size=80, location=(12, 0, 0))
    ground = bpy.context.object
    ground.data.materials.append(material("Ground", (0.07, 0.09, 0.12)))

    obstacles = []
    blue = (0.08, 0.24, 0.55)
    orange = (0.75, 0.22, 0.06)
    if scenario == "frontal-wall":
        cube("FrontalWall", (7, 0, 2), (0.3, 5, 2), orange, obstacles)
        wall_fiducials(6.68)
    elif scenario == "offset-pillar":
        cube("OffsetPillar", (7, 0.55, 2), (0.7, 0.7, 2), orange, obstacles)
    elif scenario == "doorway":
        cube("DoorLeft", (8, -2.5, 2), (0.35, 1.5, 2), blue, obstacles)
        cube("DoorRight", (8, 2.5, 2), (0.35, 1.5, 2), blue, obstacles)
        cube("DoorTop", (8, 0, 3.8), (0.35, 1.0, 0.2), blue, obstacles)
    elif scenario == "corridor":
        cube("CorridorLeft", (10, -3.0, 2), (10, 0.25, 2), blue, obstacles)
        cube("CorridorRight", (10, 3.0, 2), (10, 0.25, 2), blue, obstacles)
        cube("CorridorEnd", (18, 0.6, 2), (0.3, 2.2, 2), orange, obstacles)
        wall_fiducials(17.68, 2.0)
    elif scenario == "safe-lateral-pass":
        cube("SafeLateral", (7, -4.0, 2), (0.6, 0.6, 2), orange, obstacles)
    elif scenario == "crossing-obstacle":
        obj = cube("CrossingObstacle", (8, -4, 2), (0.6, 0.6, 2), orange, obstacles)
        obstacles[-1]["moving"] = True
        obstacles[-1]["object"] = obj
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 8))
    bpy.context.object.rotation_euler = (math.radians(20), 0, math.radians(25))
    bpy.context.object.data.energy = 2.2
    bpy.ops.object.light_add(type="AREA", location=(4, 0, 6))
    bpy.context.object.data.energy = 900
    bpy.context.object.data.shape = "DISK"
    bpy.context.object.data.size = 8

    bpy.ops.object.camera_add(location=(0, 0, 2))
    camera = bpy.context.object
    camera.data.lens = 28
    camera.data.sensor_width = 32
    scene.camera = camera
    return scene, camera, obstacles


def update_camera(camera, x, y, yaw):
    camera.location = (x, y, 2.0)
    # Blender cameras look down local -Z with local Y as up.
    direction = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def nearest_clearance(x, y, obstacles):
    nearest = float("inf")
    for obstacle in obstacles:
        obj = obstacle["object"]
        half = obstacle["half"]
        dx = max(abs(x - obj.location.x) - half[0], 0.0)
        dy = max(abs(y - obj.location.y) - half[1], 0.0)
        dz = max(abs(2.0 - obj.location.z) - half[2], 0.0)
        nearest = min(nearest, math.sqrt(dx * dx + dy * dy + dz * dz) - DRONE_RADIUS)
    return nearest


def render_bgr(scene):
    path = f"/tmp/flowguard-blender-{os.getpid()}.png"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path, check_existing=False)
    pixels = np.asarray(image.pixels[:], dtype=np.float32)
    rgba = pixels.reshape((HEIGHT, WIDTH, 4))
    rgb = np.flipud(rgba[:, :, :3])
    bgr = (np.clip(rgb[:, :, ::-1], 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
    bpy.data.images.remove(image)
    os.unlink(path)
    return bgr


def send_frame(sock, metadata, bgr):
    encoded = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    payload = struct.pack("!I", len(encoded)) + encoded + bgr
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_command(sock):
    header = receive_exact(sock, 4)
    size = struct.unpack("!I", header)[0]
    return json.loads(receive_exact(sock, size).decode("utf-8"))


def receive_exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        value = sock.recv(size - len(chunks))
        if not value:
            raise ConnectionError("FlowGuard closed the simulator connection")
        chunks.extend(value)
    return bytes(chunks)


def connect():
    for _ in range(100):
        try:
            return socket.create_connection(("127.0.0.1", 8765), timeout=5)
        except OSError:
            time.sleep(0.1)
    raise ConnectionError("cannot connect to FlowGuard at 127.0.0.1:8765")


def main():
    args = arguments()
    np.random.seed(args.seed)
    scene, camera, obstacles = setup_scene(args.scenario)
    x = y = yaw = 0.0
    speed = 2.0
    command = {"speed": 2.0, "yaw_rate": 0.0, "brake": 0.0}
    with connect() as sock:
        for frame_id in range(args.frames):
            for obstacle in obstacles:
                if obstacle.get("moving"):
                    obstacle["object"].location.y = -4.0 + frame_id * DT * 1.7
            update_camera(camera, x, y, yaw)
            clearance = nearest_clearance(x, y, obstacles)
            true_ttc = clearance / speed if speed > 0.05 and clearance >= 0 else -1.0
            metadata = {
                "schema_version": 1,
                "frame_id": frame_id,
                "simulation_time_s": frame_id * DT,
                "seed": args.seed,
                "width": WIDTH,
                "height": HEIGHT,
                "channels": 3,
                "x": x,
                "y": y,
                "z": 2.0,
                "yaw": yaw,
                "nearest_obstacle_m": clearance,
                "true_ttc_s": true_ttc,
                "collision": clearance <= 0.0,
            }
            send_frame(sock, metadata, render_bgr(scene))
            command = receive_command(sock)
            desired = min(2.0, max(0.0, float(command["speed"])))
            if float(command["brake"]) > 0.5:
                desired = 0.0
            speed += max(-3.0 * DT, min(3.0 * DT, desired - speed))
            yaw_rate = max(-60.0, min(60.0, float(command["yaw_rate"])))
            yaw += math.radians(yaw_rate) * DT
            x += math.cos(yaw) * speed * DT
            y += math.sin(yaw) * speed * DT
            scene.frame_set(frame_id + 1)


if __name__ == "__main__":
    main()
