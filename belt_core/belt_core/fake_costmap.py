#!/usr/bin/env python3
"""
PointCloud2 (x,y,z,intensity=float32) -> nav_msgs/OccupancyGrid (shaded)

Assumes:
- points are a 2D slice (z ~ constant)
- intensity = "clearance distance in meters" (smaller = more occupied)

Outputs:
- /fake/occupancy_grid (nav_msgs/OccupancyGrid)
  data values: -1 unknown, 0..100 probability-ish occupancy

Shading is achieved by mapping clearance -> occupancy smoothly rather than thresholding.
"""

import math
import struct
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header


def _field_offset(fields, name: str) -> int:
    for f in fields:
        if f.name == name:
            return int(f.offset)
    raise RuntimeError(f"PointCloud2 missing field '{name}'")


def _quat_identity() -> Quaternion:
    q = Quaternion()
    q.w = 1.0
    return q


def clearance_to_occ_linear(clearance: float, clear_occ_m: float, clear_free_m: float) -> int:
    """
    Map clearance (m) to OccupancyGrid value (0..100), linearly.
      clearance <= clear_occ_m -> 100 (occupied)
      clearance >= clear_free_m -> 0   (free)
      between -> interpolated (more clearance => lower occupancy)
    """
    if not math.isfinite(clearance):
        return -1
    if clear_free_m <= clear_occ_m:
        # fallback to a step if misconfigured
        return 100 if clearance <= clear_occ_m else 0

    if clearance <= clear_occ_m:
        return 100
    if clearance >= clear_free_m:
        return 0

    t = (clear_free_m - clearance) / (clear_free_m - clear_occ_m)  # 0..1
    return int(round(100.0 * t))


class EsdfPointcloudToOccupancyGrid(Node):
    def __init__(self):
        super().__init__("esdf_pointcloud_to_occupancy_grid")

        # Topics
        self.declare_parameter("input_topic", "/fake/combined_esdf_pointcloud")
        self.declare_parameter("output_topic", "/fake/occupancy_grid")

        # Grid config (match your fake publisher defaults)
        self.declare_parameter("x_min", 0.2)
        self.declare_parameter("x_max", 3.0)
        self.declare_parameter("y_min", -1.0)
        self.declare_parameter("y_max", 1.0)
        self.declare_parameter("resolution", 0.05)

        # Whether to use fixed bounds from params or infer bounds from each msg
        self.declare_parameter("use_bounds_from_params", True)

        # Shaded occupancy mapping params
        self.declare_parameter("clear_occ_m", 0.20)   # <= this clearance => 100
        self.declare_parameter("clear_free_m", 1.50)  # >= this clearance => 0

        # Optional: unknown handling (if intensity can be 0 or negative in your pipeline)
        self.declare_parameter("unknown_clearance_m", 0.0)  # <= -> unknown; 0 disables

        # Optional simple inflation (in grid cells)
        self.declare_parameter("inflate_radius_m", 0.0)  # 0 disables

        in_topic = self.get_parameter("input_topic").value
        out_topic = self.get_parameter("output_topic").value

        self.sub = self.create_subscription(PointCloud2, in_topic, self.cb, 10)
        self.pub = self.create_publisher(OccupancyGrid, out_topic, 10)

        self.get_logger().info(f"Subscribing: {in_topic}")
        self.get_logger().info(f"Publishing:  {out_topic}")

    def _infer_bounds(self, msg: PointCloud2, x_off: int, y_off: int) -> Optional[Tuple[float, float, float, float]]:
        n = msg.width * msg.height
        if n == 0 or len(msg.data) < msg.point_step:
            return None

        x_min = float("inf")
        x_max = float("-inf")
        y_min = float("inf")
        y_max = float("-inf")

        step = msg.point_step
        data = msg.data

        for i in range(n):
            base = i * step
            x = struct.unpack_from("<f", data, base + x_off)[0]
            y = struct.unpack_from("<f", data, base + y_off)[0]
            if math.isfinite(x) and math.isfinite(y):
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

        if not math.isfinite(x_min):
            return None

        return (x_min, x_max, y_min, y_max)

    def cb(self, msg: PointCloud2):
        try:
            x_off = _field_offset(msg.fields, "x")
            y_off = _field_offset(msg.fields, "y")
            i_off = _field_offset(msg.fields, "intensity")
        except Exception as e:
            self.get_logger().error(str(e))
            return

        res = float(self.get_parameter("resolution").value)
        if res <= 0.0:
            self.get_logger().warn("resolution must be > 0")
            return

        use_param_bounds = bool(self.get_parameter("use_bounds_from_params").value)
        if use_param_bounds:
            x_min = float(self.get_parameter("x_min").value)
            x_max = float(self.get_parameter("x_max").value)
            y_min = float(self.get_parameter("y_min").value)
            y_max = float(self.get_parameter("y_max").value)
        else:
            inferred = self._infer_bounds(msg, x_off, y_off)
            if inferred is None:
                self.get_logger().warn("Could not infer bounds from pointcloud")
                return
            x_min, x_max, y_min, y_max = inferred

        if x_max <= x_min or y_max <= y_min:
            self.get_logger().warn("Invalid bounds")
            return

        width = int(math.floor((x_max - x_min) / res)) + 1
        height = int(math.floor((y_max - y_min) / res)) + 1
        size = width * height

        # Store the minimum clearance per cell
        min_clear = [float("inf")] * size

        step = msg.point_step
        data = msg.data
        n = msg.width * msg.height

        for p in range(n):
            base = p * step
            x = struct.unpack_from("<f", data, base + x_off)[0]
            y = struct.unpack_from("<f", data, base + y_off)[0]
            clearance = struct.unpack_from("<f", data, base + i_off)[0]

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(clearance)):
                continue

            gx = int(math.floor((x - x_min) / res))
            gy = int(math.floor((y - y_min) / res))
            if gx < 0 or gx >= width or gy < 0 or gy >= height:
                continue

            idx = gy * width + gx
            if clearance < min_clear[idx]:
                min_clear[idx] = clearance

        clear_occ_m = float(self.get_parameter("clear_occ_m").value)
        clear_free_m = float(self.get_parameter("clear_free_m").value)
        unk_thr = float(self.get_parameter("unknown_clearance_m").value)

        inflate_r_m = float(self.get_parameter("inflate_radius_m").value)
        inflate_cells = int(math.ceil(inflate_r_m / res)) if inflate_r_m > 0.0 else 0

        # Build shaded occupancy grid: unknown (-1) unless we saw a point for that cell
        grid = [-1] * size

        for idx, c in enumerate(min_clear):
            if c == float("inf"):
                continue  # stay unknown
            if unk_thr > 0.0 and c <= unk_thr:
                grid[idx] = -1
                continue
            grid[idx] = clearance_to_occ_linear(c, clear_occ_m, clear_free_m)

        # Optional inflation (simple square). Keeps shading by inflating to max(occupied-ness).
        if inflate_cells > 0:
            inflated = grid[:]
            for gy in range(height):
                for gx in range(width):
                    idx = gy * width + gx
                    v = grid[idx]
                    if v < 0:
                        continue
                    # Inflate around "more occupied" cells preferentially
                    if v == 0:
                        continue
                    for oy in range(-inflate_cells, inflate_cells + 1):
                        ny = gy + oy
                        if ny < 0 or ny >= height:
                            continue
                        for ox in range(-inflate_cells, inflate_cells + 1):
                            nx = gx + ox
                            if nx < 0 or nx >= width:
                                continue
                            nidx = ny * width + nx
                            # raise occupancy (darker) but don't overwrite unknown with free
                            if inflated[nidx] < 0:
                                inflated[nidx] = v
                            else:
                                inflated[nidx] = max(inflated[nidx], v)
            grid = inflated

        out = OccupancyGrid()
        out.header = Header()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id

        info = MapMetaData()
        info.resolution = float(res)
        info.width = int(width)
        info.height = int(height)

        origin = Pose()
        origin.position.x = float(x_min)
        origin.position.y = float(y_min)
        origin.position.z = 0.0
        origin.orientation = _quat_identity()
        info.origin = origin

        out.info = info
        out.data = grid

        self.pub.publish(out)


def main():
    rclpy.init()
    node = EsdfPointcloudToOccupancyGrid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
