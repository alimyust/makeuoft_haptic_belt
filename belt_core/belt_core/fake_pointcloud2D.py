#!/usr/bin/env python3
"""
Fake ESDF-like PointCloud2 publisher for ROS 2.

Publishes sensor_msgs/PointCloud2 with fields:
- x, y, z (float32)
- intensity (float32)  # interpreted as "clearance distance in meters"

This mimics how you'd consume NVBlox ESDF pointcloud for collision urgency logic,
without requiring a camera or NVBlox running.
"""

import math
import struct
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from builtin_interfaces.msg import Time


def make_point_fields() -> List[PointField]:
    # x,y,z,intensity all float32 => 4 fields * 4 bytes = 16 bytes per point
    fields = []
    offset = 0
    for name in ["x", "y", "z", "intensity"]:
        f = PointField()
        f.name = name
        f.offset = offset
        f.datatype = PointField.FLOAT32
        f.count = 1
        fields.append(f)
        offset += 4
    return fields


def pack_point(x: float, y: float, z: float, intensity: float) -> bytes:
    # little-endian float32 pack
    return struct.pack("<ffff", x, y, z, intensity)


class FakeEsdfPointcloudPublisher(Node):
    def __init__(self) -> None:
        super().__init__("fake_esdf_pointcloud_publisher")

        # Parameters (settable via ros2 param)
        self.declare_parameter("topic", "/fake/combined_esdf_pointcloud")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("publish_rate_hz", 10.0)

        # Grid definition for the "slice"
        self.declare_parameter("x_min", 0.2)
        self.declare_parameter("x_max", 3.0)
        self.declare_parameter("y_min", -1.0)
        self.declare_parameter("y_max", 1.0)
        self.declare_parameter("resolution", 0.05)  # meters between sample points
        self.declare_parameter("z", 0.0)

        # Scenario control
        self.declare_parameter("scenario", "wall")  # clear|wall|left_blocked|right_blocked|gap_center|moving_obstacle

        # Scenario parameters
        self.declare_parameter("far_clearance", 5.0)      # intensity for free space
        self.declare_parameter("wall_distance", 1.2)      # meters ahead
        self.declare_parameter("wall_thickness", 0.15)    # meters (used for shaping)
        self.declare_parameter("blocked_clearance", 0.35) # intensity for "blocked" region
        self.declare_parameter("gap_width", 0.6)          # meters
        self.declare_parameter("moving_center_x", 1.0)    # meters
        self.declare_parameter("moving_amp_y", 0.6)       # meters
        self.declare_parameter("moving_radius", 0.35)     # meters

        topic = self.get_parameter("topic").value
        self.pub = self.create_publisher(PointCloud2, topic, 10)

        rate = float(self.get_parameter("publish_rate_hz").value)
        period = 1.0 / max(rate, 0.1)
        self.t0 = self.get_clock().now()
        self.timer = self.create_timer(period, self.on_timer)

        self.fields = make_point_fields()
        self.point_step = 16  # bytes

        self.get_logger().info(f"Publishing fake ESDF PointCloud2 on {topic}")
        self.get_logger().info("Change scenario live: ros2 param set /fake_esdf_pointcloud_publisher scenario <name>")

    def on_timer(self) -> None:
        # Read parameters each publish so you can tune live
        frame_id = self.get_parameter("frame_id").value
        scenario = self.get_parameter("scenario").value

        x_min = float(self.get_parameter("x_min").value)
        x_max = float(self.get_parameter("x_max").value)
        y_min = float(self.get_parameter("y_min").value)
        y_max = float(self.get_parameter("y_max").value)
        res = float(self.get_parameter("resolution").value)
        z0 = float(self.get_parameter("z").value)

        far = float(self.get_parameter("far_clearance").value)
        wall_d = float(self.get_parameter("wall_distance").value)
        wall_t = float(self.get_parameter("wall_thickness").value)
        blocked = float(self.get_parameter("blocked_clearance").value)
        gap_w = float(self.get_parameter("gap_width").value)

        mov_cx = float(self.get_parameter("moving_center_x").value)
        mov_ay = float(self.get_parameter("moving_amp_y").value)
        mov_r = float(self.get_parameter("moving_radius").value)

        # Time for moving scenario
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9

        # Generate points
        points_bytes = bytearray()
        width = 0

        # Sample a rect grid in front of base_link
        x = x_min
        while x <= x_max + 1e-9:
            y = y_min
            while y <= y_max + 1e-9:
                intensity = far

                if scenario == "clear":
                    intensity = far

                elif scenario == "wall":
                    # A wall at x = wall_d spanning all y: intensity decreases as you approach wall
                    # We'll approximate "clearance" as max(wall_d - x, tiny)
                    clearance = max(wall_d - x, 0.05)
                    # If past the wall plane, clamp to very small
                    if x >= wall_d:
                        clearance = 0.05
                    intensity = min(clearance, far)

                elif scenario == "left_blocked":
                    # Left side (y > 0) has close obstacle
                    if y > 0.0:
                        intensity = blocked
                    else:
                        intensity = far

                elif scenario == "right_blocked":
                    if y < 0.0:
                        intensity = blocked
                    else:
                        intensity = far

                elif scenario == "gap_center":
                    # Two walls with a central gap: blocked except |y| < gap_w/2
                    if abs(y) > (gap_w / 2.0):
                        # make it worse as you go forward (like approaching wall)
                        clearance = max(wall_d - x, 0.05)
                        if x >= wall_d:
                            clearance = 0.05
                        intensity = min(clearance, blocked)
                    else:
                        intensity = far

                elif scenario == "moving_obstacle":
                    # A moving circular "person" blob oscillating in y
                    cy = mov_ay * math.sin(0.8 * t)
                    dx = x - mov_cx
                    dy = y - cy
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < mov_r:
                        # inside obstacle -> very small clearance
                        intensity = 0.15
                    else:
                        # clearance grows with distance to blob surface
                        intensity = min(max(d - mov_r, 0.05), far)

                else:
                    # Unknown scenario -> clear
                    intensity = far

                points_bytes += pack_point(x, y, z0, intensity)
                width += 1
                y += res
            x += res

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.height = 1
        msg.width = width
        msg.fields = self.fields
        msg.is_bigendian = False
        msg.point_step = self.point_step
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = bytes(points_bytes)

        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = FakeEsdfPointcloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
