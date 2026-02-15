#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from std_msgs.msg import Float32MultiArray


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


class PointCloudHaptics(Node):
    """
    Subscribes to a PointCloud2 and outputs 3 haptic motor magnitudes [L, F, R].

    Strategy:
      - Find closest point (range-min) in the cloud (optionally only points in front).
      - Convert that point to a "camera-forward" convention (x forward, y left, z up).
      - Compute theta = atan2(y, x).
      - Choose motor: front if |theta| <= theta_front, else left if theta>0 else right.
      - Magnitude is a function of distance: closer -> higher vibration in [0,1].

    Output:
      - std_msgs/Float32MultiArray data: [left, front, right]
    """

    def __init__(self):
        super().__init__("pointcloud_haptics")

        # -------- Parameters --------
        self.declare_parameter("cloud_topic", "/camera/depth/color/points")
        self.declare_parameter("output_topic", "/haptics/motors")

        # If True, treat incoming cloud as optical frame (common for RealSense):
        # optical: x right, y down, z forward
        # we convert to: x forward, y left, z up
        self.declare_parameter("input_is_optical_frame", True)

        # Consider only points that are "in front" of the camera (after axis conversion):
        self.declare_parameter("only_in_front", True)

        # Angle deadzone for "front motor" (degrees)
        self.declare_parameter("theta_front_deg", 15.0)

        # Distance mapping (meters)
        # d <= d_min => mag=1, d >= d_max => mag=0
        self.declare_parameter("d_min", 0.35)
        self.declare_parameter("d_max", 3.0)

        # Shape of the response curve (>=1 makes it less sensitive far away, more near)
        self.declare_parameter("gamma", 1.6)

        # Performance control: process every Nth point (stride)
        self.declare_parameter("stride", 8)

        # -------- Read params --------
        self.cloud_topic = str(self.get_parameter("cloud_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.input_is_optical = bool(self.get_parameter("input_is_optical_frame").value)
        self.only_in_front = bool(self.get_parameter("only_in_front").value)

        self.theta_front = math.radians(float(self.get_parameter("theta_front_deg").value))
        self.d_min = float(self.get_parameter("d_min").value)
        self.d_max = float(self.get_parameter("d_max").value)
        self.gamma = float(self.get_parameter("gamma").value)
        self.stride = int(self.get_parameter("stride").value)

        if self.d_max <= self.d_min:
            self.get_logger().warn("d_max must be > d_min. Forcing d_max = d_min + 1.0")
            self.d_max = self.d_min + 1.0

        # -------- ROS I/O --------
        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, 10)
        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        self.get_logger().info(
            f"Listening: {self.cloud_topic} | Publishing: {self.output_topic} | "
            f"optical={self.input_is_optical} only_in_front={self.only_in_front} stride={self.stride}"
        )

    def optical_to_forward_left_up(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert optical frame (x right, y down, z forward)
        to standard camera/body-like convention (x forward, y left, z up).

        Mapping:
          x_fwd = z_opt
          y_left = -x_opt
          z_up = -y_opt
        """
        return (z, -x, -y)

    def distance_to_magnitude(self, d: float) -> float:
        """
        Map distance -> vibration in [0,1], closer = stronger.
        """
        if d <= self.d_min:
            return 1.0
        if d >= self.d_max:
            return 0.0
        x = (self.d_max - d) / (self.d_max - self.d_min)  # linear 0..1
        x = clamp(x, 0.0, 1.0)
        return x ** self.gamma

    def pick_closest_point(self, msg: PointCloud2) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (x_fwd, y_left, z_up, distance) for the closest point.
        Points are converted into forward/left/up convention if needed.
        """
        best_d = float("inf")
        best_pt = None

        # read_points yields (x, y, z, ...) in the cloud's frame
        # We'll stride by skipping most points for performance.
        i = 0
        for pt in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            if self.stride > 1 and (i % self.stride) != 0:
                i += 1
                continue
            i += 1

            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])

            # Convert axes if cloud is optical
            if self.input_is_optical:
                x_fwd, y_left, z_up = self.optical_to_forward_left_up(x, y, z)
            else:
                # Assume already x forward, y left, z up (REP-103-ish)
                x_fwd, y_left, z_up = x, y, z

            if self.only_in_front and x_fwd <= 0.05:
                continue

            d = math.sqrt(x_fwd * x_fwd + y_left * y_left + z_up * z_up)
            if d < best_d:
                best_d = d
                best_pt = (x_fwd, y_left, z_up, d)

        return best_pt

    def on_cloud(self, msg: PointCloud2) -> None:
        best = self.pick_closest_point(msg)

        # Default: no vibration
        left = front = right = 0.0

        if best is not None:
            x_fwd, y_left, z_up, d = best

            theta = math.atan2(y_left, x_fwd)  # +theta = left, -theta = right
            mag = self.distance_to_magnitude(d)

            # Motor selection
            if abs(theta) <= self.theta_front:
                front = mag
            elif theta > 0.0:
                left = mag
            else:
                right = mag

            # Optional debug (comment out if spammy)
            self.get_logger().debug(
                f"closest d={d:.2f}m theta={math.degrees(theta):.1f}deg -> "
                f"L={left:.2f} F={front:.2f} R={right:.2f}"
            )

        out = Float32MultiArray()
        out.data = [left, front, right]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = PointCloudHaptics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
