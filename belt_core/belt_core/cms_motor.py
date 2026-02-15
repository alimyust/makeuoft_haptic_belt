#!/usr/bin/env python3
"""
nvblox_ttc_haptics.py

Compute 3 haptic motor commands (Left / Center / Right) using:
- nvblox combined_map_slice (nvblox_msgs/msg/DistanceMapSlice) as a 2D ESDF slice
- velocity from Isaac ROS Visual SLAM odometry (/visual_slam/tracking/odometry)
- camera pose + yaw from TF using camera0_link as the “user forward” frame

Outputs:
- /haptics/cmd (std_msgs/UInt8MultiArray): [left, center, right] in 0..255

Assumptions (match Isaac ROS examples):
- You are running the default nvblox + visual_slam examples
- /nvblox_node/combined_map_slice publishes DistanceMapSlice
- /visual_slam/tracking/odometry publishes nav_msgs/Odometry
- TF tree includes camera0_link and the slice frame_id (usually "map")
"""

import math
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8MultiArray

from tf2_ros import Buffer, TransformListener, TransformException

# Isaac ROS nvblox
from nvblox_msgs.msg import DistanceMapSlice


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    """Yaw (Z axis rotation) from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class NvbloxTTCHaptics(Node):
    def __init__(self):
        super().__init__("nvblox_ttc_haptics")

        # -------- Parameters (tune these) --------
        self.declare_parameter("slice_topic", "/nvblox_node/combined_map_slice")
        self.declare_parameter("odom_topic", "/visual_slam/tracking/odometry")
        self.declare_parameter("camera_frame", "camera0_link")
        self.declare_parameter("publish_topic", "/haptics/cmd")

        # Direction setup (degrees relative to forward)
        self.declare_parameter("sector_angles_deg", [30.0, 0.0, -30.0])  # L, C, R
        self.declare_parameter("sector_spread_deg", 6.0)  # extra rays around each sector angle
        self.declare_parameter("rays_per_sector", 5)      # odd number recommended

        # ESDF / raycast tuning
        self.declare_parameter("surface_thresh_m", 0.20)  # ESDF <= this means “near obstacle surface”
        self.declare_parameter("max_range_m", 3.0)
        self.declare_parameter("step_cells", 1)           # step in grid cells (1 = each cell)

        # TTC mapping
        self.declare_parameter("min_speed_mps", 0.05)      # below this, treat as not moving
        self.declare_parameter("ttc_hi_s", 4.0)            # no buzz beyond this
        self.declare_parameter("ttc_lo_s", 0.7)            # max buzz at/below this

        # Output shaping
        self.declare_parameter("lowpass_alpha", 0.25)      # 0..1 (higher = more responsive, less smooth)
        self.declare_parameter("deadband", 0.05)           # intensities below this -> 0
        self.declare_parameter("unknown_is_free", True)    # True: skip unknown cells, False: treat unknown as obstacle

        # -------- State --------
        self._last_slice: Optional[DistanceMapSlice] = None
        self._last_odom: Optional[Odometry] = None
        self._prev_intensities: List[float] = [0.0, 0.0, 0.0]  # L, C, R float 0..1

        # -------- TF --------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- Pub/Sub --------
        slice_topic = self.get_parameter("slice_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        pub_topic = self.get_parameter("publish_topic").get_parameter_value().string_value

        self.slice_sub = self.create_subscription(DistanceMapSlice, slice_topic, self.on_slice, 10)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)

        self.pub = self.create_publisher(UInt8MultiArray, pub_topic, 10)

        # Drive publishing from a timer so we can smooth and handle asynchronous arrivals
        self.timer = self.create_timer(0.05, self.on_timer)  # 20 Hz

        self.get_logger().info(
            f"Started. slice_topic={slice_topic}, odom_topic={odom_topic}, camera_frame={self.get_parameter('camera_frame').value}, publish={pub_topic}"
        )

    def on_slice(self, msg: DistanceMapSlice):
        self._last_slice = msg

    def on_odom(self, msg: Odometry):
        self._last_odom = msg

    # ---------------- Core math helpers ----------------

    @staticmethod
    def grid_index(x: float, y: float, origin_x: float, origin_y: float, res: float) -> Tuple[int, int]:
        ix = int(math.floor((x - origin_x) / res))
        iy = int(math.floor((y - origin_y) / res))
        return ix, iy

    @staticmethod
    def in_bounds(ix: int, iy: int, w: int, h: int) -> bool:
        return 0 <= ix < w and 0 <= iy < h

    @staticmethod
    def linear_index(ix: int, iy: int, w: int) -> int:
        return iy * w + ix

    def ttc_to_intensity(self, ttc: float) -> float:
        """Map TTC to [0..1] intensity."""
        ttc_hi = float(self.get_parameter("ttc_hi_s").value)
        ttc_lo = float(self.get_parameter("ttc_lo_s").value)

        if not math.isfinite(ttc):
            return 0.0
        if ttc >= ttc_hi:
            return 0.0
        if ttc <= ttc_lo:
            return 1.0

        x = (ttc_hi - ttc) / (ttc_hi - ttc_lo)  # 0..1
        # Curve it to feel more urgent as it gets close
        return x * x

    def _get_camera_pose_in_slice_frame(self, slice_frame: str) -> Optional[Tuple[float, float, float]]:
        """
        Returns (x, y, yaw) of camera_frame expressed in slice_frame.
        """
        cam_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        try:
            # transform from slice_frame -> camera frame? We want camera pose in slice_frame.
            # lookup_transform(target, source): transform that takes data from source into target
            tf = self.tf_buffer.lookup_transform(
                target_frame=slice_frame,
                source_frame=cam_frame,
                time=rclpy.time.Time()
            )
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            q = tf.transform.rotation
            yaw = yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
            return tx, ty, yaw
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed ({slice_frame} <- {cam_frame}): {e}")
            return None

    def _get_forward_speed_mps(self) -> float:
        """
        Gets a usable forward speed estimate.
        Uses odometry twist. Assumes twist is in base/camera child frame as is common.
        If not, this is still often “good enough” for TTC feel; you can refine later.
        """
        if self._last_odom is None:
            return 0.0
        v = self._last_odom.twist.twist.linear
        # Common in robotics: twist.linear.x is forward speed in child frame
        return float(v.x)

    def raycast_distance_esdf(
        self,
        x0: float,
        y0: float,
        yaw: float,
        theta_rel: float,
        origin_x: float,
        origin_y: float,
        res: float,
        w: int,
        h: int,
        data: List[float],
        unknown_value: float
    ) -> float:
        """
        Raymarch along direction (yaw + theta_rel) until we find ESDF <= surface_thresh.
        Returns distance traveled (meters). inf if no hit.
        """
        surface_thresh = float(self.get_parameter("surface_thresh_m").value)
        max_range = float(self.get_parameter("max_range_m").value)
        step_cells = int(self.get_parameter("step_cells").value)
        unknown_is_free = bool(self.get_parameter("unknown_is_free").value)

        theta = yaw + theta_rel
        dx = math.cos(theta)
        dy = math.sin(theta)

        # Start cell
        ix0, iy0 = self.grid_index(x0, y0, origin_x, origin_y, res)
        if not self.in_bounds(ix0, iy0, w, h):
            return math.inf

        step_m = res * step_cells
        steps_max = int(max_range / step_m)

        for k in range(1, steps_max + 1):
            x = x0 + dx * (k * step_m)
            y = y0 + dy * (k * step_m)
            ix, iy = self.grid_index(x, y, origin_x, origin_y, res)
            if not self.in_bounds(ix, iy, w, h):
                return math.inf

            d = float(data[self.linear_index(ix, iy, w)])

            # Unknown handling (DistanceMapSlice has unknown_value)
            if math.isfinite(unknown_value) and abs(d - unknown_value) < 1e-6:
                if unknown_is_free:
                    continue
                else:
                    # Treat unknown as immediate risk at that range
                    return k * step_m

            # Near obstacle surface?
            if d <= surface_thresh:
                return k * step_m

        return math.inf

    def sector_distance(
        self,
        x0: float,
        y0: float,
        yaw: float,
        sector_center_rel: float,
        origin_x: float,
        origin_y: float,
        res: float,
        w: int,
        h: int,
        data: List[float],
        unknown_value: float
    ) -> float:
        """
        Cast multiple rays around a sector center and return the minimum hit distance.
        """
        rays_per_sector = int(self.get_parameter("rays_per_sector").value)
        rays_per_sector = max(1, rays_per_sector)
        spread_deg = float(self.get_parameter("sector_spread_deg").value)
        spread = math.radians(spread_deg)

        if rays_per_sector == 1:
            return self.raycast_distance_esdf(
                x0, y0, yaw, sector_center_rel,
                origin_x, origin_y, res, w, h, data, unknown_value
            )

        # Evenly distribute rays in [-spread, +spread] around center
        d_min = math.inf
        for i in range(rays_per_sector):
            t = 0.0 if rays_per_sector == 1 else (i / (rays_per_sector - 1))
            offset = -spread + 2.0 * spread * t
            d = self.raycast_distance_esdf(
                x0, y0, yaw, sector_center_rel + offset,
                origin_x, origin_y, res, w, h, data, unknown_value
            )
            if d < d_min:
                d_min = d
        return d_min

    # ---------------- Main loop ----------------

    def on_timer(self):
        if self._last_slice is None or self._last_odom is None:
            return

        s = self._last_slice
        slice_frame = s.header.frame_id

        pose = self._get_camera_pose_in_slice_frame(slice_frame)
        if pose is None:
            return
        x_cam, y_cam, yaw = pose

        # Slice metadata
        w = int(s.width)
        h = int(s.height)
        res = float(s.resolution)
        origin_x = float(s.origin.x)
        origin_y = float(s.origin.y)
        unknown_value = float(s.unknown_value)
        data = s.data  # float32[]

        # Get speed
        v_forward = self._get_forward_speed_mps()
        v_min = float(self.get_parameter("min_speed_mps").value)
        if abs(v_forward) < v_min:
            # not moving enough -> ramp down smoothly
            target = [0.0, 0.0, 0.0]
            self._publish_smoothed(target)
            return

        # Sector angles: L, C, R
        angs_deg = list(self.get_parameter("sector_angles_deg").value)
        if len(angs_deg) != 3:
            self.get_logger().error("sector_angles_deg must have 3 values [L,C,R].")
            return
        angs = [math.radians(float(a)) for a in angs_deg]

        # Compute directional distances and TTC
        intensities = []
        for theta_rel in angs:
            d_hit = self.sector_distance(
                x_cam, y_cam, yaw, theta_rel,
                origin_x, origin_y, res, w, h, data, unknown_value
            )

            # Closing speed along this sector:
            # simplest: assume forward velocity dominates and scale by cos(theta_rel)
            v_sector = max(0.0, v_forward * math.cos(theta_rel))
            if v_sector < v_min:
                ttc = math.inf
            else:
                ttc = d_hit / v_sector

            I = self.ttc_to_intensity(ttc)
            intensities.append(I)

        self._publish_smoothed(intensities)

    def _publish_smoothed(self, target_intensities: List[float]):
        """Low-pass + deadband + publish 0..255."""
        alpha = float(self.get_parameter("lowpass_alpha").value)
        alpha = clamp(alpha, 0.0, 1.0)
        deadband = float(self.get_parameter("deadband").value)

        out = []
        for i in range(3):
            prev = self._prev_intensities[i]
            tgt = clamp(target_intensities[i], 0.0, 1.0)
            sm = (1.0 - alpha) * prev + alpha * tgt
            if sm < deadband:
                sm = 0.0
            self._prev_intensities[i] = sm
            out.append(int(round(255.0 * sm)))

        msg = UInt8MultiArray()
        msg.data = out  # [L, C, R]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = NvbloxTTCHaptics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
