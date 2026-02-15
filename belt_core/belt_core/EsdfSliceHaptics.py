#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import Float32

from nvblox_msgs.msg import DistanceMapSlice

from tf2_ros import Buffer, TransformListener, TransformException


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def quat_conj(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_rotate(q: Tuple[float, float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    # v' = q * (v,0) * q_conj
    vx, vy, vz = v
    vq = (vx, vy, vz, 0.0)
    qc = quat_conj(q)
    return quat_mul(quat_mul(q, vq), qc)[:3]


class EsdfSliceHaptics(Node):
    """
    Subscribe to nvblox DistanceMapSlice and output a single haptic magnitude (0..1).

    We compute the distance straight ahead of the camera by sphere-tracing along the
    camera-forward direction on the ESDF slice (2D).
    """

    def __init__(self):
        super().__init__("esdf_slice_haptics")

        # --- Params ---
        self.declare_parameter("slice_topic", "/nvblox_node/combined_map_slice")
        self.declare_parameter("motor_topic", "/haptics/motor")

        # camera frame to define "straight ahead"
        self.declare_parameter("camera_frame", "camera_link")

        # If your slice frame is stable (e.g., "map"), you can force it.
        # If empty, we use msg.header.frame_id.
        self.declare_parameter("slice_frame_override", "")

        # Map distance (meters) -> vibration magnitude:
        # hit_dist <= d_min => 1.0
        # hit_dist >= d_max => 0.0
        self.declare_parameter("d_min", 0.40)
        self.declare_parameter("d_max", 3.00)
        self.declare_parameter("gamma", 1.6)

        # Sphere tracing controls
        self.declare_parameter("max_range", 6.0)      # meters
        self.declare_parameter("max_iters", 60)
        self.declare_parameter("hit_epsilon", 0.05)   # meters
        self.declare_parameter("min_step", 0.05)      # meters

        # Message coordinate convention:
        # DistanceMapSlice.msg says origin is "upper-left corner". :contentReference[oaicite:2]{index=2}
        # In typical map axes (x right, y up), moving "down rows" means y decreases.
        self.declare_parameter("origin_is_upper_left", True)

        # --- Read params ---
        self.slice_topic = str(self.get_parameter("slice_topic").value)
        self.motor_topic = str(self.get_parameter("motor_topic").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.slice_frame_override = str(self.get_parameter("slice_frame_override").value)

        self.d_min = float(self.get_parameter("d_min").value)
        self.d_max = float(self.get_parameter("d_max").value)
        self.gamma = float(self.get_parameter("gamma").value)

        self.max_range = float(self.get_parameter("max_range").value)
        self.max_iters = int(self.get_parameter("max_iters").value)
        self.hit_epsilon = float(self.get_parameter("hit_epsilon").value)
        self.min_step = float(self.get_parameter("min_step").value)

        self.origin_is_upper_left = bool(self.get_parameter("origin_is_upper_left").value)

        if self.d_max <= self.d_min:
            self.get_logger().warn("d_max must be > d_min. Forcing d_max = d_min + 1.0")
            self.d_max = self.d_min + 1.0

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- State ---
        self.slice_msg: Optional[DistanceMapSlice] = None

        # --- ROS I/O ---
        self.sub = self.create_subscription(DistanceMapSlice, self.slice_topic, self.on_slice, 10)
        self.pub = self.create_publisher(Float32, self.motor_topic, 10)

        self.timer = self.create_timer(0.05, self.on_timer)  # 20 Hz motor updates

        self.get_logger().info(f"Sub: {self.slice_topic} | Pub: {self.motor_topic} | camera_frame={self.camera_frame}")

    def on_slice(self, msg: DistanceMapSlice) -> None:
        self.slice_msg = msg

    def dist_to_mag(self, hit_dist: float) -> float:
        if hit_dist <= self.d_min:
            return 1.0
        if hit_dist >= self.d_max:
            return 0.0
        x = (self.d_max - hit_dist) / (self.d_max - self.d_min)
        x = clamp(x, 0.0, 1.0)
        return x ** self.gamma

    def sample_esdf(self, msg: DistanceMapSlice, x: float, y: float) -> Optional[float]:
        """
        Sample DistanceMapSlice at world (x,y) in msg.header.frame_id coordinates.

        origin is the upper-left corner of pixel (0,0). :contentReference[oaicite:3]{index=3}
        width along +x, height along +y axis (but "upper-left" implies row+ moves down).
        """
        res = float(msg.resolution)
        if res <= 0.0:
            return None

        ox = float(msg.origin.x)
        oy = float(msg.origin.y)

        # Convert world -> pixel indices
        px = (x - ox) / res

        if self.origin_is_upper_left:
            py = (oy - y) / res  # down rows => y decreases
        else:
            py = (y - oy) / res  # down rows => y increases

        ix = int(math.floor(px))
        iy = int(math.floor(py))

        if ix < 0 or iy < 0 or ix >= int(msg.width) or iy >= int(msg.height):
            return None

        idx = iy * int(msg.width) + ix
        d = float(msg.data[idx])

        # Unknown cells use unknown_value sentinel :contentReference[oaicite:4]{index=4}
        if math.isclose(d, float(msg.unknown_value), rel_tol=0.0, abs_tol=1e-6):
            return None
        if not math.isfinite(d):
            return None

        return d

    def sphere_trace_forward(self, msg: DistanceMapSlice, start_xy: Tuple[float, float], dir_xy: Tuple[float, float]) -> Optional[float]:
        """
        2D sphere tracing along a ray using the ESDF:
          step by d(x) (distance to nearest surface) which is guaranteed safe.
        Returns distance along ray to hit (approx), or None if no hit within max_range.
        """
        sx, sy = start_xy
        dx, dy = dir_xy

        # Normalize direction
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return None
        dx /= n
        dy /= n

        t = 0.0
        for _ in range(self.max_iters):
            x = sx + t * dx
            y = sy + t * dy

            d = self.sample_esdf(msg, x, y)
            if d is None:
                # unknown: creep forward slowly
                t += self.min_step
            else:
                if d <= self.hit_epsilon:
                    return t
                t += max(d, self.min_step)

            if t > self.max_range:
                return None

        return None

    def on_timer(self) -> None:
        msg = self.slice_msg
        out = Float32()
        out.data = 0.0

        if msg is None:
            self.pub.publish(out)
            return

        slice_frame = self.slice_frame_override if self.slice_frame_override else msg.header.frame_id
        if not slice_frame:
            self.pub.publish(out)
            return

        # Use slice timestamp for TF lookup
        stamp = Time.from_msg(msg.header.stamp)

        try:
            tf = self.tf_buffer.lookup_transform(slice_frame, self.camera_frame, stamp)
        except TransformException as e:
            # If TF not ready, just output 0
            self.get_logger().warn_throttle(2000, f"TF lookup failed ({slice_frame} <- {self.camera_frame}): {e}")
            self.pub.publish(out)
            return

        # Camera position in slice frame
        cx = float(tf.transform.translation.x)
        cy = float(tf.transform.translation.y)

        # Camera forward vector (assume +X forward in camera_frame)
        q = (
            float(tf.transform.rotation.x),
            float(tf.transform.rotation.y),
            float(tf.transform.rotation.z),
            float(tf.transform.rotation.w),
        )
        fwd = quat_rotate(q, (1.0, 0.0, 0.0))  # expressed in slice frame
        dir_xy = (fwd[0], fwd[1])

        hit_dist = self.sphere_trace_forward(msg, (cx, cy), dir_xy)

        if hit_dist is None:
            out.data = 0.0
        else:
            out.data = float(self.dist_to_mag(hit_dist))

        self.pub.publish(out)


def main():
    rclpy.init()
    node = EsdfSliceHaptics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
