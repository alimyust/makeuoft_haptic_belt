import rclpy
from rclpy.node import Node
from nav2_msgs.msg import Costmap
from std_msgs.msg import String
from geometry_msgs import PoseStamped
from nav_msgs.msg import OccupancyGrid

import numpy as np

class Nav2CostmapToMotor(Node):
    def __init__(self):
        super().__init__('nav2_costmap_to_motor')

        self.costmap_subscription = self.create_subscription(
            OccupancyGrid,
            'local_costmap/costmap',
            self.costmap_callback,
            10
        )

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            'odom',
            self.pose_callback,
            10
        )
        self.motor_publisher = self.create_publisher(String, 'esp_commands', 10)

        self.robot_pose = None
        self.map_info = None


    def costmap_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        data = msg.data
        self.map_info = msg.info
        robot_x, robot_y = self.robot_pose.position.x, self.robot_pose.position.y

        grid = np.array(data).reshape((height, width))

        robot_index_pose = self.world_to_grid(robot_x, robot_y)
        front_motor_mag = grid[robot_index_pose[0], robot_index_pose[1]]
        left_motor_mag = grid[robot_index_pose[0], max(0, robot_index_pose[1] - 3)]
        right_motor_mag = grid[robot_index_pose[0], min(width - 1, robot_index_pose[1] + 3)]

        motor_concatenated = f"{front_motor_mag},{left_motor_mag},{right_motor_mag}"
        self.get_logger().info(f"Publishing motor command: {motor_concatenated}")
        self.motor_publisher.publish(String(data=motor_concatenated))


    def world_to_grid(self, x, y):
        col = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        row = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return row, col

    def grid_to_world(self, row, col):
        x = col * self.map_info.resolution + self.map_info.origin.position.x
        y = row * self.map_info.resolution + self.map_info.origin.position.y
        return x, y

    def pose_callback(self, msg: PoseStamped):
        self.robot_pose = msg.pose

def main(args=None):
    rclpy.init(args=args)
    node = Nav2CostmapToMotor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
