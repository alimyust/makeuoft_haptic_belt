
from rclpy.node import Node

import rclpy
from std_msgs.msg import String
import serial


# Node should write to all 3 motors in one message, so esp should parse the result.
class ESP_Node(Node):
    def __init__(self, esp_port, baud_rate=115200):
        super().__init__('esp_interface_node')
        self.esp_serial = serial.Serial(esp_port, baud_rate)
        self.haptic_command_subscriber = self.create_subscription(
            String,
            'haptic_command',
            self.haptic_callback,
            10
        )

    def haptic_callback(self, msg):
        self.esp_serial.write(f"/{msg.data}".encode())

def main(args=None):
    rclpy.init(args=args)
    esp_port = '/dev/ttyUSB0'  # Update this to your ESP's serial port
    esp_node = ESP_Node(esp_port)

    try:
        rclpy.spin(esp_node)
    except KeyboardInterrupt:
        pass
    finally:
        esp_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()