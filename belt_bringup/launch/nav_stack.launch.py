
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    #fake data for testing 
    fake_esdf_node = Node(
        package='belt_core',
        executable='fake_pointcloud2D',
        name='fake_pointcloud2D',
        output='screen',
    )

    return LaunchDescription([
        fake_esdf_node
    ])