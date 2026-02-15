from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    urdf_path = PathJoinSubstitution([
        FindPackageShare("belt_bringup"),  
        "description",
        "base.urdf"
    ])

    robot_description = Command(["xacro ", urdf_path])
    rviz_config_path = PathJoinSubstitution([
        FindPackageShare("belt_bringup"),
        "rviz2",
        "belt_sim.rviz"
    ])
    
    return LaunchDescription([
        Node(
            package="belt_core",
            executable="cms_motor",
            name="cms_motor",
            output="screen",
        ),
        # Node(
        #     package="rviz2",
        #     executable="rviz2",
        #     name="rviz2",
        #     output="screen",
        #     arguments=["-d", rviz_config_path],
        # ),
    ])
