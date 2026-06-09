from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sun_shadow_detection',
            executable='sun_shadow_detection_node',
            name='sun_shadow_node',
            parameters=[
                {'white_thresh': 200},
                {'black_thresh': 50},
                {'edge_margin': 0.9},
                {'latitude': 54.6269},
                {'longitude': 39.7145}
            ],
            output='screen'
        )
    ])
