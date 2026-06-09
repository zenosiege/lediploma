import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Узел захвата видеопотока и генерации стохастических помех
        Node(
            package='sun_compass_camera_noise',
            executable='sun_compass_camera_noise_node',
            name='camera_noise_node',
            output='screen',
            parameters=[{
                'camera_id': 0,
                'frame_width': 640,
                'frame_height': 480,
                'enable_noise': True,
                'noise_intensity': 30.0
            }]
        ),

        # Узел пространственной фильтрации и морфологической сегментации
        Node(
            package='shadow_filter',
            executable='shadow_filter_node',
            name='shadow_filter_node',
            output='screen',
            parameters=[{
                'min_contour_area': 300.0,
                'max_contour_area': 50000.0,
                'radius_shrink_ratio': 0.82,
                'shadow_contrast': 6,
                'bg_blur_size': 81,
                'center_touch_tolerance': 0.3
            }]
        )
    ])
