from setuptools import setup
import os
from glob import glob

package_name = 'simple_robot_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Control nodes for simple robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_subscriber = simple_robot_control.robot_subscriber:main',
            'ground_truth_node = simple_robot_control.ground_truth_node:main',
            'mapper_node = simple_robot_control.mapper_node:main',
            'ekf_node = simple_robot_control.extended_kalman_filter:main',
            'lidar_odom = simple_robot_control.lidar_odom:main',
            'frontier_explorer = simple_robot_control.frontier_explorer:main',
            'pure_pursuit = simple_robot_control.pure_pursuit:main',
            # 'vizualize_matplotlib = simple_robot_control.vizualize_matplotlib:main'
            'vizualize = simple_robot_control.vizualize:main'
        ],
    },
)