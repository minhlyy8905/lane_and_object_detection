from setuptools import setup
import os
from glob import glob

package_name = 'lane_and_object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('lane_and_object_detection/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rb',
    maintainer_email='rb@todo.todo',
    description='Node phát hiện vạch và vật thể kết hợp YOLOv8',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_and_object_detection_node = lane_and_object_detection.lane_and_object_detection_node:main',
        ],
    },
)