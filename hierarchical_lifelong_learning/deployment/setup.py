from setuptools import setup
from glob import glob
import os

package_name = 'hierarchical_learning'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='create',
    maintainer_email='create@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'create_topomap_node = hierarchical_learning.create_topomap:main', 
            'low_level_policy = hierarchical_learning.low_level_policy:main', 
            'pd_controller_node = hierarchical_learning.pd_controller:main'
        ],
    },
)
