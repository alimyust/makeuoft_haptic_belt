from setuptools import find_packages, setup

package_name = 'belt_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alimyust',
    maintainer_email='alimyust@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'esp_interface = belt_core.esp_interface:main',
            'fake_pointcloud2D = belt_core.fake_pointcloud2D:main',
            'costmap_to_motor = belt_core.costmap_to_motor:main',
            'fake_costmap = belt_core.fake_costmap:main',
            'EsdfSliceHaptics = belt_core.EsdfSliceHaptics:main',
            
        ],
    },
)
