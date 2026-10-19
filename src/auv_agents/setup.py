from glob import glob

from setuptools import find_packages, setup

package_name = 'auv_agents'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/arenas', glob('config/arenas/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yaad Kamrul Bari',
    maintainer_email='yaadbari21@gmail.com',
    description='LangGraph multi-agent autonomy: supervisor, perception, navigation, health and diagnostics agents, human-in-the-loop approval, MCP tool server, telemetry.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
