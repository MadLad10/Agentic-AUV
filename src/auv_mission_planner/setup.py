from glob import glob

from setuptools import setup

package_name = 'auv_mission_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.tools', package_name + '.llm', package_name + '.backends'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yaad Kamrul Bari',
    maintainer_email='yaadbari21@gmail.com',
    description='LLM mission planner for the AUV',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mission = auv_mission_planner.cli:main',
            'mission_server = auv_mission_planner.mission_server_node:main',
            'tool_demo = auv_mission_planner.tool_demo:main',
            'auv_adapter = auv_mission_planner.backends.auv_adapter:main',
        ],
    },
)
