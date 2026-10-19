"""Headless stand-in sim: ign gazebo server + ros_gz bridge + AUV sim nodes.

ros2 launch auv_sim sim.launch.py arena:=default camera:=false rtf:=1.0 \
    detection_source:=ground_truth battery_start:=100.0 stack:=full
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from auv_sim.arena import load_arena
from auv_sim.world_gen import THRUSTERS, generate_world


def _setup(context):
    arena_name = LaunchConfiguration('arena').perform(context)
    camera = LaunchConfiguration('camera').perform(context).lower() == 'true'
    spectator = LaunchConfiguration('spectator').perform(context).lower() == 'true'
    rtf = float(LaunchConfiguration('rtf').perform(context))
    stack = LaunchConfiguration('stack').perform(context)
    det_src = LaunchConfiguration('detection_source').perform(context)
    battery_start = float(LaunchConfiguration('battery_start').perform(context))
    report_file = LaunchConfiguration('report_file').perform(context)
    blind = LaunchConfiguration('blind_classes').perform(context)
    verbose = LaunchConfiguration('gz_verbose').perform(context)

    arena = load_arena(arena_name)
    out_dir = os.path.join(os.environ.get('ROS_HOME', os.path.expanduser('~/.ros')), 'auv_sim')
    os.makedirs(out_dir, exist_ok=True)
    world_file = os.path.join(out_dir, f'{os.path.basename(arena_name).replace(".yaml", "")}'
                                       f'{"_cam" if camera else ""}{"_spec" if spectator else ""}.sdf')
    with open(world_file, 'w') as f:
        f.write(generate_world(arena, camera=camera, rtf=rtf, spectator=spectator))

    gz_cmd = ['ign', 'gazebo', '-s', '-r', '-v', verbose]
    if camera or spectator:
        gz_cmd.append('--headless-rendering')
    gz_cmd.append(world_file)
    gz = ExecuteProcess(cmd=gz_cmd, output='screen', name='gz_server')

    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        '/model/auv/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
        '/auv/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
    ]
    remaps = [('/model/auv/odometry', '/auv/odom')]
    for n in THRUSTERS:
        gz_topic = f'/model/auv/joint/{n}_joint/cmd_thrust'
        bridge_args.append(f'{gz_topic}@std_msgs/msg/Float64]ignition.msgs.Double')
        remaps.append((gz_topic, f'/auv/thrusters/{n}'))
    if camera:
        bridge_args.append('/auv/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image')
    if spectator:
        bridge_args.append('/auv/spectator/image@sensor_msgs/msg/Image[ignition.msgs.Image')
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', name='auv_bridge',
                  arguments=bridge_args, remappings=remaps, output='screen',
                  parameters=[{'use_sim_time': True}])

    from ament_index_python.packages import get_package_share_directory
    params = os.path.join(get_package_share_directory('auv_sim'), 'config', 'sim.yaml')
    arena_file = arena_name if os.path.isfile(arena_name) else \
        os.path.join(get_package_share_directory('auv_sim'), 'config', 'arenas', f'{arena_name}.yaml')
    common = [params, {'use_sim_time': True}]

    def node(exe, name, extra=None, remap=None):
        return Node(package='auv_sim', executable=exe, name=name, output='screen',
                    parameters=common + ([extra] if extra else []), remappings=remap or [])

    actions = [gz, bridge,
               node('controller_node', 'auv_controller',
                    remap=[('odom', '/auv/odom'), ('setpoint', '/auv/controller/setpoint'),
                           ('mode', '/auv/controller/mode')])]
    if stack == 'full':
        actions += [
            node('vehicle_server', 'auv_vehicle_server', {'arena_file': arena_file}),
            node('battery_sim', 'auv_battery', {'initial_percent': battery_start}),
            node('detector_node', 'auv_detector',
                 dict({'arena_file': arena_file, 'camera_enabled': camera, 'blind_classes': blind},
                      **({'detection_source': det_src} if det_src else {}))),
            node('referee_node', 'auv_referee', {'arena_file': arena_file, 'report_file': report_file}),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('arena', default_value='default', description='arena name or yaml path'),
        DeclareLaunchArgument('camera', default_value='false', description='render the front camera'),
        DeclareLaunchArgument('spectator', default_value='false',
                              description='third-person camera for media (/auv/spectator/image)'),
        DeclareLaunchArgument('rtf', default_value='1.0', description='real-time factor (0 = max)'),
        DeclareLaunchArgument('stack', default_value='full', description='minimal | full'),
        DeclareLaunchArgument('detection_source', default_value='',
                              description='ground_truth | camera (empty = config/sim.yaml)'),
        DeclareLaunchArgument('report_file', default_value='', description='referee JSON output'),
        DeclareLaunchArgument('blind_classes', default_value='',
                              description='comma-separated classes the detector never reports'),
        DeclareLaunchArgument('battery_start', default_value='100.0'),
        DeclareLaunchArgument('gz_verbose', default_value='1'),
        OpaqueFunction(function=_setup),
    ])
