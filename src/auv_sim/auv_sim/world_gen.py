"""Generate the Ignition Fortress world SDF from an arena YAML.

The arena YAML is the single source of truth: the same file feeds the world geometry here and the
ground-truth object poses used by the simulated detector and the referee.

Usage:  python3 -m auv_sim.world_gen <arena.yaml> <out.sdf> [--camera] [--rtf 1.0]
"""
import argparse
import math

import yaml

# Vehicle constants (also used by the controller for thrust allocation).
HULL = (0.5, 0.3, 0.2)            # collision box (m) -> displaced volume 0.03 m^3 -> 30 kg neutral
HULL_MASS = 29.4                  # + 6 * 0.05 kg propellers = 29.7 kg -> ~3 N net positive buoyancy
PROP_MASS = 0.05
MAX_THRUST_N = 40.0
# name: (x, y, z, roll, pitch, yaw) ; joint axis is the propeller link's +x
THRUSTERS = {
    'surge_left':  (-0.25, 0.17, 0.0, 0.0, 0.0, 0.0),
    'surge_right': (-0.25, -0.17, 0.0, 0.0, 0.0, 0.0),
    'heave_front': (0.18, 0.0, 0.12, 0.0, -math.pi / 2, 0.0),
    'heave_back':  (-0.18, 0.0, 0.12, 0.0, -math.pi / 2, 0.0),
    'sway_front':  (0.18, 0.0, -0.12, 0.0, 0.0, math.pi / 2),
    'sway_back':   (-0.18, 0.0, -0.12, 0.0, 0.0, math.pi / 2),
}
BUOY_COLORS = {
    'red_buoy': '1 0 0 1', 'green_buoy': '0 0.9 0 1', 'yellow_buoy': '1 0.9 0 1',
    'purple_buoy': '0.6 0 0.8 1',
}


def load_arena(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _box_inertia(m, x, y, z):
    return (m / 12 * (y * y + z * z), m / 12 * (x * x + z * z), m / 12 * (x * x + y * y))


def _static_box(name, pose, size, color, collide=True):
    col = f"""<collision name="col"><geometry><box><size>{size}</size></box></geometry></collision>""" \
        if collide else ''
    return f"""
    <model name="{name}"><static>true</static><pose>{pose}</pose>
      <link name="link">{col}
        <visual name="vis"><geometry><box><size>{size}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual>
      </link></model>"""


def _pool(arena):
    p = arena['pool']
    L, W, D, x0 = p['length'], p['width'], p['depth'], p.get('x_min', -5.0)
    cx = x0 + L / 2
    t = 0.2
    out = [_static_box('pool_floor', f'{cx} 0 {-D - t / 2} 0 0 0', f'{L} {W} {t}', '0.3 0.4 0.5 1')]
    out.append(_static_box('wall_back', f'{x0 - t / 2} 0 {-D / 2} 0 0 0', f'{t} {W} {D}', '0.5 0.6 0.7 1'))
    out.append(_static_box('wall_front', f'{x0 + L + t / 2} 0 {-D / 2} 0 0 0', f'{t} {W} {D}', '0.5 0.6 0.7 1'))
    out.append(_static_box('wall_left', f'{cx} {W / 2 + t / 2} {-D / 2} 0 0 0', f'{L} {t} {D}', '0.5 0.6 0.7 1'))
    out.append(_static_box('wall_right', f'{cx} {-W / 2 - t / 2} {-D / 2} 0 0 0', f'{L} {t} {D}', '0.5 0.6 0.7 1'))
    return ''.join(out)


def _gate(o):
    w, h, r = o['width'], o['height'], 0.05
    links = []
    for side, y in (('left', w / 2), ('right', -w / 2)):
        links.append(f"""
      <link name="post_{side}"><pose>0 {y} 0 0 0 0</pose>
        <collision name="col"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry></collision>
        <visual name="vis"><geometry><cylinder><radius>{r}</radius><length>{h}</length></cylinder></geometry>
          <material><ambient>1 0.4 0 1</ambient><diffuse>1 0.4 0 1</diffuse></material></visual></link>""")
    links.append(f"""
      <link name="top_bar"><pose>0 0 {h / 2} 1.5708 0 0</pose>
        <collision name="col"><geometry><cylinder><radius>{r}</radius><length>{w}</length></cylinder></geometry></collision>
        <visual name="vis"><geometry><cylinder><radius>{r}</radius><length>{w}</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient><diffuse>0.1 0.1 0.1 1</diffuse></material></visual></link>""")
    return f"""
    <model name="{o['id']}"><static>true</static><pose>{o['x']} {o['y']} {o['z']} 0 0 {o.get('yaw', 0.0)}</pose>{''.join(links)}
    </model>"""


def _buoy(o):
    r = o.get('radius', 0.2)
    color = BUOY_COLORS.get(o['class'], '1 1 1 1')
    # Visual only: the referee decides "touched" from geometry, so the hull can reach the centre.
    return f"""
    <model name="{o['id']}"><static>true</static><pose>{o['x']} {o['y']} {o['z']} 0 0 0</pose>
      <link name="link"><visual name="vis"><geometry><sphere><radius>{r}</radius></sphere></geometry>
        <material><ambient>{color}</ambient><diffuse>{color}</diffuse><emissive>{color}</emissive></material></visual></link>
    </model>"""


def _auv(arena, camera):
    s = arena['start']
    ixx, iyy, izz = _box_inertia(HULL_MASS, *HULL)
    hx, hy, hz = HULL
    parts = []
    for name, (x, y, z, ro, pi, ya) in THRUSTERS.items():
        parts.append(f"""
      <link name="{name}_prop"><pose>{x} {y} {z} {ro} {pi} {ya}</pose>
        <inertial><mass>{PROP_MASS}</mass><inertia><ixx>1e-4</ixx><iyy>1e-4</iyy><izz>1e-4</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <visual name="vis"><pose>0 0 0 0 1.5708 0</pose><geometry><cylinder><radius>0.04</radius><length>0.05</length></cylinder></geometry>
          <material><ambient>0.1 0.1 0.1 1</ambient></material></visual></link>
      <joint name="{name}_joint" type="revolute"><parent>base_link</parent><child>{name}_prop</child>
        <axis><xyz>1 0 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis></joint>""")
    plugins = []
    for name in THRUSTERS:
        plugins.append(f"""
      <plugin filename="ignition-gazebo-thruster-system" name="ignition::gazebo::systems::Thruster">
        <namespace>auv</namespace><joint_name>{name}_joint</joint_name>
        <thrust_coefficient>0.004422</thrust_coefficient><fluid_density>1000</fluid_density>
        <propeller_diameter>0.1</propeller_diameter><velocity_control>true</velocity_control>
        <max_thrust_cmd>{MAX_THRUST_N}</max_thrust_cmd><min_thrust_cmd>{-MAX_THRUST_N}</min_thrust_cmd>
      </plugin>""")
    cam = ''
    if camera:
        cam = """
        <sensor name="front_camera" type="camera"><pose>0.26 0 0 0 0 0</pose>
          <always_on>1</always_on><update_rate>5</update_rate><topic>/auv/camera/image</topic>
          <camera><horizontal_fov>1.396</horizontal_fov>
            <image><width>640</width><height>480</height><format>R8G8B8</format></image>
            <clip><near>0.05</near><far>30</far></clip></camera></sensor>"""
    return f"""
    <model name="auv"><pose>{s['x']} {s['y']} {s['z']} 0 0 {s.get('yaw', 0.0)}</pose>
      <link name="base_link">
        <inertial><pose>0 0 -0.04 0 0 0</pose><mass>{HULL_MASS}</mass>
          <inertia><ixx>{ixx:.4f}</ixx><iyy>{iyy:.4f}</iyy><izz>{izz:.4f}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="hull"><geometry><box><size>{hx} {hy} {hz}</size></box></geometry></collision>
        <visual name="hull_visual"><geometry><box><size>{hx} {hy} {hz}</size></box></geometry>
          <material><ambient>0.9 0.9 0.9 1</ambient><diffuse>0.9 0.9 0.9 1</diffuse></material></visual>
        <sensor name="imu" type="imu"><always_on>1</always_on><update_rate>50</update_rate><topic>/auv/imu</topic></sensor>{cam}
      </link>{''.join(parts)}{''.join(plugins)}
      <plugin filename="ignition-gazebo-hydrodynamics-system" name="ignition::gazebo::systems::Hydrodynamics">
        <link_name>base_link</link_name>
        <xDotU>-3</xDotU><yDotV>-6</yDotV><zDotW>-6</zDotW><kDotP>-0.1</kDotP><mDotQ>-0.2</mDotQ><nDotR>-0.2</nDotR>
        <xUabsU>-30</xUabsU><xU>-10</xU><yVabsV>-45</yVabsV><yV>-15</yV><zWabsW>-45</zWabsW><zW>-15</zW>
        <kPabsP>-3</kPabsP><kP>-2</kP><mQabsQ>-5</mQabsQ><mQ>-3</mQ><nRabsR>-5</nRabsR><nR>-3</nR>
      </plugin>
      <plugin filename="ignition-gazebo-odometry-publisher-system" name="ignition::gazebo::systems::OdometryPublisher">
        <odom_frame>world</odom_frame><robot_base_frame>auv/base_link</robot_base_frame>
        <odom_publish_frequency>50</odom_publish_frequency><dimensions>3</dimensions>
        <odom_topic>/model/auv/odometry</odom_topic>
      </plugin>
    </model>"""


def _spectator(arena):
    """Static third-person camera on the pool side, looking across the course (media only)."""
    xs = [o['x'] for o in arena.get('objects', [])] or [8.0]
    cx = (min(xs) + max(xs)) / 2 - 1.0
    y = -arena['pool']['width'] / 2 + 1.5
    return f"""
    <model name="spectator"><static>true</static><pose>{cx:.1f} {y:.1f} 1.2 0 0.30 1.5708</pose>
      <link name="link"><sensor name="spectator_camera" type="camera">
        <always_on>1</always_on><update_rate>4</update_rate><topic>/auv/spectator/image</topic>
        <camera><horizontal_fov>1.75</horizontal_fov><image><width>800</width><height>450</height>
          <format>R8G8B8</format></image><clip><near>0.1</near><far>60</far></clip></camera>
      </sensor></link></model>"""


def generate_world(arena, camera=False, rtf=1.0, step=0.002, spectator=False):
    objects = []
    for o in arena.get('objects', []):
        objects.append(_gate(o) if o['class'] == 'gate' else _buoy(o))
    sensors = """
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine></plugin>""" if (camera or spectator) else ''
    return f"""<?xml version="1.0" ?>
<!-- GENERATED by auv_sim.world_gen from an arena YAML. Do not edit by hand. -->
<sdf version="1.8">
  <world name="{arena['world_name']}">
    <physics name="default" type="ode">
      <max_step_size>{step}</max_step_size><real_time_factor>{rtf}</real_time_factor>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-imu-system" name="ignition::gazebo::systems::Imu"/>{sensors}
    <plugin filename="ignition-gazebo-buoyancy-system" name="ignition::gazebo::systems::Buoyancy">
      <graded_buoyancy><default_density>1000</default_density>
        <density_change><above_depth>0</above_depth><density>1</density></density_change>
      </graded_buoyancy>
    </plugin>
    <scene><ambient>0.4 0.6 0.7 1</ambient><background>0.0 0.3 0.4 1</background></scene>
    <light type="directional" name="sun"><pose>0 0 10 0 0 0</pose><diffuse>1 1 1 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.1 -0.9</direction></light>
{_pool(arena)}{''.join(objects)}{_auv(arena, camera)}{_spectator(arena) if spectator else ''}
  </world>
</sdf>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arena')
    ap.add_argument('out')
    ap.add_argument('--camera', action='store_true')
    ap.add_argument('--rtf', type=float, default=1.0)
    a = ap.parse_args()
    with open(a.out, 'w') as f:
        f.write(generate_world(load_arena(a.arena), camera=a.camera, rtf=a.rtf))
    print(a.out)


if __name__ == '__main__':
    main()
