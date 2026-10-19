"""ROS 2 mission service: other nodes submit plain-language missions.

  ros2 run auv_mission_planner mission_server
  ros2 service call /auv/submit_mission auv_mission_interfaces/srv/SubmitMission \
      "{prompt: 'pass through the gate then surface', provider: '', dry_run: false}"

Planning + validation happen inside the service call, so the response carries the validated plan
(or the refusal / rejection reason). Execution then runs in a worker thread; progress is published
on the status topic (auv_mission_interfaces/MissionStatus). One mission at a time.
"""
from __future__ import annotations

import json
import os
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter

from auv_mission_interfaces.msg import MissionStatus
from auv_mission_interfaces.srv import SubmitMission
from auv_mission_planner.config import find_workspace_root, load_config, load_dotenv, load_llm_config
from auv_mission_planner.llm.providers import make_provider
from auv_mission_planner.mission import MissionRunner
from auv_mission_planner.tools.registry import ToolRegistry
from auv_mission_planner.tools.ros_backend import RosBackend
from auv_mission_planner.tracing import new_mission_id


class MissionServer(Node):
    def __init__(self):
        super().__init__('auv_mission_server', parameter_overrides=[Parameter('use_sim_time', value=True)])
        load_dotenv()
        self.cfg = load_config(self.declare_parameter('config', '').value or None)
        self.llm_cfg = load_llm_config(self.declare_parameter('llm_config', '').value or None)
        self.timeout = self.declare_parameter('mission_timeout_s', 900.0).value
        self.trace_dir = self.declare_parameter('trace_dir', '').value or \
            os.path.join(find_workspace_root() or '.', 'traces')
        self.backend = RosBackend(self.cfg, node=self)
        self.tools = ToolRegistry(self.backend, self.cfg)
        cg = ReentrantCallbackGroup()
        mi = self.cfg['mission_interfaces']
        self.status_pub = self.create_publisher(MissionStatus, mi['status_topic'], 10)
        self.create_service(SubmitMission, mi['submit_service'], self.on_submit, callback_group=cg)
        self.busy = threading.Lock()
        self.get_logger().info(f'mission service ready on {mi["submit_service"]}')

    def status(self, mission_id, state, step, replans, detail):
        m = MissionStatus()
        m.mission_id, m.state, m.step_index, m.replans, m.detail = mission_id, state, int(step), int(replans), \
            str(detail)[:500]
        self.status_pub.publish(m)

    def on_submit(self, req, resp):
        if not self.busy.acquire(blocking=False):
            resp.accepted, resp.message = False, 'another mission is running'
            return resp
        if not self.backend.wait_ready(15.0):      # pose, battery and vehicle actions must be live first
            self.busy.release()
            resp.accepted, resp.message = False, 'vehicle not ready (no pose/battery/actions yet)'
            return resp
        mission_id = new_mission_id()
        try:
            provider = make_provider(self.llm_cfg, req.provider or None)
        except ValueError as e:
            self.busy.release()
            resp.accepted, resp.message = False, str(e)
            return resp
        runner = MissionRunner(self.cfg, self.llm_cfg, provider, self.tools, trace_dir=self.trace_dir,
                               on_status=lambda *a: self.status(mission_id, *a),
                               log=lambda m: self.get_logger().info(m))
        # plan + validate synchronously
        dry = runner.run(req.prompt, dry_run=True, mission_id=mission_id + '_plan')
        resp.mission_id = mission_id
        resp.plan_json = json.dumps(dry['final_plan']) if dry['final_plan'] else ''
        if dry['outcome'] != 'planned' or req.dry_run:
            self.busy.release()
            resp.accepted = dry['outcome'] == 'planned'
            resp.message = dry['outcome'] + (f': {dry["reason"]}' if dry['reason'] else '')
            return resp

        def work():
            try:
                runner.run(req.prompt, wall_timeout_s=self.timeout, mission_id=mission_id)
            except Exception as e:      # never leave the vehicle mid-mission silently
                self.get_logger().error(f'mission crashed: {e}; emergency surface')
                self.tools.call('abort', {'reason': f'mission crashed: {e}'})
                self.status(mission_id, 'failed', 0, 0, str(e))
            finally:
                self.busy.release()
        threading.Thread(target=work, daemon=True).start()
        resp.accepted, resp.message = True, 'executing (status on ' + self.cfg['mission_interfaces']['status_topic'] + ')'
        return resp


def main():
    from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
    rclpy.init()
    node = MissionServer()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
