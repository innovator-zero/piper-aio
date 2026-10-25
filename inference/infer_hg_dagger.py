"""DAgger-enabled inference script for the Piper dual-arm robot.

Extends infer_sync.py with the ability to pause inference, switch the arms into
drag-teach mode, record human demonstrations, and resume inference without
restarting ROS nodes.

Keyboard controls:
  Space   Pause model inference and open the interactive menu
  d       Enter DAgger mode from the interactive menu
  c/r/q   Continue, reset, or quit from the interactive menu
  Enter   Start/resume DAgger collection
  Space   Pause DAgger collection
  c       Exit DAgger mode and return to the interactive menu

Data layout (matches collect_data output format):
  {save_dir}/episode_N.hdf5
"""

import argparse
import collections
import os
import signal
import sys
import termios
import threading
import time
import tty

import dm_env
import numpy as np
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from utils import (
    InferenceDataRecorder,
    check_keyboard_input,
    get_config,
    get_rollout_observation,
    handle_interactive_mode,
    inference_fn_sync,
    process_action,
    reset_observation_window,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ros_operator import RosOperator

# ──────────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────────

shutdown_event = threading.Event()


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# DAgger data collector
# ──────────────────────────────────────────────────────────────────────────────


class DaggerCollector:
    """Buffers DAgger frames before committing them to the current episode."""

    def __init__(self):
        self.is_collecting = False
        self.timesteps = []
        self.actions = []
        self.frame_count = 0
        print("DAgger collector ready")

    def start(self):
        if self.has_pending_data():
            self.is_collecting = True
            print("DAgger collection resumed")
            return
        self.is_collecting = True
        self.timesteps = []
        self.actions = []
        self.frame_count = 0
        print("DAgger collection started")

    def add_frame(self, observation, action):
        if not self.is_collecting:
            return
        self.frame_count += 1
        step_type = dm_env.StepType.FIRST if self.frame_count == 1 else dm_env.StepType.MID
        self.timesteps.append(dm_env.TimeStep(step_type=step_type, reward=None, discount=None, observation=observation))
        self.actions.append(np.asarray(action).copy())
        print(f"Collected DAgger frames: {self.frame_count}")

    def _reset_buffers(self):
        self.is_collecting = False
        self.timesteps = []
        self.actions = []
        self.frame_count = 0

    def commit_to(self, recorder):
        if not self.has_data():
            print("\033[31mNo DAgger data to commit.\033[0m")
            return False
        for timestep, action in zip(self.timesteps, self.actions):
            recorder.add_step(timestep.observation, action, collect_label="dagger")
        print(f"\033[32mCommitted DAgger frames: {len(self.actions)}\033[0m")
        self._reset_buffers()
        return True

    def stop(self):
        self.is_collecting = False

    def has_data(self):
        return len(self.actions) > 0

    def has_pending_data(self):
        return len(self.timesteps) > 0 or len(self.actions) > 0



# ──────────────────────────────────────────────────────────────────────────────
# Main inference + DAgger loop
# ──────────────────────────────────────────────────────────────────────────────


def _record_dagger_frame(config, ros_operator, collector):
    success, result = ros_operator.get_dagger_frame()
    if not success:
        return False

    (
        img_front,
        img_left,
        img_right,
        _,
        _,
        _,
        follower_left,
        follower_right,
        follower_left_pose,
        follower_right_pose,
        leader_left,
        leader_right,
    ) = result

    observation = collections.OrderedDict()
    observation["images"] = {
        config["camera_names"][0]: img_front,
        config["camera_names"][1]: img_left,
        config["camera_names"][2]: img_right,
    }
    observation["qpos"] = np.concatenate((np.array(follower_left.position), np.array(follower_right.position)))
    observation["qvel"] = np.concatenate((np.array(follower_left.velocity), np.array(follower_right.velocity)))
    observation["effort"] = np.concatenate((np.array(follower_left.effort), np.array(follower_right.effort)))
    observation["eef_pose"] = ros_operator.build_follower_arm_pose(
        follower_left_pose, follower_right_pose, follower_left, follower_right
    )
    action = np.concatenate((np.array(leader_left.position), np.array(leader_right.position)))
    collector.add_frame(observation, action)
    return True


def _restore_exit_state(ros_operator, left0, right0):
    try:
        if not rospy.is_shutdown():
            ros_operator.move_arms_to_initial_pose(left0, right0)
        ros_operator.set_leaders_drag_teach()
    except Exception as exc:
        try:
            rospy.logerr("Failed to restore DAgger exit state: %s", exc)
        except Exception:
            print(f"Failed to restore DAgger exit state: {exc}")


def _wait_after_reset():
    print("Reset complete. Press enter/c to continue, or q to quit")
    while not rospy.is_shutdown() and not shutdown_event.is_set():
        key = sys.stdin.read(1).lower()
        if key in ("\n", "\r", "c"):
            print("Continuing...")
            return "continue"
        if key == "q":
            print("Stopping...")
            return "quit"
    return "quit"


def run_dagger_session(args, config, ros_operator, collector, recorder):
    print("Entering DAgger mode...")
    if not ros_operator.enter_dagger_mode():
        print("Failed to enter DAgger mode; returning to interactive menu")
        return "menu"

    rate = rospy.Rate(args.publish_rate)
    print("\n" + "=" * 50)
    print("DAGGER MODE")
    print("  'enter' - Start/resume collection")
    print("  'space' - Pause collection")
    print("  'c' - Exit DAgger mode and return to interactive menu")
    print("=" * 50)

    try:
        while not rospy.is_shutdown() and not shutdown_event.is_set():
            ros_operator.forward_leader_to_follower()
            key = check_keyboard_input()

            if key in ("\n", "\r"):
                if collector.is_collecting:
                    print("Already collecting")
                else:
                    collector.start()
            elif key == " ":
                if collector.is_collecting:
                    collector.stop()
                    print("DAgger collection paused")
                    print("Press enter to resume, or 'c' to exit DAgger mode")
                else:
                    print("DAgger collection already paused")
            elif key and key.lower() == "c":
                collector.stop()
                if collector.has_pending_data():
                    collector.commit_to(recorder)
                print("Exiting DAgger mode; returning to interactive menu...")
                return "menu"

            if collector.is_collecting:
                _record_dagger_frame(config, ros_operator, collector)
            rate.sleep()
        return "shutdown"
    finally:
        collector.stop()
        ros_operator.exit_dagger_mode()


def model_inference(args, config, ros_operator):
    policy = OpenpiClient(
        host=args.host, port=args.port, image_size=args.image_size, prompt=config["language_instruction"]
    )

    left0, right0 = config["left0"], config["right0"]
    ros_operator.move_arms_to_initial_pose(left0, right0)

    print("Warmup the server...")
    policy.warmup()
    print("Server warmed up")
    input("Press enter to continue")
    task_time = time.time()

    recorder = InferenceDataRecorder(args, config, shutdown_event=shutdown_event)
    collector = DaggerCollector()

    print("\n" + "=" * 50)
    print("CONTROLS")
    print("  'space' - Open interactive mode")
    print("  'd' - Enter DAgger mode from interactive mode")
    print("  'c'/'r'/'q' - Continue, reset, or quit")
    print("  'enter' - Start/resume DAgger collection")
    print("  'space' - Pause DAgger collection")
    print("  'c' - Exit DAgger mode back to interactive mode")
    print("=" * 50)

    chunk_size = config["chunk_size"]
    max_publish_step = config["episode_len"]
    rate = rospy.Rate(args.publish_rate)

    try:
        while not rospy.is_shutdown() and not shutdown_event.is_set():
            t = 0
            reset_observation_window()
            action_buffer = np.zeros([chunk_size, config["state_dim"]])
            force_replan = True
            episode_closed = False

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                key = check_keyboard_input()
                if key == " ":
                    restart_episode = False
                    while True:
                        result = handle_interactive_mode(task_time, enable_dagger=True)
                        if result == "dagger":
                            dagger_result = run_dagger_session(args, config, ros_operator, collector, recorder)
                            if dagger_result == "shutdown":
                                return
                            reset_observation_window()
                            action_buffer = np.zeros([chunk_size, config["state_dim"]])
                            force_replan = True
                            if dagger_result == "menu":
                                continue
                        if result == "reset":
                            recorder.save_episode()
                            episode_closed = True
                            ros_operator.move_arms_to_initial_pose(left0, right0)
                            if _wait_after_reset() == "quit":
                                return
                            task_time = time.time()
                            restart_episode = True
                            break
                        if result == "quit":
                            recorder.save_episode()
                            return
                        break
                    if restart_episode:
                        break
                    continue

                if force_replan or t % chunk_size == 0:
                    action_buffer = inference_fn_sync(args, config, policy, ros_operator)
                    if action_buffer is None:
                        break
                    force_replan = False

                act = action_buffer[t % chunk_size]
                obs_to_save = get_rollout_observation(args, config, ros_operator) if recorder.enabled else None
                if recorder.enabled and obs_to_save is None:
                    break

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action))
                    ros_operator.follower_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action))
                    ros_operator.follower_arm_pose_publish(left_action, right_action)

                recorder.add_step(obs_to_save, action_to_save)
                t += 1
                print("Published Step", t)
                rate.sleep()

            if not episode_closed:
                recorder.save_episode()
            if shutdown_event.is_set():
                return

    finally:
        _restore_exit_state(ros_operator, left0, right0)


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────


def get_arguments():
    parser = argparse.ArgumentParser(description="Inference with DAgger data collection for Piper dual-arm robot")
    parser.add_argument("--max_publish_step", type=int, default=10000)
    parser.add_argument("--img_front_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--img_left_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--img_right_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--img_front_depth_topic", type=str, default="/camera_f/depth/image_raw")
    parser.add_argument("--img_left_depth_topic", type=str, default="/camera_l/depth/image_raw")
    parser.add_argument("--img_right_depth_topic", type=str, default="/camera_r/depth/image_raw")
    parser.add_argument("--leader_arm_left_topic", type=str, default="/leader/joint_left")
    parser.add_argument("--leader_arm_right_topic", type=str, default="/leader/joint_right")
    parser.add_argument("--follower_arm_left_topic", type=str, default="/follower/joint_left")
    parser.add_argument("--follower_arm_right_topic", type=str, default="/follower/joint_right")
    parser.add_argument("--follower_cmd_left_topic", type=str, default="/follower_cmd/joint_left")
    parser.add_argument("--follower_cmd_right_topic", type=str, default="/follower_cmd/joint_right")
    parser.add_argument("--pos_cmd_left_topic", type=str, default="/follower/pos_cmd_left")
    parser.add_argument("--pos_cmd_right_topic", type=str, default="/follower/pos_cmd_right")
    parser.add_argument("--follower_arm_left_pose_topic", type=str, default="/follower/end_pose_euler_left")
    parser.add_argument("--follower_arm_right_pose_topic", type=str, default="/follower/end_pose_euler_right")
    parser.add_argument("--publish_rate", type=int, default=30)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--arm_steps_length", type=float, nargs=7, default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2])
    parser.add_argument("--use_depth_image", action="store_true", default=False)
    parser.add_argument("--ctrl_type", choices=["joint", "eef"], default="joint")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--image_size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(224, 224))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model", choices=["openpi"], default="openpi")
    parser.add_argument("--fix_zero", type=str, choices=["left", "right", "none"], default="none")
    # Unified data recording
    parser.add_argument("--save_dir", type=str, required=True, help="Directory for unified rollout+DAgger HDF5 files")
    # DAgger-specific
    parser.add_argument("--teach_leader_enable_left_topic", type=str, default="/teach/leader_enable_left")
    parser.add_argument("--teach_leader_enable_right_topic", type=str, default="/teach/leader_enable_right")
    parser.add_argument("--teach_leader_config_left_topic", type=str, default="/teach/leader_config_left")
    parser.add_argument("--teach_leader_config_right_topic", type=str, default="/teach/leader_config_right")
    parser.add_argument("--teach_leader_mode_left_topic", type=str, default="/teach/teach_mode_left")
    parser.add_argument("--teach_leader_mode_right_topic", type=str, default="/teach/teach_mode_right")
    parser.add_argument("--leader_cmd_left_topic", type=str, default="/leader_cmd/joint_left")
    parser.add_argument("--leader_cmd_right_topic", type=str, default="/leader_cmd/joint_right")
    return parser.parse_args()


def main():
    args = get_arguments()
    args.save_rollout = True  # rollout and kept DAgger frames share one HDF5.
    config = get_config(args)

    ros_operator = RosOperator(args, mode="dagger")

    signal.signal(signal.SIGINT, _on_sigint)

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
