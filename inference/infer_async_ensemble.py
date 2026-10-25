import argparse
import os
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque

import numpy as np
import rospy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clients import OpenpiClient
from utils import (
    InferenceDataRecorder,
    apply_fixed_arms_to_initial_pose,
    check_keyboard_input,
    get_config,
    get_inference_observation,
    get_rollout_observation,
    handle_interactive_mode,
    process_action,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ros_operator import RosOperator

observation_window = None
observation_window_lock = threading.Lock()

shutdown_event = threading.Event()
# When clear: inference thread blocks. When set: inference thread runs.
inference_paused = threading.Event()
inference_paused.clear()  # paused by default

inference_stamp = 0  # the stamp of the current inference
episode_id = 0
inference_state_lock = threading.Lock()
inference_state_cond = threading.Condition(inference_state_lock)
inflight_inference_count = 0


def _on_sigint(signum, frame):
    try:
        shutdown_event.set()
    except Exception:
        pass
    try:
        rospy.signal_shutdown("SIGINT")
    except Exception:
        pass


def reset_observation_window():
    """Reset observation window for a new episode."""
    global observation_window
    with observation_window_lock:
        observation_window = None


def begin_new_episode(wait_timeout=None):
    """Invalidate in-flight async inference and optionally wait for it to finish."""
    global episode_id
    inference_paused.clear()
    with inference_state_cond:
        episode_id += 1
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        while inflight_inference_count > 0 and not rospy.is_shutdown():
            if deadline is None:
                inference_state_cond.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                rospy.logwarn("Timed out waiting for in-flight inference; stale results will be discarded")
                break
            inference_state_cond.wait(timeout=remaining)
        return episode_id


def is_current_episode(request_episode_id):
    with inference_state_cond:
        return request_episode_id == episode_id


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = deque(maxlen=2)

            # Append the first dummy image
            observation_window.append(
                {
                    "qpos": None,
                    "images": {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
                    "eef_pose": None,
                }
            )

    observation = get_inference_observation(args, config, ros_operator)
    if observation is None:
        return False

    with observation_window_lock:
        observation_window.append(observation)

    return True


def build_policy_payload(args, config):
    with observation_window_lock:
        observation = observation_window[-1]
        image_arrs = [
            observation["images"][config["camera_names"][0]],
            observation["images"][config["camera_names"][1]],
            observation["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint":
            state = observation["qpos"]
        elif args.ctrl_type == "eef":
            state = observation["eef_pose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    return {
        "top": image_arrs[0],
        "left": image_arrs[1],
        "right": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state,
    }


class StreamActionBuffer:
    def __init__(self, delay, exec_horizon, min_smooth_steps=8):
        self.delay = delay
        self.exec_horizon = exec_horizon
        self.min_smooth_steps = min_smooth_steps
        self.lock = threading.Lock()
        self.cur_chunk = deque()
        self.k = 0
        self.epoch = 0
        self.last_launch_epoch = -1
        self.last_launch_k = 0
        self.last_action = None

    def reset(self):
        """Reset buffer state for a new episode."""
        with self.lock:
            self.cur_chunk.clear()
            self.k = 0
            self.epoch = 0
            self.last_launch_epoch = -1
            self.last_launch_k = 0
            self.last_action = None

    def mark_launch_if_ready(self):
        with self.lock:
            if self.last_launch_epoch == self.epoch:
                return False

            launch_step = max(0, self.exec_horizon - self.delay - 1)
            if self.k < launch_step:
                return False

            self.last_launch_epoch = self.epoch
            self.last_launch_k = self.k
            return True

    def integrate_first_chunk(self, actions_chunk: np.ndarray):
        with self.lock:
            self.cur_chunk = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.k = 0
            self.epoch = 0
            self.last_launch_epoch = -1
            self.last_launch_k = 0

    def integrate_new_chunk(self, actions_chunk: np.ndarray):
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or len(actions_chunk) == 0 when integrating new chunk")
            return

        with self.lock:
            drop_n = max(0, self.k - self.last_launch_k)
            if drop_n >= len(actions_chunk):
                rospy.logwarn(
                    f"Async result is too late to use: drop_n={drop_n}, actions_len={len(actions_chunk)}"
                )
                return

            new_list = [a.copy() for a in actions_chunk[drop_n:]]

            if len(self.cur_chunk) == 0:
                if self.last_action is not None:
                    old_list = [
                        np.asarray(self.last_action, dtype=float).copy() for _ in range(self.min_smooth_steps)
                    ]
                    self.last_action = None
                else:
                    self.cur_chunk = deque(new_list, maxlen=None)
                    self.k = 0
                    self.epoch += 1
                    return
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) < self.min_smooth_steps:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(self.min_smooth_steps - len(old_list))])

            overlap_len = min(len(old_list), len(new_list))
            if len(old_list) > len(new_list):
                old_list = old_list[: len(new_list)]

            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                w_old = np.linspace(1.0, 0.0, overlap_len, dtype=float)
            w_new = 1.0 - w_old

            smoothed = [
                (w_old[i] * np.asarray(old_list[i], dtype=float) + w_new[i] * np.asarray(new_list[i], dtype=float))
                for i in range(overlap_len)
            ]
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0
            self.epoch += 1

    def get_next_action(self):
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None

            action = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.last_action = action.copy()
            self.k += 1
            return action


def inference_fn_sync(args, config, policy, ros_operator):
    global inference_stamp

    if not update_observation_window(args, config, ros_operator):
        return None

    start_time = time.perf_counter()
    actions = policy.predict_action(build_policy_payload(args, config))
    actions = apply_fixed_arms_to_initial_pose(actions, config, target="action")
    print(f"[Sync   {inference_stamp:2d}] Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms")
    inference_stamp += 1

    return actions


def inference_fn_async(args, config, policy, ros_operator, action_buffer):
    global inference_stamp, inflight_inference_count

    while not rospy.is_shutdown():
        try:
            inference_paused.wait()
            if rospy.is_shutdown():
                break

            with inference_state_cond:
                request_episode_id = episode_id
                inflight_inference_count += 1

            print(f"[Async  {inference_stamp:2d}] Start inference")

            if not update_observation_window(args, config, ros_operator):
                inference_paused.clear()
                continue

            start_time = time.perf_counter()
            actions = policy.predict_action(build_policy_payload(args, config))
            actions = apply_fixed_arms_to_initial_pose(actions, config, target="action")
            print(
                f"[Async  {inference_stamp:2d}] Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms"
            )

            if not is_current_episode(request_episode_id):
                rospy.logwarn(
                    f"Discard async inference result from stale episode {request_episode_id}; current={episode_id}"
                )
            elif actions is not None and len(actions) > 0:
                action_buffer.integrate_new_chunk(actions[: config["chunk_size"]])
            else:
                print("actions is None or len(actions) == 0")

            inference_stamp += 1
            inference_paused.clear()

        except Exception as e:
            rospy.logwarn(f"[inference_fn_async] {e}")
            inference_paused.clear()
            time.sleep(0.1)
            continue
        finally:
            with inference_state_cond:
                if inflight_inference_count > 0:
                    inflight_inference_count -= 1
                inference_state_cond.notify_all()


def start_inference_thread(args, config, policy, ros_operator, action_buffer):
    inference_thread = threading.Thread(
        target=inference_fn_async, args=(args, config, policy, ros_operator, action_buffer)
    )
    inference_thread.daemon = True
    inference_thread.start()


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    global inference_stamp

    if args.model == "openpi":
        policy = OpenpiClient(
            host=args.host,
            port=args.port,
            prompt=config["language_instruction"],
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    max_publish_step = config["episode_len"]

    left0 = config["left0"]
    right0 = config["right0"]

    print(config)

    ros_operator.follower_arm_publish_continuous(left0, right0)

    print("Warmup the server...")
    policy.warmup()
    print("Server warmed up")

    input("Press enter to continue")
    task_time = time.time()
    ros_operator.follower_arm_publish_continuous(left0, right0)

    # Create action buffer once (outside the loop)
    action_buffer = StreamActionBuffer(
        delay=config["delay"],
        exec_horizon=config["exec_horizon"],
        min_smooth_steps=args.min_smooth_steps,
    )

    # Start inference thread once (outside the loop)
    start_inference_thread(args, config, policy, ros_operator, action_buffer)
    recorder = InferenceDataRecorder(args, config, shutdown_event=shutdown_event)

    try:
        # Inference loop
        while not rospy.is_shutdown():
            # The current time step
            t = 0
            rate = rospy.Rate(args.publish_rate)

            # Reset observation window and action buffer for new episode
            begin_new_episode(wait_timeout=5.0)
            reset_observation_window()
            action_buffer.reset()

            inference_stamp = 0
            episode_closed = False

            # At beginning, launch sync inference
            actions = inference_fn_sync(args, config, policy, ros_operator)
            assert actions is not None, "Initial sync inference returned None"
            assert actions.shape[0] >= config["chunk_size"], (
                f"Initial actions length {actions.shape[0]} is smaller than {config['chunk_size']}"
            )
            action_buffer.integrate_first_chunk(actions[: config["chunk_size"]])

            last_valid_act = None

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                print(
                    f"[Step {t:4d}] k={action_buffer.k:3d} | cur_chunk={len(action_buffer.cur_chunk):3d} | epoch={action_buffer.epoch:3d}"
                )
                # Check for keyboard input (space to enter interactive mode)
                key = check_keyboard_input()
                if key == " ":
                    inference_paused.clear()
                    result = handle_interactive_mode(task_time)
                    if result == "reset":
                        recorder.save_episode()
                        episode_closed = True
                        # Reset to starting position
                        ros_operator.follower_arm_publish_continuous(left0, right0)
                        input("Press enter to continue")
                        task_time = time.time()
                        break  # Break inner loop to restart
                    elif result == "quit":
                        recorder.save_episode()
                        return  # Exit the function entirely
                    # 'continue' just resumes the loop

                if not inference_paused.is_set() and action_buffer.mark_launch_if_ready():
                    inference_paused.set()
                    time.sleep(0.001)

                act = action_buffer.get_next_action()

                if act is None:
                    rospy.logwarn(f"[Step {t:4d}] act is None")
                    if last_valid_act is not None:
                        act = last_valid_act
                    else:
                        rate.sleep()
                        continue

                observation_to_save = get_rollout_observation(args, config, ros_operator) if recorder.enabled else None
                if recorder.enabled and observation_to_save is None:
                    break

                if args.ctrl_type == "joint":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    ros_operator.follower_arm_publish(left_action, right_action)
                elif args.ctrl_type == "eef":
                    left_action, right_action = process_action(config["task"], act)
                    action_to_save = np.concatenate((left_action, right_action), axis=0)
                    ros_operator.follower_arm_pose_publish(left_action, right_action)

                recorder.add_step(observation_to_save, action_to_save)
                t += 1
                last_valid_act = act
                rate.sleep()

            if not episode_closed:
                recorder.save_episode()
            if shutdown_event.is_set():
                return
    finally:
        ros_operator.follower_arm_publish_continuous(left0, right0)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_publish_step",
        action="store",
        type=int,
        help="Maximum number of action publishing steps",
        default=10000,
        required=False,
    )
    parser.add_argument(
        "--img_front_topic",
        action="store",
        type=str,
        help="img_front_topic",
        default="/camera_f/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_topic",
        action="store",
        type=str,
        help="img_left_topic",
        default="/camera_l/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_topic",
        action="store",
        type=str,
        help="img_right_topic",
        default="/camera_r/color/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_front_depth_topic",
        action="store",
        type=str,
        help="img_front_depth_topic",
        default="/camera_f/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_left_depth_topic",
        action="store",
        type=str,
        help="img_left_depth_topic",
        default="/camera_l/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--img_right_depth_topic",
        action="store",
        type=str,
        help="img_right_depth_topic",
        default="/camera_r/depth/image_raw",
        required=False,
    )
    parser.add_argument(
        "--leader_arm_left_topic",
        action="store",
        type=str,
        help="leader_arm_left_topic",
        default="/leader/joint_left",
        required=False,
    )
    parser.add_argument(
        "--leader_arm_right_topic",
        action="store",
        type=str,
        help="leader_arm_right_topic",
        default="/leader/joint_right",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_left_topic",
        action="store",
        type=str,
        help="follower_arm_left_topic",
        default="/follower/joint_left",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_right_topic",
        action="store",
        type=str,
        help="follower_arm_right_topic",
        default="/follower/joint_right",
        required=False,
    )
    parser.add_argument(
        "--pos_cmd_left_topic",
        action="store",
        type=str,
        help="pos_cmd_left_topic",
        default="/follower/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--pos_cmd_right_topic",
        action="store",
        type=str,
        help="pos_cmd_right_topic",
        default="/follower/pos_cmd_right",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_left_pose_topic",
        action="store",
        type=str,
        default="/follower/end_pose_euler_left",
        required=False,
    )
    parser.add_argument(
        "--follower_arm_right_pose_topic",
        action="store",
        type=str,
        default="/follower/end_pose_euler_right",
        required=False,
    )
    parser.add_argument(
        "--publish_rate",
        action="store",
        type=int,
        help="The rate at which to publish the actions",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--chunk_size",
        action="store",
        type=int,
        help="Action chunk size",
        default=50,
        required=False,
    )
    parser.add_argument(
        "--arm_steps_length",
        action="store",
        type=float,
        nargs=7,
        help="The maximum change allowed for each joint per timestep (7 values)",
        default=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.2],
        required=False,
    )
    parser.add_argument(
        "--use_depth_image",
        action="store_true",
        help="Whether to use depth images",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--save_rollout",
        action="store_true",
        help="Save rollout observations/actions to HDF5 episodes",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--save_dir",
        action="store",
        type=str,
        help="Directory used when --save_rollout is set.",
        default="",
        required=False,
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["joint", "eef"],
        help="Control type for the robot arm",
        default="joint",
    )
    parser.add_argument(
        "--host",
        action="store",
        type=str,
        help="Websocket server host",
        default="127.0.0.1",
        required=False,
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        help="Websocket server port",
        default=8000,
        required=False,
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        help="Desired image size for OpenPI preprocessing",
        default=(224, 224),
        required=False,
    )
    parser.add_argument(
        "--task",
        action="store",
        type=str,
        help="Task name",
        required=True,
    )
    parser.add_argument(
        "--delay",
        type=int,
        help="Delay in steps",
        default=4,
        required=False,
    )
    parser.add_argument(
        "--exec_horizon",
        type=int,
        help="Execution horizon in steps",
        default=25,
        required=False,
    )
    parser.add_argument(
        "--min_smooth_steps",
        type=int,
        help="Minimum number of actions used to smooth async chunk transitions",
        default=8,
        required=False,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openpi"],
        help="Model to use",
        default="openpi",
        required=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args, mode="inference")
    config = get_config(args)

    signal.signal(signal.SIGINT, _on_sigint)

    # Set terminal to raw mode for non-blocking keyboard input
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
