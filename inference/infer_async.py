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


class StreamActionBuffer:

    def __init__(self, delay, exec_horizon, state_dim):
        self.delay = delay
        self.exec_horizon = exec_horizon
        self.lock = threading.Lock()

        self.cur_chunk = np.zeros((exec_horizon, state_dim))
        self.next_chunk = np.zeros((exec_horizon, state_dim))
        self.cur_len = 0
        self.next_len = 0

        self.cur_step = 0  # current step in cur_chunk
        self.cur_stamp = 0  # indicate the origin inference stamp of the current chunk
        self.last_launch_stamp = -1

    def reset(self):
        """Reset buffer state for a new episode."""
        with self.lock:
            self.cur_chunk.fill(0.0)
            self.next_chunk.fill(0.0)
            self.cur_len = 0
            self.next_len = 0
            self.cur_step = 0
            self.cur_stamp = 0
            self.last_launch_stamp = -1

    def mark_launch_if_ready(self, require_full_current=True):
        with self.lock:
            if self.last_launch_stamp == self.cur_stamp:
                return False

            if require_full_current and self.cur_len != self.exec_horizon:
                return False

            if self.exec_horizon > self.delay:
                launch_step = self.exec_horizon - self.delay - 1
                ready = self.cur_step >= launch_step
            else:
                # In continuous mode, launch once per chunk as soon as the async thread is free.
                ready = self.cur_step >= 0

            if not ready:
                return False

            self.last_launch_stamp = self.cur_stamp
            return True

    def integrate_first_chunk(self, actions_chunk: np.ndarray):
        # only for the first chunk inferenced by sync inference, already selected [0:s)
        with self.lock:
            assert self.cur_len == 0, "cur_len should be 0 when starting"
            assert self.cur_stamp == 0, "cur_stamp should be 0 when starting"
            assert actions_chunk.shape[0] == self.exec_horizon, f"{actions_chunk.shape[0]} != {self.exec_horizon}"
            self.cur_chunk[: actions_chunk.shape[0]] = actions_chunk
            self.cur_len = self.exec_horizon

    def integrate_new_chunk(self, actions_chunk: np.ndarray):
        # only for the new chunk inferenced by async inference, already selected [d:s+d)
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or len(actions_chunk) == 0 when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            assert L == self.exec_horizon, f"{L} != {self.exec_horizon}"
            if self.cur_len == 0:
                # warning: this should not happen
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                self.cur_chunk[:L] = actions_chunk
                self.cur_len = L
            else:
                self.next_chunk[:L] = actions_chunk
                self.next_len = L

    def integrate_new_chunk_streaming(self, actions_chunk: np.ndarray, stamp: int):
        # only for the new chunk inferenced by async streaming inference, already removed [0:d), but can be different lengths
        if actions_chunk is None or actions_chunk.shape[0] == 0:
            rospy.logwarn("actions_chunk is None or len(actions_chunk) == 0 when integrating new chunk")
            return
        L = actions_chunk.shape[0]
        with self.lock:
            if self.cur_len == 0:
                # warning: this should not happen
                rospy.logwarn("cur_len is 0 when integrating new chunk")
                safe_L = min(L, self.exec_horizon)
                self.cur_chunk[:safe_L] = actions_chunk[:safe_L]
                self.cur_len = safe_L
                print(f"cur_chunk extend to {self.cur_len} at stamp {stamp}")
            else:
                if self.cur_stamp == stamp:
                    # current chunk is already executing, extend it
                    remaining_len = self.exec_horizon - self.cur_len
                    if remaining_len > 0:
                        safe_L = min(L, remaining_len)
                        self.cur_chunk[self.cur_len : self.cur_len + safe_L] = actions_chunk[:safe_L]
                        self.cur_len += safe_L
                        print(f"cur_chunk extend to {self.cur_len} with {safe_L} new actions at stamp {stamp}")
                    else:
                        print(f"cur_chunk is already enough at stamp {stamp}")
                else:
                    remaining_len = self.exec_horizon - self.next_len
                    if remaining_len > 0:
                        safe_L = min(L, remaining_len)
                        self.next_chunk[self.next_len : self.next_len + safe_L] = actions_chunk[:safe_L]
                        self.next_len += safe_L
                        print(f"next_chunk extend to {self.next_len} with {safe_L}/{L} new actions at stamp {stamp}")
                    else:
                        print(f"next_chunk is already enough at stamp {stamp}")

    def get_next_action(self):
        with self.lock:
            if self.cur_step >= self.cur_len:
                return None

            action = self.cur_chunk[self.cur_step]
            self.cur_step += 1

            # should only execute [0:s) of the current chunk, switch to next chunk
            if self.cur_step == self.exec_horizon:
                self.cur_chunk, self.next_chunk = self.next_chunk, self.cur_chunk
                self.cur_len = self.next_len
                self.next_len = 0
                self.cur_step = 0
                self.cur_stamp += 1

            return action


def inference_fn_sync(args, config, policy, ros_operator):
    global inference_stamp

    if not update_observation_window(args, config, ros_operator):
        return None

    start_time = time.perf_counter()

    with observation_window_lock:
        # fetch images in sequence [front, left, right]
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint":
            # state: Abs Joint 14dim
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            # state: Abs EEF 14dim
            state = observation_window[-1]["eef_pose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    payload = {
        "top": image_arrs[0],
        "left": image_arrs[1],
        "right": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state,
        "action_prefix": None,
        "delay": None,
    }

    if args.streaming:
        actions = policy.predict_action_streaming(payload)
    else:
        actions = policy.predict_action(payload)
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

            d = config["delay"]
            s = config["exec_horizon"]

            # Use action_buffer's internal lock to safely read cur_chunk
            with action_buffer.lock:
                if config["mode"] == "naive":
                    action_prefix = None
                    delay = None
                else:
                    assert action_buffer.cur_len == s, (
                        f"Current chunk must be full before RTC inference: {action_buffer.cur_len} != {s}"
                    )
                    action_prefix = action_buffer.cur_chunk[(s - d) : s].copy()  # last d actions of the current chunk
                    assert action_prefix.shape[0] == d, f"{action_prefix.shape[0]} != {d}"
                    delay = np.array(d)

            if not update_observation_window(args, config, ros_operator):
                inference_paused.clear()
                continue

            start_time = time.perf_counter()

            with observation_window_lock:
                # fetch images in sequence [front, left, right]
                image_arrs = [
                    observation_window[-1]["images"][config["camera_names"][0]],
                    observation_window[-1]["images"][config["camera_names"][1]],
                    observation_window[-1]["images"][config["camera_names"][2]],
                ]

                if args.ctrl_type == "joint":
                    # state: Abs Joint 14dim
                    state = observation_window[-1]["qpos"]
                elif args.ctrl_type == "eef":
                    # state: Abs EEF 14dim
                    state = observation_window[-1]["eef_pose"]
                else:
                    raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

            payload = {
                "top": image_arrs[0],
                "left": image_arrs[1],
                "right": image_arrs[2],
                "instruction": config["language_instruction"],
                "state": state,
                "action_prefix": action_prefix,
                "delay": delay,
            }

            def on_actions_ready(actions_chunk):
                if not is_current_episode(request_episode_id):
                    rospy.logwarn(
                        f"Discard async streaming chunk from stale episode {request_episode_id}; current={episode_id}"
                    )
                    return
                action_buffer.integrate_new_chunk_streaming(actions_chunk, stamp=inference_stamp)

            if args.streaming:
                policy.predict_action_streaming(payload, on_actions_ready=on_actions_ready)
                print(
                    f"[Async  {inference_stamp:2d}] Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms"
                )
            else:
                actions = policy.predict_action(payload)
                print(
                    f"[Async  {inference_stamp:2d}] Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms"
                )

                if not is_current_episode(request_episode_id):
                    rospy.logwarn(
                        f"Discard async inference result from stale episode {request_episode_id}; current={episode_id}"
                    )
                elif actions is not None and len(actions) > 0:
                    assert actions.shape[0] >= s + d, f"Async actions length {actions.shape[0]} is smaller than {s + d}"
                    action_buffer.integrate_new_chunk(actions[d : s + d])
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
        policy = OpenpiClient(host=args.host, port=args.port)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    max_publish_step = config["episode_len"]

    left0 = config["left0"]
    right0 = config["right0"]

    print(config)

    ros_operator.follower_arm_publish_continuous(left0, right0)

    print("Warmup the server...")
    policy.warmup(rtc=(args.mode == "rtc"), streaming=args.streaming)
    print("Server warmed up")

    input("Press enter to continue")
    task_time = time.time()
    ros_operator.follower_arm_publish_continuous(left0, right0)

    # Create action buffer once (outside the loop)
    action_buffer = StreamActionBuffer(
        delay=config["delay"], exec_horizon=config["exec_horizon"], state_dim=config["state_dim"]
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
            assert actions.shape[0] >= config["exec_horizon"], (
                f"Initial actions length {actions.shape[0]} is smaller than {config['exec_horizon']}"
            )
            action_buffer.integrate_first_chunk(actions[: config["exec_horizon"]])

            last_valid_act = None

            while t < max_publish_step and not rospy.is_shutdown() and not shutdown_event.is_set():
                print(
                    f"[Step {t:4d}] cur_step={action_buffer.cur_step:3d} | cur_chunk={action_buffer.cur_len:3d} | next_chunk={action_buffer.next_len:3d} | cur_stamp={action_buffer.cur_stamp:3d}"
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

                require_full_current = config["mode"] != "naive"
                if (
                    not inference_paused.is_set()
                    and action_buffer.mark_launch_if_ready(require_full_current=require_full_current)
                ):
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
        "--mode",
        action="store",
        type=str,
        choices=["naive", "rtc"],
        help="Mode of the inference",
        default="rtc",
        required=False,
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Whether to use streaming inference",
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
