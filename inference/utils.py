import argparse
import os
from pathlib import Path
import select
import sys
import threading
import time
from collections import deque

import cv2
import dm_env
import numpy as np
import rospy
import yaml

TASK_CONFIG_PATH = Path(__file__).with_name("task_configs.yaml")


# Task Configuration


def _load_task_configs():
    with TASK_CONFIG_PATH.open("r", encoding="utf-8") as file:
        task_configs = yaml.safe_load(file)
    if not isinstance(task_configs, dict):
        raise ValueError(f"Invalid task config format in {TASK_CONFIG_PATH}")
    return task_configs


TASK_CONFIGS = _load_task_configs()
FIXED_ARM_TARGETS = {"state", "action"}


def get_config(args):
    task_config = TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {args.task}")

    language_instruction = task_config.get("language_instruction")
    left0 = task_config.get("left0")
    right0 = task_config.get("right0")
    if language_instruction is None or left0 is None or right0 is None:
        raise ValueError(f"Task config for {args.task} is missing required fields")

    action_postprocess = task_config.get("action_postprocess", {})
    fixed_arms = get_fixed_arms(action_postprocess)
    fixed_arm_targets = get_fixed_arm_targets(action_postprocess) if fixed_arms else set()
    if fixed_arms and args.ctrl_type == "eef":
        raise ValueError(
            f"Task {args.task} uses fix_arms_to_initial_pose={sorted(fixed_arms)}, "
            "which is only supported with --ctrl_type joint"
        )

    state_dim = 14

    config = {
        "episode_len": args.max_publish_step,
        "state_dim": state_dim,
        "left0": left0,
        "right0": right0,
        "action_postprocess": action_postprocess,
        "fixed_arms_to_initial_pose": fixed_arms,
        "fixed_arms_to_initial_pose_targets": fixed_arm_targets,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "task": args.task,
        "language_instruction": language_instruction,
        "ctrl_type": args.ctrl_type,
        "chunk_size": args.chunk_size,  # action chunk size from model
        "delay": getattr(args, "delay", None),
        "exec_horizon": getattr(args, "exec_horizon", None),  # execution horizon/inference interval
        "mode": getattr(args, "mode", None),
        "model": getattr(args, "model", None),
    }
    return config


# Action Post-Processing


def get_fixed_arms(action_postprocess):
    fixed_arms = action_postprocess.get("fix_arms_to_initial_pose", [])
    if isinstance(fixed_arms, str):
        fixed_arms = [fixed_arms]
    fixed_arms = set(fixed_arms)
    invalid_arms = fixed_arms - {"left", "right"}
    if invalid_arms:
        raise ValueError(f"Invalid fixed arms: {sorted(invalid_arms)}; expected left and/or right")
    return fixed_arms


def get_fixed_arm_targets(action_postprocess):
    targets = action_postprocess.get("fix_arms_to_initial_pose_targets", sorted(FIXED_ARM_TARGETS))
    if isinstance(targets, str):
        targets = [targets]
    targets = set(targets)
    invalid_targets = targets - FIXED_ARM_TARGETS
    if invalid_targets:
        raise ValueError(
            f"Invalid fixed arm targets: {sorted(invalid_targets)}; expected state and/or action"
        )
    return targets


def apply_fixed_arms_to_initial_pose(array, config, target=None):
    fixed_arms = config.get("fixed_arms_to_initial_pose", set())
    if not fixed_arms or array is None:
        return array

    if target is not None:
        if target not in FIXED_ARM_TARGETS:
            raise ValueError(f"Invalid fixed arm target: {target}; expected state or action")
        fixed_arm_targets = set(config.get("fixed_arms_to_initial_pose_targets", FIXED_ARM_TARGETS))
        if target not in fixed_arm_targets:
            return array

    result = np.asarray(array).copy()
    if "left" in fixed_arms:
        result[..., :7] = np.asarray(config["left0"])
    if "right" in fixed_arms:
        result[..., 7:14] = np.asarray(config["right0"])
    return result


def _apply_gripper_rules(gripper_value, rules):
    for rule in rules:
        condition = rule.get("when")
        threshold = rule.get("threshold")
        if condition == "below" and not gripper_value < threshold:
            continue
        if condition == "above" and not gripper_value > threshold:
            continue

        if "set" in rule:
            gripper_value = rule["set"]
        if "add" in rule:
            gripper_value += rule["add"]
    return max(0, gripper_value)


def process_action(task, action):
    action = action.copy()
    left_action = action[:7]
    right_action = action[7:14]

    task_config = TASK_CONFIGS.get(task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {task}")

    action_postprocess = task_config.get("action_postprocess", {})
    left_action[6] = _apply_gripper_rules(left_action[6], action_postprocess.get("left_gripper", []))
    right_action[6] = _apply_gripper_rules(right_action[6], action_postprocess.get("right_gripper", []))

    fixed_arm_targets = get_fixed_arm_targets(action_postprocess)
    fixed_arms = get_fixed_arms(action_postprocess) if "action" in fixed_arm_targets else set()
    if "left" in fixed_arms:
        left_action[:] = task_config["left0"]
    if "right" in fixed_arms:
        right_action[:] = task_config["right0"]

    return left_action, right_action


# Keyboard Interaction


def check_keyboard_input():
    """Check if a key was pressed without blocking."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


_interactive_timer_state = {
    "task_time": None,
    "total_elapsed": 0.0,
    "run_started_at": None,
}


def _pause_interactive_timer(task_time):
    now = time.time()

    if _interactive_timer_state["task_time"] != task_time:
        _interactive_timer_state["task_time"] = task_time
        _interactive_timer_state["total_elapsed"] = 0.0
        _interactive_timer_state["run_started_at"] = task_time

    run_started_at = _interactive_timer_state["run_started_at"]
    if run_started_at is None:
        run_started_at = now

    segment_elapsed = max(0.0, now - run_started_at)
    total_elapsed = _interactive_timer_state["total_elapsed"] + segment_elapsed

    _interactive_timer_state["total_elapsed"] = total_elapsed
    _interactive_timer_state["run_started_at"] = None

    return total_elapsed, segment_elapsed


def _resume_interactive_timer():
    _interactive_timer_state["run_started_at"] = time.time()


def handle_interactive_mode(task_time, enable_dagger=False, policy_switcher=None):
    """Handle interactive mode when space is pressed."""
    total_elapsed, segment_elapsed = _pause_interactive_timer(task_time)
    use_policy_switch = policy_switcher is not None

    print("\n" + "=" * 50)
    print(f"Total task time: {total_elapsed:.1f} s")
    print(f"Time since last continue: {segment_elapsed:.1f} s")
    print("INTERACTIVE MODE")
    if use_policy_switch:
        print(f"Active policy: {policy_switcher.active_name}")
    print("  'c' - Continue running")
    if use_policy_switch:
        print("  's' - Switch to student and continue")
        print("  't' - Switch to teacher and continue")
    print("  'r' - Reset to starting point and restart")
    print("  'q' - Quit/Stop")
    if enable_dagger:
        print("  'd' - Enter DAgger mode")
    print("=" * 50)

    def format_result(command, policy_changed=False):
        if use_policy_switch:
            return command, policy_changed
        return command

    while True:
        key = sys.stdin.read(1).lower()
        if key == "c":
            _resume_interactive_timer()
            if use_policy_switch:
                print(f"Continuing with {policy_switcher.active_name} policy...")
            else:
                print("Continuing...")
            return format_result("continue")
        if key == "s" and use_policy_switch:
            policy_changed = policy_switcher.switch_to("student")
            _resume_interactive_timer()
            print("Continuing...")
            return format_result("continue", policy_changed)
        if key == "t" and use_policy_switch:
            policy_changed = policy_switcher.switch_to("teacher")
            _resume_interactive_timer()
            print("Continuing...")
            return format_result("continue", policy_changed)
        if key == "r":
            print("Restarting...")
            return format_result("reset")
        if key == "q":
            print("Stopping...")
            return format_result("quit")
        if key == "d" and enable_dagger:
            print("Entering DAgger mode...")
            return format_result("dagger")


# Observation And Sync Inference


def build_observation(observation, config, ros_operator):
    if observation is None:
        return None

    (
        img_front,
        img_left,
        img_right,
        follower_arm_left,
        follower_arm_right,
        follower_arm_left_pose,
        follower_arm_right_pose,
    ) = observation

    qpos = np.concatenate(
        (np.array(follower_arm_left.position), np.array(follower_arm_right.position)),
        axis=0,
    )
    eef_pose = ros_operator.build_follower_arm_pose(
        follower_arm_left_pose,
        follower_arm_right_pose,
        follower_arm_left,
        follower_arm_right,
    )
    qpos = apply_fixed_arms_to_initial_pose(qpos, config, target="state")

    return {
        "qpos": qpos,
        "eef_pose": eef_pose,
        "images": {
            config["camera_names"][0]: img_front,
            config["camera_names"][1]: img_left,
            config["camera_names"][2]: img_right,
        },
    }


def get_inference_observation(args, config, ros_operator):
    from ros_operator import get_ros_observation

    return build_observation(get_ros_observation(args, ros_operator), config, ros_operator)


def get_rollout_observation(args, config, ros_operator):
    from ros_operator import get_latest_ros_observation

    return build_observation(get_latest_ros_observation(args, ros_operator), config, ros_operator)


observation_window = None
observation_window_lock = threading.Lock()


def reset_observation_window():
    """Reset observation window for a new episode."""
    global observation_window
    with observation_window_lock:
        observation_window = None


def update_observation_window(args, config, ros_operator):
    global observation_window
    with observation_window_lock:
        if observation_window is None:
            observation_window = deque(maxlen=2)
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


def inference_fn_sync(args, config, policy, ros_operator):
    if not update_observation_window(args, config, ros_operator):
        return None

    start_time = time.perf_counter()

    with observation_window_lock:
        image_arrs = [
            observation_window[-1]["images"][config["camera_names"][0]],
            observation_window[-1]["images"][config["camera_names"][1]],
            observation_window[-1]["images"][config["camera_names"][2]],
        ]

        if args.ctrl_type == "joint":
            state = observation_window[-1]["qpos"]
        elif args.ctrl_type == "eef":
            state = observation_window[-1]["eef_pose"]
        else:
            raise ValueError(f"Unknown ctrl_type: {args.ctrl_type}")

    payload = {
        "top": image_arrs[0],
        "left": image_arrs[1],
        "right": image_arrs[2],
        "instruction": config["language_instruction"],
        "state": state,
    }

    actions = policy.predict_action(payload)
    actions = apply_fixed_arms_to_initial_pose(actions, config, target="action")
    print(f"Model inference time: {(time.perf_counter() - start_time)*1000:.3f} ms")

    return actions


# Data Recording


def save_inference_data(args, timesteps, actions, dataset_path, collect_labels=None):
    import h5py

    data_size = len(actions)
    first_observation = timesteps[0].observation if timesteps else None
    if first_observation is None:
        raise ValueError("No timesteps available for saving")

    data_dict = {
        "/observations/qpos": [],
        "/observations/eef_pose": [],
        "/action": [],
        "/collect": [],
    }

    for cam_name in args.camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []

    if collect_labels is None:
        collect_labels = ["rollout"] * data_size
    else:
        collect_labels = list(collect_labels)
        if len(collect_labels) != data_size:
            raise ValueError(f"collect_labels length {len(collect_labels)} does not match actions length {data_size}")

    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        collect_label = collect_labels.pop(0)

        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/eef_pose"].append(ts.observation["eef_pose"])
        data_dict["/action"].append(action)
        data_dict["/collect"].append(collect_label)

        for cam_name in args.camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(ts.observation["images"][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in args.camera_names:
            _ = image.create_dataset(
                cam_name,
                (data_size, 480, 640, 3),
                dtype="uint8",
                chunks=(1, 480, 640, 3),
            )

        _ = obs.create_dataset("qpos", (data_size, 14))
        _ = obs.create_dataset("eef_pose", (data_size, 14))
        _ = root.create_dataset("action", (data_size, 14))
        _ = root.create_dataset("collect", (data_size,), dtype=h5py.string_dtype(encoding="utf-8"))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f"\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n" % dataset_path)


class InferenceDataRecorder:
    def __init__(self, args, config, shutdown_event=None):
        self.enabled = args.save_rollout
        self.shutdown_event = shutdown_event
        self.save_args = argparse.Namespace(camera_names=config["camera_names"])
        self.save_dir = os.path.expanduser(args.save_dir)
        self.episode_idx = 0
        if self.enabled:
            from collect_data.collect_data import get_next_episode_idx

            os.makedirs(self.save_dir, exist_ok=True)
            self.episode_idx = get_next_episode_idx(self.save_dir)
            print(f"Rollout recording enabled: {self.save_dir}, next episode {self.episode_idx}")
        self.reset()

    def reset(self):
        self.timesteps = []
        self.actions = []
        self.collect_labels = []

    def add_step(self, observation, action, collect_label="rollout"):
        if not self.enabled or observation is None:
            return

        step_type = dm_env.StepType.FIRST if len(self.timesteps) == 0 else dm_env.StepType.MID
        self.timesteps.append(
            dm_env.TimeStep(
                step_type=step_type,
                reward=None,
                discount=None,
                observation=observation,
            )
        )
        self.actions.append(np.asarray(action).copy())
        self.collect_labels.append(collect_label)

        if len(self.actions) % 50 == 0:
            print(f"Recorded inference frames: {len(self.actions)}")

    def should_stop_waiting(self):
        return rospy.is_shutdown() or (self.shutdown_event is not None and self.shutdown_event.is_set())

    def wait_save_choice(self):
        print(
            "\n\033[33m\nRollout paused. Press 's' to SAVE or 'q' to DISCARD: \033[0m",
            end="",
            flush=True,
        )
        while not self.should_stop_waiting():
            key = sys.stdin.read(1).lower()
            if key in {"s", "q"}:
                print(key)
                return key
        return "q"

    def save_episode(self):
        if not self.enabled:
            return
        if len(self.actions) == 0:
            print("\033[31m\nNo inference data to save (0 frames recorded).\033[0m")
            self.reset()
            return

        print("len(timesteps): ", len(self.timesteps))
        print("len(actions)  : ", len(self.actions))
        if self.wait_save_choice() != "s":
            print(f"\033[31m\nEpisode discarded. {len(self.actions)} frames thrown away.\033[0m")
            self.reset()
            return

        dataset_path = os.path.join(self.save_dir, f"episode_{self.episode_idx}")
        save_inference_data(
            self.save_args,
            self.timesteps.copy(),
            self.actions.copy(),
            dataset_path,
            self.collect_labels.copy(),
        )
        print(f"\033[32mEpisode {self.episode_idx} saved successfully!\033[0m")
        self.episode_idx += 1
        self.reset()


# Image Utilities


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(im: np.ndarray, height: int, width: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize one image (H, W, C) to target height/width without distortion by padding with zeros."""
    cur_height, cur_width = im.shape[0], im.shape[1]
    if cur_width == width and cur_height == height:
        return np.ascontiguousarray(im)

    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)

    resized = cv2.resize(im, (resized_width, resized_height), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    out = np.zeros((height, width, resized.shape[2]), dtype=im.dtype)
    pad_top = (height - resized_height) // 2
    pad_left = (width - resized_width) // 2
    out[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return out
