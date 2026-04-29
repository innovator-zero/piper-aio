from collections import deque
import threading

import cv2
import dm_env
import numpy as np
import rospy
from cv_bridge import CvBridge
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


class RosOperator:
    queue_size_limit = 2000

    def __init__(self, args, mode):
        self.args = args
        if mode not in {"collection", "inference"}:
            raise ValueError(f"Unsupported RosOperator mode: {mode}")
        self.mode = mode
        self.bridge = CvBridge()

        self.follower_arm_left_publisher = None
        self.follower_arm_right_publisher = None
        self.follower_arm_left_pose_publisher = None
        self.follower_arm_right_pose_publisher = None
        self.follower_arm_publish_thread = None
        self.follower_arm_publish_lock = threading.Lock()
        self.follower_arm_publish_lock.acquire()

        self._init_queues()
        self.init_ros()

    def _queue_attribute_names(self):
        queue_names = [
            "front_image_queue",
            "left_image_queue",
            "right_image_queue",
            "follower_left_arm_queue",
            "follower_right_arm_queue",
        ]

        if self.args.use_depth_image:
            queue_names.extend(
                [
                    "front_depth_image_queue",
                    "left_depth_image_queue",
                    "right_depth_image_queue",
                ]
            )

        if self.mode == "collection":
            queue_names.extend(
                [
                    "leader_left_arm_queue",
                    "leader_right_arm_queue",
                    "follower_arm_left_pose_queue",
                    "follower_arm_right_pose_queue",
                ]
            )
        else:
            queue_names.extend(
                [
                    "follower_arm_left_pose_queue",
                    "follower_arm_right_pose_queue",
                ]
            )
        return queue_names

    def _init_queues(self):
        for attr_name in self._queue_attribute_names():
            setattr(self, attr_name, deque())

    def init(self):
        self.reset()

    def reset(self):
        self._init_queues()

    def _append_to_queue(self, queue, msg):
        if len(queue) >= self.queue_size_limit:
            queue.popleft()
        queue.append(msg)

    def _make_queue_callback(self, attr_name, stamp_now=False):
        def _callback(msg):
            queue = getattr(self, attr_name)
            data = (rospy.Time.now(), msg) if stamp_now else msg
            self._append_to_queue(queue, data)

        return _callback

    def _register_subscribers(self, subscriber_specs):
        for topic, msg_type, callback in subscriber_specs:
            rospy.Subscriber(topic, msg_type, callback, queue_size=1000, tcp_nodelay=True)

    def _get_msg_stamp(self, msg):
        if isinstance(msg, tuple):
            return msg[0].to_sec()
        return msg.header.stamp.to_sec()

    def _format_sync_failure(self, stream_map, frame_time=None):
        missing_streams = []
        stale_streams = []
        stream_states = []
        for name, queue in stream_map.items():
            if len(queue) == 0:
                missing_streams.append(name)
                stream_states.append(f"{name}:len=0")
                continue
            latest_stamp = self._get_msg_stamp(queue[-1])
            if frame_time is not None and latest_stamp < frame_time:
                stale_streams.append(f"{name}={latest_stamp:.6f}")
            if frame_time is None:
                stream_states.append(f"{name}:len={len(queue)},latest={latest_stamp:.6f}")
            else:
                stream_states.append(
                    f"{name}:len={len(queue)},latest={latest_stamp:.6f},delta={latest_stamp - frame_time:+.4f}s"
                )

        details = []
        if frame_time is not None:
            details.append(f"frame_time={frame_time:.6f}")
        if missing_streams:
            details.append("missing=" + ",".join(missing_streams))
        if stale_streams:
            details.append("stale=" + ",".join(stale_streams))
        if stream_states:
            details.append("streams=[" + "; ".join(stream_states) + "]")
        return " | ".join(details) if details else "unknown sync state"

    def _resolve_frame_time(self, camera_queues):
        return min(self._get_msg_stamp(queue[-1]) for queue in camera_queues)

    def _validate_stream_sync(self, stream_map, frame_time):
        for queue in stream_map.values():
            if len(queue) == 0 or self._get_msg_stamp(queue[-1]) < frame_time:
                return False
        return True

    def _pop_synced_message(self, queue, frame_time):
        while self._get_msg_stamp(queue[0]) < frame_time:
            queue.popleft()
        return queue.popleft()

    def _pop_synced_image(self, queue, frame_time, border_padding=None):
        image = self.bridge.imgmsg_to_cv2(self._pop_synced_message(queue, frame_time), "passthrough")
        if border_padding is not None:
            top, bottom, left, right = border_padding
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return image

    def _peek_latest_message(self, queue):
        return queue[-1]

    def _peek_latest_image(self, queue, border_padding=None):
        image = self.bridge.imgmsg_to_cv2(queue[-1], "passthrough")
        if border_padding is not None:
            top, bottom, left, right = border_padding
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return image

    def _format_zero_value_failure(self, named_values):
        zero_fields = [name for name, values in named_values if np.allclose(values, 0.0)]
        if not zero_fields:
            return None
        return "all_zero=" + ",".join(zero_fields)

    def _build_single_arm_pose(self, follower_arm_pose, follower_arm_joint_state):
        return np.array(
            [
                follower_arm_pose.x,
                follower_arm_pose.y,
                follower_arm_pose.z,
                follower_arm_pose.roll,
                follower_arm_pose.pitch,
                follower_arm_pose.yaw,
                follower_arm_joint_state.position[-1],
            ]
        )

    def build_follower_arm_pose(self, follower_arm_left_pose, follower_arm_right_pose, follower_arm_left, follower_arm_right):
        left_pose = self._build_single_arm_pose(follower_arm_left_pose, follower_arm_left)
        right_pose = self._build_single_arm_pose(follower_arm_right_pose, follower_arm_right)
        return np.concatenate((left_pose, right_pose), axis=0)

    def _camera_queues(self):
        camera_queues = [self.front_image_queue, self.left_image_queue, self.right_image_queue]
        if self.args.use_depth_image:
            camera_queues.extend(
                [
                    self.front_depth_image_queue,
                    self.left_depth_image_queue,
                    self.right_depth_image_queue,
                ]
            )
        return camera_queues

    def _collection_stream_map(self):
        stream_map = {
            "img_front": self.front_image_queue,
            "img_left": self.left_image_queue,
            "img_right": self.right_image_queue,
            "leader_arm_left": self.leader_left_arm_queue,
            "leader_arm_right": self.leader_right_arm_queue,
            "follower_arm_left": self.follower_left_arm_queue,
            "follower_arm_right": self.follower_right_arm_queue,
            "follower_arm_left_pose": self.follower_arm_left_pose_queue,
            "follower_arm_right_pose": self.follower_arm_right_pose_queue,
        }
        if self.args.use_depth_image:
            stream_map.update(
                {
                    "img_front_depth": self.front_depth_image_queue,
                    "img_left_depth": self.left_depth_image_queue,
                    "img_right_depth": self.right_depth_image_queue,
                }
            )
        return stream_map

    def _inference_stream_map(self):
        stream_map = {
            "img_front": self.front_image_queue,
            "img_left": self.left_image_queue,
            "img_right": self.right_image_queue,
            "follower_arm_left": self.follower_left_arm_queue,
            "follower_arm_right": self.follower_right_arm_queue,
            "follower_arm_left_pose": self.follower_arm_left_pose_queue,
            "follower_arm_right_pose": self.follower_arm_right_pose_queue,
        }
        if self.args.use_depth_image:
            stream_map.update(
                {
                    "img_front_depth": self.front_depth_image_queue,
                    "img_left_depth": self.left_depth_image_queue,
                    "img_right_depth": self.right_depth_image_queue,
                }
            )
        return stream_map

    def get_collection_frame(self):
        stream_map = self._collection_stream_map()
        if any(len(queue) == 0 for queue in stream_map.values()):
            return False, self._format_sync_failure(stream_map)

        frame_time = self._resolve_frame_time(self._camera_queues())
        if not self._validate_stream_sync(stream_map, frame_time):
            return False, self._format_sync_failure(stream_map, frame_time)

        img_front = self._pop_synced_image(self.front_image_queue, frame_time)
        img_left = self._pop_synced_image(self.left_image_queue, frame_time)
        img_right = self._pop_synced_image(self.right_image_queue, frame_time)
        leader_arm_left = self._pop_synced_message(self.leader_left_arm_queue, frame_time)
        leader_arm_right = self._pop_synced_message(self.leader_right_arm_queue, frame_time)
        follower_arm_left = self._pop_synced_message(self.follower_left_arm_queue, frame_time)
        follower_arm_right = self._pop_synced_message(self.follower_right_arm_queue, frame_time)
        _, follower_arm_left_pose = self._pop_synced_message(self.follower_arm_left_pose_queue, frame_time)
        _, follower_arm_right_pose = self._pop_synced_message(self.follower_arm_right_pose_queue, frame_time)

        img_front_depth = None
        img_left_depth = None
        img_right_depth = None
        if self.args.use_depth_image:
            border_padding = (40, 40, 0, 0)
            img_front_depth = self._pop_synced_image(self.front_depth_image_queue, frame_time, border_padding)
            img_left_depth = self._pop_synced_image(self.left_depth_image_queue, frame_time, border_padding)
            img_right_depth = self._pop_synced_image(self.right_depth_image_queue, frame_time, border_padding)

        return (
            True,
            (
                img_front,
                img_left,
                img_right,
                img_front_depth,
                img_left_depth,
                img_right_depth,
                follower_arm_left,
                follower_arm_right,
                follower_arm_left_pose,
                follower_arm_right_pose,
                leader_arm_left,
                leader_arm_right,
            ),
        )

    def get_inference_frame(self):
        stream_map = self._inference_stream_map()
        if any(len(queue) == 0 for queue in stream_map.values()):
            return False, self._format_sync_failure(stream_map)

        frame_time = self._resolve_frame_time(self._camera_queues())
        if not self._validate_stream_sync(stream_map, frame_time):
            return False, self._format_sync_failure(stream_map, frame_time)

        img_front = self._pop_synced_image(self.front_image_queue, frame_time)
        img_left = self._pop_synced_image(self.left_image_queue, frame_time)
        img_right = self._pop_synced_image(self.right_image_queue, frame_time)
        follower_arm_left = self._pop_synced_message(self.follower_left_arm_queue, frame_time)
        follower_arm_right = self._pop_synced_message(self.follower_right_arm_queue, frame_time)
        _, follower_arm_left_pose = self._pop_synced_message(self.follower_arm_left_pose_queue, frame_time)
        _, follower_arm_right_pose = self._pop_synced_message(self.follower_arm_right_pose_queue, frame_time)

        img_front_depth = None
        img_left_depth = None
        img_right_depth = None
        if self.args.use_depth_image:
            img_front_depth = self._pop_synced_image(self.front_depth_image_queue, frame_time)
            img_left_depth = self._pop_synced_image(self.left_depth_image_queue, frame_time)
            img_right_depth = self._pop_synced_image(self.right_depth_image_queue, frame_time)

        return (
            True,
            (
                img_front,
                img_left,
                img_right,
                img_front_depth,
                img_left_depth,
                img_right_depth,
                follower_arm_left,
                follower_arm_right,
                follower_arm_left_pose,
                follower_arm_right_pose,
            ),
        )

    def get_latest_inference_frame(self):
        stream_map = self._inference_stream_map()
        if any(len(queue) == 0 for queue in stream_map.values()):
            return False, self._format_sync_failure(stream_map)

        img_front = self._peek_latest_image(self.front_image_queue)
        img_left = self._peek_latest_image(self.left_image_queue)
        img_right = self._peek_latest_image(self.right_image_queue)
        follower_arm_left = self._peek_latest_message(self.follower_left_arm_queue)
        follower_arm_right = self._peek_latest_message(self.follower_right_arm_queue)
        _, follower_arm_left_pose = self._peek_latest_message(self.follower_arm_left_pose_queue)
        _, follower_arm_right_pose = self._peek_latest_message(self.follower_arm_right_pose_queue)

        img_front_depth = None
        img_left_depth = None
        img_right_depth = None
        if self.args.use_depth_image:
            img_front_depth = self._peek_latest_image(self.front_depth_image_queue)
            img_left_depth = self._peek_latest_image(self.left_depth_image_queue)
            img_right_depth = self._peek_latest_image(self.right_depth_image_queue)

        return (
            True,
            (
                img_front,
                img_left,
                img_right,
                img_front_depth,
                img_left_depth,
                img_right_depth,
                follower_arm_left,
                follower_arm_right,
                follower_arm_left_pose,
                follower_arm_right_pose,
            ),
        )

    def get_frame(self):
        if self.mode == "collection":
            return self.get_collection_frame()
        return self.get_inference_frame()

    def init_ros(self):
        node_name = "record_episodes" if self.mode == "collection" else "joint_state_publisher"
        rospy.init_node(node_name, anonymous=True)

        subscriber_specs = [
            (self.args.img_front_topic, Image, self._make_queue_callback("front_image_queue")),
            (self.args.img_left_topic, Image, self._make_queue_callback("left_image_queue")),
            (self.args.img_right_topic, Image, self._make_queue_callback("right_image_queue")),
            (self.args.follower_arm_left_topic, JointState, self._make_queue_callback("follower_left_arm_queue")),
            (self.args.follower_arm_right_topic, JointState, self._make_queue_callback("follower_right_arm_queue")),
        ]
        if self.args.use_depth_image:
            subscriber_specs.extend(
                [
                    (self.args.img_front_depth_topic, Image, self._make_queue_callback("front_depth_image_queue")),
                    (self.args.img_left_depth_topic, Image, self._make_queue_callback("left_depth_image_queue")),
                    (self.args.img_right_depth_topic, Image, self._make_queue_callback("right_depth_image_queue")),
                ]
            )

        if self.mode == "collection":
            subscriber_specs.extend(
                [
                    (self.args.leader_arm_left_topic, JointState, self._make_queue_callback("leader_left_arm_queue")),
                    (self.args.leader_arm_right_topic, JointState, self._make_queue_callback("leader_right_arm_queue")),
                    (
                        self.args.follower_arm_left_pose_topic,
                        PosCmd,
                        self._make_queue_callback("follower_arm_left_pose_queue", True),
                    ),
                    (
                        self.args.follower_arm_right_pose_topic,
                        PosCmd,
                        self._make_queue_callback("follower_arm_right_pose_queue", True),
                    ),
                ]
            )
        else:
            subscriber_specs.extend(
                [
                    (
                        self.args.follower_arm_left_pose_topic,
                        PosCmd,
                        self._make_queue_callback("follower_arm_left_pose_queue", True),
                    ),
                    (
                        self.args.follower_arm_right_pose_topic,
                        PosCmd,
                        self._make_queue_callback("follower_arm_right_pose_queue", True),
                    ),
                ]
            )

        self._register_subscribers(subscriber_specs)

        if self.mode == "inference":
            self.follower_arm_left_publisher = rospy.Publisher(self.args.leader_arm_left_topic, JointState, queue_size=10)
            self.follower_arm_right_publisher = rospy.Publisher(
                self.args.leader_arm_right_topic, JointState, queue_size=10
            )
            self.follower_arm_left_pose_publisher = rospy.Publisher(self.args.pos_cmd_left_topic, PosCmd, queue_size=10)
            self.follower_arm_right_pose_publisher = rospy.Publisher(self.args.pos_cmd_right_topic, PosCmd, queue_size=10)

    def process(self, keyboard_handler):
        """
        Collect data continuously until space is pressed.
        Returns: (timesteps, actions, user_choice)
            - user_choice: 's' to save, 'q' to discard
        """
        timesteps = []
        actions = []
        count = 0

        rate = rospy.Rate(self.args.frame_rate)
        last_warning = None

        print("\n\033[36m=== Recording started ===\033[0m")
        print("\033[33mPress SPACE to stop recording, then 's' to save or 'q' to discard\033[0m\n")

        while not rospy.is_shutdown():
            key = keyboard_handler.get_key()
            if key == " ":
                print(
                    "\n\033[33m\nRecording paused. Press 's' to SAVE or 'q' to DISCARD: \033[0m",
                    end="",
                    flush=True,
                )
                choice = keyboard_handler.wait_for_key(["s", "q", "S", "Q"])
                choice = choice.lower()
                print(choice)
                print("len(timesteps): ", len(timesteps))
                print("len(actions)  : ", len(actions))
                return timesteps, actions, choice

            success, result = self.get_collection_frame()
            if not success:
                warning = f"syn fail: {result}"
                if warning != last_warning:
                    print(warning)
                    last_warning = warning
                rate.sleep()
                continue

            count += 1
            (
                img_front,
                img_left,
                img_right,
                img_front_depth,
                img_left_depth,
                img_right_depth,
                follower_arm_left,
                follower_arm_right,
                follower_arm_left_pose,
                follower_arm_right_pose,
                leader_arm_left,
                leader_arm_right,
            ) = result

            observation = {
                "images": {
                    self.args.camera_names[0]: img_front,
                    self.args.camera_names[1]: img_left,
                    self.args.camera_names[2]: img_right,
                }
            }
            if self.args.use_depth_image:
                observation["images_depth"] = {
                    self.args.camera_names[0]: img_front_depth,
                    self.args.camera_names[1]: img_left_depth,
                    self.args.camera_names[2]: img_right_depth,
                }

            qpos = np.concatenate((np.array(follower_arm_left.position), np.array(follower_arm_right.position)), axis=0)
            qvel = np.concatenate((np.array(follower_arm_left.velocity), np.array(follower_arm_right.velocity)), axis=0)
            effort = np.concatenate((np.array(follower_arm_left.effort), np.array(follower_arm_right.effort)), axis=0)
            action = np.concatenate((np.array(leader_arm_left.position), np.array(leader_arm_right.position)), axis=0)
            eef_pose = self.build_follower_arm_pose(
                follower_arm_left_pose,
                follower_arm_right_pose,
                follower_arm_left,
                follower_arm_right,
            )

            zero_value_error = self._format_zero_value_failure(
                [
                    ("follower_arm_left.position", np.array(follower_arm_left.position)),
                    ("follower_arm_right.position", np.array(follower_arm_right.position)),
                    ("leader_arm_left.position", np.array(leader_arm_left.position)),
                    ("leader_arm_right.position", np.array(leader_arm_right.position)),
                ]
            )
            if zero_value_error:
                warning = f"arm data invalid: {zero_value_error}"
                if warning != last_warning:
                    print(warning)
                    last_warning = warning
                rate.sleep()
                continue

            last_warning = None
            observation["qpos"] = qpos
            observation["qvel"] = qvel
            observation["effort"] = effort
            observation["eef_pose"] = eef_pose

            if count == 1:
                timesteps.append(
                    dm_env.TimeStep(
                        step_type=dm_env.StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=observation,
                    )
                )
                continue

            timesteps.append(
                dm_env.TimeStep(
                    step_type=dm_env.StepType.MID,
                    reward=None,
                    discount=None,
                    observation=observation,
                )
            )
            actions.append(action)
            print("Frame data: ", count)
            if rospy.is_shutdown():
                exit(-1)
            rate.sleep()

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions, "q"

    def follower_arm_publish(self, left_joint_positions, right_joint_positions):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        joint_state_msg.position = left_joint_positions
        self.follower_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right_joint_positions
        self.follower_arm_right_publisher.publish(joint_state_msg)

    def follower_arm_pose_publish(self, left_pose, right_pose):
        pose_msg = PosCmd()
        pose_msg.x, pose_msg.y, pose_msg.z = left_pose[:3]
        pose_msg.roll, pose_msg.pitch, pose_msg.yaw = left_pose[3:6]
        pose_msg.gripper = left_pose[6]
        self.follower_arm_left_pose_publisher.publish(pose_msg)

        pose_msg.x, pose_msg.y, pose_msg.z = right_pose[:3]
        pose_msg.roll, pose_msg.pitch, pose_msg.yaw = right_pose[3:6]
        pose_msg.gripper = right_pose[6]
        self.follower_arm_right_pose_publisher.publish(pose_msg)

    def follower_arm_publish_continuous(self, left_joint_positions, right_joint_positions):
        rate = rospy.Rate(self.args.publish_rate)
        current_left_arm = None
        current_right_arm = None
        while not rospy.is_shutdown():
            if len(self.follower_left_arm_queue) != 0:
                current_left_arm = list(self.follower_left_arm_queue[-1].position)
            if len(self.follower_right_arm_queue) != 0:
                current_right_arm = list(self.follower_right_arm_queue[-1].position)
            if current_left_arm is None or current_right_arm is None:
                rate.sleep()
                continue
            break

        left_direction = [
            1 if left_joint_positions[i] - current_left_arm[i] > 0 else -1 for i in range(len(left_joint_positions))
        ]
        right_direction = [
            1 if right_joint_positions[i] - current_right_arm[i] > 0 else -1 for i in range(len(right_joint_positions))
        ]
        moving = True
        step = 0
        while moving and not rospy.is_shutdown():
            if self.follower_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left_joint_positions[i] - current_left_arm[i]) for i in range(len(left_joint_positions))]
            right_diff = [
                abs(right_joint_positions[i] - current_right_arm[i]) for i in range(len(right_joint_positions))
            ]
            moving = False
            for i in range(len(left_joint_positions)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    current_left_arm[i] = left_joint_positions[i]
                else:
                    current_left_arm[i] += left_direction[i] * self.args.arm_steps_length[i]
                    moving = True
            for i in range(len(right_joint_positions)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    current_right_arm[i] = right_joint_positions[i]
                else:
                    current_right_arm[i] += right_direction[i] * self.args.arm_steps_length[i]
                    moving = True
            self.follower_arm_publish(current_left_arm, current_right_arm)
            step += 1
            print("follower_arm_publish_continuous:", step)
            rate.sleep()


def get_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True

    while not rospy.is_shutdown():
        result = ros_operator.get_inference_frame()
        success, frame_or_error = result
        if not success:
            if print_flag:
                print(f"syn fail when get_ros_observation: {frame_or_error}")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            follower_arm_left,
            follower_arm_right,
            follower_arm_left_pose,
            follower_arm_right_pose,
        ) = frame_or_error
        return (
            img_front,
            img_left,
            img_right,
            follower_arm_left,
            follower_arm_right,
            follower_arm_left_pose,
            follower_arm_right_pose,
        )


def get_latest_ros_observation(args, ros_operator):
    rate = rospy.Rate(args.publish_rate)
    print_flag = True

    while not rospy.is_shutdown():
        success, frame_or_error = ros_operator.get_latest_inference_frame()
        if not success:
            if print_flag:
                print(f"latest obs unavailable: {frame_or_error}")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (
            img_front,
            img_left,
            img_right,
            img_front_depth,
            img_left_depth,
            img_right_depth,
            follower_arm_left,
            follower_arm_right,
            follower_arm_left_pose,
            follower_arm_right_pose,
        ) = frame_or_error
        return (
            img_front,
            img_left,
            img_right,
            follower_arm_left,
            follower_arm_right,
            follower_arm_left_pose,
            follower_arm_right_pose,
        )
