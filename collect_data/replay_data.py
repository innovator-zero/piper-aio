# coding=utf-8
import argparse
import os

import h5py
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from piper_msgs.msg import PosCmd


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
        eef_pose = root["/observations/eef_pose"][()]

    return qpos, action, eef_pose


def publish_joint_replay(
    joint_state_msg,
    timestamp,
    action,
    joint_left_publisher,
    joint_right_publisher,
):
    joint_state_msg.header.stamp = timestamp

    joint_state_msg.position = action[:7]
    joint_left_publisher.publish(joint_state_msg)

    joint_state_msg.position = action[7:]
    joint_right_publisher.publish(joint_state_msg)


def publish_eef_replay(left_publisher, right_publisher, eef_pose):
    left_msg = PosCmd()
    left_msg.x = eef_pose[0]
    left_msg.y = eef_pose[1]
    left_msg.z = eef_pose[2]
    left_msg.roll = eef_pose[3]
    left_msg.pitch = eef_pose[4]
    left_msg.yaw = eef_pose[5]
    left_msg.gripper = eef_pose[6]
    left_msg.mode1 = 0
    left_msg.mode2 = 0
    left_publisher.publish(left_msg)

    right_msg = PosCmd()
    right_msg.x = eef_pose[7]
    right_msg.y = eef_pose[8]
    right_msg.z = eef_pose[9]
    right_msg.roll = eef_pose[10]
    right_msg.pitch = eef_pose[11]
    right_msg.yaw = eef_pose[12]
    right_msg.gripper = eef_pose[13]
    right_msg.mode1 = 0
    right_msg.mode2 = 0
    right_publisher.publish(right_msg)


def main(args):
    rospy.init_node("replay_node")

    joint_left_publisher = rospy.Publisher(args.control_arm_left_topic, JointState, queue_size=10)
    joint_right_publisher = rospy.Publisher(args.control_arm_right_topic, JointState, queue_size=10)

    eef_left_publisher = rospy.Publisher(args.control_arm_left_pose_topic, PosCmd, queue_size=10)
    eef_right_publisher = rospy.Publisher(args.control_arm_right_pose_topic, PosCmd, queue_size=10)

    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    dataset_name = f"episode_{args.episode_idx}"
    qposs, actions, eef_poses = load_hdf5(dataset_dir, dataset_name)

    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

    rate = rospy.Rate(args.frame_rate)

    for index in range(len(actions)):
        if rospy.is_shutdown():
            break

        if args.replay_mode == "joint":
            publish_joint_replay(
                joint_state_msg,
                rospy.Time.now(),
                actions[index],
                joint_left_publisher,
                joint_right_publisher,
            )
            print(
                "joint replay",
                "left_action:",
                np.round(actions[index][:7], 4),
                "right_action:",
                np.round(actions[index][7:], 4),
            )
        else:
            publish_eef_replay(
                eef_left_publisher,
                eef_right_publisher,
                eef_poses[index],
            )
            print(
                "eef replay",
                "left_eef:",
                np.round(eef_poses[index][:7], 4),
                "right_eef:",
                np.round(eef_poses[index][7:], 4),
            )

        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Dataset dir.",
        default="",
        required=False,
    )
    parser.add_argument(
        "--task_name",
        action="store",
        type=str,
        help="Task name.",
        default="test",
        required=False,
    )
    parser.add_argument(
        "--episode_idx",
        action="store",
        type=int,
        help="Episode index.",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--replay_mode",
        action="store",
        type=str,
        choices=["joint", "eef"],
        default="joint",
        help="Replay using joint commands or end-effector pose commands.",
        required=False,
    )
    parser.add_argument(
        "--control_arm_left_topic",
        action="store",
        type=str,
        help="control_arm_left_topic",
        default="/leader/joint_left",
        required=False,
    )
    parser.add_argument(
        "--control_arm_right_topic",
        action="store",
        type=str,
        help="control_arm_right_topic",
        default="/leader/joint_right",
        required=False,
    )
    parser.add_argument(
        "--control_arm_left_pose_topic",
        action="store",
        type=str,
        help="control_arm_left_pose_topic",
        default="/follower/pos_cmd_left",
        required=False,
    )
    parser.add_argument(
        "--control_arm_right_pose_topic",
        action="store",
        type=str,
        help="control_arm_right_pose_topic",
        default="/follower/pos_cmd_right",
        required=False,
    )
    parser.add_argument(
        "--frame_rate",
        action="store",
        type=int,
        help="frame_rate",
        default=30,
        required=False,
    )

    main(parser.parse_args())
