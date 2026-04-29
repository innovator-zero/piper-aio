# coding=utf-8
import argparse
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

ARM_STATE_NAMES = [
    "joint0",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "gripper",
]
EEF_STATE_NAMES = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
]


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
        eef_pose = root["/observations/eef_pose"][()]
        image_dict = {
            cam_name: root[f"/observations/images/{cam_name}"][()] for cam_name in root["/observations/images"].keys()
        }

    return qpos, action, eef_pose, image_dict


def save_video(image_dict, action, dt, video_path):
    cam_names = list(image_dict.keys())
    merged_video = np.concatenate([image_dict[cam_name] for cam_name in cam_names], axis=2)

    n_frames, height, width, _ = merged_video.shape
    fps = int(round(1 / dt))
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_idx in range(n_frames):
        image = merged_video[frame_idx][:, :, [2, 1, 0]]
        writer.write(image)
        cv2.imshow("images", image)
        cv2.waitKey(max(1, int(dt * 1000)))
        print(
            "frame:",
            frame_idx,
            "left_action:",
            np.round(action[frame_idx][:7], 3),
            "right_action:",
            np.round(action[frame_idx][7:], 3),
        )

    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved video to: {video_path}")


def _plot_series(series_list, labels, names, plot_path, title_prefix, colors):
    values_list = [np.asarray(series) for series in series_list]
    _, num_dim = values_list[0].shape
    fig, axes = plt.subplots(num_dim, 1, figsize=(10, 2.2 * num_dim), squeeze=False)
    axes = axes[:, 0]

    for dim_idx in range(num_dim):
        ax = axes[dim_idx]
        for values, label, color in zip(values_list, labels, colors):
            ax.plot(values[:, dim_idx], label=label, color=color)
        ax.set_title(f"{title_prefix} {dim_idx}: {names[dim_idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved plot to: {plot_path}")


def visualize_joints(qpos, action, plot_path):
    joint_names = [name + "_left" for name in ARM_STATE_NAMES] + [name + "_right" for name in ARM_STATE_NAMES]
    _plot_series(
        [qpos, action],
        ["qpos", "action"],
        joint_names,
        plot_path,
        "Joint",
        ["orangered", "royalblue"],
    )


def visualize_eef(eef_pose, plot_path):
    eef_names = [name + "_left" for name in EEF_STATE_NAMES] + [name + "_right" for name in EEF_STATE_NAMES]
    _plot_series([eef_pose], ["eef_pose"], eef_names, plot_path, "EEF", ["royalblue"])


def main(args):
    dataset_dir = os.path.join(args.dataset_dir, args.task_name)
    dataset_name = f"episode_{args.episode_idx}"
    dt = 1.0 / args.frame_rate

    qpos, action, eef_pose, image_dict = load_hdf5(dataset_dir, dataset_name)
    print("hdf5 loaded!!")

    save_video(
        image_dict,
        action,
        dt,
        video_path=os.path.join(dataset_dir, dataset_name + "_video.mp4"),
    )
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + "_qpos.png"))
    visualize_eef(eef_pose, plot_path=os.path.join(dataset_dir, dataset_name + "_eef_pose.png"))


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
        "--frame_rate",
        action="store",
        type=int,
        help="frame_rate used when replaying video",
        default=30,
        required=False,
    )

    main(parser.parse_args())
