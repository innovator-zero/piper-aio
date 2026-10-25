import os
import sys
import shutil
import tempfile
from datetime import datetime
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from contextlib import contextmanager
from typing import List

from features import FEATURES
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

INSTRUCTION = "Example"
DEFAULT_COLLECT = "teleop"


def get_arm_slice(zero_arm: str) -> slice | None:
    if zero_arm == "left":
        return slice(0, 7)
    if zero_arm == "right":
        return slice(7, 14)
    return None


def _decode_collect_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_collect_labels(hdf5_file, num_frames: int) -> list[str]:
    if "collect" in hdf5_file:
        collect = [_decode_collect_value(value) for value in np.array(hdf5_file["collect"])]
    else:
        collect = [DEFAULT_COLLECT] * num_frames

    if len(collect) != num_frames:
        raise ValueError(f"collect frame count {len(collect)} does not match action frame count {num_frames}")

    return collect


def find_collect_label_ranges(collect: list[str], collect_label: str | None = None) -> list[tuple[int, int]]:
    """Return source frame ranges as [start, stop) spans for an optional collect label."""

    if collect_label is None:
        return [(0, len(collect))]

    ranges = []
    start = None
    for idx, label in enumerate(collect):
        if label == collect_label:
            if start is None:
                start = idx
        elif start is not None:
            ranges.append((start, idx))
            start = None

    if start is not None:
        ranges.append((start, len(collect)))

    return ranges


@contextmanager
def suppress_stderr_on_success():
    """Hide noisy native stderr output from video encoding, but replay it if an error occurs."""
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")

    try:
        os.dup2(tmp.fileno(), stderr_fd)
        try:
            yield
        except Exception:
            os.dup2(saved_stderr_fd, stderr_fd)
            tmp.seek(0)
            stderr_output = tmp.read().decode("utf-8", errors="replace").strip()
            if stderr_output:
                print("Captured ffmpeg/libav stderr:", file=sys.stderr)
                print(stderr_output, file=sys.stderr)
            raise
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)
        tmp.close()


def _load_hdf5_arrays(episode_path: str | Path) -> dict:
    with h5py.File(episode_path) as f:
        actions = np.array(f["action"])
        data = {
            "state_images_cam_high": np.array(f["observations/images/cam_high"]),
            "state_images_cam_left_wrist": np.array(f["observations/images/cam_left_wrist"]),
            "state_images_cam_right_wrist": np.array(f["observations/images/cam_right_wrist"]),
            "state_qpos": np.array(f["observations/qpos"]),
            "state_eef_pose": np.array(f["observations/eef_pose"]),
            "actions": actions,
            "collect": _read_collect_labels(f, actions.shape[0]),
        }

    if not (
        data["state_images_cam_high"].shape[0]
        == data["state_images_cam_left_wrist"].shape[0]
        == data["state_images_cam_right_wrist"].shape[0]
        == data["state_qpos"].shape[0]
        == data["state_eef_pose"].shape[0]
        == data["actions"].shape[0]
        == len(data["collect"])
    ):
        raise ValueError("Mismatch in dataset lengths.")

    if data["state_qpos"].shape[1] != 14:
        raise ValueError(f"Expected qpos to have 14 dimensions, but got {data['state_qpos'].shape[1]}.")

    if data["state_eef_pose"].shape[1] != 14:
        raise ValueError(f"Expected eef_pose to have 14 dimensions, but got {data['state_eef_pose'].shape[1]}.")

    if data["actions"].shape[1] != 14:
        raise ValueError(f"Expected actions to have 14 dimensions, but got {data['actions'].shape[1]}.")

    return data


def _apply_zero_arm(data: dict, zero_arm: str) -> None:
    arm_slice = get_arm_slice(zero_arm)
    if arm_slice is None:
        return

    data["state_qpos"][:, arm_slice] = 0.0
    data["state_eef_pose"][:, arm_slice] = 0.0
    data["actions"][:, arm_slice] = 0.0


def _build_frames_from_arrays(
    data: dict,
    cut_head: bool = False,
    cut_tail: bool = False,
    frame_range: tuple[int, int] | None = None,
) -> tuple[list, dict]:
    total_file_frames = data["state_qpos"].shape[0]
    if frame_range is None:
        segment_start = 0
        segment_stop = total_file_frames
    else:
        segment_start, segment_stop = frame_range

    if not (0 <= segment_start < segment_stop <= total_file_frames):
        raise ValueError(
            f"Invalid frame range [{segment_start}, {segment_stop}) for dataset with {total_file_frames} frames."
        )

    segment_len = segment_stop - segment_start
    state_qpos = data["state_qpos"][segment_start:segment_stop]

    # [Optional] We skip the first few still steps
    EPS = 5e-3
    # Get the idx of the first qpos whose delta exceeds the threshold
    if cut_head:
        qpos_delta = np.abs(state_qpos - state_qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")
    else:
        first_idx = 1

    if cut_tail:
        qpos_delta = np.abs(state_qpos - state_qpos[-1:])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            last_idx = indices[-1]
        else:
            raise ValueError("Found no qpos that exceeds the threshold.")
    else:
        last_idx = segment_len - 1

    local_st = first_idx - 1
    local_end = min(last_idx, segment_len - 1)

    if local_end <= local_st:
        raise ValueError("Need at least 2 frames after trimming to build next-frame eef actions.")

    st = segment_start + local_st
    end = segment_start + local_end

    frames = [
        {
            "observation.state": data["state_qpos"][i].reshape(-1),
            "observation.eef_pose": data["state_eef_pose"][i].reshape(-1),
            "observation.images.cam_high": data["state_images_cam_high"][i],
            "observation.images.cam_left_wrist": data["state_images_cam_left_wrist"][i],
            "observation.images.cam_right_wrist": data["state_images_cam_right_wrist"][i],
            "action": data["actions"][i].reshape(-1),
            "eef_action": data["state_eef_pose"][i + 1].reshape(-1),
            "collect": data["collect"][i],
        }
        for i in range(st, end)
    ]

    cut_info = {
        "total_frames": int(segment_len),
        "file_total_frames": int(total_file_frames),
        "source_start_index": int(segment_start),
        "source_stop_index_exclusive": int(segment_stop),
        "start_index": int(st),
        "end_index_inclusive": int(end),
        "kept_frames": int(len(frames)),
        "cut_head_frames": int(local_st),
        "cut_tail_frames": int(segment_len - 1 - local_end),
        "cut_head_enabled": cut_head,
        "cut_tail_enabled": cut_tail,
    }

    return frames, cut_info


def load_hdf5_dataset(
    episode_path: str | Path,
    cut_head: bool = False,
    cut_tail: bool = False,
    zero_arm: str = "none",
    frame_range: tuple[int, int] | None = None,
) -> tuple[list, dict]:
    """Load hdf5 dataset and return per-frame observations with aligned actions."""

    data = _load_hdf5_arrays(episode_path)
    _apply_zero_arm(data, zero_arm)
    return _build_frames_from_arrays(data, cut_head=cut_head, cut_tail=cut_tail, frame_range=frame_range)


def main(
    src_path: str,
    tgt_path: str,
    repo_ids: List[str],
    save_repoid: str,
    cut_head: bool = False,
    cut_tail: bool = False,
    zero_arm: str = "none",
    collect_label: str | None = None,
):
    target_dir = f"{tgt_path}/{save_repoid}"
    error_log_path = Path(f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    if os.path.exists(target_dir):
        print(f"[Init] Target folder exists, deleting: {target_dir}")
        shutil.rmtree(target_dir)
        print(f"[Init] Deleted: {target_dir}")

    print(f"[Init] Creating LeRobot dataset at: {target_dir}")
    print(f"[Init] Error log file: {error_log_path}")
    if collect_label is not None:
        print(f"[Init] Converting contiguous collect == {collect_label!r} segments as episodes")

    dataset = LeRobotDataset.create(
        repo_id=f"{tgt_path}/{save_repoid}",
        fps=30,
        robot_type="agilex",
        features=FEATURES,
        image_writer_processes=24,
        image_writer_threads=12,
        video_backend="torchcodec",
    )
    exclude_files = [
        # list of hdf5 files to exclude
    ]

    for repo_id in repo_ids:
        hdf5_source_path = os.path.join(src_path, repo_id)
        hdf5_files = []
        repo_total_frames = 0
        repo_kept_frames = 0
        repo_head_cut_frames = 0
        repo_tail_cut_frames = 0
        repo_success_episodes = 0
        repo_candidate_episodes = 0

        if os.path.exists(hdf5_source_path):
            hdf5_files = sorted([f.as_posix() for f in Path(hdf5_source_path).glob("*.hdf5")])
        else:
            print(f"[Skip] Source folder not found: {hdf5_source_path}")
            continue

        for exclude_file in exclude_files:
            if exclude_file in hdf5_files:
                hdf5_files.remove(exclude_file)

        print(f"[Repo] {repo_id}: {len(hdf5_files)} episode files to process")

        for hdf5_file in tqdm(hdf5_files, total=len(hdf5_files), desc="Processing episodes"):
            try:
                data = _load_hdf5_arrays(hdf5_file)
                _apply_zero_arm(data, zero_arm)
                frame_ranges = find_collect_label_ranges(data["collect"], collect_label)
            except Exception as e:
                print(f"[Error] Failed to process episode file: {hdf5_file}")
                print(f"[Error] {e}")
                with open(error_log_path, "a") as f:
                    f.write(f"Error processing episode file {hdf5_file}: {e}\n")
                continue

            if not frame_ranges:
                print(f"[Skip] {Path(hdf5_file).name}: no frames with collect == {collect_label!r}")
                continue

            repo_candidate_episodes += len(frame_ranges)

            for segment_idx, frame_range in enumerate(frame_ranges):
                try:
                    frames, cut_info = _build_frames_from_arrays(
                        data,
                        cut_head=cut_head,
                        cut_tail=cut_tail,
                        frame_range=frame_range,
                    )

                    for frame in frames:
                        dataset.add_frame(frame, task=INSTRUCTION)

                    with suppress_stderr_on_success():
                        dataset.save_episode()

                    segment_text = ""
                    if collect_label is not None:
                        segment_text = (
                            f" segment={segment_idx} "
                            f"source=[{cut_info['source_start_index']},{cut_info['source_stop_index_exclusive']})"
                        )

                    print(
                        "[Episode] "
                        f"{Path(hdf5_file).name}:"
                        f"{segment_text} "
                        f"head_cut={cut_info['cut_head_frames']} "
                        f"tail_cut={cut_info['cut_tail_frames']} "
                        f"kept={cut_info['kept_frames']}/{cut_info['total_frames']} "
                        f"(range: {cut_info['start_index']}:{cut_info['end_index_inclusive']} inclusive, "
                        f"cut_head={cut_info['cut_head_enabled']}, "
                        f"cut_tail={cut_info['cut_tail_enabled']})"
                    )
                    repo_total_frames += cut_info["total_frames"]
                    repo_kept_frames += cut_info["kept_frames"]
                    repo_head_cut_frames += cut_info["cut_head_frames"]
                    repo_tail_cut_frames += cut_info["cut_tail_frames"]
                    repo_success_episodes += 1
                except Exception as e:
                    print(
                        f"[Error] Failed to process segment {segment_idx} "
                        f"[{frame_range[0]},{frame_range[1]}) from: {hdf5_file}"
                    )
                    print(f"[Error] {e}")
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"Error processing segment {segment_idx} [{frame_range[0]},{frame_range[1]}) "
                            f"from {hdf5_file}: {e}\n"
                        )

        if repo_success_episodes > 0:
            avg_kept_frames = repo_kept_frames / repo_success_episodes
            episode_denominator = len(hdf5_files) if collect_label is None else repo_candidate_episodes
            print(
                "[Repo Summary] "
                f"{repo_id}: "
                f"episodes={repo_success_episodes}/{episode_denominator} "
                f"head_cut_total={repo_head_cut_frames} "
                f"tail_cut_total={repo_tail_cut_frames} "
                f"kept_total={repo_kept_frames}/{repo_total_frames} "
                f"avg_kept={avg_kept_frames:.1f}"
            )
        else:
            print(f"[Repo Summary] {repo_id}: no episode converted successfully")

    print("[Done] Conversion finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=False,
        default="",
        help="src path for the original dataset",
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=False,
        default="",
        help="tgt path to save the converted dataset",
    )
    parser.add_argument("--repo_ids", nargs="+", required=True, help="repo ids")
    parser.add_argument(
        "--save_repoid", type=str, default=None, help="save repoid, default as the first repo_id in the list"
    )
    parser.add_argument(
        "--cut_head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cut head of the dataset",
    )
    parser.add_argument(
        "--cut_tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cut tail of the dataset",
    )
    parser.add_argument(
        "--zero_arm",
        choices=["none", "left", "right"],
        default="none",
        help="Overwrite the selected arm dimensions with zeros during conversion.",
    )
    parser.add_argument(
        "--collect_label",
        type=str,
        default=None,
        help="Only convert contiguous segments whose /collect label exactly matches this value.",
    )
    args = parser.parse_args()
    repo_ids = args.repo_ids
    save_repoid = args.save_repoid
    if save_repoid is None:
        save_repoid = repo_ids[0]

    main(
        src_path=args.src_path,
        tgt_path=args.tgt_path,
        repo_ids=repo_ids,
        save_repoid=save_repoid,
        cut_head=args.cut_head,
        cut_tail=args.cut_tail,
        zero_arm=args.zero_arm,
        collect_label=args.collect_label,
    )
