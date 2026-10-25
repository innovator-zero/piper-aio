# coding=utf-8
import argparse
import os
from dataclasses import dataclass

import h5py
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
FRAME_RATE = 30
JPEG_QUALITY = 90
WEB_PORT = 9090
GRPC_PORT = 9876


@dataclass
class EpisodeData:
    qpos: np.ndarray
    action: np.ndarray
    eef_pose: np.ndarray
    image_dict: dict[str, np.ndarray]
    depth_image_dict: dict[str, np.ndarray]
    collect: np.ndarray | None = None


@dataclass
class EpisodeJob:
    dataset_name: str
    dataset_path: str


@dataclass
class EpisodeSpec:
    dataset_name: str
    dataset_path: str
    cam_names: list[str]
    has_depth: bool


def _decode_collect_label(label):
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def _arm_slice(arm_name):
    if arm_name == "left":
        return slice(0, 7)
    if arm_name == "right":
        return slice(7, 14)
    raise ValueError(f"Unsupported arm: {arm_name}")


def _join_path(prefix, *parts):
    suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
    if prefix:
        return f"{prefix.rstrip('/')}/{suffix}"
    return f"/{suffix}"


def _dataset_name_from_path(dataset_path):
    return os.path.splitext(os.path.basename(dataset_path))[0]


def load_hdf5(dataset_path):
    dataset_path = os.path.abspath(dataset_path)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
        eef_pose = root["/observations/eef_pose"][()]
        image_dict = {
            cam_name: root[f"/observations/images/{cam_name}"][()]
            for cam_name in root["/observations/images"].keys()
        }
        depth_image_dict = {}
        if "/observations/images_depth" in root:
            depth_image_dict = {
                cam_name: root[f"/observations/images_depth/{cam_name}"][()]
                for cam_name in root["/observations/images_depth"].keys()
            }
        collect = root["/collect"][()] if "/collect" in root else None

    return EpisodeData(
        qpos=qpos,
        action=action,
        eef_pose=eef_pose,
        image_dict=image_dict,
        depth_image_dict=depth_image_dict,
        collect=collect,
    )


def read_episode_spec(job):
    with h5py.File(job.dataset_path, "r") as root:
        cam_names = list(root["/observations/images"].keys())
        has_depth = "/observations/images_depth" in root

    return EpisodeSpec(
        dataset_name=job.dataset_name,
        dataset_path=job.dataset_path,
        cam_names=cam_names,
        has_depth=has_depth,
    )


def _collect_labels(root):
    if "/collect" not in root:
        return []
    return [_decode_collect_label(label) for label in root["/collect"][()]]


def _job_has_collect_label(job, collect_label):
    if collect_label is None:
        return True

    with h5py.File(job.dataset_path, "r") as root:
        return collect_label in _collect_labels(root)


def _filter_jobs_by_collect_label(jobs, collect_label):
    if collect_label is None:
        return jobs

    filtered_jobs = []
    for job in jobs:
        if _job_has_collect_label(job, collect_label):
            filtered_jobs.append(job)
        else:
            print(f"[{job.dataset_name}] Skipping: no frames with collect label '{collect_label}'")

    if not filtered_jobs:
        raise ValueError(f"No frames found with collect label '{collect_label}'")

    return filtered_jobs


def _validate_episode_shapes(episode):
    num_frames = episode.action.shape[0]
    frame_aligned_arrays = {
        "qpos": episode.qpos,
        "action": episode.action,
        "eef_pose": episode.eef_pose,
    }

    for name, values in frame_aligned_arrays.items():
        if values.shape[0] != num_frames:
            raise ValueError(f"{name} frame count {values.shape[0]} does not match action frame count {num_frames}")
        if values.ndim != 2 or values.shape[1] != 14:
            raise ValueError(f"{name} must have shape (T, 14), got {values.shape}")

    for cam_name, images in episode.image_dict.items():
        if images.shape[0] != num_frames:
            raise ValueError(f"Camera {cam_name} frame count {images.shape[0]} does not match action frame count {num_frames}")
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Camera {cam_name} must have shape (T, H, W, 3), got {images.shape}")

    for cam_name, images in episode.depth_image_dict.items():
        if images.shape[0] != num_frames:
            raise ValueError(f"Depth camera {cam_name} frame count {images.shape[0]} does not match action frame count {num_frames}")
        if images.ndim != 3:
            raise ValueError(f"Depth camera {cam_name} must have shape (T, H, W), got {images.shape}")

    if episode.collect is not None and episode.collect.shape[0] != num_frames:
        raise ValueError(f"collect frame count {episode.collect.shape[0]} does not match action frame count {num_frames}")


def _frame_indices_for_collect_label(episode, collect_label):
    if collect_label is None:
        return np.arange(episode.action.shape[0])
    if episode.collect is None:
        return np.array([], dtype=int)

    labels = np.array([_decode_collect_label(label) for label in episode.collect])
    return np.flatnonzero(labels == collect_label)


def _episode_layout(rrb, name, prefix, cam_names, has_depth):
    camera_views = [
        rrb.Spatial2DView(origin=_join_path(prefix, "cameras", "rgb", cam_name), name=f"rgb/{cam_name}")
        for cam_name in cam_names
    ]
    if has_depth:
        camera_views.extend(
            rrb.Spatial2DView(origin=_join_path(prefix, "cameras", "depth", cam_name), name=f"depth/{cam_name}")
            for cam_name in cam_names
        )

    return rrb.Grid(
        rrb.Grid(*camera_views, grid_columns=max(1, min(3, len(camera_views))), name="Cameras"),
        rrb.Grid(
            rrb.TimeSeriesView(origin=_join_path(prefix, "joints", "left", "qpos"), name="left/qpos"),
            rrb.TimeSeriesView(origin=_join_path(prefix, "joints", "right", "qpos"), name="right/qpos"),
            rrb.TimeSeriesView(origin=_join_path(prefix, "joints", "left", "action"), name="left/action"),
            rrb.TimeSeriesView(origin=_join_path(prefix, "joints", "right", "action"), name="right/action"),
            rrb.TimeSeriesView(origin=_join_path(prefix, "eef", "left", "pose"), name="left/eef_pose"),
            rrb.TimeSeriesView(origin=_join_path(prefix, "eef", "right", "pose"), name="right/eef_pose"),
            rrb.TextLogView(origin=_join_path(prefix, "collect"), name="Collect Labels"),
            rrb.TextDocumentView(origin=_join_path(prefix, "summary"), name="Episode"),
            grid_columns=2,
            name="Signals",
        ),
        row_shares=[3, 2],
        name=name,
    )


def _make_blueprint(specs):
    import rerun.blueprint as rrb

    combined = len(specs) > 1
    layouts = [
        _episode_layout(
            rrb,
            spec.dataset_name,
            _join_path("/episodes", spec.dataset_name) if combined else "",
            spec.cam_names,
            spec.has_depth,
        )
        for spec in specs
    ]

    if combined and len(layouts) > 1 and hasattr(rrb, "Tabs"):
        root = rrb.Tabs(*layouts, name="Episodes")
    elif combined and len(layouts) > 1:
        root = rrb.Grid(*layouts, grid_columns=1, name="Episodes")
    else:
        root = layouts[0]

    return rrb.Blueprint(
        root,
        rrb.TimePanel(expanded=True),
        collapse_panels=True,
    )


def _series_colors(num_series):
    palette = [
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ]
    return palette[:num_series]


def _log_series_metadata(recording, path, names):
    import rerun as rr

    recording.log(
        path,
        rr.SeriesLines(names=names, colors=_series_colors(len(names))),
        static=True,
    )


def _log_static_metadata(recording, job, episode, prefix, frame_indices, collect_label):
    import rerun as rr

    lines = [
        f"# {job.dataset_name}",
        "",
        f"- path: `{job.dataset_path}`",
        f"- frames: `{episode.action.shape[0]}`",
        f"- visualized frames: `{len(frame_indices)}`",
        f"- frame_rate: `{FRAME_RATE}` Hz",
        f"- cameras: `{', '.join(episode.image_dict.keys())}`",
        f"- qpos shape: `{episode.qpos.shape}`",
        f"- action shape: `{episode.action.shape}`",
        f"- eef_pose shape: `{episode.eef_pose.shape}`",
    ]
    if collect_label is not None:
        lines.append(f"- collect filter: `{collect_label}`")
    if episode.collect is not None:
        labels, counts = np.unique([_decode_collect_label(label) for label in episode.collect], return_counts=True)
        label_summary = ", ".join(f"{label}:{count}" for label, count in zip(labels, counts))
        lines.append(f"- collect labels: `{label_summary}`")

    recording.log(
        _join_path(prefix, "summary"),
        rr.TextDocument("\n".join(lines), media_type="text/markdown"),
        static=True,
    )

    for arm_name in ("left", "right"):
        _log_series_metadata(recording, _join_path(prefix, "joints", arm_name, "qpos"), ARM_STATE_NAMES)
        _log_series_metadata(recording, _join_path(prefix, "joints", arm_name, "action"), ARM_STATE_NAMES)
        _log_series_metadata(recording, _join_path(prefix, "eef", arm_name, "pose"), EEF_STATE_NAMES)


def _maybe_compressed_image(image):
    import rerun as rr

    rerun_image = rr.Image(image)
    if JPEG_QUALITY <= 0:
        return rerun_image
    return rerun_image.compress(jpeg_quality=JPEG_QUALITY)


def _call_with_supported_kwargs(function, **kwargs):
    import inspect

    signature = inspect.signature(function)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if accepts_kwargs:
        return function(**{key: value for key, value in kwargs.items() if value is not None})

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }
    return function(**supported_kwargs)


def _start_web_viewer(recording, rr, blueprint):
    if hasattr(recording, "serve_web"):
        _call_with_supported_kwargs(
            recording.serve_web,
            open_browser=True,
            web_port=WEB_PORT,
            grpc_port=GRPC_PORT,
            default_blueprint=blueprint,
        )
        print(f"Serving Rerun Web Viewer at http://127.0.0.1:{WEB_PORT}")
        return

    if hasattr(recording, "serve_grpc") and hasattr(rr, "serve_web_viewer"):
        try:
            _call_with_supported_kwargs(recording.serve_grpc, grpc_port=GRPC_PORT, port=GRPC_PORT)
            _call_with_supported_kwargs(
                rr.serve_web_viewer,
                open_browser=True,
                web_port=WEB_PORT,
                port=WEB_PORT,
                grpc_port=GRPC_PORT,
                connect_to=f"127.0.0.1:{GRPC_PORT}",
                default_blueprint=blueprint,
            )
        except TypeError as exc:
            raise RuntimeError(
                "This rerun-sdk version has partial Web Viewer APIs, but their signatures are not compatible. "
                "Upgrade with: pip install -U rerun-sdk"
            ) from exc
        print(f"Serving Rerun Web Viewer at http://127.0.0.1:{WEB_PORT}")
        return

    version = getattr(rr, "__version__", "unknown")
    raise RuntimeError(
        f"rerun-sdk {version} does not support Web Viewer serving in this script. "
        "Upgrade with: pip install -U rerun-sdk."
    )


def _log_frame(recording, episode, frame_idx, elapsed_seconds, prefix):
    import rerun as rr

    recording.set_time("frame", sequence=frame_idx)
    recording.set_time("time", duration=elapsed_seconds)

    for cam_name, images in episode.image_dict.items():
        recording.log(
            _join_path(prefix, "cameras", "rgb", cam_name),
            _maybe_compressed_image(images[frame_idx]),
        )
    for cam_name, images in episode.depth_image_dict.items():
        recording.log(
            _join_path(prefix, "cameras", "depth", cam_name),
            rr.DepthImage(images[frame_idx], meter=1000.0),
        )

    for arm_name in ("left", "right"):
        arm_slice = _arm_slice(arm_name)
        recording.log(_join_path(prefix, "joints", arm_name, "qpos"), rr.Scalars(episode.qpos[frame_idx, arm_slice]))
        recording.log(_join_path(prefix, "joints", arm_name, "action"), rr.Scalars(episode.action[frame_idx, arm_slice]))
        recording.log(_join_path(prefix, "eef", arm_name, "pose"), rr.Scalars(episode.eef_pose[frame_idx, arm_slice]))

    if episode.collect is not None:
        recording.log(_join_path(prefix, "collect"), rr.TextLog(_decode_collect_label(episode.collect[frame_idx])))


def _log_episode(recording, job, episode, prefix, collect_label):
    _validate_episode_shapes(episode)
    frame_indices = _frame_indices_for_collect_label(episode, collect_label)
    if len(frame_indices) == 0:
        print(f"[{job.dataset_name}] Skipping: no frames with collect label '{collect_label}'")
        return 0

    _log_static_metadata(recording, job, episode, prefix, frame_indices, collect_label)

    dt = 1.0 / FRAME_RATE
    num_frames = len(frame_indices)
    for log_idx, frame_idx in enumerate(frame_indices):
        _log_frame(
            recording=recording,
            episode=episode,
            frame_idx=frame_idx,
            elapsed_seconds=frame_idx * dt,
            prefix=prefix,
        )
        if log_idx % 100 == 0 or log_idx == num_frames - 1:
            print(f"[{job.dataset_name}] Logged {log_idx + 1}/{num_frames} matching frames to Rerun")

    return num_frames


def _keep_web_viewer_alive():
    print("Keep this process running while viewing in the browser. Press Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopped Rerun Web Viewer.")


def visualize_live(jobs, collect_label):
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError("rerun-sdk is required. Install it with: pip install rerun-sdk") from exc

    jobs = _filter_jobs_by_collect_label(jobs, collect_label)
    combined = len(jobs) > 1
    specs = [read_episode_spec(job) for job in jobs]
    blueprint = _make_blueprint(specs)

    recording = rr.RecordingStream("piper_episode_visualizer")
    _start_web_viewer(recording, rr, blueprint)

    for job in jobs:
        episode = load_hdf5(job.dataset_path)
        print(f"Loaded {job.dataset_path}")
        _log_episode(
            recording=recording,
            job=job,
            episode=episode,
            prefix=_join_path("/episodes", job.dataset_name) if combined else "",
            collect_label=collect_label,
        )

    recording.flush()
    _keep_web_viewer_alive()


def _resolve_jobs(dataset_path):
    dataset_path = os.path.abspath(dataset_path)

    if os.path.isfile(dataset_path):
        if not dataset_path.endswith(".hdf5"):
            raise ValueError(f"--dataset_path must point to a .hdf5 file or a directory, got {dataset_path}")
        return [EpisodeJob(_dataset_name_from_path(dataset_path), dataset_path)]

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"--dataset_path does not exist: {dataset_path}")

    hdf5_paths = sorted(
        os.path.join(dataset_path, name)
        for name in os.listdir(dataset_path)
        if name.endswith(".hdf5") and os.path.isfile(os.path.join(dataset_path, name))
    )
    if not hdf5_paths:
        raise FileNotFoundError(f"No .hdf5 files found directly under {dataset_path}")

    return [EpisodeJob(_dataset_name_from_path(path), path) for path in hdf5_paths]


def main(args):
    visualize_live(_resolve_jobs(args.dataset_path), args.collect_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        help="Path to a .hdf5 episode file, or a directory containing .hdf5 files.",
        required=True,
    )
    parser.add_argument(
        "--collect_label",
        action="store",
        type=str,
        help="Only visualize frames whose /collect label exactly matches this value.",
        default=None,
        required=False,
    )

    try:
        main(parser.parse_args())
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
