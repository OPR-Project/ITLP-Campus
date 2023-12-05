import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

matplotlib.use("Agg")


def parse_args() -> Tuple[Path, Path, Path, Path, Path, float]:
    """Parse input CLI arguments.

    Raises:
        ValueError: If the given input directory does not exist.

    Returns:
        Path: Front camera directory path.
        Path: Back camera directory path.
        Path: LIDAR directory path.
        Path: Poses file path.
        Path: Output directory which is going to be created.
        float: Distance threshold for subsampling.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=Path, required=True, help="Path to the unpacked data.")
    parser.add_argument("-o", "--out_dir", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument(
        "--dist_threshold",
        type=float,
        default=5.0,
        required=False,
        help="Distance threshold for subsampling.",
    )
    args = parser.parse_args()

    input_dir = Path(args.dir)
    if not input_dir.exists():
        raise ValueError("Given input directory does not exist.")

    front_cam_dir = input_dir / "front_cam"
    back_cam_dir = input_dir / "back_cam"
    lidar_dir = input_dir / "lidar"
    poses_file = input_dir / "trajectory.tum"

    output_dir = Path(args.out_dir)

    dist_threshold = args.dist_threshold

    return front_cam_dir, back_cam_dir, lidar_dir, poses_file, output_dir, dist_threshold


def check_in_test_set(
    northing: float,
    easting: float,
    test_boundary_points: List[Tuple[float, float]],
    boundary_width: Tuple[float, float],
) -> bool:
    """Checks whether the given point is in the test set.

    Args:
        northing (float): x coordinate of the point.
        easting (float): y coordinate of the point.
        test_boundary_points (List[Tuple[float, float]]): List of boundary points of the test set.
        boundary_width (Tuple[float, float]): Boundary width.

    Returns:
        bool: Whether the given point is in the test set.
    """
    in_test_set = False
    x_width, y_width = boundary_width
    for boundary_point in test_boundary_points:
        if (
            boundary_point[0] - x_width < northing < boundary_point[0] + x_width
            and boundary_point[1] - y_width < easting < boundary_point[1] + y_width
        ):
            in_test_set = True
            break
    return in_test_set


def read_poses_tum(filepath: Path) -> DataFrame:
    """Read csv file with ground-truth poses.

    Args:
        filepath (Path): The path to the file.

    Returns:
        DataFrame: Pandas DataFrame with following columns: `timestamp`, `tx`, `ty`, `tz`,
        `qx`, `qy`, `qz`, `qw`.
    """
    colnames = ["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    dtypes_dict = {
        "timestamp": np.float64,
        "tx": np.float64,
        "ty": np.float64,
        "tz": np.float64,
        "qx": np.float64,
        "qy": np.float64,
        "qz": np.float64,
        "qw": np.float64,
    }
    df = pd.read_csv(
        filepath, sep=" ", header=None, names=colnames, dtype=dtypes_dict, skiprows=1
    )  # skip first pose cause it may be incorrect
    if filepath.name == "spring-day-1p.tum":  # kek
        t0 = float(df["timestamp"].to_numpy()[0])
        max_diff = 3620.0
        diffs = df["timestamp"].to_numpy() - t0
        mask = diffs <= max_diff
        df = df[mask]
    if filepath.name == "campus_spring_night_1p.tum":  # kek2
        t0 = float(df["timestamp"].to_numpy()[0])
        max_diff = 3593.0
        diffs = df["timestamp"].to_numpy() - t0
        mask = diffs <= max_diff
        df = df[mask]
    if filepath.name == "campus_winter_twilight.tum":  # kek3
        t0 = float(df["timestamp"].to_numpy()[0])
        max_diff = 4150.0
        diffs = df["timestamp"].to_numpy() - t0
        mask = diffs <= max_diff
        df = df[mask]
    df["timestamp"] = (df["timestamp"] * 1000000000).astype(np.int64)  # convert to nanoseconds format
    return df


def get_files_list(directory: Path, ext: str) -> List[Path]:
    if ext[0] != ".":
        ext = "." + ext
    files_list = [f for f in directory.iterdir() if f.suffix == ext]
    return sorted(files_list)


def closest_values_indices(in_array: np.ndarray, from_array: np.ndarray) -> np.ndarray:
    """For each element in the first array find the closest value from the second array.

    Args:
        in_array (np.ndarray): First array.
        from_array (np.ndarray): Second array.

    Returns:
        np.ndarray: Indices of elements from `from_array` that are closest to
        corresponding values in `in_array`.
    """
    closest_idxs = np.zeros(len(in_array), dtype=np.int64)
    for i, a_val in enumerate(in_array):  # memory-optimized version
        abs_diffs = np.abs(from_array - a_val)
        closest_idxs[i] = np.argmin(abs_diffs)
    return closest_idxs


def filter_timestamps(
    pose_ts: np.ndarray,
    front_ts: np.ndarray,
    back_ts: np.ndarray,
    lidar_ts: np.ndarray,
    max_diff: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter timestamps by values.

    For each timestamp in `pose_ts`, `front_ts`, `back_ts`, and `lidar_ts`, find the closest corresponding
        timestamps in the other arrays that differ less than `max_diff`. Returns the indices
        of the filtered elements.

    Args:
        pose_ts (np.ndarray): Array of pose timestamps.
        front_ts (np.ndarray): Array of front timestamps.
        back_ts (np.ndarray): Array of back timestamps.
        lidar_ts (np.ndarray): Array of lidar timestamps.
        max_diff (int): Maximum allowed difference between corresponding timestamps.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple of arrays containing the indices
        of the filtered timestamps for each sensor in the order (pose_ts, front_ts, back_ts, lidar_ts).
    """
    # Initialize arrays to hold filtered indices
    filtered_pose_idxs = []
    filtered_front_idxs = []
    filtered_back_idxs = []
    filtered_lidar_idxs = []

    # Iterate over timestamps in pose_ts
    for i, ts in enumerate(pose_ts):
        # Find the closest corresponding timestamps in the other arrays
        front_idx = closest_values_indices(np.array([ts]), front_ts)[0]
        back_idx = closest_values_indices(np.array([ts]), back_ts)[0]
        lidar_idx = closest_values_indices(np.array([ts]), lidar_ts)[0]

        # Check if the differences between the timestamps are less than max_diff
        front_diff = np.abs(ts - front_ts[front_idx])
        back_diff = np.abs(ts - back_ts[back_idx])
        lidar_diff = np.abs(ts - lidar_ts[lidar_idx])

        if front_diff <= max_diff and back_diff <= max_diff and lidar_diff <= max_diff:
            filtered_pose_idxs.append(i)
            filtered_front_idxs.append(front_idx)
            filtered_back_idxs.append(back_idx)
            filtered_lidar_idxs.append(lidar_idx)

    # Convert filtered index lists to NumPy arrays and return as a list
    return (
        np.array(filtered_pose_idxs),
        np.array(filtered_front_idxs),
        np.array(filtered_back_idxs),
        np.array(filtered_lidar_idxs),
    )


def filter_by_distance_indices(utm_points: np.ndarray, distance: float = 5.0) -> np.ndarray:
    """Filter points so that each point is approximatly `distance` meters away from the previous.

    Args:
        utm_points (np.ndarray): The array of UTM coordinates.
        distance (float): The desirable distance between points. Defaults to 5.0.

    Returns:
        np.ndarray: The indices of the filtered points.
    """
    filtered_points = np.array([0], dtype=int)  # start with the index of the first point
    for i in range(1, utm_points.shape[0]):
        # calculate the Euclidean distance between the current and previous point
        right_dist = np.linalg.norm(utm_points[i] - utm_points[filtered_points[-1]])
        if right_dist >= distance:  # we found the point to the right of the 'ideal point'
            left_dist = np.linalg.norm(utm_points[i - 1] - utm_points[filtered_points[-1]])
            if np.abs(right_dist - distance) < np.abs(left_dist - distance):
                filtered_points = np.append(filtered_points, i)  # the point to the right is closer
            else:
                filtered_points = np.append(filtered_points, i - 1)  # the point to the left is closer
    return filtered_points


def plot_track_map(utms: np.ndarray) -> np.ndarray:
    x, y = utms[:, 0], utms[:, 1]
    x_min, x_max = np.min(x) - 2, np.max(x) + 2
    y_min, y_max = np.min(y) - 2, np.max(y) + 2
    fig, ax = plt.subplots(dpi=200)
    ax.scatter(x, y, s=0.5, c="blue")
    ax.set_xlabel("x")
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("y")
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    fig.canvas.draw()
    # convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    # convert from RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    front_cam_dir, back_cam_dir, lidar_dir, poses_file, output_dir, dist_threshold = parse_args()

    poses_df = read_poses_tum(poses_file)
    front_cam_files = get_files_list(front_cam_dir, ext=".png")
    back_cam_files = get_files_list(back_cam_dir, ext=".png")
    lidar_files = get_files_list(lidar_dir, ext=".bin")
    front_cam_timestamps = np.array(sorted([int(f.stem) for f in front_cam_files]))
    back_cam_timestamps = np.array(sorted([int(f.stem) for f in back_cam_files]))
    lidar_timestamps = np.array(sorted([int(f.stem) for f in lidar_files]))

    assert len(front_cam_timestamps) == len(front_cam_files)
    assert len(back_cam_timestamps) == len(back_cam_files)
    assert len(lidar_timestamps) == len(lidar_files)

    filtered_indices = filter_timestamps(
        pose_ts=poses_df["timestamp"].to_numpy(),
        front_ts=front_cam_timestamps,
        back_ts=back_cam_timestamps,
        lidar_ts=lidar_timestamps,
        max_diff=60000000,  # 60 ms
    )

    poses_df = poses_df.iloc[filtered_indices[0]]
    front_cam_timestamps = front_cam_timestamps[filtered_indices[1]]
    front_cam_files = [front_cam_files[i] for i in filtered_indices[1]]
    back_cam_timestamps = back_cam_timestamps[filtered_indices[2]]
    back_cam_files = [back_cam_files[i] for i in filtered_indices[2]]
    lidar_timestamps = lidar_timestamps[filtered_indices[3]]
    lidar_files = [lidar_files[i] for i in filtered_indices[3]]

    distance_filtered_indices = filter_by_distance_indices(
        poses_df[["tx", "ty", "tz"]].to_numpy(), distance=dist_threshold
    )

    poses_df = poses_df.iloc[distance_filtered_indices]
    front_cam_timestamps = front_cam_timestamps[distance_filtered_indices]
    front_cam_files = [front_cam_files[i] for i in distance_filtered_indices]
    back_cam_timestamps = back_cam_timestamps[distance_filtered_indices]
    back_cam_files = [back_cam_files[i] for i in distance_filtered_indices]
    lidar_timestamps = lidar_timestamps[distance_filtered_indices]
    lidar_files = [lidar_files[i] for i in distance_filtered_indices]

    track_map_img = plot_track_map(poses_df[["tx", "ty"]].to_numpy())

    if output_dir.exists():
        print(f"Given output dir already exist: {output_dir}")
    else:
        print(f"Creating output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "track_map.png"), track_map_img)

    out_df = poses_df
    out_df["front_cam_ts"] = front_cam_timestamps
    out_df["back_cam_ts"] = back_cam_timestamps
    out_df["lidar_ts"] = lidar_timestamps
    out_df = out_df[
        ["timestamp", "front_cam_ts", "back_cam_ts", "lidar_ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    ]
    out_df.to_csv(output_dir / "track.csv")

    front_cam_dst_dir = output_dir / "front_cam"
    front_cam_dst_dir.mkdir(exist_ok=True)
    for src_file, timestamp in tqdm(
        zip(front_cam_files, front_cam_timestamps), desc="front camera", total=len(front_cam_timestamps)
    ):
        shutil.copy(str(src_file), str(front_cam_dst_dir / f"{timestamp}.png"))

    back_cam_dst_dir = output_dir / "back_cam"
    back_cam_dst_dir.mkdir(exist_ok=True)
    for src_file, timestamp in tqdm(
        zip(back_cam_files, back_cam_timestamps), desc="back camera", total=len(back_cam_timestamps)
    ):
        shutil.copy(str(src_file), str(back_cam_dst_dir / f"{timestamp}.png"))

    lidar_dst_dir = output_dir / "lidar"
    lidar_dst_dir.mkdir(exist_ok=True)
    for src_file, timestamp in tqdm(
        zip(lidar_files, lidar_timestamps), desc="lidar", total=len(lidar_timestamps)
    ):
        shutil.copy(str(src_file), str(lidar_dst_dir / f"{timestamp}.bin"))
