from typing import Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


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
