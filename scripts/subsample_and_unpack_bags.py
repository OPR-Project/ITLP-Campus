"""Unpack images and point clouds from a ROS bag file."""
import argparse
from os import PathLike
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
from pandas import DataFrame
from sensor_msgs.point_cloud2 import read_points
from tqdm import tqdm

from itlp_campus.preprocessing import (
    filter_by_distance_indices,
    filter_timestamps,
    plot_track_map,
)

FRONT_CAM_TOPIC = "/zed_node/left/image_rect_color/compressed"
BACK_CAM_TOPIC = "/realsense_back/color/image_raw/compressed"
LIDAR_TOPIC = "/velodyne_points"
TRAJECTORY_TOPIC = "/global_trajectory_0"


def merge_dicts(dict1: Dict[str, List[int]], dict2: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Merges two dictionaries.

    Args:
        dict1 (Dict[str: List[int]]): First dictionary.
        dict2 (Dict[str: List[int]]): Second dictionary.

    Returns:
        Dict[str: List[int]]: Merged dictionary.
    """
    for key in dict2:
        dict1[key] += dict2[key]

    return dict1


def list_images_and_points(bag_file_path: Union[str, PathLike]) -> Dict[str, List[int]]:
    """Return images and lidar timestamps from a ROS bag file.

    Args:
        bag_file_path (Union[str, PathLike]): Path to the ROS bag file.

    Returns:
        Dict[str, List[int]]: Dictionary containing the timestamps of the images and the point clouds.
    """
    bag_file_path = Path(bag_file_path)
    bag = rosbag.Bag(bag_file_path)

    out_dict = {"front_cam": [], "back_cam": [], "lidar": []}

    for topic, msg, t in tqdm(
        bag.read_messages(topics=[FRONT_CAM_TOPIC, BACK_CAM_TOPIC, LIDAR_TOPIC]),
        desc=bag_file_path.name,
        position=1,
        leave=False,
    ):
        if topic == FRONT_CAM_TOPIC:
            out_dict["front_cam"].append(t.to_nsec())
        elif topic == BACK_CAM_TOPIC:
            out_dict["back_cam"].append(t.to_nsec())
        elif topic == LIDAR_TOPIC:
            out_dict["lidar"].append(t.to_nsec())
    bag.close()

    return out_dict


def export_from_bag(
    bag_file_path: Union[str, PathLike],
    output_dir: Union[str, PathLike],
    timestamps_dict: Dict[str, List[int]],
) -> None:
    """Return images and lidar timestamps from a ROS bag file.

    Args:
        bag_file_path (Union[str, PathLike]): Path to the ROS bag file.

    Returns:
        Dict[str, List[int]]: Dictionary containing the timestamps of the images and the point clouds.
    """
    bag_file_path = Path(bag_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    front_cam_dir = output_dir / "front_cam"
    front_cam_dir.mkdir(exist_ok=True)
    back_cam_dir = output_dir / "back_cam"
    back_cam_dir.mkdir(exist_ok=True)
    lidar_dir = output_dir / "lidar"
    lidar_dir.mkdir(exist_ok=True)

    bag = rosbag.Bag(bag_file_path)
    bridge = CvBridge()
    for topic, msg, t in tqdm(
        bag.read_messages(topics=[FRONT_CAM_TOPIC, BACK_CAM_TOPIC, LIDAR_TOPIC]),
        desc=bag_file_path.name,
        position=1,
        leave=False,
    ):
        if topic == FRONT_CAM_TOPIC:
            if t.to_nsec() in timestamps_dict["front_cam"]:
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                image_file_path = front_cam_dir / f"{t.to_nsec()}.png"
                cv2.imwrite(str(image_file_path), cv_image)
        elif topic == BACK_CAM_TOPIC:
            if t.to_nsec() in timestamps_dict["back_cam"]:
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                image_file_path = back_cam_dir / f"{t.to_nsec()}.png"
                cv2.imwrite(str(image_file_path), cv_image)
        elif topic == LIDAR_TOPIC:
            if t.to_nsec() in timestamps_dict["lidar"]:
                points_file_path = lidar_dir / f"{t.to_nsec()}.bin"
                points = np.array(list(read_points(msg)), dtype=np.float32)
                points[:, 3] /= 255.0  # to keep compatibility with KITTI
                points[:, :4].tofile(points_file_path)  # x, y, z, intensity
    bag.close()


def read_trajectory_bag(filepath: Path) -> DataFrame:
    """Reads a trajectory from a ROS bag file.

    Args:
        filepath (Path): Path to the ROS bag file.

    Returns:
        DataFrame: Trajectory dataframe.
    """
    bag = rosbag.Bag(filepath)
    data = {"timestamp": [], "tx": [], "ty": [], "tz": [], "qx": [], "qy": [], "qz": [], "qw": []}
    for _, msg, t in bag.read_messages(topics=[TRAJECTORY_TOPIC]):
        row = {
            "timestamp": [t.to_nsec()],
            "tx": [msg.transform.translation.x],
            "ty": [msg.transform.translation.y],
            "tz": [msg.transform.translation.z],
            "qx": [msg.transform.rotation.x],
            "qy": [msg.transform.rotation.y],
            "qz": [msg.transform.rotation.z],
            "qw": [msg.transform.rotation.w],
        }
        data = merge_dicts(data, row)
    df = DataFrame(data=data)
    return df


def read_trajectory_tum(filepath: Path) -> DataFrame:
    """Reads a trajectory from a file in tum format.

    Args:
        filepath (Path): Path to the trajectory file.

    Returns:
        DataFrame: Trajectory dataframe.
    """
    data = {"timestamp": [], "tx": [], "ty": [], "tz": [], "qx": [], "qy": [], "qz": [], "qw": []}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            timestamp = int(float(line[0]) * 1e9)
            tx, ty, tz = [float(x) for x in line[1:4]]
            qx, qy, qz, qw = [float(x) for x in line[4:8]]
            row = {
                "timestamp": [timestamp],
                "tx": [tx],
                "ty": [ty],
                "tz": [tz],
                "qx": [qx],
                "qy": [qy],
                "qz": [qz],
                "qw": [qw],
            }
            data = merge_dicts(data, row)
    df = DataFrame(data=data)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and point clouds from a ROS bag file.")
    parser.add_argument(
        "-d", "--dir", type=str, required=True, help="Path to the directory containing the ROS bag files."
    )
    parser.add_argument(
        "-t",
        "--trajectory",
        type=str,
        required=True,
        help="Path to the ROS bag file containing the trajectory.",
    )
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument(
        "--dist_threshold",
        type=float,
        default=5.0,
        required=False,
        help="Distance threshold for subsampling.",
    )
    args = parser.parse_args()

    input_dir = Path(args.dir)
    output_dir = Path(args.out_dir)
    dist_threshold = float(args.dist_threshold)

    bag_files_list = sorted([f for f in input_dir.iterdir() if f.suffix == ".bag" and f.stem != "trajectory"])
    trajectory_bag_file = Path(args.trajectory)

    timestamps_dict = {"front_cam": [], "back_cam": [], "lidar": []}

    for bag_file_path in tqdm(
        bag_files_list, desc="Read images and point clouds timestamps", position=0, leave=True
    ):
        timestamps_dict = merge_dicts(timestamps_dict, list_images_and_points(bag_file_path))

    timestamps_dict = {key: np.array(value) for key, value in timestamps_dict.items()}

    if trajectory_bag_file.suffix == ".bag":
        poses_df = read_trajectory_bag(trajectory_bag_file)
    elif trajectory_bag_file.suffix == ".tum" or trajectory_bag_file.suffix == ".txt":
        poses_df = read_trajectory_tum(trajectory_bag_file)
    else:
        raise ValueError(f"Unsupported trajectory file format: {trajectory_bag_file.suffix}")

    filtered_indices = filter_timestamps(
        pose_ts=poses_df["timestamp"].to_numpy(),
        front_ts=timestamps_dict["front_cam"],
        back_ts=timestamps_dict["back_cam"],
        lidar_ts=timestamps_dict["lidar"],
        max_diff=60000000,  # 60 ms
    )

    poses_df = poses_df.iloc[filtered_indices[0]]
    timestamps_dict["front_cam"] = timestamps_dict["front_cam"][filtered_indices[1]]
    timestamps_dict["back_cam"] = timestamps_dict["back_cam"][filtered_indices[2]]
    timestamps_dict["lidar"] = timestamps_dict["lidar"][filtered_indices[3]]

    distance_filtered_indices = filter_by_distance_indices(
        poses_df[["tx", "ty", "tz"]].to_numpy(), distance=dist_threshold
    )

    poses_df = poses_df.iloc[distance_filtered_indices]
    timestamps_dict["front_cam"] = timestamps_dict["front_cam"][distance_filtered_indices]
    timestamps_dict["back_cam"] = timestamps_dict["back_cam"][distance_filtered_indices]
    timestamps_dict["lidar"] = timestamps_dict["lidar"][distance_filtered_indices]

    track_map_img = plot_track_map(poses_df[["tx", "ty"]].to_numpy())

    if output_dir.exists():
        print(f"Given output dir already exist: {output_dir}")
    else:
        print(f"Creating output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / "track_map.png"), track_map_img)

    out_df = poses_df
    out_df["front_cam_ts"] = timestamps_dict["front_cam"]
    out_df["back_cam_ts"] = timestamps_dict["back_cam"]
    out_df["lidar_ts"] = timestamps_dict["lidar"]
    out_df = out_df[
        ["timestamp", "front_cam_ts", "back_cam_ts", "lidar_ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    ]
    out_df.to_csv(output_dir / "track.csv", index=False)

    for bag_file_path in tqdm(
        bag_files_list, desc="Extracting images and point clouds from bags", position=0, leave=True
    ):
        export_from_bag(bag_file_path, output_dir, timestamps_dict)
