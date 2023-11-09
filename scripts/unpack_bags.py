"""Unpack images and point clouds from a ROS bag file."""
import argparse
from os import PathLike
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.point_cloud2 import read_points
from tqdm import tqdm

FRONT_CAM_TOPIC = "/zed_node/left/image_rect_color/compressed"
BACK_CAM_TOPIC = "/realsense_back/color/image_raw/compressed"
LIDAR_TOPIC = "/velodyne_points"


def extract_images_and_points(bag_file_path: Union[str, PathLike], output_dir: Union[str, PathLike]) -> None:
    """Extracts images and lidar points from a ROS bag file and saves them to disk.

    Args:
        bag_file_path (Union[str, PathLike]): Path to the ROS bag file.
        output_dir (Union[str, PathLike]): Path to the output directory.
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
        if topic == FRONT_CAM_TOPIC or topic == BACK_CAM_TOPIC:
            cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image_file_path = (
                front_cam_dir / f"{t.to_nsec()}.png"
                if topic == FRONT_CAM_TOPIC
                else back_cam_dir / f"{t.to_nsec()}.png"
            )
            cv2.imwrite(str(image_file_path), cv_image)

        elif topic == LIDAR_TOPIC:
            points_file_path = lidar_dir / f"{t.to_nsec()}.bin"
            points = np.array(list(read_points(msg)), dtype=np.float32)
            points[:, 3] /= 255.0  # to keep compatibility with KITTI
            points[:, :4].tofile(points_file_path)  # x, y, z, intensity

    bag.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images and point clouds from a ROS bag file.")
    parser.add_argument(
        "-d", "--dir", type=str, required=True, help="Path to the directory containing the ROS bag files."
    )
    parser.add_argument("-o", "--out_dir", type=str, required=True, help="Path to the output directory.")
    args = parser.parse_args()

    input_dir = Path(args.dir)
    output_dir = Path(args.out_dir)

    bag_files_list = [f for f in input_dir.iterdir() if f.suffix == ".bag"]

    for bag_file_path in tqdm(
        bag_files_list, desc="Extracting images and point clouds", position=0, leave=True
    ):
        extract_images_and_points(bag_file_path, output_dir)
