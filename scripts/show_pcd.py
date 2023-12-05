import argparse
from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
import open3d as o3d


def create_coordinate_frame(size: float = 0.1) -> o3d.cuda.pybind.geometry.TriangleMesh:
    """Creates a coordinate frame with the specified size.

    Args:
        size (float): The size of the coordinate frame.

    Returns:
        o3d.cuda.pybind.geometry.TriangleMesh: The coordinate frame mesh.
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return mesh_frame


def read_tum_file(file_path: Union[str, PathLike]) -> np.ndarray:
    """Reads a TUM file and returns an array of poses.

    Args:
        file_path (Union[str, PathLike]): The path to the TUM file.

    Returns:
        np.ndarray: An array of poses.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    poses = []
    for line in lines:
        if not line.startswith("#"):
            pose = np.fromstring(line, sep=" ")
            poses.append(pose)
    return np.array(poses)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix.

    Args:
        q: A 4-element numpy array representing the quaternion.

    Returns:
        A 3x3 numpy array representing the rotation matrix.
    """
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    rotation_matrix = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )
    return rotation_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a point cloud")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the PCD file")
    parser.add_argument(
        "-t", "--trajectory", type=str, default=None, required=False, help="Path to the TUM file"
    )
    parser.add_argument(
        "-s", "--voxel_size", type=float, default=0.5, required=False, help="Downsample voxel size"
    )
    args = parser.parse_args()

    input_file_path = Path(args.file)
    trajectory_file_path = Path(args.trajectory) if args.trajectory else None

    pcd = o3d.io.read_point_cloud(str(input_file_path))
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    poses = read_tum_file(trajectory_file_path) if trajectory_file_path else None

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(pcd)

    if poses is not None:
        for pose in poses[::10]:
            pose_matrix = np.eye(4)
            pose_matrix[:3, 3] = pose[1:4]  # translation
            pose_matrix[:3, :3] = quaternion_to_rotation_matrix(pose[4:])  # rotation
            pose_frame = create_coordinate_frame(size=1.0)
            pose_frame.transform(pose_matrix)
            vis.add_geometry(pose_frame)

    render_options = vis.get_render_option()
    render_options.point_size = 1.0  # Adjust the size as needed
    vis.update_renderer()

    # Run the visualizer
    vis.run()
    vis.destroy_window()
