"""Extract trajectories from a rosbag file and save it in TUM format."""
import argparse
from pathlib import Path

import rosbag

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract trajectories from a rosbag file and save it in TUM format."
    )
    parser.add_argument("-f", "--file", type=str, required=True, help="path to the rosbag file")
    args = parser.parse_args()

    bagfile_path = Path(args.file)

    bag = rosbag.Bag(args.file)
    output_file_path = bagfile_path.parent / (bagfile_path.stem + ".tum")

    with open(output_file_path, "w") as f:
        for _, msg, t in bag.read_messages(topics=["/global_trajectory_0"]):
            f.write(
                f"{t.to_sec()} {msg.transform.translation.x} {msg.transform.translation.y} "
                f"{msg.transform.translation.z} {msg.transform.rotation.x} {msg.transform.rotation.y} "
                f"{msg.transform.rotation.z} {msg.transform.rotation.w}\n"
            )

    bag.close()
