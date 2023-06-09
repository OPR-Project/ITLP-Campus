"""Custom datasets implementations."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from src.visualizations import get_colored_mask
from src.transforms import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
    DefaultSemanticTransform,
)


class ITLPCampus(Dataset):
    """ITLP Campus dataset implementation."""

    dataset_root: Path
    dataset_df: DataFrame
    front_cam_text_descriptions_df: Optional[DataFrame]
    back_cam_text_descriptions_df: Optional[DataFrame]
    front_cam_text_labels_df: Optional[DataFrame]
    back_cam_text_labels_df: Optional[DataFrame]
    front_cam_aruco_labels_df: Optional[DataFrame]
    back_cam_aruco_labels_df: Optional[DataFrame]
    sensors: Tuple[str, ...]
    images_subdir: str = ""
    clouds_subdir: str = "lidar"
    semantic_subdir: str = "masks"
    text_descriptions_subdir: str = "text_descriptions"
    text_labels_subdir: str = "text_labels"
    aruco_labels_subdir: str = "aruco_labels"
    image_transform: DefaultImageTransform
    cloud_transform: DefaultCloudTransform
    cloud_set_transform: DefaultCloudSetTransform
    mink_quantization_size: Optional[float]
    load_semantics: bool
    load_text_descriptions: bool
    load_text_labels: bool
    load_aruco_labels: bool
    indoor: bool

    def __init__(
        self,
        dataset_root: Union[str, Path],
        sensors: Union[str, Tuple[str, ...]] = ("front_cam", "lidar"),
        mink_quantization_size: Optional[float] = 0.5,
        load_semantics: bool = False,
        load_text_descriptions: bool = False,
        load_text_labels: bool = False,
        load_aruco_labels: bool = False,
        indoor: bool = False,
    ) -> None:
        """ITLP Campus dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset track root directory.
            sensors (Union[str, Tuple[str, ...]]): List of sensors for which the data should be loaded.
                Defaults to ("front_cam", "lidar").
            mink_quantization_size (Optional[float]): The quantization size for point clouds. Defaults to 0.5.
            load_semantics (bool): Wether to load semantic masks for camera images. Defaults to False.
            load_text_descriptions (bool): Wether to load text descriptions for camera images.
                Defaults to False.
            load_text_labels (bool): Wether to load detected text for camera images. Defaults to False.
            load_aruco_labels (bool): Wether to load detected aruco labels for camera images.
                Defaults to False.
            indoor (bool): Wether to load indoor or outdoor dataset track. Defaults to False.

        Raises:
            FileNotFoundError: If dataset_root doesn't exist.
            FileNotFoundError: If there is no csv file for given subset (track).
        """
        super().__init__()

        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        subset_csv = self.dataset_root / "track.csv"
        self.dataset_df = pd.read_csv(subset_csv, index_col=0)

        if isinstance(sensors, str):
            sensors = tuple([sensors])
        self.sensors = sensors

        self.mink_quantization_size = mink_quantization_size
        self.load_semantics = load_semantics

        self.load_text_descriptions = load_text_descriptions
        if self.load_text_descriptions:
            if "front_cam" in self.sensors:
                self.front_cam_text_descriptions_df = pd.read_csv(
                    self.dataset_root / self.text_descriptions_subdir / "front_cam_text.csv"
                )
            if "back_cam" in self.sensors:
                self.back_cam_text_descriptions_df = pd.read_csv(
                    self.dataset_root / self.text_descriptions_subdir / "back_cam_text.csv"
                )

        self.load_text_labels = load_text_labels
        if self.load_text_labels:
            if "front_cam" in self.sensors:
                self.front_cam_text_labels_df = pd.read_csv(
                    self.dataset_root / self.text_labels_subdir / "front_cam_text_labels.csv"
                )
            if "back_cam" in self.sensors:
                self.back_cam_text_labels_df = pd.read_csv(
                    self.dataset_root / self.text_labels_subdir / "back_cam_text_labels.csv"
                )

        self.load_aruco_labels = load_aruco_labels
        if self.load_aruco_labels:
            if "front_cam" in self.sensors:
                self.front_cam_aruco_labels_df = pd.read_csv(
                    self.dataset_root / self.aruco_labels_subdir / "front_cam_aruco_labels.csv", sep="\t"
                )
            if "back_cam" in self.sensors:
                self.back_cam_aruco_labels_df = pd.read_csv(
                    self.dataset_root / self.aruco_labels_subdir / "back_cam_aruco_labels.csv", sep="\t"
                )

        self.indoor = indoor

        self.image_transform = DefaultImageTransform(resize=(320, 192))
        self.semantic_transform = DefaultSemanticTransform(resize=(320, 192))
        self.cloud_transform = DefaultCloudTransform()
        self.cloud_set_transform = DefaultCloudSetTransform()

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        data: Dict[str, Union[int, Tensor]] = {"idx": idx}
        row = self.dataset_df.iloc[idx]
        data["pose"] = torch.tensor(
            row[["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
        )
        if "front_cam" in self.sensors:
            image_ts = int(row["front_cam_ts"])
            im_filepath = self.dataset_root / self.images_subdir / "front_cam" / f"{image_ts}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image_front_cam"] = im
            if self.load_semantics:
                im_filepath = (
                    self.dataset_root / self.semantic_subdir / "front_cam" / f"{image_ts}.png"
                )  # image id is equal to semantic mask id~
                im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
                im = self.semantic_transform(im)
                data["semantic_front_cam"] = im
            if self.load_text_labels:
                text_labels_df = self.front_cam_text_labels_df[
                    self.front_cam_text_labels_df["path"] == f"{image_ts}.png"
                ]
                data["text_labels_front_cam_df"] = text_labels_df
            if self.load_text_descriptions:
                text_description_df = self.front_cam_text_descriptions_df[
                    self.front_cam_text_descriptions_df["path"] == f"{image_ts}.png"
                ]
                data["text_description_front_cam_df"] = text_description_df
            if self.load_aruco_labels:
                aruco_labels_df = self.front_cam_aruco_labels_df[
                    self.front_cam_aruco_labels_df["image_name"] == f"{image_ts}.png"
                ]
                data["aruco_labels_front_cam_df"] = aruco_labels_df
        if "back_cam" in self.sensors:
            image_ts = int(row["back_cam_ts"])
            im_filepath = self.dataset_root / self.images_subdir / "back_cam" / f"{image_ts}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image_back_cam"] = im
            if self.load_semantics:
                im_filepath = (
                    self.dataset_root / self.semantic_subdir / "back_cam" / f"{image_ts}.png"
                )  # image id is equal to semantic mask id~
                im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
                im = self.semantic_transform(im)
                data["semantic_back_cam"] = im
            if self.load_text_labels:
                text_labels_df = self.back_cam_text_labels_df[
                    self.back_cam_text_labels_df["path"] == f"{image_ts}.png"
                ]
                data["text_labels_back_cam_df"] = text_labels_df
            if self.load_text_descriptions:
                text_description_df = self.back_cam_text_descriptions_df[
                    self.back_cam_text_descriptions_df["path"] == f"{image_ts}.png"
                ]
                data["text_description_back_cam_df"] = text_description_df
            if self.load_aruco_labels:
                aruco_labels_df = self.back_cam_aruco_labels_df[
                    self.back_cam_aruco_labels_df["image_name"] == f"{image_ts}.png"
                ]
                data["aruco_labels_back_cam_df"] = aruco_labels_df
        if "lidar" in self.sensors:
            pc_filepath = self.dataset_root / self.clouds_subdir / f"{int(row['lidar_ts'])}.bin"
            pc = self._load_pc(pc_filepath)
            data["cloud"] = pc
        return data

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset_df)

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float32).reshape((-1, 4))[:, :-1]
        in_range_idx = np.all(
            np.logical_and(-100 <= pc, pc <= 100),  # select points in range [-100, 100] meters
            axis=1,
        )
        pc = pc[in_range_idx]
        pc_tensor = torch.tensor(pc, dtype=torch.float32)
        return pc_tensor

    def visualize_semantic_mask(
        self,
        idx: int,
        camera: Literal["front_cam", "back_cam"] = "front_cam",
        alpha: float = 0.7,
        show: bool = False,
    ) -> np.ndarray:
        """Method to visualize semantic segmenation mask blended with camera image.

        Args:
            idx (int): Index of dataset element to visualize.
            camera (Literal["front_cam", "back_cam"]): Which camera should be used. Defaults to "front_cam".
            alpha (float): Ratio of mask vs original image. Defaults to 0.7.
            show (bool): Show image using `plt.show()`. Defaults to False.

        Raises:
            ValueError: If given 'camera' argument is not in `("front_cam", "back_cam")`.

        Returns:
            np.ndarray: Semantic mask image blended with camera image in `cv2` RGB format: (H, W, 3).
        """
        if camera not in ("front_cam", "back_cam"):
            raise ValueError("Wrong 'camera' argument given: you should select 'front_cam' or 'back_cam'")
        if self.indoor:
            dataset_type = "ade20k"
        else:
            dataset_type = "mapillary"
        row = self.dataset_df.iloc[idx]
        image_ts = int(row[f"{camera}_ts"])

        im_filepath = self.dataset_root / self.images_subdir / camera / f"{image_ts}.png"
        im = cv2.cvtColor(cv2.imread(str(im_filepath)), cv2.COLOR_BGR2RGB)

        mask_filepath = self.dataset_root / self.semantic_subdir / camera / f"{image_ts}.png"
        mask = cv2.imread(str(mask_filepath), cv2.IMREAD_UNCHANGED)
        rgb_mask = get_colored_mask(mask, dataset=dataset_type)

        blended_img = cv2.addWeighted(src1=rgb_mask, alpha=alpha, src2=im, beta=(1 - alpha), gamma=0)

        if show:
            plt.imshow(blended_img)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return blended_img

    def visualize_pointcloud(self, idx: int, show: bool = False) -> np.ndarray:
        """Method to visualize LiDAR's point cloud.

        Args:
            idx (int): Index of dataset element to visualize.
            show (bool): Show pointcloud using `plt.show()`. Defaults to False.

        Returns:
            np.ndarray: Point cloud image in `cv2` RGB format: (H, W, 3).
        """
        row = self.dataset_df.iloc[idx]
        pc_filepath = self.dataset_root / self.clouds_subdir / f"{int(row['lidar_ts'])}.bin"
        pc = self._load_pc(pc_filepath).numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_zlim([-10, 20])

        dist = np.sqrt(np.sum(np.square(pc), axis=1))
        norm = plt.Normalize(dist.min(), dist.max())
        colors = plt.cm.jet(norm(dist))

        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.2, c=colors, marker="o")
        ax.view_init(elev=45, azim=180)
        ax.set_axis_off()
        plt.tight_layout()
        fig.canvas.draw()
        img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2RGB)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if show:
            plt.show()
        plt.close(fig)
        return img

    def visualize_text_labels(
        self, idx: int, camera: Literal["front_cam", "back_cam"] = "front_cam", show: bool = False
    ) -> np.ndarray:
        """Method to visualize detected text labels on the image.

        Args:
            idx (int): Index of dataset element to visualize.
            camera (Literal["front_cam", "back_cam"]): Which camera should be used. Defaults to "front_cam".
            show (bool, optional): Show image with detections using `plt.show()`. Defaults to False.

        Raises:
            ValueError: If tried to load text labels with load_text_labels=False
            ValueError: If wrong 'camera' argument is given.

        Returns:
            np.ndarray: Image with detected labels in `cv2` RGB format: (H, W, 3).
        """
        if not self.load_text_labels:
            raise ValueError("Tried to load text labels with load_text_labels=False")
        row = self.dataset_df.iloc[idx]
        image_name = f"{int(row[f'{camera}_ts'])}.png"
        im_filepath = self.dataset_root / self.images_subdir / camera / image_name
        im = cv2.cvtColor(cv2.imread(str(im_filepath)), cv2.COLOR_BGR2RGB)
        if camera == "front_cam":
            text_labels_df = self.front_cam_text_labels_df[
                self.front_cam_text_labels_df["path"] == image_name
            ]
        elif camera == "back_cam":
            text_labels_df = self.back_cam_text_labels_df[self.back_cam_text_labels_df["path"] == image_name]
        else:
            raise ValueError("Wrong 'camera' argument given: you should select 'front_cam' or 'back_cam'")

        bboxes = (
            text_labels_df[["x_lt", "y_lt", "x_rt", "y_rt", "x_rb", "y_rb", "x_lb", "y_lb"]]
            .to_numpy()
            .astype(int)
        )
        texts = text_labels_df[["label"]].to_numpy().tolist()
        for bbox, text in zip(bboxes, texts):
            im = cv2.polylines(
                im, pts=[bbox.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2
            )
            im = cv2.putText(
                im,
                text=str(text[0]),
                org=bbox.reshape(-1, 2)[0],
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )

        if show:
            plt.imshow(im)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return im

    def visualize_aruco_labels(
        self, idx: int, camera: Literal["front_cam", "back_cam"] = "front_cam", show: bool = False
    ) -> np.ndarray:
        """Method to visualize detected aruco labels on the image.

        Args:
            idx (int): Index of dataset element to visualize.
            camera (Literal["front_cam", "back_cam"]): Which camera should be used. Defaults to "front_cam".
            show (bool, optional): Show image with detections using `plt.show()`. Defaults to False.

        Raises:
            ValueError: If tried to load text labels with load_aruco_labels=False
            ValueError: If wrong 'camera' argument is given.

        Returns:
            np.ndarray: Image with detected aruco labels in `cv2` RGB format: (H, W, 3).
        """
        if not self.load_aruco_labels:
            raise ValueError("Tried to load text labels with load_aruco_labels=False")
        row = self.dataset_df.iloc[idx]
        image_name = f"{int(row[f'{camera}_ts'])}.png"
        im_filepath = self.dataset_root / self.images_subdir / camera / image_name
        im = cv2.cvtColor(cv2.imread(str(im_filepath)), cv2.COLOR_BGR2RGB)
        if camera == "front_cam":
            aruco_labels_df = self.front_cam_aruco_labels_df[
                self.front_cam_aruco_labels_df["image_name"] == image_name
            ]
        elif camera == "back_cam":
            aruco_labels_df = self.back_cam_aruco_labels_df[
                self.back_cam_aruco_labels_df["image_name"] == image_name
            ]
        else:
            raise ValueError("Wrong 'camera' argument given: you should select 'front_cam' or 'back_cam'")

        bboxes = aruco_labels_df[["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]].to_numpy().astype(int)
        texts = aruco_labels_df[["aruco_id"]].to_numpy().tolist()
        for bbox, text in zip(bboxes, texts):
            im = cv2.polylines(
                im, pts=[bbox.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2
            )
            im = cv2.putText(
                im,
                text=str(text[0]),
                org=bbox.reshape(-1, 2)[0],
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=2,
            )

        if show:
            plt.imshow(im)
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return im

    @staticmethod
    def download_data(out_dir: Path) -> None:
        outdoor_tracks_dict = {
            "00_2023-02-10": "",
            "01_2023-02-21": "",
            "02_2023-03-15": "",
            "03_2023-04-11": "",
            "04_2023-04-13": "",
        }
        indoor_tracks_dict = {
            "00_2023-03-13": "",
        }

        if not out_dir.exists():
            print(f"Creating output directory: {out_dir}")
            out_dir.mkdir(parents=True)
        else:
            print(f"Will download in existing directory: {out_dir}")

        (out_dir / "ITLP_Campus_outdoor").mkdir(exist_ok=True)
        for track_name, url in outdoor_tracks_dict.items():
            gdown.download(url, output=str(out_dir / f"{track_name}.zip"), quiet=False, fuzzy=True)
        (out_dir / "ITLP_Campus_indoor").mkdir(exist_ok=True)
        for track_name, url in indoor_tracks_dict.items():
            gdown.download(url, output=str(out_dir / f"{track_name}.zip"), quiet=False, fuzzy=True)
