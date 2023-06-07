"""Custom datasets implementations."""
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from src.transforms import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
)


class ITLPCampus(Dataset):
    """ITLP Campus dataset implementation."""

    dataset_root: Path
    dataset_df: DataFrame
    sensors: Tuple[str, ...]
    images_subdir: str = ""
    clouds_subdir: str = "lidar"
    semantic_subdir: str = "labels"
    image_transform: DefaultImageTransform
    cloud_transform: DefaultCloudTransform
    cloud_set_transform: DefaultCloudSetTransform
    mink_quantization_size: Optional[float]
    load_semantics: bool

    def __init__(
        self,
        dataset_root: Union[str, Path],
        sensors: Union[str, Tuple[str, ...]] = ("front_cam", "lidar"),
        mink_quantization_size: Optional[float] = 0.5,
        load_semantics: bool = False,
    ) -> None:
        """ITLP Campus dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (str): Which track to load.
            sensors (Union[str, Tuple[str, ...]]): List of sensors for which the data should be loaded.
                Defaults to ("front_cam", "lidar").
            mink_quantization_size (Optional[float]): The quantization size for point clouds. Defaults to 0.5.

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

        self.image_transform = DefaultImageTransform(resize=(320, 192))
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
                data["semantic_front_cam"] = im
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
                data["semantic_back_cam"] = im
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
