"""SECAD / SERCAD 数据集：``ae_train.hdf5``、``ae_test.hdf5`` 与 ``train_names.npz``、``test_names.npz``。"""

import os
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
import torch.utils.data


def _normalize_voxels(vox: torch.Tensor) -> torch.Tensor:
    x = vox.float()
    if x.dim() >= 5 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x.unsqueeze(1)


class GTSamples(torch.utils.data.Dataset):
    """
    与 SERCAD ``GTSamples`` 一致：HDF5 含 ``voxels``、``points_{grid_sample}``（末维为 xyz + 占用）。
    """

    def __init__(
        self,
        data_source: str,
        grid_sample: int = 64,
        test_flag: bool = False,
        name_keys: Optional[Sequence[str]] = None,
    ):
        print("data source", data_source)
        self.data_source = data_source

        if test_flag:
            h5_path = os.path.join(data_source, "ae_test.hdf5")
            npz_path = os.path.join(data_source, "test_names.npz")
            keys = list(name_keys) if name_keys else ("test_names",)
        else:
            h5_path = os.path.join(data_source, "ae_train.hdf5")
            npz_path = os.path.join(data_source, "train_names.npz")
            keys = list(name_keys) if name_keys else ("train_names",)

        npz = np.load(npz_path)
        self.data_names = None
        for k in keys:
            if k in npz.files:
                self.data_names = npz[k]
                break
        if self.data_names is None:
            raise KeyError(
                f"No name array in {npz_path}; tried {keys}; files={npz.files}"
            )

        pk = "points_" + str(grid_sample)
        with h5py.File(h5_path, "r") as f:
            print(sorted(f.keys()))
            print("grid_sample", grid_sample)
            if "voxels" not in f:
                raise KeyError(f"Missing voxels in {h5_path}")
            if pk not in f:
                raise KeyError(f"Missing {pk} in {h5_path}")
            data_voxels = torch.from_numpy(f["voxels"][:])
            self.data_points = torch.from_numpy(f[pk][:]).float()

        self.data_voxels = _normalize_voxels(data_voxels)
        print("Loaded voxels shape, ", self.data_voxels.shape)
        print("Loaded points shape, ", self.data_points.shape)

    def __len__(self) -> int:
        return len(self.data_voxels)

    def __getitem__(self, idx: int):
        return {"voxels": self.data_voxels[idx], "occ_data": self.data_points[idx]}


def dataset_from_specs(
    specs: dict,
    test_flag: bool = False,
    grid_sample: int = 64,
) -> GTSamples:
    """``specs`` 可选 ``DatasetNameKeys``：npz 内名称数组的键名列表。"""
    name_keys = specs.get("DatasetNameKeys")
    return GTSamples(
        specs["DataSource"],
        grid_sample=grid_sample,
        test_flag=test_flag,
        name_keys=name_keys,
    )
