import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch


@dataclass
class DataConfig:
    sample_size: tuple[int] = (16, 64)
    undersampling: float = 0.5  # (64, 32) -> (16, 32) measurement
    measurement_noise_std: float = 0.01

    num_samples: int = 1000

    distribution: str = "CDL-C"
    data_dir: Path | None = None
    data_tag: str | None = None

    def __post_init__(self):
        if self.distribution not in ["gaussian", "CDL-A", "CDL-B", "CDL-C", "CDL-D"]:
            raise ValueError(f"Unsupported data distribution {self.distribution}!")


def get_data(cfg: DataConfig) -> torch.Tensor:
    if cfg.distribution == "gaussian":
        data = torch.randn(
            cfg.num_samples,
            *cfg.sample_size,
            dtype=torch.complex64,
        )
    elif "CDL" in cfg.distribution:
        if cfg.data_tag is None or cfg.data_dir is None:
            raise ValueError("A tag and folder must be specified when trying to load CDL data!")
        filename = (
            cfg.data_dir
            / f"{cfg.distribution}_rx{cfg.sample_size[0]}_tx{cfg.sample_size[1]}_{cfg.data_tag}.h5"
        )
        with h5py.File(filename, "r") as f:
            data = np.asarray(f["data"])
        # Convert to tensor
        data = torch.tensor(data, dtype=torch.complex64)
    else:
        raise NotImplementedError("Other data distributions are not yet supported!")

    return data


def complex_to_real(data: torch.Tensor) -> torch.Tensor:
    output = torch.view_as_real(data)
    output = torch.moveaxis(output, -1, 1)
    return output


def real_to_complex(data: torch.Tensor) -> torch.Tensor:
    output = torch.moveaxis(data, 1, -1)
    output = torch.view_as_complex(output.contiguous())
    return output
