import h5py
import numpy as np
from pathlib import Path

import torch

from mimo.utils import ChannelConfig


def get_data(load_path: Path) -> tuple[torch.Tensor, ChannelConfig]:
    if not load_path.is_file():
        raise ValueError(f"Load path {load_path} must be a file written by mimo.cdl!")

    with h5py.File(load_path, "r") as f:
        data = np.asarray(f["data"])
        config = ChannelConfig(
            cdl_model=f["cdl_model"][()].decode(),  # type: ignore
            num_rx=int(f["num_rx"][()]),  # type: ignore
            num_tx=int(f["num_tx"][()]),  # type: ignore
        )

    # Convert data to tensor and complex conjugate
    data = torch.tensor(data, dtype=torch.complex64)
    data = torch.conj(torch.transpose(data, -1, -2)).contiguous()
    return data, config


def generate_measurements(
    samples: torch.Tensor, undersampling: float, noise_std: float
) -> tuple[torch.Tensor, torch.Tensor]:
    pilots_real = torch.randn(
        samples.shape[0],
        samples.shape[-1],
        int(samples.shape[-1] * undersampling),
        device=samples.device,
    ).sign()
    pilots_imag = torch.randn(
        samples.shape[0],
        samples.shape[-1],
        int(samples.shape[-1] * undersampling),
        device=samples.device,
    ).sign()
    pilots = 1 / np.sqrt(2) * (pilots_real + 1j * pilots_imag)
    clean_y = 1 / np.sqrt(samples.shape[-1]) * torch.matmul(samples, pilots)
    noisy_y = clean_y + noise_std * torch.randn_like(clean_y)
    return noisy_y, pilots


def complex_to_real(data: torch.Tensor) -> torch.Tensor:
    output = torch.view_as_real(data)
    output = torch.moveaxis(output, -1, 1)
    return output


def real_to_complex(data: torch.Tensor) -> torch.Tensor:
    output = torch.moveaxis(data, 1, -1)
    output = torch.view_as_complex(output.contiguous())
    return output
