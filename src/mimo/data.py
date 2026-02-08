from dataclasses import dataclass

import torch


@dataclass
class DataConfig:
    sample_size: tuple[int] = (16, 64)

    num_samples: int = 10000
    distribution: str = "gaussian"


def get_data(cfg: DataConfig) -> torch.Tensor:
    if cfg.distribution == "gaussian":
        data = torch.randn(
            cfg.num_samples,
            *cfg.sample_size,
            dtype=torch.complex64,
        )
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
