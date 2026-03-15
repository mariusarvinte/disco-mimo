from dataclasses import dataclass
from typing import Union, Optional

import torch
from diffusers import UNet2DModel

from mimo.data import DataConfig


@dataclass
class UNetConfig:
    device: str = "cuda"

    num_channels: int = 2

    block_out_channels: tuple[int] = (16, 32, 48, 64)
    norm_num_groups: int = 16
    layers_per_block: int = 8


class UNet2DModelNCSN(UNet2DModel):
    def __init__(self, noise_levels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_levels = noise_levels

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        outputs = super().forward(sample, timestep, class_labels, return_dict)

        # Unpack output and apply NCSNv2 normalization trick
        output = outputs["sample"]
        extra_dims = len(output.shape) - len(timestep.shape)
        output = output / self.noise_levels[timestep][..., *[None] * extra_dims]
        return output


def get_model(
    cfg_model: UNetConfig, cfg_data: DataConfig, noise_levels: torch.Tensor
) -> UNet2DModel:
    model = UNet2DModelNCSN(
        noise_levels=noise_levels,
        sample_size=cfg_data.sample_size,
        in_channels=cfg_model.num_channels,
        out_channels=cfg_model.num_channels,
        block_out_channels=cfg_model.block_out_channels,
        layers_per_block=cfg_model.layers_per_block,
        norm_num_groups=cfg_model.norm_num_groups,
    )

    return model
