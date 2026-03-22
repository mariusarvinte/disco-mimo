import numpy as np

from dataclasses import dataclass, field
from typing import Union, Optional
from pathlib import Path

import torch
from diffusers import UNet2DModel
from omegaconf import MISSING

from ncsnv2.models.ncsnv2 import NCSNv2Deepest


@dataclass
class UNetConfig:
    noise_levels: torch.Tensor = MISSING
    sample_size: tuple[int] = MISSING

    channels: int = 2
    block_out_channels: tuple[int] = (16, 32, 48, 64)
    norm_num_groups: int = 16
    layers_per_block: int = 8


@dataclass
class NCSNv2Config:
    @dataclass
    class Model:
        num_classes: int = MISSING
        sigma_begin: float = MISSING
        sigma_end: float = MISSING
        sigma_rate: float = MISSING
        sigma_dist: str = MISSING

        ngf: int = 32
        normalization: str = "InstanceNorm++"
        nonlinearity: str = "relu"

    @dataclass
    class Data:
        logit_transform: bool = False
        channels: int = 2
        rescaled: bool = False

    device: str = MISSING
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)


class UNet2DModelNCSN(UNet2DModel):
    def __init__(self, noise_levels: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas = noise_levels

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
        output = output / self.sigmas[timestep][..., *[None] * extra_dims]
        return output


def get_model(
    arch: str,
    max_noise_level: float,
    num_noise_levels: float,
    noise_step_factor: float,
    noise_distribution: str = "geometric",
    num_rx: int | None = None,
    num_tx: int | None = None,
    device: str = "cuda",
    filename: Path | None = None,
) -> torch.nn.Module:
    match arch:
        case "unet2d-diffusers":
            noise_levels = np.exp(
                np.linspace(
                    np.log(max_noise_level),
                    np.log(max_noise_level * noise_step_factor ** (num_noise_levels - 1)),
                    num_noise_levels,
                )
            )
            noise_levels = torch.tensor(noise_levels, device=device, dtype=torch.float32)
            model_config = UNetConfig(noise_levels=noise_levels, sample_size=(num_rx, num_tx))
            model = get_diffusers_model(model_config)

        case "ncsnv2":
            model_config = NCSNv2Config(device=device)
            # Populate required inner configurations
            model_config.model.num_classes = num_noise_levels
            model_config.model.sigma_begin = max_noise_level
            model_config.model.sigma_rate = noise_step_factor
            model_config.model.sigma_dist = noise_distribution
            model_config.model.sigma_end = (
                model_config.model.sigma_begin
                * model_config.model.sigma_rate ** (model_config.model.num_classes - 1)
            )
            model = get_ncsnv2_model(model_config)
        case _:
            raise ValueError("Invalid model architecture!")

    # Load pretrained model state if specified
    if filename:
        contents = torch.load(filename, map_location="cpu")
        model.load_state_dict(contents["model_state_dict"], strict=True)

    return model


def get_diffusers_model(cfg: UNetConfig) -> UNet2DModelNCSN:
    model = UNet2DModelNCSN(
        noise_levels=cfg.noise_levels,
        sample_size=cfg.sample_size,
        in_channels=cfg.channels,
        out_channels=cfg.channels,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block,
        norm_num_groups=cfg.norm_num_groups,
    )
    return model


def get_ncsnv2_model(cfg: NCSNv2Config) -> NCSNv2Deepest:
    model = NCSNv2Deepest(cfg)
    return model
