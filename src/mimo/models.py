from dataclasses import dataclass, field
from typing import Union, Optional
from pathlib import Path

import torch
from diffusers import UNet2DModel

# from score_sde_pytorch.models.ncsnv2 import NCSNv2
from ncsnv2.models.ncsnv2 import NCSNv2Deepest


@dataclass
class UNetConfig:
    num_channels: int = 2

    block_out_channels: tuple[int] = (16, 32, 48, 64)
    norm_num_groups: int = 16
    layers_per_block: int = 8


@dataclass
class NCSNv2Config:
    @dataclass
    class Model:
        ngf: int = 32
        num_classes: int = 2311
        normalization: str = "InstanceNorm++"
        nonlinearity: str = "relu"
        sigma_dist: str = "geometric"

        sigma_begin: float = 39.15
        sigma_rate: float = 0.995

    @dataclass
    class Data:
        logit_transform: bool = False
        channels: int = 2
        rescaled: bool = False

    device: str = "cuda:0"
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)

    def __post_init__(self):
        self.model.sigma_end = self.model.sigma_begin * self.model.sigma_rate ** (
            self.model.num_classes - 1
        )


@dataclass
class ModelConfig:
    device: str = "cuda"
    arch: str = "ncsnv2"
    config: UNetConfig | NCSNv2Config | None = None

    filename: Path | None = None

    def __post_init__(self):
        match self.arch:
            case "unet2d-diffusers":
                self.config = UNetConfig()
            case "ncsnv2":
                self.config = NCSNv2Config()
            case _:
                raise ValueError("Invalid model architecture!")


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
    cfg_model: ModelConfig,
    noise_levels: torch.Tensor,
) -> UNet2DModel:
    match cfg_model.arch:
        case "unet2d-diffusers":
            model = UNet2DModelNCSN(
                noise_levels=noise_levels,
                sample_size=cfg_model.sample_size,
                in_channels=cfg_model.num_channels,
                out_channels=cfg_model.num_channels,
                block_out_channels=cfg_model.block_out_channels,
                layers_per_block=cfg_model.layers_per_block,
                norm_num_groups=cfg_model.norm_num_groups,
            )
        case "ncsnv2":
            model = NCSNv2Deepest(cfg_model.config)
        case _:
            raise ValueError("Invalid model architecture!")

    # Load pretrained model state if specified
    if cfg_model.filename:
        contents = torch.load(cfg_model.filename, map_location="cpu")
        model.load_state_dict(contents["model_state_dict"], strict=True)

    return model
