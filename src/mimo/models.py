from dataclasses import dataclass

from diffusers import UNet2DModel

from mimo.data import DataConfig


@dataclass
class UNetConfig:
    device: str = "cuda"

    num_channels: int = 2

    block_out_channels: tuple[int] = (16, 32, 48, 64)
    norm_num_groups: int = 16
    layers_per_block: int = 8


def get_model(cfg_model: UNetConfig, cfg_data: DataConfig) -> UNet2DModel:
    model = UNet2DModel(
        sample_size=cfg_data.sample_size,
        in_channels=cfg_model.num_channels,
        out_channels=cfg_model.num_channels,
        block_out_channels=cfg_model.block_out_channels,
        layers_per_block=cfg_model.layers_per_block,
        norm_num_groups=cfg_model.norm_num_groups,
    )

    return model
