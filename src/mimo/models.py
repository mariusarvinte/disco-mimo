from dataclasses import dataclass

import torch
from diffusers import UNet2DModel

from mimo.data import DataConfig, get_data
from mimo.data import complex_to_real, real_to_complex


@dataclass
class UNetConfig:
    device: str = "cuda"

    num_channels: int = 2

    block_out_channels: tuple[int] = (16, 32, 48, 64)
    norm_num_groups: int = 16
    layers_per_block: int = 2


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


def main():
    cfg_model = UNetConfig()
    cfg_data = DataConfig()
    model = get_model(cfg_model, cfg_data).to(cfg_model.device)

    # Display number of model weights
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")

    # Get data
    data = get_data(cfg_data)

    # Run some data through the model
    with torch.inference_mode():
        # Prepare data
        data_subset = data[:10].to(cfg_model.device)
        data_subset_real = complex_to_real(data_subset)

        # Pass through model
        outputs = model(sample=data_subset_real, timestep=1)
        output = outputs["sample"]

        # Post-process output back to complex values
        output = real_to_complex(output)

    print(f"Input shape is {data_subset.shape} and data type is {data_subset.dtype}")
    print(f"Output shape is {output.shape} and data type is {output.dtype}")


if __name__ == "__main__":
    main()
