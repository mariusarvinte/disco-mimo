from dataclasses import dataclass

import torch
from diffusers import UNet2DModel


@dataclass
class UNetConfig:
    device: str = "cuda"

    sample_size: tuple[int] = (64, 64)
    num_channels: int = 2

    block_out_channels: tuple[int] = (16, 32, 48, 64)
    layers_per_block: int = 2


def get_model(cfg: UNetConfig) -> UNet2DModel:
    model = UNet2DModel(
        sample_size=cfg.sample_size,
        in_channels=cfg.num_channels,
        out_channels=cfg.num_channels,
        block_out_channels=cfg.block_out_channels,
        layers_per_block=cfg.layers_per_block,
    )

    return model


def main():
    cfg = UNetConfig()
    model = get_model(cfg).to(cfg.device)

    # Display number of model weights
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")

    # Generate random input data of shape [B, C, H, W]
    batch_size = 1
    X = torch.randn(batch_size, cfg.num_channels, *cfg.sample_size, device=cfg.device)

    # Run the data through the model
    with torch.inference_mode():
        outputs = model(sample=X, timestep=1)
        Y = outputs["sample"]

    print(f"Input shape is: {X.shape}\nOutput shape is: {Y.shape}")


if __name__ == "__main__":
    main()
