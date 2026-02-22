from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mimo.models import UNetConfig, get_model
from mimo.data import DataConfig, get_data
from mimo.data import complex_to_real, real_to_complex

from mimo.losses import add_noise_to_data
from mimo.losses import score_training_loss

from mimo.sampling import SamplingConfig, sample_unconditional


@dataclass
class TrainConfig:
    lr: float = 0.0003

    batch_size: int = 16
    num_steps: int = 1000

    max_noise_level: float = 10.0
    num_noise_levels: int = 1000
    r: float = 0.99

    loss_verbose: int = 50
    sampling_verbose: int = 100
    sampling_batch: int = 4


def main(
    cfg_train: TrainConfig,
    cfg_model: UNetConfig,
    cfg_sampling: SamplingConfig,
    cfg_data: DataConfig,
):
    # Instantiate model
    model = get_model(cfg_model, cfg_data).to(cfg_model.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")

    noise_levels = torch.tensor(cfg_train.max_noise_level) * torch.tensor(
        cfg_train.r
    ) ** torch.arange(cfg_train.num_noise_levels).to(model.device)

    # Instantiate optimizer
    optimizer = Adam(model.parameters(), lr=cfg_train.lr)

    # Get clean dataset
    data = get_data(cfg_data)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=cfg_train.batch_size, shuffle=True)

    # Training loop
    for step in range(cfg_train.num_steps):
        batch = next(iter(dataloader))[0]
        batch = batch.to(cfg_model.device)

        # Pick noise levels uniformly at random
        stddev_idx = torch.multinomial(
            torch.ones_like(noise_levels), num_samples=len(batch), replacement=True
        )
        stddev = noise_levels[stddev_idx]

        # Run some noisy data through the model and get the loss function
        batch_noisy, noise = add_noise_to_data(batch, stddev)
        batch_noisy = complex_to_real(batch_noisy)

        # Pass through model
        outputs = model(sample=batch_noisy, timestep=1)
        output = outputs["sample"]

        # Post-process output back to complex values
        output = real_to_complex(output)

        # Compute the loss function
        loss = score_training_loss(output, noise, stddev.square())
        if step % cfg_train.loss_verbose == 0:
            print(f"Loss function on at step {step} is {loss.item()}")

        # Update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Sample from the model
        if step % cfg_train.sampling_verbose == 0:
            with torch.inference_mode():
                init = torch.randn(
                    (cfg_train.sampling_batch, *cfg_data.sample_size),
                    dtype=torch.complex64,
                    device=model.device,
                )
                val_samples = sample_unconditional(
                    model,
                    complex_to_real(init),
                    cfg_sampling,
                    noise_levels,
                )
                val_samples = real_to_complex(val_samples)


if __name__ == "__main__":
    cfg_train = TrainConfig()
    cfg_model = UNetConfig()
    cfg_sampling = SamplingConfig(
        num_steps_outer=cfg_train.num_noise_levels, alpha_0=1e-3, r=cfg_train.r
    )
    cfg_data = DataConfig(data_dir=Path("data"), data_tag="train")

    main(cfg_train, cfg_model, cfg_sampling, cfg_data)
