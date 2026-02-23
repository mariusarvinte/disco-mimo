import os
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mimo.models import UNetConfig, get_model
from mimo.data import DataConfig, get_data
from mimo.data import complex_to_real, real_to_complex

from mimo.losses import add_noise_to_data
from mimo.losses import score_training_loss

from mimo.sampling import SamplingConfig, sample_from_model


@dataclass
class TrainConfig:
    lr: float = 0.0003

    batch_size: int = 16
    num_steps: int = 1000

    max_noise_level: float = 10.0
    num_noise_levels: int = 1000
    r: float = 0.99

    loss_verbose: int = 50
    val_batch_size: int = 4
    sampling_verbose: int = 500
    sampling_batch: int = 4

    save_dir: Path = Path("models")


def main(
    cfg_train: TrainConfig,
    cfg_model: UNetConfig,
    cfg_sampling: SamplingConfig,
    cfg_data_train: DataConfig,
    cfg_data_val: DataConfig,
):
    # Instantiate model
    model = get_model(cfg_model, cfg_data_train).to(cfg_model.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")
    os.makedirs(cfg_train.save_dir, exist_ok=True)

    noise_levels = torch.tensor(cfg_train.max_noise_level) * torch.tensor(
        cfg_train.r
    ) ** torch.arange(cfg_train.num_noise_levels).to(model.device)

    # Instantiate optimizer
    optimizer = Adam(model.parameters(), lr=cfg_train.lr)

    # Get data
    train_data = get_data(cfg_data_train)
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg_train.batch_size, shuffle=True)

    val_data = get_data(cfg_data_val)
    val_dataset = TensorDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg_train.val_batch_size, shuffle=True)

    # Training loop
    for step in range(cfg_train.num_steps):
        batch = next(iter(train_dataloader))[0]
        batch = batch.to(model.device)

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
            print(f"Loss function at step {step} is {loss.item()}")

        # Update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate and sample from the model
        if (step + 1) % cfg_train.sampling_verbose == 0:
            with torch.inference_mode():
                # Get a batch of validation samples
                val_samples = next(iter(val_dataloader))[0]
                val_samples = val_samples.to(model.device)

                # Unconditional
                init = torch.randn(
                    (cfg_train.val_batch_size, *cfg_data_train.sample_size),
                    dtype=torch.complex64,
                    device=model.device,
                )
                synthetic_samples = sample_from_model(
                    model,
                    complex_to_real(init),
                    cfg_sampling,
                    noise_levels,
                )
                synthetic_samples = real_to_complex(synthetic_samples)
                # Plot the synthetic data
                plt.figure()
                for i in range(cfg_train.val_batch_size):
                    plt.subplot(2, cfg_train.val_batch_size, i + 1)
                    plt.imshow(val_samples[i].abs().cpu().numpy())
                    plt.axis("off")
                    plt.subplot(2, cfg_train.val_batch_size, i + 1 + cfg_train.val_batch_size)
                    plt.imshow(synthetic_samples[i].abs().cpu().numpy())
                    plt.axis("off")
                plt.tight_layout()
                plt.savefig(
                    cfg_train.save_dir / f"unconditional_step{step}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Conditional
                pilots_real = torch.randn(
                    val_samples.shape[0],
                    val_samples.shape[-1],
                    int(val_samples.shape[-1] * cfg_data_val.undersampling),
                    device=model.device,
                ).sign()
                pilots_imag = torch.randn(
                    val_samples.shape[0],
                    val_samples.shape[-1],
                    int(val_samples.shape[-1] * cfg_data_val.undersampling),
                    device=model.device,
                ).sign()
                pilots = 1 / np.sqrt(2) * (pilots_real + 1j * pilots_imag)
                clean_y = torch.matmul(val_samples, pilots)
                noisy_y = clean_y + cfg_data_val.measurement_noise_std * torch.randn_like(
                    clean_y
                )

                recon_samples = sample_from_model(
                    model,
                    complex_to_real(val_samples),
                    cfg_sampling,
                    noise_levels,
                    noisy_y,
                    pilots,
                    cfg_data_val.measurement_noise_std,
                )
                recon_samples = real_to_complex(recon_samples)
                # Measure reconstruction and plot if everything went ok
                if not recon_samples.isnan().any():
                    recon_mse = torch.sum(
                        torch.square(torch.abs(recon_samples - val_samples)), dim=(-1, -2)
                    )
                    print(f"Validation step {step}, MSE {recon_mse.cpu().numpy()}")
                    plt.figure()
                    for i in range(cfg_train.val_batch_size):
                        plt.subplot(1, cfg_train.val_batch_size, i + 1)
                        plt.imshow(recon_samples[i].abs().cpu().numpy())
                        plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(
                        cfg_train.save_dir / f"conditional_step{step}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()


if __name__ == "__main__":
    cfg_train = TrainConfig()
    cfg_model = UNetConfig()
    cfg_sampling = SamplingConfig(
        num_steps_outer=cfg_train.num_noise_levels, alpha_0=1e-3, r=cfg_train.r
    )
    cfg_data_train = DataConfig(data_dir=Path("data"), data_tag="train")
    cfg_data_val = DataConfig(data_dir=Path("data"), data_tag="val")

    main(cfg_train, cfg_model, cfg_sampling, cfg_data_train, cfg_data_val)
