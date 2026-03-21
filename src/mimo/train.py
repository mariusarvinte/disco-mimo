import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mimo.models import ModelConfig, get_model
from mimo.data import DataConfig, get_data
from mimo.data import complex_to_real, real_to_complex

from mimo.losses import add_noise_to_data
from mimo.losses import score_training_loss

from mimo.sampling import SamplingConfig, sample_from_model
from mimo.sampling import plot_paired_data


@dataclass
class TrainConfig:
    lr: float = 0.0001

    batch_size: int = 32
    num_steps: int = 50000

    max_noise_level: float = 39.15
    noise_step_factor: float = 0.995
    num_noise_levels: int = 2311

    loss_verbose: int = 100
    val_batch_size: int = 4
    sample_unconditional: bool = True
    sample_conditional: bool = False
    sampling_verbose: int = 2000
    sampling_batch: int = 4

    save_dir: Path = Path("models")


def main(
    cfg_train: TrainConfig,
    cfg_model: ModelConfig,
    cfg_sampling: SamplingConfig,
    cfg_data_train: DataConfig,
    cfg_data_val: DataConfig,
):
    # Instantiate model
    noise_levels = np.exp(
        np.linspace(
            np.log(cfg_train.max_noise_level),
            np.log(
                cfg_train.max_noise_level
                * cfg_train.noise_step_factor ** (cfg_train.num_noise_levels - 1)
            ),
            cfg_train.num_noise_levels,
        )
    )
    noise_levels = torch.tensor(noise_levels, device=cfg_model.device, dtype=torch.float32)

    model = get_model(cfg_model, cfg_data_train, noise_levels).to(cfg_model.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")
    os.makedirs(cfg_train.save_dir, exist_ok=True)

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
    trailing_loss = 0.0
    train_loss_log, val_loss_log = [], []
    for step in range(cfg_train.num_steps):
        model.train()
        batch = next(iter(train_dataloader))[0]
        batch = batch.to(cfg_model.device)

        # Pick noise levels uniformly at random
        stddev_idx = torch.multinomial(
            torch.ones_like(noise_levels), num_samples=len(batch), replacement=True
        )
        stddev = noise_levels[stddev_idx]
        batch_noisy, noise = add_noise_to_data(batch, stddev)
        batch_noisy = complex_to_real(batch_noisy)

        # Pass through model
        output = model(batch_noisy, stddev_idx)
        output = real_to_complex(output)

        # Compute the loss function
        loss = score_training_loss(output, noise, stddev.square())
        trailing_loss = loss.item() if step == 0 else trailing_loss * 0.99 + loss.item() * 0.01
        train_loss_log.append(trailing_loss)

        # Update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate and sample from the model
        if (step + 1) % cfg_train.loss_verbose == 0:
            model.eval()
            with torch.inference_mode():
                # Get a batch of validation samples
                val_samples = next(iter(val_dataloader))[0]
                val_samples = val_samples.to(cfg_model.device)

                # Compute validation loss
                stddev_idx = torch.multinomial(
                    torch.ones_like(noise_levels),
                    num_samples=len(val_samples),
                    replacement=True,
                )
                stddev = noise_levels[stddev_idx]
                val_samples_noisy, noise = add_noise_to_data(val_samples, stddev)
                val_samples_noisy = complex_to_real(val_samples_noisy)

                # Pass through model
                output = model(val_samples_noisy, stddev_idx)
                output = real_to_complex(output)
                # Compute the loss function
                val_loss = score_training_loss(output, noise, stddev.square())
                print(
                    f"Step {step}, Train Loss {trailing_loss:.2f}, Val. Loss {val_loss.item():.2f}"
                )
                val_loss_log.append(val_loss)

                # Unconditional sampling
                if (step + 1) % cfg_train.sampling_verbose == 0:
                    synthetic_samples = (
                        sample_from_model(
                            model,
                            cfg_sampling,
                            noise_levels,
                            batch_size=cfg_train.val_batch_size,
                            sample_size=cfg_data_train.sample_size,
                        )
                        if cfg_train.sample_unconditional
                        else torch.randn_like(val_samples)
                    )
                    # Plot the synthetic data
                    plot_paired_data(
                        val_samples,
                        synthetic_samples,
                        cfg_train.save_dir / f"unconditional_step{step}.png",
                    )
                    # Save model weights to disk
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                            "cfg_train": cfg_train,
                            "train_loss_log": train_loss_log,
                            "val_loss_log": val_loss_log,
                        },
                        cfg_train.save_dir / f"weights_step{step}.pt",
                    )


if __name__ == "__main__":
    cfg_train = TrainConfig()
    cfg_model = ModelConfig(arch="ncsnv2")

    if cfg_model.arch == "ncsnv2":
        # Populate sigma values
        cfg_model.config.set_sigmas(
            cfg_train.max_noise_level,
            cfg_train.max_noise_level
            * cfg_train.noise_step_factor ** (cfg_train.num_noise_levels - 1),
            cfg_train.num_noise_levels,
        )

    cfg_sampling = SamplingConfig(
        num_steps_outer=cfg_train.num_noise_levels,
        alpha_0=1e-11,
        r=cfg_train.noise_step_factor,
    )
    cfg_data_train = DataConfig(data_dir=Path("data"), data_tag="train")
    if cfg_model.arch == "ncsnv2":
        cfg_model.config.set_image_size(min(cfg_data_train.sample_size))
    cfg_data_val = DataConfig(data_dir=Path("data"), data_tag="val")

    main(cfg_train, cfg_model, cfg_sampling, cfg_data_train, cfg_data_val)
