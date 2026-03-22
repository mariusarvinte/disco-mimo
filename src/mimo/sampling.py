import os

from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt

import scipy
import torch
from torch.utils.data import DataLoader, TensorDataset

from mimo.data import DataConfig, complex_to_real, real_to_complex, get_data
from mimo.train import TrainConfig
from mimo.models import ModelConfig, get_model

from mimo.losses import add_noise_to_data
from mimo.losses import score_training_loss


@dataclass
class SamplingConfig:
    alpha_0: float
    r: float
    num_steps_outer: int

    num_steps_inner: int = 1
    beta: float = 1


def compute_epsilon(
    num_steps_inner: int, noise_step_factor: float, min_noise_level: float
) -> float:
    # Cost function
    def cost(epsilon: float) -> float:
        ratio = epsilon / min_noise_level**2
        denominator = min_noise_level**2 - min_noise_level**2 * (1 - ratio) ** 2
        fraction = 2 * epsilon / denominator

        value = (1 - ratio) ** (2 * num_steps_inner) * (
            (1 / noise_step_factor) ** 2 - fraction
        ) + fraction
        return abs(1 - value)

    # Line search
    result = scipy.optimize.brute(cost, ((1e-11, 1e-8),), Ns=100000, full_output=True)
    return result[0]


def sample_from_model(
    model: torch.nn.Module,
    config: SamplingConfig,
    noise_levels: list[float],
    batch_size: int,
    sample_size: tuple[int],
    measurements: torch.Tensor | None = None,
    pilots: torch.Tensor | None = None,
    measurement_noise_std: float | None = None,
) -> torch.Tensor:
    # Initialize process directly with real-valued signals
    current = noise_levels[0] * torch.randn(
        (batch_size, 2, *sample_size),
        dtype=torch.float32,
        device=noise_levels.device,
    )

    # Check if there's a mismatch between number of outer steps and noise levels
    if config.num_steps_outer != len(noise_levels):
        raise RuntimeError("Unequally spaced sampling not yet supported!")

    for outer_step in tqdm(range(config.num_steps_outer)):
        step_size = torch.tensor(
            config.alpha_0 * config.r ** (2 * outer_step), device=noise_levels.device
        )
        noise_std = (
            torch.sqrt(2 * torch.tensor(config.beta, device=noise_levels.device) * step_size)
            * noise_levels[outer_step]
        )
        for _ in range(config.num_steps_inner):
            # Predict with diffusion model
            output = model(
                current,
                outer_step * torch.ones(len(current), device=current.device, dtype=torch.long),
            )
            noise = torch.randn_like(output)

            # Apply unconditional update equation
            current = current + step_size * output + noise_std * noise

            # Apply conditional update equation
            if measurements is not None:
                if pilots is None or measurement_noise_std is None:
                    raise RuntimeError(
                        "Pilots and measurement noise must be passed together with measurements!"
                    )
                reconstructed_measurements = torch.matmul(real_to_complex(current), pilots)
                conditional = torch.matmul(
                    reconstructed_measurements - measurements,
                    pilots.transpose(-1, -2).conj(),
                ) / (measurement_noise_std**2 + noise_std**2)

                current = current - step_size * complex_to_real(conditional)

        # Early exit if we encounter NaN
        if current.isnan().any():
            print("WARNING: Sampling exited early because of NaN values!")
            return current

    return real_to_complex(current)


def plot_paired_data(top_row: torch.Tensor, bottom_row: torch.Tensor, save_file: Path):
    if top_row.shape != bottom_row.shape:
        raise ValueError("Cannot plot differently shaped top/bottom row samples!")
    b, h, w = top_row.shape

    plot_ratio = (b * w) / (2 * h)
    plt.figure(figsize=(4 * plot_ratio, 4))
    for i in range(b):
        plt.subplot(2, b, i + 1)
        plt.imshow(top_row[i].abs().cpu().numpy())
        plt.axis("off")
        plt.subplot(2, b, i + 1 + b)
        plt.imshow(bottom_row[i].abs().cpu().numpy())
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_file,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(cfg_model: ModelConfig, cfg_train: TrainConfig, cfg_data_val: DataConfig):
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
    model = get_model(cfg_model, noise_levels).to(cfg_model.device)

    cfg_sampling = SamplingConfig(
        alpha_0=3e-11
        * (1 / cfg_train.noise_step_factor) ** (2 * (cfg_train.num_noise_levels - 1)),
        r=cfg_train.noise_step_factor,
        num_steps_outer=cfg_train.num_noise_levels,
    )

    # Load real validation
    val_data = get_data(cfg_data_val)
    val_dataset = TensorDataset(val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg_train.val_batch_size, shuffle=True)
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
    print(f"Val. Loss {val_loss.item():.2f}")

    # Generate synthetic data
    with torch.inference_mode():
        synthetic_samples = sample_from_model(
            model,
            cfg_sampling,
            noise_levels,
            batch_size=cfg_train.val_batch_size,
            sample_size=cfg_train.sample_size,
        )

    # Plot the synthetic data
    save_dir = Path("samples")
    os.makedirs(save_dir, exist_ok=True)
    plot_paired_data(
        val_samples,
        synthetic_samples,
        save_dir / "unconditional.png",
    )


if __name__ == "__main__":
    cfg_model = ModelConfig(arch="ncsnv2", filename=Path("models") / "weights_step99999.pt")
    cfg_train = TrainConfig()

    if cfg_model.arch == "ncsnv2":
        # Populate sigma values
        cfg_model.config.set_sigmas(
            cfg_train.max_noise_level,
            cfg_train.max_noise_level
            * cfg_train.noise_step_factor ** (cfg_train.num_noise_levels - 1),
            cfg_train.num_noise_levels,
        )

    cfg_data_val = DataConfig(data_dir=Path("data"), data_tag="val")
    if cfg_model.arch == "ncsnv2":
        cfg_model.config.set_image_size(min(cfg_data_val.sample_size))

    main(cfg_model, cfg_train, cfg_data_val)
