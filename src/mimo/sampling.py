from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

from matplotlib import pyplot as plt

import torch

from mimo.data import complex_to_real, real_to_complex


@dataclass
class SamplingConfig:
    num_steps_outer: int
    alpha_0: float
    r: float

    num_steps_inner: int = 1
    beta: float = 1


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
        device=model.device,
    )

    # Check if there's a mismatch between number of outer steps and noise levels
    if config.num_steps_outer != len(noise_levels):
        raise RuntimeError("Unequally spaced sampling not yet supported!")

    for outer_step in tqdm(range(config.num_steps_outer)):
        step_size = torch.tensor(config.alpha_0 * config.r**outer_step, device=model.device)
        noise_std = (
            torch.sqrt(2 * torch.tensor(config.beta, device=model.device) * step_size)
            * noise_levels[outer_step]
        )
        for _ in range(config.num_steps_inner):
            # Predict with diffusion model
            output = model(current, timestep=1)
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
