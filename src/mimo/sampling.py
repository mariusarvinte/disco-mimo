from dataclasses import dataclass
from tqdm import tqdm

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
    init: torch.Tensor,
    config: SamplingConfig,
    noise_levels: list[float],
    measurements: torch.Tensor | None = None,
    pilots: torch.Tensor | None = None,
    measurement_noise_std: float | None = None,
) -> torch.Tensor:
    # Initialize process
    current = init

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
            output = output["sample"]
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

    return current
