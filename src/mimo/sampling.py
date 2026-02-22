from dataclasses import dataclass
from tqdm import tqdm

import torch


@dataclass
class SamplingConfig:
    num_steps_outer: int
    alpha_0: float
    r: float

    num_steps_inner: int = 1
    beta: float = 1


def sample_unconditional(
    model: torch.nn.Module,
    init: torch.Tensor,
    config: SamplingConfig,
    noise_levels: list[float],
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

            # Apply update equation
            current = current + step_size * output + noise_std * noise
    return current
