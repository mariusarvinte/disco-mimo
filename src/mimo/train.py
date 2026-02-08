from dataclasses import dataclass

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from mimo.models import UNetConfig, get_model
from mimo.data import DataConfig, get_data
from mimo.data import complex_to_real, real_to_complex

from mimo.losses import add_noise_to_data
from mimo.losses import score_training_loss


@dataclass
class TrainConfig:
    lr: float = 0.0003

    batch_size: int = 16
    num_steps: int = 1000

    loss_verbose: int = 50


def main(cfg_train: TrainConfig, cfg_model: UNetConfig, cfg_data: DataConfig):
    # Instantiate model
    model = get_model(cfg_model, cfg_data).to(cfg_model.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")

    # Instantiate optimizer
    optimizer = Adam(model.parameters(), lr=cfg_train.lr)

    # Get clean dataset
    data = get_data(cfg_data)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=cfg_train.batch_size, shuffle=True)

    # Training loop
    for step in range(cfg_train.num_steps):
        # TODO: Write this cleanly
        batch = next(iter(dataloader))[0]
        batch = batch.to(cfg_model.device)
        stddev = torch.rand(len(batch), device=batch.device)

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


if __name__ == "__main__":
    cfg_train = TrainConfig()
    cfg_model = UNetConfig()
    cfg_data = DataConfig()

    main(cfg_train, cfg_model, cfg_data)
