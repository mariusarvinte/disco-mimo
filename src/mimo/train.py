import torch


from mimo.models import UNetConfig, get_model
from mimo.data import DataConfig, get_data
from mimo.data import complex_to_real, real_to_complex

from mimo.losses import add_noise_to_data
from mimo.losses import score_training_loss


def main():
    cfg_model = UNetConfig()
    cfg_data = DataConfig()
    model = get_model(cfg_model, cfg_data).to(cfg_model.device)

    # Display number of model weights
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")

    # Get clean data
    data = get_data(cfg_data)

    # Run some noisy data through the model and get the loss function
    data_subset = data[:10].to(cfg_model.device)
    stddev = torch.rand(data_subset.shape[0], device=data_subset.device)
    data_subset_noisy, noise = add_noise_to_data(data_subset, stddev)
    data_subset_real = complex_to_real(data_subset_noisy)

    # Pass through model
    outputs = model(sample=data_subset_real, timestep=1)
    output = outputs["sample"]

    # Post-process output back to complex values
    output = real_to_complex(output)

    # Compute the loss function
    loss = score_training_loss(output, noise, stddev.square())

    print(f"Input shape is {data_subset.shape} and data type is {data_subset.dtype}")
    print(f"Output shape is {output.shape} and data type is {output.dtype}")
    print(f"Loss function on batch is {loss.item()}")


if __name__ == "__main__":
    main()
