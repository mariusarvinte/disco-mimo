import torch


from mimo.models import UNetConfig, get_model
from mimo.data import DataConfig, get_data
from mimo.data import complex_to_real, real_to_complex


def main():
    cfg_model = UNetConfig()
    cfg_data = DataConfig()
    model = get_model(cfg_model, cfg_data).to(cfg_model.device)

    # Display number of model weights
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params} weights!")

    # Get data
    data = get_data(cfg_data)

    # Run some data through the model
    with torch.inference_mode():
        # Prepare data
        data_subset = data[:10].to(cfg_model.device)
        data_subset_real = complex_to_real(data_subset)

        # Pass through model
        outputs = model(sample=data_subset_real, timestep=1)
        output = outputs["sample"]

        # Post-process output back to complex values
        output = real_to_complex(output)

    print(f"Input shape is {data_subset.shape} and data type is {data_subset.dtype}")
    print(f"Output shape is {output.shape} and data type is {output.dtype}")


if __name__ == "__main__":
    main()
