import torch


def add_noise_to_data(data: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
    noise = stddev[..., None, None, None] * torch.randn_like(data)
    noisy_data = data + noise

    return noisy_data, noise


def score_training_loss(
    model_pred: torch.Tensor, noise: torch.Tensor, stddev: torch.Tensor
) -> torch.Tensor:
    """
    Computes a score-based generative model training loss

    :param model_pred: The model prediction s(h + z)
    :type model_pred: torch.complex
    :shape model_pred: [B, 2, H, W]
    :param noise: The added noise z
    :type noise: torch.complex
    :shape noise: [B, 2, H, W]
    """

    loss = model_pred + noise / stddev.square()[..., None, None, None]
    loss = 1 / 2.0 * torch.sum(torch.square(loss), dim=(-1, -2, -3))
    weighted_loss = stddev.square() * loss
    average_loss = torch.mean(weighted_loss)

    return average_loss
