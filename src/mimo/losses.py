import torch


def add_noise_to_data(data: torch.Tensor, stddev: torch.Tensor) -> torch.Tensor:
    noise = stddev[..., None, None] * torch.randn_like(data)
    noisy_data = data + noise

    return noisy_data, noise


def score_training_loss(
    model_pred: torch.Tensor, noise: torch.Tensor, coefficient: torch.Tensor
) -> torch.Tensor:
    """
    Computes a score-based generative model training loss

    :param model_pred: The model prediction s(h + z)
    :type model_pred: torch.complex
    :shape model_pred: [B, H, W]
    :param noise: The added noise z
    :type noise: torch.complex
    :shape noise: [B, H, W]
    :param coefficient: Noise weight coefficient
    :type coefficient: torch.float
    :shape coefficient: [B]
    """

    loss = model_pred + noise / coefficient[..., None, None]
    loss = torch.linalg.norm(loss, axis=(-1, -2)).square()
    weighted_loss = coefficient * loss
    average_loss = torch.mean(weighted_loss)

    return average_loss
