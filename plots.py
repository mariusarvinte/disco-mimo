import torch
import h5py
import hdf5storage
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


def load_loss(path: Path, legacy: bool) -> npt.NDArray:
    contents = torch.load(path, map_location="cpu")
    if legacy:
        raw_loss = contents["train_loss"]
        # EMA with 0.99
        loss = [raw_loss[0]]
        for value in raw_loss[1:]:
            loss.append(loss[-1] * 0.99 + value * 0.01)
    else:
        loss = contents["train_loss_log"]

    return loss


def load_data(path: Path, use_hdf5: bool) -> npt.NDArray:
    if use_hdf5:
        contents = hdf5storage.loadmat(path)
        channel = np.asarray(contents["output_h"], dtype=np.complex64)
        channel = channel[:, 0]
    else:
        with h5py.File(path, "r") as f:
            channel = np.asarray(f["data"], dtype=np.complex64)

    return channel


def capacity(channels: npt.NDArray[np.complex64], snr: float = 1.0) -> npt.NDArray[np.float32]:
    return np.log2(
        np.linalg.det(
            np.eye(channels.shape[-2])
            + snr
            * np.real(np.matmul(channels, np.transpose(np.conj(channels), axes=(0, 2, 1))))
        )
    )


def get_cdf(data: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32]]:
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_cdfs(
    paths: list[Path],
    labels: list[str],
    uses_hdf5: list[bool],
    transpose: list[bool],
    linestyles: list[str],
    name: str = "cdf_results.png",
    normalize: bool = True,
) -> None:
    if not len(paths) == len(labels) == len(uses_hdf5):
        raise ValueError("List of files and plot labels must be synchronized!")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    for path, label, use_hdf5, tp, linestyle in zip(
        paths, labels, uses_hdf5, transpose, linestyles
    ):
        # Load data
        channel = load_data(path, use_hdf5)

        # Data transformations
        if tp:
            channel = np.conj(np.transpose(channel, axes=(0, 2, 1)))
        if normalize:
            channel = channel / np.sqrt(np.mean(np.abs(channel) ** 2))

        # Compute channel capacity
        local_capacity = capacity(channel)
        x, y = get_cdf(local_capacity)
        ax.plot(x, y, label=label, linewidth=3, linestyle=linestyle)

    ax.set_title("Cumulative Distribution Function")
    ax.set_xlabel("Capacity [bits/Hz/channel use]")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(42.5, 85)

    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    paths: list[Path],
    labels: list[str],
    legacy: list[bool],
    name: str = "training_results.png",
) -> None:
    if not len(paths) == len(labels) == len(legacy):
        raise ValueError("List of files and plot labels must be synchronized!")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))

    for path, label, is_legacy in zip(paths, labels, legacy):
        # Load data
        loss = load_loss(path, is_legacy)
        ax.loglog(range(1, len(loss) + 1), loss, label=label, linewidth=3)

    ax.set_title("Training loss (exponential moving average)")
    ax.set_xlabel("Step")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=16)

    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Global font size
    plt.rcParams["font.size"] = 16

    paths = [
        Path("../") / "score-based-channels/data/CDL-C_Nt64_Nr16_ULA0.50_seed1234.mat",
        Path("data-new") / "train.h5",
        Path("data-new") / "unconditional-samples-original.h5",
        Path("data-new") / "unconditional-samples.h5",
        Path("data-new") / "unconditional-samples-repro-strict.h5",
    ]
    labels = [
        "Original (MATLAB)\ntraining data",
        "Refactored (Sionna)\ntraining data",
        "Original code\nsynthetic data",
        "Reproduced paper\nsynthetic data",
        "Reproduced code\nsynthetic data",
    ]
    uses_hdf5 = [True, False, False, False, False]
    transpose = [False, False, True, True, True]
    linestyles = ["--", "--", "-", "-", "-"]

    plot_cdfs(paths, labels, uses_hdf5, transpose, linestyles, name="cdf_plot.png")

    training_paths = [
        Path("../") / "score-based-channels/models/score/CDL-C/final_model.pt",
        Path("models-paper") / "CDL-C" / "weights_step124799.pt",
    ]
    training_labels = [
        "Original training",
        "Refactored training",
    ]
    plot_training_curves(
        training_paths, training_labels, legacy=[True, False], name="training.png"
    )


if __name__ == "__main__":
    main()
