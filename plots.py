import h5py
import hdf5storage
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


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
    name: str = "cdf_results.png",
    normalize: bool = True,
) -> None:
    if not len(paths) == len(labels) == len(uses_hdf5):
        raise ValueError("List of files and plot labels must be synchronized!")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    for path, label, use_hdf5, tp in zip(paths, labels, uses_hdf5, transpose):
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
        ax.plot(x, y, label=label, linewidth=2.5)

    ax.set_title("Cumulative Distribution Function", fontsize=14)
    ax.set_xlabel("Capacity [bits/Hz/channel use]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    # ax.set_xlim(45, 75)

    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    paths: list[Path],
    labels: list[str],
    uses_hdf5: list[bool],
    name: str = "training_results.png",
) -> None:
    pass


def main() -> None:
    paths = [
        Path("../") / "score-based-channels/data/CDL-C_Nt64_Nr16_ULA0.50_seed1234.mat",
        Path("data-new") / "train.h5",
        Path("data-new") / "unconditional-samples-original.h5",
        Path("data-new") / "unconditional-samples.h5",
        Path("data-new") / "unconditional-samples-repro-strict.h5",
    ]
    labels = [
        "Original (MATLAB) training data",
        "Refactored (Sionna) training data",
        "Original code synthetic data",
        "Reproduced paper synthetic data",
        "Reproduced code synthetic data",
    ]
    uses_hdf5 = [True, False, False, False, False]
    transpose = [False, False, True, True, True]

    plot_cdfs(paths, labels, uses_hdf5, transpose, name="penta_plot.png")


if __name__ == "__main__":
    main()
