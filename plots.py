import h5py
import hdf5storage
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


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


def plot_cdfs(paths: list[Path], labels: list[str], uses_hdf5: list[bool]) -> None:
    if not len(paths) == len(labels) == len(uses_hdf5):
        raise ValueError("List of files and plot labels must be synchronized!")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for path, label, use_hdf5 in zip(paths, labels, uses_hdf5):
        if use_hdf5:
            contents = hdf5storage.loadmat(path)
            channel = np.asarray(contents["output_h"], dtype=np.complex64)
            channel = channel[:, 0]
            channel = channel / np.sqrt(np.mean(np.abs(channel) ** 2))
            local_capacity = capacity(channel)
        else:
            with h5py.File(path, "r") as f:
                channel = np.asarray(f["data"], dtype=np.complex64)
            local_capacity = capacity(channel)

        x, y = get_cdf(local_capacity)
        ax.plot(x, y, label=label, linewidth=2.5)

    ax.set_title("Cumulative Distribution Function", fontsize=14)
    ax.set_xlabel("Capacity", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(45, 75)

    plt.tight_layout()
    plt.savefig("cdf_results.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    # Load reference (Matlab) CDL-C channels
    ref_file = Path("../") / "score-based-channels/data/CDL-C_Nt64_Nr16_ULA0.50_seed1234.mat"
    contents = hdf5storage.loadmat(ref_file)
    ref_channels = np.asarray(contents["output_h"], dtype=np.complex64)
    # Only use the first subcarrier
    ref_channels = ref_channels[:, 0]
    # Normalize channels
    ref_channels = ref_channels / np.sqrt(np.mean(np.abs(ref_channels) ** 2))
    ref_capacity = capacity(ref_channels)

    # Load reproduced (Sionna) CDL-C and CDL-B channels
    repro_file = Path("data-new") / "train.h5"
    with h5py.File(repro_file, "r") as f:
        channels = np.asarray(f["data"], dtype=np.complex64)
    repro_capacity = capacity(channels)

    # Calculate CDFs
    x1, y1 = get_cdf(ref_capacity)
    x2, y2 = get_cdf(repro_capacity)

    # Plot lines
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x1, y1, label="Original", linewidth=2)
    ax.plot(x2, y2, label="Reproduced", linewidth=2)
    ax.set_title("Cumulative Distribution Function", fontsize=14)
    ax.set_xlabel("Capacity", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("cdf_results.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
