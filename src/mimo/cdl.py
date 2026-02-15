import os

from dataclasses import dataclass
from pathlib import Path

import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sionna.phy.ofdm import ResourceGrid
from sionna.phy.channel.tr38901 import AntennaArray, CDL
from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel


@dataclass
class CDLConfig:
    num_rx: int = 16
    num_tx: int = 64

    cdl_model: str = "C"
    num_samples: int = 1000

    verbose: bool = False
    save_dir: Path = Path("data")
    save_tag: str = "train"

    max_chunk_product: int = 10000


def plot_tensor_grid(
    data: np.array,
    cmap: str = "viridis",
    spacing: float = 0.05,
    plot_dir: Path = Path("plots"),
):
    _, _, outer_dim, inner_dim = data.shape

    # Create the figure
    # We want 'delay' rows and 'time' columns
    fig, axes = plt.subplots(
        nrows=outer_dim,
        ncols=inner_dim,
        figsize=(outer_dim * 1.5, inner_dim * 1.5),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": spacing, "hspace": spacing},
    )

    # Global min/max for consistent color scaling
    v_min, v_max = data.min(), data.max()

    for o in range(outer_dim):
        for i in range(inner_dim):
            ax = axes[o, i]
            # Extract [rx, tx] slice
            slice_data = data[:, :, o, i]
            ax.imshow(slice_data, aspect="auto", cmap=cmap, vmin=v_min, vmax=v_max)

            # Clean up internal ticks to save space
            ax.set_xticks([])
            ax.set_yticks([])

            # Label the outer grid edges only
            if o == outer_dim - 1:
                ax.set_xlabel(f"F{i}", fontsize=8)
            if i == 0:
                ax.set_ylabel(f"T{o}", fontsize=8)

    # Add big labels for the entire grid
    fig.supxlabel("Frequency Axis", fontsize=12, fontweight="bold")
    fig.supylabel("Time Axis", fontsize=12, fontweight="bold")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(plot_dir / "frequency.png", dpi=300, bbox_inches="tight")
    plt.close()


def main(cfg: CDLConfig):
    # Define the number of UT and BS antennas
    num_ut_ant = num_streams_per_tx = cfg.num_rx
    num_bs_ant = cfg.num_tx
    rg = ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=76,
        subcarrier_spacing=15e3,
        num_tx=1,
        num_streams_per_tx=num_streams_per_tx,
        cyclic_prefix_length=6,
        num_guard_carriers=[5, 6],
        dc_null=True,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
    )

    carrier_frequency = 2.6e9  # [Hz]
    ut_array = AntennaArray(
        num_rows=1,
        num_cols=int(num_ut_ant / 2),
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency,
    )
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=int(num_bs_ant / 2),
        polarization="dual",
        polarization_type="cross",
        antenna_pattern="38.901",
        carrier_frequency=carrier_frequency,
    )

    delay_spread = 300e-9  # Nominal delay spread in [s]. Please see the CDL documentation
    # about how to choose this value.
    direction = "downlink"
    cdl_model = cfg.cdl_model
    speed = 10  # UT speed [m/s]

    # Configure a channel impulse reponse (CIR) generator for the CDL model.
    # cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
    cdl = CDL(
        cdl_model,
        delay_spread,
        carrier_frequency,
        ut_array,
        bs_array,
        direction,
        min_speed=speed,
    )

    # Chunk the channel generation process
    dataset = np.zeros((cfg.num_samples, cfg.num_rx, cfg.num_tx), dtype=np.complex128)
    if (size_prod := cfg.num_rx * cfg.num_tx) > cfg.max_chunk_product:
        print(
            f"Warning: the array sizes {cfg.num_rx, cfg.num_tx = } are larger than the chunk size, which may lead to OOM errors!"
        )
    chunk_size = cfg.max_chunk_product // size_prod
    num_chunks = int(np.ceil(cfg.num_samples / chunk_size))
    for i in tqdm(range(num_chunks)):
        samples_in_chunk = min((i + 1) * chunk_size, cfg.num_samples) - i * chunk_size
        gains, tau = cdl(
            batch_size=samples_in_chunk,
            num_time_steps=rg.num_ofdm_symbols,
            sampling_frequency=1 / rg.ofdm_symbol_duration,
        )
        # Move to frequency domain
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, gains, tau, normalize=True)
        h_freq = tf.squeeze(h_freq).numpy()

        # Subsample data at random in the time-frequency grid
        random_times = np.random.randint(0, h_freq.shape[-2], size=samples_in_chunk)
        random_freqs = np.random.randint(0, h_freq.shape[-1], size=samples_in_chunk)
        dataset[i * chunk_size : i * chunk_size + samples_in_chunk, ...] = h_freq[
            range(samples_in_chunk), ..., random_times, random_freqs
        ]

    # Save dataset to disk
    os.makedirs(cfg.save_dir, exist_ok=True)
    filename = f"CDL-{cfg.cdl_model}_rx{cfg.num_rx}_tx{cfg.num_tx}_{cfg.save_tag}.h5"
    with h5py.File(cfg.save_dir / filename, "w") as f:
        f.create_dataset("data", data=dataset)

    # Visualize gain matrix in the time-delay domain
    if cfg.verbose:
        print("Shape of the path gains: ", gains.shape)
        print("Shape of the delays:", tau.shape)
        print("Shape of the frequency-domain channel", h_freq.shape)
        plot_tensor_grid(np.abs(h_freq[0, :, :, :, ::8]))


if __name__ == "__main__":
    cfg = CDLConfig()
    main(cfg)
