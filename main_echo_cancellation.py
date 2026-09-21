#!/usr/bin/env python3
"""
Acoustic echo cancellation with NLMS and an ideal double-talk detector
======================================================================
Cancels the echo of a far-end signal in a microphone recording with a
normalized LMS (NLMS) adaptive FIR filter, and compares two variants:

  * continuous adaptation, and
  * adaptation frozen from the sample where near-end speech starts
    (an *ideal* double-talk detector: the onset index is known a priori).

The script measures the baseline SNR, grid-searches the step size ``mu`` and
the filter order ``p`` for both variants, exports the cancelled audio and
plots the results.

Usage:
    python3 main_echo_cancellation.py

Author: Daniel Pereira Riquelme
Context: Digital Signal Processing course project
License: MIT
"""

from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import freqz

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

SAMPLE_RATE_HZ = 8000
NEAR_END_ONSET = 2150  # first sample of near-end speech (double-talk starts)
EPSILON = 1e-8         # regularization that avoids division by zero in NLMS
MU_GRID = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
ORDER_GRID = [2, 3, 4, 5, 6, 7]
FIGURE_DPI = 300


class NlmsResult(NamedTuple):
    """Output of one NLMS run."""
    error: np.ndarray    # cancelled signal e[n]
    weights: np.ndarray  # tap history, shape (n_samples, order)


class BestRun(NamedTuple):
    """Best grid-search point for one NLMS variant."""
    snr_db: float
    mu: float
    order: int
    result: NlmsResult
    snr_grid: np.ndarray  # shape (len(MU_GRID), len(ORDER_GRID))


def compute_snr(reference: np.ndarray, distortion: np.ndarray) -> float:
    """Return the SNR in dB of ``reference`` against ``distortion``."""
    return 10.0 * np.log10(np.sum(reference**2) / np.sum(distortion**2))


def run_nlms(far_end: np.ndarray, mic: np.ndarray, mu: float, order: int,
             freeze_from: Optional[int] = None) -> NlmsResult:
    """Run an NLMS adaptive FIR filter.

    Args:
        far_end: Far-end reference u[n] (loudspeaker signal).
        mic: Microphone signal s[n] = x[n] + y[n].
        mu: Step size (0 < mu < 2).
        order: Number of filter taps.
        freeze_from: If given, the weights stop adapting from this sample on.

    Returns:
        The error signal e[n] = s[n] - w^T u[n] and the tap history.
    """
    n_samples = len(mic)
    w = np.zeros(order)
    regressor = np.zeros(order)
    error = np.zeros(n_samples)
    weights = np.zeros((n_samples, order))

    for n in range(n_samples):
        regressor[1:] = regressor[:-1]
        regressor[0] = far_end[n]

        error[n] = mic[n] - w @ regressor

        if freeze_from is None or n < freeze_from:
            w += (mu * error[n] / (regressor @ regressor + EPSILON)) * regressor

        weights[n] = w

    return NlmsResult(error, weights)


def grid_search(far_end: np.ndarray, mic: np.ndarray, near_end: np.ndarray,
                freeze_from: Optional[int]) -> BestRun:
    """Sweep ``MU_GRID`` x ``ORDER_GRID`` and keep the run with the best SNR."""
    snr_grid = np.zeros((len(MU_GRID), len(ORDER_GRID)))
    best: Optional[BestRun] = None

    for i, mu in enumerate(MU_GRID):
        for j, order in enumerate(ORDER_GRID):
            result = run_nlms(far_end, mic, mu, order, freeze_from)
            snr = compute_snr(near_end, result.error - near_end)
            snr_grid[i, j] = snr
            if best is None or snr > best.snr_db:
                best = BestRun(snr, mu, order, result, snr_grid)

    return best._replace(snr_grid=snr_grid)


def _add_onset_marker(ax, x: float, label: Optional[str] = None) -> None:
    ax.axvline(x, color="black", linestyle="--", alpha=0.7, label=label)


def plot_time_domain(t: np.ndarray, mic: np.ndarray, near_end: np.ndarray,
                     snr_raw: float, plain: BestRun, dtd: BestRun) -> None:
    """Plot the microphone, both outputs and the clean near-end speech."""
    onset_s = NEAR_END_ONSET / SAMPLE_RATE_HZ
    panels = [
        (mic, "#d62728",
         f"Microphone signal $s[n] = x[n] + y[n]$ (SNR = {snr_raw:.2f} dB)"),
        (plain.result.error, "#ff7f0e",
         rf"NLMS, continuous adaptation ($\mu={plain.mu}$, $p={plain.order}$): "
         f"SNR = {plain.snr_db:.2f} dB"),
        (dtd.result.error, "#2ca02c",
         rf"NLMS, frozen at $n={NEAR_END_ONSET}$ ($\mu={dtd.mu}$, $p={dtd.order}$): "
         f"SNR = {dtd.snr_db:.2f} dB"),
        (near_end, "#1f77b4", "Clean near-end speech $x[n]$"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, dpi=FIGURE_DPI)
    for ax, (signal, color, title) in zip(axes, panels):
        ax.plot(t, signal, color=color, linewidth=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude")
        _add_onset_marker(ax, onset_s)
    axes[0].lines[-1].set_label(f"Near-end speech onset ($n={NEAR_END_ONSET}$)")
    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Time [s]")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "time_domain_signals.png", bbox_inches="tight")
    plt.close(fig)


def plot_tap_evolution(plain: BestRun, dtd: BestRun) -> None:
    """Plot the tap trajectories of both variants side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=FIGURE_DPI)
    panels = [
        (plain, "Continuous adaptation\n[near-end speech perturbs the taps]",
         "Near-end speech onset"),
        (dtd, "Adaptation frozen at the onset\n[taps are preserved]",
         "Freeze point"),
    ]
    for ax, (run, subtitle, marker_label) in zip(axes, panels):
        for k in range(run.order):
            ax.plot(run.result.weights[:, k], label=f"$w_{k}[n]$", linewidth=1.2)
        ax.axvline(NEAR_END_ONSET, color="black", linestyle="--",
                   label=f"{marker_label} ($n={NEAR_END_ONSET}$)")
        ax.set_title(rf"NLMS taps ($\mu={run.mu}$, $p={run.order}$)" + "\n" + subtitle,
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample index $n$")
        ax.set_ylabel("Tap weight")
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "filter_coefficient_evolution.png", bbox_inches="tight")
    plt.close(fig)


def plot_grid_search(plain: BestRun, dtd: BestRun) -> None:
    """Plot output SNR versus filter order for every step size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=FIGURE_DPI)
    panels = [
        (axes[0], plain, "o", "upper right", "Continuous adaptation"),
        (axes[1], dtd, "s", "lower right", "Adaptation frozen at the onset"),
    ]
    for ax, run, marker, legend_loc, title in panels:
        for i, mu in enumerate(MU_GRID):
            ax.plot(ORDER_GRID, run.snr_grid[i], marker=marker,
                    label=rf"$\mu={mu}$", linewidth=1.2)
        ax.set_title(f"{title}: SNR vs filter order $p$", fontsize=11, fontweight="bold")
        ax.set_xlabel("Filter order $p$")
        ax.set_ylabel("Output SNR [dB]")
        ax.legend(loc=legend_loc, fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "snr_grid_search_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_channel_response(dtd: BestRun) -> None:
    """Plot the frequency response of the filter taken at the freeze point."""
    taps = dtd.result.weights[NEAR_END_ONSET]
    freqs_hz, h = freqz(b=taps, a=1, worN=1024, fs=SAMPLE_RATE_HZ)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)
    phase_deg = np.degrees(np.unwrap(np.angle(h)))

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                           dpi=FIGURE_DPI)
    ax_mag.plot(freqs_hz, magnitude_db, color="#1f77b4", linewidth=1.5)
    ax_mag.set_title(rf"Estimated acoustic path $H(e^{{j\omega}})$ ($p={dtd.order}$ taps)",
                     fontsize=11, fontweight="bold")
    ax_mag.set_ylabel("Magnitude [dB]")

    ax_phase.plot(freqs_hz, phase_deg, color="#d62728", linewidth=1.5)
    ax_phase.set_title("Phase response", fontsize=11, fontweight="bold")
    ax_phase.set_ylabel("Unwrapped phase [deg]")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_xlim(0, SAMPLE_RATE_HZ / 2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "acoustic_channel_frequency_response.png",
                bbox_inches="tight")
    plt.close(fig)


def write_wav(path: Path, signal: np.ndarray) -> None:
    """Write ``signal`` as 16-bit PCM at the project sample rate."""
    wavfile.write(path, SAMPLE_RATE_HZ, np.clip(signal, -32768, 32767).astype(np.int16))


def load_wav(name: str) -> np.ndarray:
    """Load an 8 kHz WAV file from ``data/`` as float64."""
    rate, samples = wavfile.read(DATA_DIR / name)
    if rate != SAMPLE_RATE_HZ:
        raise ValueError(f"{name}: expected {SAMPLE_RATE_HZ} Hz, got {rate} Hz")
    return samples.astype(np.float64)


def main() -> None:
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    near_end = load_wav("near_end.wav")
    far_end = load_wav("far_end.wav")
    mic = load_wav("mic.wav")
    t = np.arange(len(mic)) / SAMPLE_RATE_HZ

    snr_raw = compute_snr(near_end, mic - near_end)
    plain = grid_search(far_end, mic, near_end, freeze_from=None)
    dtd = grid_search(far_end, mic, near_end, freeze_from=NEAR_END_ONSET)

    print("Acoustic echo cancellation: NLMS with ideal double-talk detection")
    print(f"  Unprocessed SNR:                {snr_raw:6.2f} dB")
    for label, run in (("Continuous NLMS", plain), (f"NLMS frozen @ n={NEAR_END_ONSET}", dtd)):
        print(f"  {label:<31} {run.snr_db:6.2f} dB  "
              f"(+{run.snr_db - snr_raw:.2f} dB, mu={run.mu}, p={run.order})")

    write_wav(OUTPUT_DIR / "signal_canceled_nlms.wav", plain.result.error)
    write_wav(OUTPUT_DIR / "signal_canceled_dtd.wav", dtd.result.error)

    plot_time_domain(t, mic, near_end, snr_raw, plain, dtd)
    plot_tap_evolution(plain, dtd)
    plot_grid_search(plain, dtd)
    plot_channel_response(dtd)
    print(f"  Audio written to {OUTPUT_DIR}, figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
