#!/usr/bin/env python3
"""
Acoustic Echo Cancellation (AEC) Engine
========================================
Normalized Least Mean Squares (NLMS) adaptive filtering with Double-Talk Detection (DTD).
Evaluates baseline SNR, performs 2D parameter optimization, and estimates acoustic channel response.

Author: Daniel Pereira Riquelme
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import freqz

# Plot configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def compute_snr(signal: np.ndarray, distortion: np.ndarray) -> float:
    """Calculates Signal-to-Noise Ratio (SNR) in dB."""
    return 10.0 * np.log10(np.sum(signal**2) / np.sum(distortion**2))


def run_nlms(u_far: np.ndarray, s_mic: np.ndarray, mu: float, p: int, dtd_sample: int | None = None):
    """
    Executes the Normalized Least Mean Squares (NLMS) adaptive FIR filter.

    Parameters:
        u_far: Far-end speech reference signal u[n]
        s_mic: Microphone pickup signal s[n] = x[n] + y[n]
        mu: Step-size convergence parameter
        p: Filter order (number of taps M = p)
        dtd_sample: If provided, freezes filter adaptation for n >= dtd_sample
    """
    n_samples = len(s_mic)
    w = np.zeros(p, dtype=np.float64)
    buffer = np.zeros(p, dtype=np.float64)
    e = np.zeros(n_samples, dtype=np.float64)
    w_hist = np.zeros((n_samples, p), dtype=np.float64)

    eps = 1e-8  # Regularization constant to prevent division by zero

    for n in range(n_samples):
        # Update tapped delay line buffer
        buffer[1:] = buffer[:-1]
        buffer[0] = u_far[n]

        # Filter output (estimated echo)
        y_hat = np.dot(w, buffer)

        # Error signal (cancelled output)
        e[n] = s_mic[n] - y_hat

        # Adapt filter weights if DTD is inactive
        if dtd_sample is None or n < dtd_sample:
            pwr = np.dot(buffer, buffer) + eps
            w += (mu * e[n] / pwr) * buffer

        w_hist[n, :] = w

    return e, w_hist


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Audio WAV Files
    fs_loc, local = wavfile.read(DATA_DIR / "local.wav")
    fs_rem, remota = wavfile.read(DATA_DIR / "remota.wav")
    fs_sig, signal_mic = wavfile.read(DATA_DIR / "signal.wav")

    assert fs_loc == fs_rem == fs_sig == 8000, "Sampling rate must be 8 kHz."

    local = local.astype(np.float64)
    remota = remota.astype(np.float64)
    signal_mic = signal_mic.astype(np.float64)
    n_samples = len(signal_mic)
    time_axis = np.arange(n_samples) / fs_loc

    print("=" * 70)
    print("  ACOUSTIC ECHO CANCELLATION (NLMS + DTD) OPTIMIZATION")
    print("=" * 70)

    # Question 1: Unprocessed Baseline SNR
    raw_distortion = signal_mic - local
    snr_raw = compute_snr(local, raw_distortion)
    print(f"\n[Task 1] Baseline SNR without Cancellation:")
    print(f"  -> Unprocessed SNR: {snr_raw:.2f} dB")

    # Question 2 & 3: 2D Grid Search
    u_vec = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]
    p_vec = [2, 3, 4, 5, 6, 7]

    snr_no_dtd_matrix = np.zeros((len(u_vec), len(p_vec)))
    snr_dtd_matrix = np.zeros((len(u_vec), len(p_vec)))

    best_snr_no_dtd = -np.inf
    u_opt_no_dtd, p_opt_no_dtd = 0, 0
    w_opt_no_dtd, e_opt_no_dtd = None, None

    best_snr_dtd = -np.inf
    u_opt_dtd, p_opt_dtd = 0, 0
    w_opt_dtd, e_opt_dtd = None, None

    dtd_index = 2150  # Sample where near-end local speech begins

    for i, u in enumerate(u_vec):
        for j, p in enumerate(p_vec):
            # Standard NLMS (No DTD)
            e_no_dtd, w_hist_no_dtd = run_nlms(remota, signal_mic, u, p, dtd_sample=None)
            snr_no_dtd = compute_snr(local, e_no_dtd - local)
            snr_no_dtd_matrix[i, j] = snr_no_dtd

            if snr_no_dtd > best_snr_no_dtd:
                best_snr_no_dtd = snr_no_dtd
                u_opt_no_dtd, p_opt_no_dtd = u, p
                w_opt_no_dtd = w_hist_no_dtd
                e_opt_no_dtd = e_no_dtd

            # DTD-Gated NLMS (Freeze @ n=2150)
            e_dtd, w_hist_dtd = run_nlms(remota, signal_mic, u, p, dtd_sample=dtd_index)
            snr_dtd = compute_snr(local, e_dtd - local)
            snr_dtd_matrix[i, j] = snr_dtd

            if snr_dtd > best_snr_dtd:
                best_snr_dtd = snr_dtd
                u_opt_dtd, p_opt_dtd = u, p
                w_opt_dtd = w_hist_dtd
                e_opt_dtd = e_dtd

    print(f"\n[Task 2] Standard NLMS (Continuous Adaptation, No DTD):")
    print(f"  -> Optimal Parameters: mu = {u_opt_no_dtd}, p = {p_opt_no_dtd}")
    print(f"  -> Maximum Output SNR: {best_snr_no_dtd:.2f} dB (Improvement: +{best_snr_no_dtd - snr_raw:.2f} dB)")

    print(f"\n[Task 3] DTD-Gated NLMS (Adaptation Frozen @ n >= {dtd_index}):")
    print(f"  -> Optimal Parameters: mu = {u_opt_dtd}, p = {p_opt_dtd}")
    print(f"  -> Maximum Output SNR: {best_snr_dtd:.2f} dB (Improvement: +{best_snr_dtd - snr_raw:.2f} dB)")
    print("=" * 70)

    # Export Processed Audio Files
    wavfile.write(OUTPUT_DIR / "signal_canceled_nlms.wav", fs_loc, np.clip(e_opt_no_dtd, -32768, 32767).astype(np.int16))
    wavfile.write(OUTPUT_DIR / "signal_canceled_dtd.wav", fs_loc, np.clip(e_opt_dtd, -32768, 32767).astype(np.int16))
    print(f"\n[Export] Saved audio files to {OUTPUT_DIR}/")

    # -------------------------------------------------------------
    # Plot 1: Time Domain Waveforms
    # -------------------------------------------------------------
    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True, dpi=300)
    axs[0].plot(time_axis, signal_mic, color='#d62728', linewidth=0.85)
    axs[0].set_title(f'Contaminated Microphone Signal $s[n] = x[n] + y[n]$ (Raw SNR = {snr_raw:.2f} dB)', fontsize=11, fontweight='bold')
    axs[0].set_ylabel('Amplitude', fontsize=10)
    axs[0].axvline(dtd_index / fs_loc, color='black', linestyle='--', alpha=0.7, label=f'Near-End Speech Start ($n={dtd_index}$)')
    axs[0].legend(loc='upper right', frameon=True)

    axs[1].plot(time_axis, e_opt_no_dtd, color='#ff7f0e', linewidth=0.85)
    axs[1].set_title(rf'Standard NLMS Output (No DTD, $\mu={u_opt_no_dtd}, p={p_opt_no_dtd}$) — Output SNR = {best_snr_no_dtd:.2f} dB', fontsize=11, fontweight='bold')
    axs[1].set_ylabel('Amplitude', fontsize=10)
    axs[1].axvline(dtd_index / fs_loc, color='black', linestyle='--', alpha=0.7)

    axs[2].plot(time_axis, e_opt_dtd, color='#2ca02c', linewidth=0.85)
    axs[2].set_title(rf'DTD-Gated NLMS Output (Freeze @ $n={dtd_index}$, $\mu={u_opt_dtd}, p={p_opt_dtd}$) — Output SNR = {best_snr_dtd:.2f} dB', fontsize=11, fontweight='bold')
    axs[2].set_ylabel('Amplitude', fontsize=10)
    axs[2].axvline(dtd_index / fs_loc, color='black', linestyle='--', alpha=0.7)

    axs[3].plot(time_axis, local, color='#1f77b4', linewidth=0.85)
    axs[3].set_title('Clean Near-End Ground Truth Speech $x[n]$', fontsize=11, fontweight='bold')
    axs[3].set_ylabel('Amplitude', fontsize=10)
    axs[3].set_xlabel('Time [seconds]', fontsize=10)
    axs[3].axvline(dtd_index / fs_loc, color='black', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "time_domain_signals.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Plot 2: Tap Coefficient Convergence Comparison
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    for k in range(p_opt_no_dtd):
        ax1.plot(w_opt_no_dtd[:, k], label=f'$w_{k}[n]$', linewidth=1.2)
    ax1.axvline(dtd_index, color='black', linestyle='--', label=f'Near-End Speech Start ($n={dtd_index}$)')
    ax1.set_title(rf'Standard NLMS Tap Adaptation ($\mu={u_opt_no_dtd}, p={p_opt_no_dtd}$)' + '\n[Near-End Interference Causes Tap Divergence]', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Sample Index $n$', fontsize=10)
    ax1.set_ylabel('Filter Tap Weight', fontsize=10)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

    for k in range(p_opt_dtd):
        ax2.plot(w_opt_dtd[:, k], label=f'$w_{k}[n]$', linewidth=1.2)
    ax2.axvline(dtd_index, color='black', linestyle='--', label=f'DTD Freeze Trigger ($n={dtd_index}$)')
    ax2.set_title(rf'DTD-Gated NLMS Adaptation ($\mu={u_opt_dtd}, p={p_opt_dtd}$)' + '\n[Adaptation Frozen to Protect Near-End Speech]', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Sample Index $n$', fontsize=10)
    ax2.set_ylabel('Filter Tap Weight', fontsize=10)
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "filter_coefficient_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Plot 3: 2D Grid Search SNR Analysis
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    for i, u in enumerate(u_vec):
        ax1.plot(p_vec, snr_no_dtd_matrix[i, :], marker='o', label=rf'$\mu={u}$', linewidth=1.2)
    ax1.set_title('Standard NLMS (No DTD): SNR vs Order $p$', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Filter Order $p$', fontsize=10)
    ax1.set_ylabel('Output SNR [dB]', fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, frameon=True)

    for i, u in enumerate(u_vec):
        ax2.plot(p_vec, snr_dtd_matrix[i, :], marker='s', label=rf'$\mu={u}$', linewidth=1.2)
    ax2.set_title('DTD-Gated NLMS: SNR vs Order $p$', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Filter Order $p$', fontsize=10)
    ax2.set_ylabel('Output SNR [dB]', fontsize=10)
    ax2.legend(loc='lower right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "snr_grid_search_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # Plot 4: Frequency & Impulse Response of Estimated Acoustic Channel
    # -------------------------------------------------------------
    w_final = w_opt_dtd[dtd_index, :]
    w_rad, h = freqz(b=w_final, a=1, worN=1024, fs=fs_loc)
    h_db = 20 * np.log10(np.abs(h) + 1e-12)
    phase_deg = np.unwrap(np.angle(h)) * 180 / np.pi

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=300)
    ax1.plot(w_rad, h_db, color='#1f77b4', linewidth=1.5)
    ax1.set_title(rf'Estimated Room Acoustic Channel Response $H(e^{{j\omega}})$ (Optimal FIR $p={p_opt_dtd}$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Magnitude [dB]', fontsize=10)
    ax1.grid(True)

    ax2.plot(w_rad, phase_deg, color='#d62728', linewidth=1.5)
    ax2.set_title('Phase Response (Linear Phase Characteristic)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Unwrapped Phase [deg]', fontsize=10)
    ax2.set_xlabel('Frequency [Hz]', fontsize=10)
    ax2.set_xlim([0, fs_loc / 2])
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "acoustic_channel_frequency_response.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[Done] Generated all 4 figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
