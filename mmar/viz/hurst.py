"""Animate how H changes the model's fractional-Gaussian driver ``z^(H)``."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .. import hurst as model
from . import OBSERVED_COLOR, PREDICTION_PURPLE, SECONDARY_COLOR


def matched_fractional_noise(hurst_values, num_observations=model.WINDOW, seed=20260907):
    """Filter the same IID Gaussian seed ``g`` at several H values.

    Holding the full seed fixed ensures that changes between driver paths
    ``z^(H)`` come from H rather than unrelated Monte Carlo draws. This is the
    Gaussian Davies--Harte stage of the Hurst MMAR; the cascade and Student-t
    radial scale are deliberately omitted.
    """
    hurst_values = np.asarray(hurst_values, dtype="float64")
    if hurst_values.ndim != 1 or not len(hurst_values):
        raise ValueError("hurst_values must be a non-empty one-dimensional sequence.")
    if np.any((hurst_values <= 0) | (hurst_values >= 1)):
        raise ValueError("Every Hurst coefficient must lie strictly between zero and one.")

    embedding_size = 2 * num_observations
    gaussian_seed = np.random.default_rng(seed).normal(size=embedding_size)
    frequencies = np.fft.rfft(gaussian_seed)
    driver_paths = np.empty((len(hurst_values), num_observations))

    for i, hurst in enumerate(hurst_values):
        autocovariance = model.fractional_gaussian_autocovariance(
            np.array([hurst]), num_observations
        )[0]
        circulant_row = np.concatenate(
            (autocovariance, np.zeros(1), autocovariance[1:][::-1])
        )
        eigenvalues = np.maximum(np.fft.rfft(circulant_row).real, 0.0)
        driver_paths[i] = np.fft.irfft(
            frequencies * np.sqrt(eigenvalues), n=embedding_size
        )[:num_observations]
    return driver_paths


def save_hurst_gif(
    path="gifs/hurst_memory.gif",
    hurst_values=None,
    num_observations=model.WINDOW,
    max_lag=24,
    seed=20260907,
    fps=10,
):
    """Animate the diagnostic driver path and its partial sum as H changes."""
    if hurst_values is None:
        hurst_values = np.linspace(0.35, 0.75, 41)
    hurst_values = np.asarray(hurst_values, dtype="float64")
    driver_paths = matched_fractional_noise(hurst_values, num_observations, seed)
    partial_sums = np.column_stack(
        (np.zeros(len(driver_paths)), np.cumsum(driver_paths, axis=1))
    )
    lags = np.arange(1, max_lag + 1)
    autocorrelations = model.fractional_gaussian_autocovariance(
        hurst_values, max_lag + 1
    )[:, 1:]

    forward = list(range(len(hurst_values)))
    backward = list(range(len(hurst_values) - 2, 0, -1))
    neutral = int(np.argmin(np.abs(hurst_values - 0.5)))
    frames = (
        [0] * 8
        + forward[: neutral + 1]
        + [neutral] * 8
        + forward[neutral + 1 :]
        + [len(hurst_values) - 1] * 8
        + backward
    )

    cmap = LinearSegmentedColormap.from_list(
        "hurst_memory", [SECONDARY_COLOR, OBSERVED_COLOR, PREDICTION_PURPLE]
    )
    norm = Normalize(hurst_values.min(), hurst_values.max())
    day = np.arange(1, num_observations + 1)
    partial_sum_day = np.arange(num_observations + 1)
    driver_limit = 1.08 * np.max(np.abs(driver_paths))
    partial_sum_limit = 1.08 * np.max(np.abs(partial_sums))
    correlation_limit = 1.15 * np.max(np.abs(autocorrelations))

    fig = plt.figure(figsize=(11.0, 6.5), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, width_ratios=(2.15, 1), height_ratios=(1, 1), hspace=0.35, wspace=0.28
    )
    driver_axis = fig.add_subplot(grid[0, 0])
    partial_sum_axis = fig.add_subplot(grid[1, 0])
    correlation_axis = fig.add_subplot(grid[:, 1])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.14)
    fig.suptitle("How the Hurst coefficient changes signed memory", fontsize=20, y=0.97)
    regime_label = fig.text(0.5, 0.895, "", ha="center", fontsize=14, weight="bold")
    fig.text(
        0.5,
        0.04,
        "Same Gaussian seed in every frame  ·  cascade and Student-t radial scale omitted",
        ha="center",
        fontsize=11,
        color=OBSERVED_COLOR,
    )

    def draw(frame):
        hurst = hurst_values[frame]
        driver = driver_paths[frame]
        path_color = cmap(norm(hurst))
        if np.isclose(hurst, 0.5):
            regime = "no linear memory — successive signs are unrelated"
        elif hurst < 0.5:
            regime = "anti-persistent — reversals are favored"
        else:
            regime = "persistent — same-direction runs are favored"
        regime_label.set_text(rf"$H={hurst:.2f}$  ·  {regime}")
        regime_label.set_color(path_color)

        driver_axis.clear()
        driver_axis.plot(day, driver, color=path_color, linewidth=1.0)
        driver_axis.axhline(0, color=OBSERVED_COLOR, linewidth=0.9, alpha=0.65)
        driver_axis.set(
            xlim=(1, num_observations),
            ylim=(-driver_limit, driver_limit),
            title="Fractional-Gaussian driver",
            ylabel=r"$z_t^{(H)}$",
        )

        partial_sum_axis.clear()
        partial_sum_axis.plot(
            partial_sum_day, partial_sums[frame], color=path_color, linewidth=2.0
        )
        partial_sum_axis.axhline(0, color=OBSERVED_COLOR, linewidth=0.9, alpha=0.65)
        partial_sum_axis.set(
            xlim=(0, num_observations),
            ylim=(-partial_sum_limit, partial_sum_limit),
            title=r"Diagnostic partial sum of $z_t^{(H)}$",
            xlabel="Trading day",
            ylabel="Partial sum",
        )

        correlation_axis.clear()
        correlation_axis.bar(
            lags,
            autocorrelations[frame],
            width=0.78,
            color=path_color,
            alpha=0.9,
        )
        correlation_axis.axhline(0, color=OBSERVED_COLOR, linewidth=1.0)
        correlation_axis.set(
            xlim=(0.25, max_lag + 0.75),
            ylim=(-correlation_limit, correlation_limit),
            title="Theoretical driver autocorrelation",
            xlabel="Lag (days)",
            ylabel=r"$\gamma_H(k)$",
        )
        correlation_axis.text(
            0.96,
            0.95,
            rf"$\gamma_H(1)={autocorrelations[frame, 0]:+.2f}$",
            transform=correlation_axis.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            color=path_color,
            weight="bold",
        )

        for axis in (driver_axis, partial_sum_axis, correlation_axis):
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=10)
        return driver_axis, partial_sum_axis, correlation_axis, regime_label

    animation = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, repeat=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(save_hurst_gif())
