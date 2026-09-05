"""Animate the univariate cascade: python -m mmar.viz.cascade."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ..univariate import CASCADE_BLOCK, WINDOW
from . import OBSERVED_COLOR, PREDICTION_PURPLE, SECONDARY_COLOR


def cascade_stages(q=0.72, seed=20260905):
    """The model's seeded splits and phase, with intermediate levels retained."""
    rng = np.random.default_rng(seed)
    depth = int(np.log2(WINDOW // CASCADE_BLOCK))
    stages = [np.ones(1)]
    for _ in range(depth):
        parent = stages[-1]
        left = rng.random(parent.shape) < 0.5
        children = np.column_stack(
            (
                parent * np.where(left, 2 * q, 2 * (1 - q)),
                parent * np.where(left, 2 * (1 - q), 2 * q),
            )
        ).ravel()
        stages.append(children)
    return [stage / stage.mean() for stage in stages], int(rng.integers(WINDOW))


def save_cascade_gif(path="gifs/fractal_cascade.gif", q=0.72, seed=20260905):
    """Show mass-preserving splits, followed by the model's circular shift."""
    stages, phase = cascade_stages(q, seed)
    depth = len(stages) - 1
    daily = np.repeat(stages[-1], CASCADE_BLOCK)
    # Intermediate frames explain construction; they are not additional latent states.
    frames = [(0, 1.0, 0)] * 10
    for level in range(1, depth + 1):
        frames.extend((level, t, 0) for t in np.linspace(0, 1, 13)[1:])
        frames.extend([(level, 1.0, 0)] * 8)
    frames.extend((depth, 1.0, int(s)) for s in np.linspace(0, phase, 17)[1:])
    frames.extend([(depth, 1.0, phase)] * 20)

    cmap = LinearSegmentedColormap.from_list(
        "cascade", [SECONDARY_COLOR, "#E2D7F2", PREDICTION_PURPLE]
    )
    norm = Normalize(0, daily.max())
    fig, (tree, clock) = plt.subplots(
        2, 1, figsize=(10.5, 7.0), gridspec_kw={"height_ratios": [1.1, 1]}
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.83, bottom=0.10, hspace=0.25)
    fig.suptitle("From fractal splits to clustered volatility", fontsize=20, y=0.98)
    fig.text(
        0.5,
        0.91,
        f"q = {q:.2f}    ·    high intensity × {2*q:.2f}    ·    low intensity × {2*(1-q):.2f}",
        ha="center",
        fontsize=12,
        color=OBSERVED_COLOR,
    )

    def draw(frame):
        level, progress, shift = frame
        tree.clear()
        clock.clear()
        tree.set(xlim=(0, 1), ylim=(-0.5, depth + 0.5))
        tree.axis("off")
        for d in range(level + 1):
            for j, value in enumerate(stages[d]):
                x, y = (j + 0.5) / 2**d, depth - d
                alpha = progress if d == level and d > 0 else 1.0
                if d:
                    parent = stages[d - 1][j // 2]
                    color = PREDICTION_PURPLE if value > parent else SECONDARY_COLOR
                    tree.plot(
                        [(j // 2 + 0.5) / 2 ** (d - 1), x],
                        [y + 1, y],
                        color=color,
                        linewidth=2,
                        alpha=alpha,
                    )
                tree.scatter(x, y, s=34, color=OBSERVED_COLOR, alpha=alpha, zorder=3)
                tree.text(x, y + 0.13, f"{value:.2f}", ha="center", fontsize=11, alpha=alpha)

        if shift:
            values = np.roll(daily, -shift)
            label = f"Random circular shift: {shift} of {phase} days"
        elif level == 0:
            values = stages[0]
            label = "Start with uniform trading-time intensity"
        else:
            values = (1 - progress) * np.repeat(stages[level - 1], 2) + progress * stages[level]
            label = f"Split {level}/{depth}  ·  {2**level} intervals × {WINDOW // 2**level} days"
        width = WINDOW / len(values)
        clock.bar(
            np.arange(len(values)) * width,
            values,
            width=width,
            align="edge",
            color=cmap(norm(values)),
            linewidth=0,
        )
        clock.axhline(
            1, color=OBSERVED_COLOR, linestyle="--", linewidth=1.2, label="Mean intensity = 1"
        )
        clock.set(
            xlim=(0, WINDOW),
            ylim=(0, 1.17 * daily.max()),
            xlabel="Trading day",
            ylabel=r"Trading-time intensity $\theta_t$",
        )
        clock.set_title(label, fontsize=13, pad=10)
        clock.set_xticks(np.arange(0, WINDOW + 1, CASCADE_BLOCK))
        clock.tick_params(labelsize=11)
        clock.spines[["top", "right"]].set_visible(False)
        clock.legend(loc="upper right", frameon=False, fontsize=11)
        return tree, clock

    animation = FuncAnimation(fig, draw, frames=frames, interval=100, repeat=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=10), dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(save_cascade_gif())
