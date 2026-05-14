"""
visualizer.py — Generate all heatmaps, plots, and overlay images.
Returns PIL Images or numpy arrays. NO st.* calls anywhere.
All matplotlib figures are closed after use to prevent memory leaks.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
PLOT_BG_COLOR: str = "#0d0d0d"
PLOT_TEXT_COLOR: str = "#e0e0e0"
PLOT_ACCENT: str = "#00ff88"
PLOT_DANGER: str = "#ff3333"
PLOT_WARN: str = "#ffcc00"


def _apply_dark_style(fig: plt.Figure, ax) -> None:
    """Apply consistent dark forensics theme to matplotlib figure."""
    fig.patch.set_facecolor(PLOT_BG_COLOR)
    if isinstance(ax, np.ndarray):
        axes = ax.ravel()
    elif hasattr(ax, '__iter__'):
        axes = list(ax)
    else:
        axes = [ax]
    for a in axes:
        a.set_facecolor("#111111")
        a.tick_params(colors=PLOT_TEXT_COLOR, labelsize=8)
        for spine in a.spines.values():
            spine.set_edgecolor("#333333")
        a.xaxis.label.set_color(PLOT_TEXT_COLOR)
        a.yaxis.label.set_color(PLOT_TEXT_COLOR)
        a.title.set_color(PLOT_ACCENT)


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Render a matplotlib figure to a PIL Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    img = Image.open(buf).copy()
    plt.close(fig)
    return img


# ── ELA Heatmap ───────────────────────────────────────────────────────────────

def render_ela_heatmap(ela_pil: Image.Image, mean_ela: float) -> Image.Image:
    """
    Render the ELA difference image as a styled heatmap figure.

    Args:
        ela_pil:  PIL Image of raw ELA difference.
        mean_ela: Mean ELA intensity value for annotation.

    Returns:
        PIL Image of the styled ELA heatmap.
    """
    ela_arr = np.array(ela_pil.convert("L"))

    fig, ax = plt.subplots(figsize=(4, 3))
    _apply_dark_style(fig, ax)
    im = ax.imshow(ela_arr, cmap="inferno", vmin=0, vmax=80)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04).ax.yaxis.set_tick_params(color=PLOT_TEXT_COLOR)
    ax.set_title(f"ELA HEATMAP  |  Mean: {mean_ela:.1f}", fontsize=9, pad=6)
    ax.axis("off")
    return fig_to_pil(fig)


# ── FFT Spectrum ──────────────────────────────────────────────────────────────

def render_fft_spectrum(log_magnitude: Optional[np.ndarray], periodicity_score: float) -> Image.Image:
    """
    Render the FFT log-magnitude spectrum as a styled plot.

    Args:
        log_magnitude:     2D numpy array of normalised log magnitude.
        periodicity_score: Measured periodicity score for annotation.

    Returns:
        PIL Image of the FFT spectrum plot.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _apply_dark_style(fig, ax)

    if log_magnitude is not None:
        im = ax.imshow(log_magnitude, cmap="hot", origin="lower")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04).ax.yaxis.set_tick_params(color=PLOT_TEXT_COLOR)
    else:
        ax.text(0.5, 0.5, "FFT data unavailable", ha="center", va="center",
                color=PLOT_WARN, transform=ax.transAxes)

    ax.set_title(
        f"FREQUENCY DOMAIN — GAN FINGERPRINT DETECTION\nPeriodicity Score: {periodicity_score:.3f}",
        fontsize=9, pad=6,
    )
    ax.axis("off")

    # Annotation
    note = "Periodic grid patterns = GAN artifact signature"
    fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="#888888", style="italic")
    return fig_to_pil(fig)


# ── Color Channel Bar Chart ───────────────────────────────────────────────────

def render_color_chart(face_stats: dict, border_stats: dict, delta: float) -> Image.Image:
    """
    Render side-by-side R/G/B bar charts for face vs border color comparison.

    Args:
        face_stats:   Color stats dict for the face region.
        border_stats: Color stats dict for the surrounding border.
        delta:        Sum of absolute channel mean differences.

    Returns:
        PIL Image of the matplotlib bar chart.
    """
    channels = ["R", "G", "B"]
    face_means = [face_stats.get(f"{c}_mean", 0) for c in channels]
    border_means = [border_stats.get(f"{c}_mean", 0) for c in channels]

    x = np.arange(len(channels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 2.8))
    _apply_dark_style(fig, ax)

    bars1 = ax.bar(x - width / 2, face_means, width, label="Face ROI",
                   color=["#cc2222", "#22cc22", "#2222cc"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, border_means, width, label="Border Region",
                   color=["#ff6666", "#66ff66", "#6666ff"], alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(["Red", "Green", "Blue"])
    ax.set_ylabel("Mean Pixel Value", color=PLOT_TEXT_COLOR, fontsize=8)
    ax.set_title(f"COLOR CHANNEL COMPARISON  |  Δ={delta:.1f}", fontsize=9, pad=6)
    ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333333",
              labelcolor=PLOT_TEXT_COLOR)
    ax.set_ylim(0, 275)
    return fig_to_pil(fig)


# ── Score Gauge ───────────────────────────────────────────────────────────────

def render_score_gauge(fake_score: float, is_fake: bool) -> Image.Image:
    """
    Render a semi-circular gauge showing the fake score.

    Args:
        fake_score: Final fake score 0-100.
        is_fake:    Whether the image was classified as fake.

    Returns:
        PIL Image of the gauge.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw={"aspect": "equal"})
    _apply_dark_style(fig, ax)
    ax.axis("off")

    # Background arc
    theta1, theta2 = 180, 0
    bg = mpatches.Wedge((0.5, 0.3), 0.4, theta2, theta1,
                        width=0.12, transform=ax.transAxes,
                        facecolor="#1a1a1a", edgecolor="#333333", lw=1)
    ax.add_patch(bg)

    # Score arc
    score_angle = 180 - (fake_score / 100.0) * 180
    color = PLOT_DANGER if is_fake else PLOT_ACCENT
    score_arc = mpatches.Wedge((0.5, 0.3), 0.4, score_angle, theta1,
                               width=0.12, transform=ax.transAxes,
                               facecolor=color, edgecolor=color, lw=0, alpha=0.9)
    ax.add_patch(score_arc)

    ax.text(0.5, 0.42, f"{fake_score:.1f}", ha="center", va="center",
            fontsize=22, color=color, fontweight="bold",
            fontfamily="monospace", transform=ax.transAxes)
    ax.text(0.5, 0.28, "FAKE SCORE", ha="center", va="center",
            fontsize=7, color="#888888", transform=ax.transAxes)
    ax.text(0.5, 0.08, "AUTHENTIC" if not is_fake else "DEEPFAKE",
            ha="center", va="center", fontsize=9, color=color,
            fontweight="bold", transform=ax.transAxes)

    return fig_to_pil(fig)


# ── LBP Texture Visual ────────────────────────────────────────────────────────

def render_lbp_visual(lbp_pil: Optional[Image.Image], entropy: float) -> Image.Image:
    """
    Render the LBP texture map with entropy annotation.

    Args:
        lbp_pil: Grayscale PIL Image of the LBP map.
        entropy: Computed Shannon entropy value.

    Returns:
        PIL Image of the styled LBP visual.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3))
    _apply_dark_style(fig, ax)

    if lbp_pil is not None:
        lbp_arr = np.array(lbp_pil)
        im = ax.imshow(lbp_arr, cmap="gray")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04).ax.yaxis.set_tick_params(color=PLOT_TEXT_COLOR)
    else:
        ax.text(0.5, 0.5, "LBP data unavailable", ha="center", va="center",
                color=PLOT_WARN, transform=ax.transAxes)

    ax.set_title(f"LBP TEXTURE MAP  |  Entropy: {entropy:.3f} bits", fontsize=9, pad=6)
    ax.axis("off")
    return fig_to_pil(fig)
