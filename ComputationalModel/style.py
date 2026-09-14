"""Shared publication style for all figures in a manuscript. Import in every
figure script so typography, sizing, colour, and export stay identical across
panels:

    from style import setup, save_panel, figsize_mm, DIV_CMAP, SEQ_CMAP
    setup()
    fig, ax = plt.subplots(figsize=figsize_mm(85, 55))   # final size, in mm
    ...
    save_panel(fig, OUT_DIR / "panel_A")   # -> SVG + PDF (+ PNG), editable text

Design target: a vector pipeline for high-impact journals
(Python -> SVG/PDF panels -> Inkscape assembly -> final figure). Therefore:

  * panels are generated at final publication size -- never rescale later, or
    text and line weights drift away from the journal's specs
  * Arial, 7-8 pt; text stays editable in the SVG (``svg.fonttype = "none"``)
    so co-authors can fix a typo in Inkscape without re-running Python
  * fonts, line widths, and colours are centralized here, not repeated per
    script -- one edit re-themes the whole figure set
  * two canonical colormaps cover all data so the figures read as one family:
        DIV_CMAP  brightRdBu  -- diverging: brain maps, connectivity, contrasts
        SEQ_CMAP  brightPMY   -- sequential: effect size / responsiveness
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Importing custom_cmaps registers brightRdBu / brightPMY (+ _r, _lin) with
# matplotlib, so they are usable by string name (e.g. in surfplot) too.
from custom_cmaps import brightRdBu, brightPMY, brightRdBu_lin, brightPMY_lin  # noqa: F401


# ---- figure sizing (millimetres) ----------------------------------------
MM = 1.0 / 25.4                  # millimetres -> inches
SINGLE_COL_MM = 85.0             # typical single-column width
ONEHALF_COL_MM = 114.0
DOUBLE_COL_MM = 174.0            # typical double-column width


def figsize_mm(w_mm: float, h_mm: float) -> tuple[float, float]:
    """Figure size in inches from a width/height given in millimetres."""
    return (w_mm * MM, h_mm * MM)


# ---- rcParams ------------------------------------------------------------
def setup():
    """Apply the publication rcParams. Call once at the top of every script."""
    plt.rcParams.update({
        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.titleweight": "regular",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "figure.labelsize": 8,
        # lines / axes / ticks
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # vector cleanliness + editable text
        "svg.fonttype": "none",      # keep text as text in SVG (editable)
        "pdf.fonttype": 42,          # embed TrueType so text stays editable
        "ps.fonttype": 42,
        # output: transparent by default for clean Inkscape compositing
        "figure.dpi": 150,
        "savefig.dpi": 600,          # only affects rasterized insets (e.g. brains)
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "savefig.transparent": True,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
    })


# ---- canonical colormaps -------------------------------------------------
# Diverging blue <-> red for anything centred on zero (brain maps, connectivity).
DIV_CMAP = brightRdBu
# Sequential purple -> yellow for one-sided magnitudes (responsiveness, effect).
SEQ_CMAP = brightPMY
# Registered string names, for libraries that need a name (e.g. surfplot/VTK).
DIV_CMAP_NAME = "brightRdBu"
SEQ_CMAP_NAME = "brightPMY"


# ---- export --------------------------------------------------------------
def save_panel(fig, path, *, formats=("svg", "pdf"), png_preview=True, dpi=600,
               close=True):
    """Export a modular panel for Inkscape assembly.

    Writes one file per format with a transparent background and editable text
    (vector SVG by default, plus PDF). ``path`` may omit an extension.

    Parameters
    ----------
    formats      : vector formats to write (default SVG + PDF).
    png_preview  : also drop a raster PNG for quick visual checks.
    dpi          : raster resolution (PNG, and any rasterized image insets).
    close        : close the figure afterwards (avoids state leaking).
    """
    stem = Path(path).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    exts = list(formats) + (["png"] if png_preview else [])
    for ext in exts:
        fig.savefig(stem.with_suffix(f".{ext}"),
                    bbox_inches="tight", transparent=True,
                    dpi=(dpi if ext == "png" else None))
    if close:
        plt.close(fig)
    print(f"  -> {stem.name}.{{{','.join(exts)}}}")
    return stem


# ---- coherent categorical palette ---------------------------------------
# Neutrals for non-data ink: lines, guides, schematic fills.
INK = "#1a1a1a"        # primary dark: single emphasis lines, text
GUIDE = "#9aa0a8"      # dashed guide lines / subtle dividers
MUTE = "#f2f2f5"       # neutral light fill (schematic boxes)

# Categorical colours come from seaborn's "muted" qualitative palette: soft,
# colorblind-friendly, distinct. Keeping every discrete slot on one source
# (including the Yeo networks below) avoids a rainbow of clashing hues and
# reserves the bright data colormaps for actual data.
import seaborn as _sns  # noqa: E402
_CAT_QUAL = _sns.color_palette("muted")        # 10 soft qualitative colours


def categorical(n, palette=None):
    """``n`` distinct, soft, colorblind-friendly colours for any 'few discrete
    categories' slot -- group labels, bar groups, grouped scatters. Drawn from
    seaborn's muted palette so every categorical slot stays coherent."""
    pal = palette if palette is not None else _CAT_QUAL
    return list(pal[:n]) if n <= len(pal) else _sns.color_palette("muted", n)


def sequential_colors(n, cmap=SEQ_CMAP, lo=0.05, hi=0.95):
    """``n`` colours for an ordered stack / gradient (e.g. stacked traces).
    Defaults to brightPMY so a continuous gradient reads as the sequential
    data family."""
    return [cmap(x) for x in np.linspace(lo, hi, n)]


def text_on(color):
    """Return black or white, whichever is legible on ``color`` -- for labels
    drawn on top of coloured markers/bars."""
    r, g, b = mpl.colors.to_rgb(color)
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "#ffffff"


# Distinct, soft colours for the 7 Yeo resting-state networks, from the same
# muted palette -- distinguishable but not garish. Exposed as a list and dict.
YEO7_NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn",
                 "Cont", "Default", "Limbic"]
YEO7_COLOR_LIST = _sns.color_palette("muted", len(YEO7_NETWORKS))
YEO7_COLORS = dict(zip(YEO7_NETWORKS, YEO7_COLOR_LIST))

# Binary warm/cool accents (e.g. condition A vs B). Colorblind-friendly pair.
ACCENT_WARM = "#dd8452"          # muted orange
ACCENT_COOL = "#4c72b0"          # muted blue


# ---- small axis helpers --------------------------------------------------
def clean_axes(ax, keep=("left", "bottom")):
    """Hide non-data spines; keep only the ones in ``keep``."""
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(width=0.8, direction="out", length=3)


def rounded_box(ax, x, y, w, h, fc=MUTE, ec=INK, lw=1.0, rounding=0.05):
    """A rounded rectangle for schematic boxes (data-coords by default)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    return box
