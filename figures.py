"""
Figure generation for the MII tool — reproduces manuscript Figures 2 and 4.

Figure 2: Source-level PRIOR (Level 4) distributions, small multiples,
          including engine/turbine exhaust (kg/mmbtu).
Figure 4: PRIOR (blue) vs POSTERIOR (orange) overlaid per category.
          Exhaust and other non-updated categories appear prior-only.

Colors follow the manuscript: prior/L4 = blue, posterior/L5 = orange.
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless / stlite safe
import matplotlib.pyplot as plt

PRIOR_COLOR = "#1f5fbf"     # blue (Level 4)
POST_COLOR  = "#e07b00"     # orange (Level 5)

# Colors for the seven source-category components in the per-station stacked
# bar chart. Chosen to be distinguishable from each other and from the
# prior/posterior blue-orange scheme used elsewhere.
STATION_COMPONENT_COLORS = {
    "exhaust":    "#4C72B0",   # blue
    "seals":      "#DD8452",   # orange
    "bdv":        "#55A868",   # green
    "iso":        "#C44E52",   # red
    "tanks":      "#8172B2",   # purple
    "leaks":      "#937860",   # brown
    "pneumatics": "#DA8BC3",   # pink
}
STATION_COMPONENT_LABELS = {
    "exhaust":    "Combustion slip",
    "seals":      "Compressor seals",
    "bdv":        "Blowdown valve",
    "iso":        "Isolation valve",
    "tanks":      "Tanks",
    "leaks":      "Equipment leaks",
    "pneumatics": "Pneumatics",
}
STATION_COMPONENT_ORDER = ["exhaust", "seals", "pneumatics", "bdv", "iso", "tanks", "leaks"]

# Exhaust categories are prior-only and carry different units (kg/mmbtu).
EXHAUST_LABELS = {
    "2SLB": "Engine exhaust — 2SLB",
    "4SLB": "Engine exhaust — 4SLB",
    "Turbine": "Turbine exhaust",
}
EXHAUST_ORDER = ["2SLB", "4SLB", "Turbine"]

CATEGORY_LABELS = {
    "RPV_Op": "RPV operating", "RPV_Stdby": "RPV standby",
    "DSV_Op": "DSV operating", "DSV_Stdby": "DSV standby",
    "WSV_Op": "WSV operating",
    "BDV_Cent_Op": "BDV cent. operating", "BDV_Cent_Stdby": "BDV cent. standby",
    "BDV_Recip_Op": "BDV recip. operating", "BDV_Recip_Stdby": "BDV recip. standby",
    "ISO_Recip": "ISO recip.", "ISO_Cent": "ISO cent.",
    "Tank": "Tanks", "Leak": "Leaks", "Pneumatic": "Malf. pneumatics",
}

CATEGORY_ORDER = [
    "RPV_Op", "RPV_Stdby", "DSV_Op", "DSV_Stdby", "WSV_Op",
    "BDV_Recip_Op", "BDV_Recip_Stdby", "BDV_Cent_Op", "BDV_Cent_Stdby",
    "ISO_Recip", "ISO_Cent", "Tank", "Leak", "Pneumatic",
]


def _ordered_categories(sources):
    present = [c for c in CATEGORY_ORDER if c in sources]
    extras = [c for c in sources if c not in CATEGORY_ORDER]
    return present + sorted(extras)


def _ordered_exhaust(engines):
    """Exhaust driver types present with a non-degenerate distribution."""
    if not engines:
        return []
    out = []
    for d in EXHAUST_ORDER:
        arr = engines.get(d)
        if arr is not None and np.ptp(arr) > 0:   # skip all-zero / constant
            out.append(d)
    return out


def _grid(n):
    cols = min(4, n) if n else 1
    rows = math.ceil(n / cols) if n else 1
    return rows, cols


# ---------------------------------------------------------------------------
# Figure 2 — prior distributions (exhaust + all source categories)
# ---------------------------------------------------------------------------

def figure2_prior_distributions(result, bins=40):
    sources = result.sources
    engines = result.engines or {}
    exhaust = _ordered_exhaust(engines)
    cats = _ordered_categories(sources)

    # Panels: exhaust first (kg/mmbtu), then vent/other categories (kg/hr).
    panels = [("exhaust", d) for d in exhaust] + [("cat", c) for c in cats]
    rows, cols = _grid(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for i, (kind, key) in enumerate(panels):
        ax = axes[i]
        if kind == "exhaust":
            data = engines[key]
            label = EXHAUST_LABELS.get(key, key)
            xunit = "kg mmbtu$^{-1}$"
        else:
            data = sources[key]["prior"]
            label = CATEGORY_LABELS.get(key, key)
            xunit = None
        ax.hist(data, bins=bins, color=PRIOR_COLOR, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        m = float(np.mean(data))
        ax.axvline(m, color="red", linestyle="--", linewidth=1.2)
        ax.text(0.97, 0.92, f"{m:.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="red", fontweight="bold")
        title = label + (f"\n({xunit})" if xunit else "")
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Figure 2 — Source-level prior (BU / Level 4) mean emission-rate distributions",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.supxlabel("Category-mean emission rate — kg h$^{-1}$ (exhaust panels: kg mmbtu$^{-1}$)",
                  fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — prior vs posterior overlaid (exhaust prior-only)
# ---------------------------------------------------------------------------

def figure4_prior_posterior(result, bins=40):
    sources = result.sources
    engines = result.engines or {}
    exhaust = _ordered_exhaust(engines)
    cats = _ordered_categories(sources)

    panels = [("exhaust", d) for d in exhaust] + [("cat", c) for c in cats]
    rows, cols = _grid(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for i, (kind, key) in enumerate(panels):
        ax = axes[i]
        if kind == "exhaust":
            # Prior-only (exhaust not TD-updated).
            prior = engines[key]
            ax.hist(prior, bins=bins, color=PRIOR_COLOR, alpha=0.55,
                    edgecolor="white", linewidth=0.3)
            ax.axvline(float(np.mean(prior)), color=PRIOR_COLOR, linestyle="--", linewidth=1.2)
            label = EXHAUST_LABELS.get(key, key)
            ax.set_title(label + "\n(kg mmbtu$^{-1}$)", fontsize=8.5, fontweight="bold")
        else:
            prior = sources[key]["prior"]
            post = sources[key]["posterior"]
            updated = not np.array_equal(prior, post)
            lo = min(np.min(prior), np.min(post))
            hi = max(np.max(prior), np.max(post))
            edges = np.linspace(lo, hi, bins + 1)
            ax.hist(prior, bins=edges, color=PRIOR_COLOR, alpha=0.55,
                    edgecolor="white", linewidth=0.3)
            ax.axvline(float(np.mean(prior)), color=PRIOR_COLOR, linestyle="--", linewidth=1.2)
            if updated:
                ax.hist(post, bins=edges, color=POST_COLOR, alpha=0.55,
                        edgecolor="white", linewidth=0.3)
                ax.axvline(float(np.mean(post)), color="gray", linestyle="--", linewidth=1.2)
            ax.set_title(CATEGORY_LABELS.get(key, key), fontsize=8.5, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PRIOR_COLOR, alpha=0.55),
        plt.Rectangle((0, 0), 1, 1, color=POST_COLOR, alpha=0.55),
    ]
    fig.legend(handles, ["Prior (Level 4)", "Posterior (Level 5)"],
               loc="upper right", fontsize=9, frameon=False)
    fig.suptitle("Figure 4 — Prior (L4) vs posterior (L5) source-category distributions",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.supxlabel("Category-mean emission rate — kg h$^{-1}$ (exhaust panels: kg mmbtu$^{-1}$)",
                  fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-station stacked bar — source-category contribution to each station total
# ---------------------------------------------------------------------------

def figure_station_source_breakdown(result, sort_by_total=False):
    """Stacked bar chart: one bar per station, segments colored by source
    category. Stacking order is fixed across all bars: combustion slip
    (bottom), compressor seals, pneumatics, then the remaining categories
    (blowdown valve, isolation valve, tanks, equipment leaks) in a fixed order.

    Stations appear in a fixed order (by station_id) rather than sorted by
    total, so the same station always appears in the same x-position across
    different runs/reports.

    Uses result.per_station, which already carries the per-category mean
    contribution (mt/yr) for each station.
    """
    df = result.per_station.copy()
    cols_present = [c for c in STATION_COMPONENT_ORDER if f"{c}_mt_yr" in df.columns]
    if not cols_present:
        raise ValueError("per_station has no source-category columns to plot.")

    if sort_by_total and "mean_mt_yr" in df.columns:
        df = df.sort_values("mean_mt_yr", ascending=False)
    else:
        df = df.sort_values("station_id", ascending=True, key=lambda s: s.astype(str))

    stations = df["station_id"].tolist()
    n = len(stations)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n + 2), 5.5), dpi=150)

    x = np.arange(n)
    bottom = np.zeros(n)
    for c in cols_present:
        vals = df[f"{c}_mt_yr"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=STATION_COMPONENT_COLORS[c],
              label=STATION_COMPONENT_LABELS[c], width=0.68,
              edgecolor="white", linewidth=0.4)
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Annual emissions (mt CH$_4$ yr$^{-1}$)", fontsize=10)
    ax.set_title("Source-category contribution to each station's annual emissions",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center",
             bbox_to_anchor=(0.5, -0.22))
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: build all and save
# ---------------------------------------------------------------------------

def generate_all(result, outdir="."):
    import os
    figs = {
        "Figure2_prior_distributions": figure2_prior_distributions(result),
        "Figure4_prior_posterior":     figure4_prior_posterior(result),
        "Figure_station_source_breakdown": figure_station_source_breakdown(result),
    }
    paths = {}
    for name, fig in figs.items():
        p = os.path.join(outdir, f"{name}.png")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths[name] = p
    return paths


if __name__ == "__main__":
    import sys
    from mii_model import run_inventory, Config
    path = sys.argv[1] if len(sys.argv) > 1 else "Inventory_Template_FILLED.xlsx"
    result = run_inventory(path, config=Config(n_resamples=4000))
    paths = generate_all(result, outdir=".")
    for name, p in paths.items():
        print(f"saved {p}")
