"""
MII Tool — Streamlit UI
=======================
Thin UI layer over mii_model.run_inventory(). Runs identically as:
  - a normal Streamlit app (`streamlit run app.py`), and
  - a static browser app via stlite (see index.html).

Nothing here does computation itself — all modeling lives in mii_model.py.
"""
import io
import pandas as pd
import streamlit as st

from mii_model import run_inventory, Config

st.set_page_config(page_title="EEMDL MII Tool", layout="wide")

# ---- Brand theme (white + UT Austin burnt orange) ------------------------
# Background/body text stay white/near-black for readability; burnt orange is
# reserved for headings, links, buttons, and other accent/interactive
# elements (also set as the Streamlit theme primaryColor in
# .streamlit/config.toml, which covers buttons, sliders, and the active-tab
# underline that CSS alone can't reach).
BURNT_ORANGE = "#BF5700"
st.markdown(f"""
<style>
  h1, h2, h3 {{
      color: {BURNT_ORANGE} !important;
  }}
  h1 {{
      border-bottom: 3px solid {BURNT_ORANGE};
      padding-bottom: 0.3rem;
  }}
  a, a:visited {{
      color: {BURNT_ORANGE} !important;
  }}
  [data-testid="stMetricValue"] {{
      color: {BURNT_ORANGE};
  }}
  [data-testid="stSidebar"] {{
      background-color: #F7F3EF;
  }}
</style>
""", unsafe_allow_html=True)

try:
    st.image("eemdl_logo.png", width=320)
except Exception:
    pass

st.title("Measurement-Integrated Inventory Tool For Compressor Station")
# st.caption(
#     "Upload your BU and TD data as the input template; estimate annual "
#     "facility-level methane emissions with quantified uncertainty. "
#     "**Your data never leaves your browser** — all computation runs locally."
# )

st.markdown(
    " "
    " "
    "This tool implements the Bayesian "
    "integration framework developed in the accompanying manuscript. It updates "
    "bottom-up (BU) source-level data "
    "with top-down (TD) aerial measurements "
    "using Bayesian data-augmentation approach. "
    "Posterior distributions were propagated through unit-level activity data and "
    "source-specific information via Monte Carlo simulation to estimate annual inventory. " 
    "The result is a "
    "station- and asset-level annual methane inventory with quantified "
    "uncertainty, broken out by compressor unit and source "
    "category, consistent with OGMP 2.0 Level 4/5 reporting."
)
st.caption(
    "Read the manuscript: "
    "[Measurement-Integrated Inventory Preprint]"
    "(https://chemrxiv.org/doi/full/10.26434/chemrxiv.15008933/v1)"
)

# ---- Sidebar: run controls ---------------------------------------------
with st.sidebar:
    st.header("Run settings")
    quick = st.toggle("Quick preview (fewer iterations)", value=True,
                      help="2,000 bootstrap iterations instead of 9,999. "
                           "Faster, slightly noisier. Use the full run for final numbers.")
    estimate_type = st.radio("Estimate", ["Posterior (BU + TD)", "Prior (BU only)"],
                             help="Posterior integrates aerial TD detections; "
                                  "prior uses operator BU data only.")
    seed = st.number_input("Random seed", value=10, step=1,
                           help="Fix for reproducible results.")

    st.divider()
    st.markdown("**Need the template?**")
    _xlsx_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    try:
        with open("Inventory_Template.xlsx", "rb") as f:
            st.download_button("Download blank template",
                               f.read(), "Inventory_Template.xlsx",
                               mime=_xlsx_mime)
    except FileNotFoundError:
        st.info("Template file not bundled with this deployment.")
    # Filled sample so first-time testers can run the tool immediately.
    try:
        with open("Inventory_Template_FILLED.xlsx", "rb") as f:
            st.download_button("Download filled sample (test data)",
                               f.read(), "Inventory_Template_FILLED.xlsx",
                               mime=_xlsx_mime)
        st.caption("Sample contains synthetic data for testing only.")
    except FileNotFoundError:
        pass

# ---- Main: upload -------------------------------------------------------
uploaded = st.file_uploader("Upload your filled-in input template (.xlsx)", type=["xlsx"])

if uploaded is None:
    st.info("Upload a completed template to begin. Download a blank one from the sidebar.")
    st.stop()

# ---- Run ----------------------------------------------------------------
cfg = Config(
    n_resamples=2000 if quick else 9999,
    random_state=int(seed),
)
use_posterior = estimate_type.startswith("Posterior")

with st.spinner("Running bootstrap + Monte Carlo… (a full run can take 1–2 minutes in-browser)"):
    try:
        # run_inventory accepts a path or a file-like buffer (pd.ExcelFile handles both)
        result = run_inventory(uploaded, config=cfg, use_posterior=use_posterior)
    except Exception as e:
        st.error(f"Could not process the file: {e}")
        st.exception(e)
        st.stop()

import figures as figmod


# ---- Figure display helper ----------------------------------------------
# st.pyplot rasterizes at a modest dpi and the browser then rescales it, which
# looks soft. Instead we render each figure to a 300-dpi PNG ourselves, show
# that image (the browser downsamples a large image = crisp), and offer the
# very same high-res PNG for download.
FIG_DPI = 300

def _show(fig, fname):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=FIG_DPI, facecolor="white")
    png = buf.getvalue()
    st.image(png, use_container_width=True)
    st.download_button(f"Download {fname}", png, fname, mime="image/png",
                       key=f"dl_{fname}")

# ---- Asset summary ------------------------------------------------------
st.header("Asset total")
fs = result.asset_summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean (mt CH₄/yr)", f"{fs['mean_mt_yr']:,.0f}")
c2.metric("10th percentile", f"{fs['p10_mt_yr']:,.0f}")
c3.metric("Median", f"{fs['p50_mt_yr']:,.0f}")
c4.metric("90th percentile", f"{fs['p90_mt_yr']:,.0f}")

# Asset distribution histogram
st.subheader("Asset-total distribution")
import matplotlib.pyplot as _plt
_fig, _ax = _plt.subplots(figsize=(7, 3.5), dpi=150)
_samples = result.asset_total_samples
_ax.hist(_samples, bins=40, color="#1f5fbf", alpha=0.85, edgecolor="white", linewidth=0.3)
_ax.axvline(fs["mean_mt_yr"], color="red", linestyle="--", linewidth=1.4,
            label=f"Mean {fs['mean_mt_yr']:,.0f}")
_ax.axvline(fs["p10_mt_yr"], color="gray", linestyle=":", linewidth=1.2,
            label=f"P10 {fs['p10_mt_yr']:,.0f}")
_ax.axvline(fs["p90_mt_yr"], color="gray", linestyle=":", linewidth=1.2,
            label=f"P90 {fs['p90_mt_yr']:,.0f}")
_ax.set_xlabel("Asset annual emissions (mt CH$_4$ yr$^{-1}$)")
_ax.set_ylabel("Frequency")
_ax.legend(fontsize=8, frameon=False)
_ax.set_yticks([])
_fig.tight_layout()
_show(_fig, "Asset_total_distribution.png")

# ---- Asset breakdown by source category ---------------------------------
st.header("Asset total by source category")
st.caption("How much each source category contributes to the asset total, "
           "and how the estimate shifted from bottom-up (L4) to measurement-"
           "integrated (L5).")

fbc = result.asset_by_category
if fbc is not None and len(fbc):
    _fig2, _ax2 = _plt.subplots(figsize=(7.5, 4.2), dpi=150)
    _y = range(len(fbc))
    _ax2.barh(_y, fbc["L4_mt_yr"], height=0.38, color="#1f5fbf", alpha=0.85,
             label="Bottom-up (L4)", align="edge")
    _ax2.barh([i - 0.38 for i in _y], fbc["L5_mt_yr"], height=0.38, color="#e07b00",
             alpha=0.85, label="Measurement-integrated (L5)", align="edge")
    _ax2.set_yticks(_y)
    _ax2.set_yticklabels(fbc["source_category"], fontsize=9)
    _ax2.invert_yaxis()
    _ax2.set_xlabel("Asset annual emissions (mt CH$_4$ yr$^{-1}$)")
    _ax2.legend(fontsize=8, frameon=False)
    _ax2.grid(True, axis="x", linewidth=0.3, alpha=0.4)
    _fig2.tight_layout()
    _show(_fig2, "Asset_total_by_source_category.png")

    st.table(fbc.round(1).rename(columns={
        "source_category": "Source category",
        "L4_mt_yr": "L4 (mt/yr)",
        "L5_mt_yr": "L5 (mt/yr)",
        "pct_of_asset_L5": "% of asset (L5)",
        "revision_mt_yr": "Revision (mt/yr)",
        "pct_of_revision": "% of total revision",
    }).set_index("Source category"))
else:
    st.info("Asset-by-category breakdown not available.")

# ---- Per-station source-category stacked bar (moved up, right below the
# L4/L5 asset breakdown figure) --------------------------------------------
st.subheader("Source-category contribution by station")
st.caption("Each bar is one station; segments show that station's source-category mix.")
_show(figmod.figure_station_source_breakdown(result), "Figure_station_source_breakdown.png")

# st.table renders as plain HTML (reliable in stlite/browser); st.dataframe
# uses an interactive grid component that sometimes fails to load in the
# browser build. We round numeric columns so the plain table stays readable.
def _display_table(df):
    d = df.copy()
    for c in d.select_dtypes(include="number").columns:
        d[c] = d[c].round(2)
    # Set station_id / unit_id as index just for tidy display (st.table always
    # shows an index; using an ID keeps it meaningful instead of 0..n-1).
    for id_col in ("station_id", "unit_id"):
        if id_col in d.columns:
            d = d.set_index(id_col)
            break
    return d

# ---- Per-station --------------------------------------------------------
st.header("Per-station estimates")
st.table(_display_table(result.per_station))

# ---- Per-compressor -----------------------------------------------------
st.header("Per-compressor breakdown")
st.caption("Each row is one physical compressor unit. Exhaust is zero for "
           "electric-driven units by construction.")
st.table(_display_table(result.per_compressor))

# ---- Category prior vs posterior ---------------------------------------
st.header("Source categories: prior vs posterior")
st.caption("Mean emission rate (kg/hr) before and after integrating TD detections.")
st.table(_display_table(result.category_summary))

# ---- Figures (manuscript 2, 4) -------------------------------------------
st.header("Figures")
st.caption("Prior/posterior distributions, matching the manuscript.")

tab2, tab4 = st.tabs(["Fig 2 — Priors", "Fig 4 — Prior vs Posterior"])

with tab2:
    _show(figmod.figure2_prior_distributions(result), "Figure2_prior_distributions.png")
with tab4:
    _show(figmod.figure4_prior_posterior(result), "Figure4_prior_posterior.png")

# ---- Downloads ----------------------------------------------------------
st.header("Export results")

def build_results_xlsx() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        result.per_station.to_excel(xw, sheet_name="Per_Station", index=False)
        result.per_compressor.to_excel(xw, sheet_name="Per_Compressor", index=False)
        result.category_summary.to_excel(xw, sheet_name="Categories", index=False)
        if result.asset_by_category is not None:
            result.asset_by_category.to_excel(xw, sheet_name="Asset_By_Category", index=False)
        pd.DataFrame([result.asset_summary]).to_excel(xw, sheet_name="Asset_Total", index=False)
        # provenance: record the exact config used
        pd.DataFrame([{
            "n_resamples": cfg.n_resamples,
            "random_state": cfg.random_state,
            "estimate": "posterior" if use_posterior else "prior",
        }]).to_excel(xw, sheet_name="Run_Config", index=False)
    return buf.getvalue()

st.download_button("Download results (.xlsx)", build_results_xlsx(),
                   "MII_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
