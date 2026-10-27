"""
Measurement-Integrated Inventory (MII) model — v2
================================================

Bayesian integration of bottom-up (BU) and top-down (TD) methane emission data
for midstream compressor stations.

CHANGES FROM v1:
- Compressor vents are now computed per-unit, not per-station. A new
  `Compressors` sheet enumerates each physical compressor with its own
  operating / standby-pressurized hours and depressurization events.
- A single set of hours per unit drives ALL vent categories for that unit
  (RPV/DSV and BDV share the same operational-state hours because they
  occur at the same physical moments).
- Engine fuel throughput moves from `Stations` to the `Compressors` sheet
  (each unit carries its own driver's fuel).
- Electric drives are explicit: driver_type = "Electric" -> zero exhaust,
  but vents still counted normally.

Public API:
    run_inventory(template_path, config=None) -> InventoryResult
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    n_resamples: int = 9999
    random_state: int = 10
    confidence_level: float = 0.90
    percentiles: tuple = (10, 50, 90)
    scf_to_kg_ch4: float = 0.0192
    kg_to_mt: float = 1e-3
    survey_interval_hours: float = 2160.0   # 90 days x 24 hr between LDAR surveys
    surveys_per_year: int = 4               # default quarterly LDAR


# Standard routine-pneumatic emission factors (scf CH4 / hr) per device type.
# Values are EPA standard emission factors. Term-2 pneumatics use these fixed
# EFs; only malfunctioning devices use the measured posterior.
PNEUMATIC_EF_SCFHR = {
    "intermittent_bleed":  2.3,   # EPA intermittent-bleed controller
    "low_bleed":           6.8,   # EPA low-bleed controller
    "high_bleed":         30.0,   # EPA high-bleed controller
    "pump":               13.3,   # EPA pneumatic pump
}


# ---------------------------------------------------------------------------
# Sheet contract
# ---------------------------------------------------------------------------

SHEET_STACK       = "BU_StackTests"
SHEET_VENTS_BU    = "BU_VentEmissions"
SHEET_TANKS_BU    = "BU_Tanks"
SHEET_LEAKS_BU    = "BU_Leaks"
SHEET_PNEUM_BU    = "BU_Pneumatics"
SHEET_TD          = "TD_Detections"
SHEET_STATIONS    = "Stations"          # tanks/leaks/pneumatics only now
SHEET_COMPRESSORS = "Compressors"       # NEW: one row per physical compressor
SHEET_CONFIG      = "Config"

TD_UPDATED_CATEGORIES = {
    "RPV_Op", "RPV_Stdby",
    "DSV_Op", "DSV_Stdby",
    "BDV_Cent_Op", "BDV_Cent_Stdby",
    "BDV_Recip_Op", "BDV_Recip_Stdby",
    "ISO_Recip", "ISO_Cent",
    "Tank", "Leak", "Pneumatic",
}


@dataclass
class InventoryResult:
    per_station: pd.DataFrame
    per_compressor: pd.DataFrame
    asset_total_samples: np.ndarray
    asset_summary: dict
    category_summary: pd.DataFrame
    station_samples: dict
    config: Config
    # Extra fields for figure generation (populated by run_inventory):
    sources: dict = None                    # {cat: {'prior': arr, 'posterior': arr}} kg/hr
    engines: dict = None                     # {driver_type: EF prior array} kg/mmbtu (prior-only)
    asset_by_category: pd.DataFrame = None   # per-category asset L4/L5 annual totals (mt/yr)
    station_prior_samples: dict = None       # sid -> prior (L4) annual samples (mt/yr)
    station_dominant_category: dict = None   # sid -> dominant source-group name (L5)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _clean(series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s[s != 0].to_numpy()


def _emit_frequency(series: pd.Series) -> float:
    """Fraction of inspected sources that were emitting.

    Uses the FULL series (including zeros): zeros are inspected-but-not-emitting
    observations, nonzeros are emitting. This is the frequency-of-emission term.
    Returns 0.0 if there are no valid observations.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    return float((s != 0).sum() / len(s))


def _boot(samples: np.ndarray, cfg: Config, seed_offset: int = 0) -> np.ndarray:
    """Bootstrap the mean of `samples`.

    seed_offset makes each category's resampling independent while keeping the
    whole run reproducible. Using one shared random_state for every category
    would make all category bootstraps perfectly correlated, which inflates the
    variance of any sum across categories (station and asset totals). A distinct
    deterministic offset per category restores independence.
    """
    if len(samples) < 2:
        return np.full(cfg.n_resamples, float(np.mean(samples)) if len(samples) else 0.0)
    res = bootstrap(
        (samples,), np.mean,
        confidence_level=cfg.confidence_level,
        n_resamples=cfg.n_resamples,
        random_state=cfg.random_state + seed_offset,
    )
    return res.bootstrap_distribution


# ---------------------------------------------------------------------------
# Category-level distributions
# ---------------------------------------------------------------------------

def build_engine_distributions(stack_df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    """EF distributions (kg CH4 / mmbtu) per engine driver type. Electric = 0."""
    out = {}
    for i, driver_type in enumerate(("2SLB", "4SLB", "Turbine")):
        ef = _clean(stack_df.loc[stack_df["Driver Type"] == driver_type, "ch4_kg_mmbtu"])
        out[driver_type] = _boot(ef, cfg, seed_offset=100 + i)
    out["Electric"] = np.zeros(cfg.n_resamples)
    return out


def build_source_distributions(bu_df: pd.DataFrame, td_df: pd.DataFrame,
                               cfg: Config) -> dict:
    dists = {}
    categories = sorted(set(bu_df["category"]).union(td_df["category"]))
    for i, cat in enumerate(categories):
        bu = _clean(bu_df.loc[bu_df["category"] == cat, "rate_kgh"])
        td = _clean(td_df.loc[td_df["category"] == cat, "rate_kgh"])
        # Distinct offsets per category (and prior vs posterior) so no two
        # bootstrap distributions are spuriously correlated.
        prior = _boot(bu, cfg, seed_offset=2 * i + 1)
        if cat in TD_UPDATED_CATEGORIES and len(td) > 0:
            posterior = _boot(np.concatenate([bu, td]), cfg, seed_offset=2 * i + 2)
        else:
            posterior = prior
        dists[cat] = {"prior": prior, "posterior": posterior}
    return dists


def build_emission_frequencies(bu_df: pd.DataFrame) -> dict:
    """Per-category empirical frequency of emission (nonzero obs / total obs).

    Uses the FULL bu_df (zeros retained) so non-emitting inspections count in
    the denominator. Applied to valve categories that are designed to seal
    (BDV, ISO): a valve emits only when its sealing element fails to seat, so
    only a subset is leaking at any time.

    Returns {category: frequency in [0,1]}. Categories with no observations
    default to 0.0.
    """
    freqs = {}
    for cat in bu_df["category"].unique():
        freqs[cat] = _emit_frequency(bu_df.loc[bu_df["category"] == cat, "rate_kgh"])
    return freqs


# ---------------------------------------------------------------------------
# Per-compressor annual estimate
# ---------------------------------------------------------------------------

def compressor_annual_estimate(row, engines, sources, cfg, use_posterior=True,
                                frequencies=None):
    """One compressor unit's annual contribution (kg/yr).

    Three physically distinct venting mechanisms, separated by operational state:

    1. SEAL VENTS (RPV/DSV/WSV) — continuous by design; emit whenever the unit
       is pressurized. Full rate, no frequency scaling.
           seal_op x op_hrs + seal_stdby x stdby_pressurized_hrs

    2. BLOWDOWN VALVE (BDV) leak-through — designed to seal; emits only when it
       fails to seat. Scaled by empirical emission frequency (split op/stdby to
       match the posterior-rate split), applied over pressurized hours.
           freq_bdv_op    x bdv_rate_op    x op_hrs
         + freq_bdv_stdby x bdv_rate_stdby x stdby_pressurized_hrs

    3. ISOLATION VALVE (ISO) leak-by — designed to seal; emits only when it
       fails to seat, over DEPRESSURIZED hours (the valve holds back station gas
       while the unit sits isolated and blown down). One frequency per category.
           freq_iso x iso_rate x depressurized_hrs

       (The one-time blowdown VENT event is a separate BU-inventory term, not
        included here.)

    Returns dict with 'exhaust' and 'vents' distributions (each length n_resamples).
    """
    key = "posterior" if use_posterior else "prior"
    freqs = frequencies or {}
    def g(col, default=0):
        val = row.get(col, default)
        try:
            if pd.isna(val):
                return default
        except (TypeError, ValueError):
            pass
        return val
    def sd(cat):
        if cat in sources:
            return sources[cat][key]
        return np.zeros(cfg.n_resamples)   # always array, never scalar
    def fq(cat):
        return float(freqs.get(cat, 0.0))

    driver = str(g("driver_type", "Electric"))
    compressor_type = str(g("compressor_type", "")).strip().lower()
    seal_type = str(g("seal_type", "")).strip().upper()

    op_hrs      = float(g("operating_hours", 0) or 0)
    stdby_hrs   = float(g("standby_pressurized_hours", 0) or 0)
    fuel        = float(g("fuel_mmbtu_yr", 0) or 0)

    # Depressurized hours: explicit column preferred, else computed as remainder.
    depress_hrs = g("depressurized_hours", None)
    try:
        depress_hrs = float(depress_hrs)
        if pd.isna(depress_hrs):
            raise ValueError
    except (TypeError, ValueError):
        depress_hrs = max(0.0, 8760.0 - op_hrs - stdby_hrs)

    # Exhaust: EF x annual fuel. Zero for Electric.
    exhaust = engines.get(driver, engines["Electric"]) * fuel

    # Resolve category names by compressor/seal type.
    if compressor_type.startswith("recip"):
        seal_op_cat, seal_stdby_cat = "RPV_Op", "RPV_Stdby"
        bdv_op_cat, bdv_stdby_cat   = "BDV_Recip_Op", "BDV_Recip_Stdby"
        iso_cat                     = "ISO_Recip"
    elif compressor_type.startswith("centri"):
        if seal_type == "WSV":
            seal_op_cat, seal_stdby_cat = "WSV_Op", None
        else:
            seal_op_cat, seal_stdby_cat = "DSV_Op", "DSV_Stdby"
        bdv_op_cat, bdv_stdby_cat = "BDV_Cent_Op", "BDV_Cent_Stdby"
        iso_cat                   = "ISO_Cent"
    else:
        # Unknown type -> no compressor vents
        z = np.zeros(cfg.n_resamples)
        return {"exhaust": exhaust, "vents": z, "seals": z, "bdv": z, "iso": z}

    # 1. Seal vents — continuous, full rate.
    seals = sd(seal_op_cat) * op_hrs
    if seal_stdby_cat:
        seals = seals + sd(seal_stdby_cat) * stdby_hrs

    # 2. BDV leak-through — frequency-scaled (expected value per valve; one BDV
    #    per unit), over pressurized hours.
    bdv = (
        fq(bdv_op_cat)    * sd(bdv_op_cat)    * op_hrs +
        fq(bdv_stdby_cat) * sd(bdv_stdby_cat) * stdby_hrs
    )

    # 3. ISO leak-by — frequency-scaled (expected value per valve; one ISO per
    #    unit), over depressurized hours.
    iso = fq(iso_cat) * sd(iso_cat) * depress_hrs

    vents = seals + bdv + iso
    return {"exhaust": exhaust, "vents": vents,
            "seals": seals, "bdv": bdv, "iso": iso}


# ---------------------------------------------------------------------------
# Per-station "other" sources: tanks, leaks, pneumatics
# ---------------------------------------------------------------------------

def station_other_estimate(row, sources, cfg, use_posterior=True,
                           leak_surveys=None, default_tank_emit_freq=0.0):
    """Non-compressor sources: tanks + leaks + pneumatics (kg/yr).

    TANKS:
        emit_frequency x tank_count x posterior_tank_rate x survey_interval_hours
        emit_frequency comes from the station's Tank_emit_frequency column if
        provided; otherwise falls back to default_tank_emit_freq (derived
        asset-wide from BU_Tanks).

    LEAKS (two modes):
        If leak_surveys is provided (per-survey rows for this station), annual
        leak emissions = sum over surveys of:
            leaks_found_s x posterior_leak_rate x interval_hours_s
        Otherwise fall back to the single-value mode:
            surveys_per_year x Leak_per_survey x posterior_leak_rate x survey_interval_hours

    PNEUMATICS = malfunction term + routine term (unchanged):
        term 1 (measured):  malfunction_device_hours x posterior_malfunc_rate
        term 2 (standard):  sum_type[ count_type x EF_type_scfhr x 8760 x 0.0192 ]
    """
    key = "posterior" if use_posterior else "prior"
    def g(col, default=0):
        val = row.get(col, default)
        try:
            if pd.isna(val):
                return default
        except (TypeError, ValueError):
            pass
        return val
    def sd(cat):
        if cat in sources:
            return sources[cat][key]
        return np.zeros(cfg.n_resamples)
    f = lambda col, d=0: float(g(col, d) or 0)

    # ---- Tanks ----
    # Station override if present and non-blank, else asset-derived default.
    tank_freq = g("Tank_emit_frequency", None)
    try:
        tank_freq = float(tank_freq)
    except (TypeError, ValueError):
        tank_freq = default_tank_emit_freq
    tanks = tank_freq * f("Tank_count") * sd("Tank") * cfg.survey_interval_hours

    # ---- Leaks ----
    if leak_surveys is not None and len(leak_surveys) > 0:
        # Per-survey mode: each survey weighted by its own interval.
        leaks = 0.0
        for _, s in leak_surveys.iterrows():
            n = float(pd.to_numeric(s.get("leaks_found", 0), errors="coerce") or 0)
            interval = s.get("interval_hours", cfg.survey_interval_hours)
            try:
                interval = float(interval)
                if pd.isna(interval):
                    interval = cfg.survey_interval_hours
            except (TypeError, ValueError):
                interval = cfg.survey_interval_hours
            leaks = leaks + n * sd("Leak") * interval
    else:
        # Single-value fallback: annualize a representative per-survey count.
        leaks = (
            cfg.surveys_per_year
            * f("Leak_per_survey")
            * sd("Leak")
            * cfg.survey_interval_hours
        )

    # ---- Pneumatics: term 1 (malfunctioning, measured) ----
    malfunc_hours = f("Pneumatic_malfunc_hours")
    if malfunc_hours == 0:
        malfunc_hours = f("Pneumatic_malfunc_count") * f("Pneumatic_malfunc_fraction") * 8760
    pneum_malfunc = malfunc_hours * sd("Pneumatic")

    # ---- Pneumatics: term 2 (routine, standard EPA EFs) ----
    pneum_routine = 0.0
    for dev_key, ef in PNEUMATIC_EF_SCFHR.items():
        count = f(f"Pneumatic_{dev_key}_count")
        pneum_routine += count * ef * 8760 * cfg.scf_to_kg_ch4

    return {
        "tanks": tanks,
        "leaks": leaks,
        "pneumatics": pneum_malfunc + pneum_routine,
        "total": tanks + leaks + pneum_malfunc + pneum_routine,
    }


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_inventory(template_path, config=None, use_posterior=True, preserve_run_settings=False):
    cfg = config or Config()

    xls = pd.ExcelFile(template_path)
    stack_df    = xls.parse(SHEET_STACK)
    bu_df       = xls.parse(SHEET_VENTS_BU)
    td_df       = xls.parse(SHEET_TD)
    tanks_bu    = xls.parse(SHEET_TANKS_BU)
    leaks_bu    = xls.parse(SHEET_LEAKS_BU)
    pneum_bu    = xls.parse(SHEET_PNEUM_BU)
    stations    = xls.parse(SHEET_STATIONS, dtype={"station_id": str})
    compressors = xls.parse(SHEET_COMPRESSORS, dtype={"station_id": str, "unit_id": str})

    # Optional per-survey leak sheet: columns station_id, survey_date, leaks_found, interval_hours
    if "Leaks_Surveys" in xls.sheet_names:
        leaks_surveys_df = xls.parse("Leaks_Surveys", dtype={"station_id": str})
    else:
        leaks_surveys_df = None

    extras = pd.concat([
        tanks_bu.assign(category="Tank")[["category", "rate_kgh"]],
        leaks_bu.assign(category="Leak")[["category", "rate_kgh"]],
        pneum_bu.assign(category="Pneumatic")[["category", "rate_kgh"]],
    ], ignore_index=True)
    bu_long = pd.concat([bu_df[["category", "rate_kgh"]], extras], ignore_index=True)

    # Derive asset-wide tank emission frequency from BU_Tanks (zeros = inspected,
    # not emitting). Used as the default when a station leaves Tank_emit_frequency blank.
    asset_tank_emit_freq = _emit_frequency(tanks_bu["rate_kgh"])

    if SHEET_CONFIG in xls.sheet_names:
        cfg_df = xls.parse(SHEET_CONFIG).set_index("param")["value"]
        # Browser/UI callers can lock the interactive run controls while still
        # allowing operational assumptions below to come from the workbook.
        if not preserve_run_settings:
            if "n_resamples" in cfg_df.index:
                cfg.n_resamples = int(cfg_df["n_resamples"])
            if "random_state" in cfg_df.index:
                cfg.random_state = int(cfg_df["random_state"])
        if "survey_interval_hours" in cfg_df.index:
            cfg.survey_interval_hours = float(cfg_df["survey_interval_hours"])
        if "surveys_per_year" in cfg_df.index:
            cfg.surveys_per_year = int(cfg_df["surveys_per_year"])

    engines = build_engine_distributions(stack_df, cfg)
    sources = build_source_distributions(bu_long, td_df, cfg)
    # Empirical emission frequencies for valves designed to seal (BDV, ISO).
    # Built from BU_VentEmissions with zeros retained (non-emitting observations).
    frequencies = build_emission_frequencies(bu_df[["category", "rate_kgh"]])

    # ---- Helper: run a full pass (all stations) under prior OR posterior ----
    # Returns (station_samples dict mt/yr, asset array mt/yr, per_station_group_kg dict).
    # per_station_group_kg[sid][group] = kg/yr contribution of that source group.
    def run_pass(use_post):
        st_samples = {}
        asset_arr = np.zeros(cfg.n_resamples)
        group_contrib = {}   # sid -> {group: kg/yr array} (3 groups, for coloring)
        comp_contrib = {}    # sid -> {seals,bdv,iso,exhaust: kg/yr array}

        # Compressor contributions per station, split into all mechanisms
        comp_by_station = {}
        for _, row in compressors.iterrows():
            sid = str(row["station_id"])
            parts = compressor_annual_estimate(row, engines, sources, cfg, use_post,
                                               frequencies=frequencies)
            z = lambda: np.zeros(cfg.n_resamples)
            d = comp_by_station.setdefault(sid, {"exhaust": z(), "seals": z(),
                                                 "bdv": z(), "iso": z()})
            for k in ("exhaust", "seals", "bdv", "iso"):
                d[k] = d[k] + parts[k]

        # Iterate the UNION of station IDs from both sheets so a compressor
        # whose station is missing from Stations still has its emissions counted
        # (previously such "orphan" compressors were silently dropped from totals).
        station_rows_by_id = {str(r["station_id"]): r for _, r in stations.iterrows()}
        all_sids = list(dict.fromkeys(          # preserve order, dedupe
            [str(r["station_id"]) for _, r in stations.iterrows()] +
            list(comp_by_station.keys())
        ))
        for sid in all_sids:
            srow = station_rows_by_id.get(sid)
            if srow is not None:
                ls = leaks_surveys_df[leaks_surveys_df["station_id"] == sid] if leaks_surveys_df is not None else None
                other = station_other_estimate(srow, sources, cfg, use_post,
                                               leak_surveys=ls,
                                               default_tank_emit_freq=asset_tank_emit_freq)
            else:
                # Compressor references a station not present on the Stations sheet:
                # count its compressor emissions, zero for tanks/leaks/pneumatics.
                z0 = np.zeros(cfg.n_resamples)
                other = {"tanks": z0, "leaks": z0, "pneumatics": z0, "total": z0}
            z = np.zeros(cfg.n_resamples)
            cg = comp_by_station.get(sid, {"exhaust": z, "seals": z, "bdv": z, "iso": z})

            # Full six-component breakdown (kg/yr arrays)
            comp_contrib[sid] = {
                "exhaust":    cg["exhaust"],
                "seals":      cg["seals"],
                "bdv":        cg["bdv"],
                "iso":        cg["iso"],
                "tanks":      other["tanks"],
                "leaks":      other["leaks"],
                "pneumatics": other["pneumatics"],
            }
            # Three coarse groups (for Fig 5 coloring / asset-by-category)
            comp_vents = cg["seals"] + cg["bdv"] + cg["iso"]
            groups = {
                "Combustion":             cg["exhaust"],
                "Compressor vents":       comp_vents,
                "Tanks/Leaks/Pneumatics": other["total"],
            }
            group_contrib[sid] = groups
            total_kg = cg["exhaust"] + comp_vents + other["total"]
            total_mt = total_kg * cfg.kg_to_mt
            st_samples[sid] = total_mt
            asset_arr = asset_arr + total_mt
        return st_samples, asset_arr, group_contrib, comp_contrib

    # ---- Posterior (primary) and prior (baseline) passes ----
    station_samples, asset, post_groups, post_components = run_pass(True)
    station_prior_samples, asset_prior, prior_groups, prior_components = run_pass(False)

    # Select the component breakdown matching the requested mode.
    primary_components = post_components if use_posterior else prior_components

    # If the caller asked for prior as the primary estimate, swap totals too.
    if not use_posterior:
        station_samples, asset = station_prior_samples, asset_prior

    # ---- Per-compressor table (uses the primary mode) ----
    comp_rows = []
    for _, row in compressors.iterrows():
        sid = str(row["station_id"])
        parts = compressor_annual_estimate(row, engines, sources, cfg, use_posterior,
                                            frequencies=frequencies)
        total_kg = parts["exhaust"] + parts["vents"]
        comp_rows.append({
            "station_id":       sid,
            "unit_id":          row.get("unit_id", ""),
            "driver_type":      row.get("driver_type", ""),
            "compressor_type":  row.get("compressor_type", ""),
            "seal_type":        row.get("seal_type", ""),
            "exhaust_mt_yr":    float(np.mean(parts["exhaust"])) * cfg.kg_to_mt,
            "seals_mt_yr":      float(np.mean(parts["seals"]))   * cfg.kg_to_mt,
            "bdv_mt_yr":        float(np.mean(parts["bdv"]))     * cfg.kg_to_mt,
            "iso_mt_yr":        float(np.mean(parts["iso"]))     * cfg.kg_to_mt,
            "total_mt_yr":      float(np.mean(total_kg)) * cfg.kg_to_mt,
        })
    per_compressor = pd.DataFrame(comp_rows)

    # ---- Per-station summary table (with source-component breakdown) ----
    station_rows = []
    for sid, total_mt in station_samples.items():
        comp = primary_components.get(sid, {})
        row = {
            "station_id": sid,
            "mean_mt_yr": float(np.mean(total_mt)),
            **{f"p{p}_mt_yr": float(np.percentile(total_mt, p)) for p in cfg.percentiles},
        }
        # Component means (mt/yr) so the total is transparent without hand-tracing
        for comp_name in ("exhaust", "seals", "bdv", "iso", "tanks", "leaks", "pneumatics"):
            arr = comp.get(comp_name)
            row[f"{comp_name}_mt_yr"] = float(np.mean(arr)) * cfg.kg_to_mt if arr is not None else 0.0
        station_rows.append(row)
    per_station = pd.DataFrame(station_rows)

    # ---- Dominant source group per station (from posterior groups) ----
    station_dominant_category = {}
    for sid, groups in post_groups.items():
        means = {g: float(np.mean(arr)) for g, arr in groups.items()}
        station_dominant_category[sid] = max(means, key=means.get) if means else "n/a"

    orphan_sids = set(post_groups) - set(stations["station_id"].astype(str))
    if orphan_sids:
        print(f"WARNING: compressor(s) reference station_id not in Stations sheet: {orphan_sids}")

    cat_rows = [{
        "category":         cat,
        "prior_mean_kgh":   float(np.mean(d["prior"])),
        "posterior_mean_kgh": float(np.mean(d["posterior"])),
        "td_updated":       cat in TD_UPDATED_CATEGORIES,
    } for cat, d in sources.items()]
    category_summary = pd.DataFrame(cat_rows).sort_values("category").reset_index(drop=True)

    # ---- Asset annual emissions by source category (L4 vs L5), full resolution ----
    category_names = ["exhaust", "seals", "bdv", "iso", "tanks", "leaks", "pneumatics"]
    category_display = {
        "exhaust": "Combustion slip",
        "seals": "Compressor seals (RPV/DSV/WSV)",
        "bdv": "Blowdown valve",
        "iso": "Isolation valve",
        "tanks": "Tanks",
        "leaks": "Equipment leaks",
        "pneumatics": "Pneumatics",
    }
    fbc_rows = []
    for cname in category_names:
        l4 = sum(float(np.mean(prior_components[sid][cname])) for sid in prior_components) * cfg.kg_to_mt
        l5 = sum(float(np.mean(post_components[sid][cname]))  for sid in post_components)  * cfg.kg_to_mt
        fbc_rows.append({
            "source_category": category_display[cname],
            "L4_mt_yr": l4,
            "L5_mt_yr": l5,
        })
    asset_by_category = pd.DataFrame(fbc_rows)
    # Percent of asset total (L5), for a quick contribution readout.
    _asset_l5_total = asset_by_category["L5_mt_yr"].sum()
    if _asset_l5_total > 0:
        asset_by_category["pct_of_asset_L5"] = (asset_by_category["L5_mt_yr"] / _asset_l5_total * 100).round(1)
    else:
        asset_by_category["pct_of_asset_L5"] = 0.0

    # Percent of the total asset REVISION (L5 - L4) each category accounts for.
    # A category with no TD update (e.g. combustion) contributes 0 revision by
    # construction, so it should show 0% here even though it may be a large
    # share of the L5 total above.
    asset_by_category["revision_mt_yr"] = asset_by_category["L5_mt_yr"] - asset_by_category["L4_mt_yr"]
    _total_revision = asset_by_category["revision_mt_yr"].sum()
    if _total_revision > 0:
        asset_by_category["pct_of_revision"] = (asset_by_category["revision_mt_yr"] / _total_revision * 100).round(1)
    else:
        # No net revision (prior == posterior asset-wide) -> undefined share; report 0.
        asset_by_category["pct_of_revision"] = 0.0

    asset_by_category = asset_by_category.sort_values("L5_mt_yr", ascending=False).reset_index(drop=True)

    asset_summary = {
        "mean_mt_yr": float(np.mean(asset)),
        **{f"p{p}_mt_yr": float(np.percentile(asset, p)) for p in cfg.percentiles},
    }

    return InventoryResult(
        per_station=per_station,
        per_compressor=per_compressor,
        asset_total_samples=asset,
        asset_summary=asset_summary,
        category_summary=category_summary,
        station_samples=station_samples,
        config=cfg,
        sources=sources,
        engines=engines,
        asset_by_category=asset_by_category,
        station_prior_samples=station_prior_samples,
        station_dominant_category=station_dominant_category,
    )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "Inventory_Template.xlsx"
    result = run_inventory(path)
    print("\n=== ASSET SUMMARY (mt CH4 / year) ===")
    for k, v in result.asset_summary.items():
        print(f"  {k:15s} {v:>12,.2f}")
    print("\n=== PER-STATION ===")
    print(result.per_station.to_string(index=False))
    print("\n=== PER-COMPRESSOR (mean mt CH4/yr) ===")
    print(result.per_compressor.to_string(index=False))
    print("\n=== CATEGORY PRIOR vs POSTERIOR (kg/hr) ===")
    print(result.category_summary.to_string(index=False))
