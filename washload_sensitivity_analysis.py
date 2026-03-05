"""
Washload Sensitivity Analysis
==============================
Assesses the effect of washload inclusion (0–100% in 10% increments) on:
  - Total sediment flux (kg/s)
  - Bed material fraction
  - Transit length (x_tran, km) — median across reaches
  - Total transit time (ttot, yr) — full river Monte Carlo distribution
  - Total transit velocity (utot = river_length / ttot, km/yr) — distribution

Rivers: Beni (10 reaches), Bermejo (20 reaches)

Outputs are saved to:
  <working_directory>/RiverMapping/Mobility/<river>/WashloadSensitivity/
  with washload % in each filename.

The figure (10 panels, matching publication style) is saved to the same folder.

CONFIGURATION — edit the block below before running.
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# =============================================================================
# USER CONFIGURATION
# =============================================================================

RIVERS = {
    "Beni": {
        "working_directory": r"D:\Dissertation\Data",   # <-- edit
        "num_reaches": 10,
        "reach_start": 1,
        "bulk_density": 1600.0,       # kg/m³ — edit if different
    },
    "Bermejo": {
        "working_directory": r"D:\Dissertation\Data",   # <-- edit
        "num_reaches": 20,
        "reach_start": 1,
        "bulk_density": 1600.0,       # kg/m³ — edit if different
    },
}

NUM_ITERATIONS = 10_000   # Monte Carlo iterations for ttot
WASHLOAD_FRACTIONS = [w / 100.0 for w in range(0, 101, 10)]   # 0.0 … 1.0

# Colors matching the figure
COLORS = {"Beni": "#D4AF37", "Bermejo": "#CC3333"}

# =============================================================================
# HELPER: load all reach-level data for one river
# =============================================================================

def load_river_data(river_name: str, cfg: dict) -> dict:
    """Load WBMsed, hydraulic geometry, mobility metrics, channel belt areas,
    real reach lengths, and Tstor distributions for a river."""
    wd = cfg["working_directory"]

    # WBMsed
    wbmsed = pd.read_csv(
        os.path.join(wd, "WBMsed", "Extracted_Rivers", f"{river_name}_wbmsed.csv")
    )

    # Hydraulic geometry
    hg = pd.read_csv(
        os.path.join(wd, "RiverMapping", "HydraulicGeometry", river_name,
                     f"{river_name}_hydraulic_geometry.csv")
    )
    hg = hg.rename(columns={"length_m": "GQBF_reach_length_m"})

    # Mobility metrics (contains Tw_yr, Pswitch, median_width_m, depth_for_calcs_m)
    mob = pd.read_csv(
        os.path.join(wd, "RiverMapping", "Mobility", river_name,
                     f"{river_name}_mobility_metrics.csv")
    )

    # Channel belt areas (km²)
    cb = pd.read_csv(
        os.path.join(wd, "ChannelBelts", "Extracted_ChannelBelts", river_name,
                     f"{river_name}_channelbelt_areas.csv")
    )

    # Real reach lengths — load from the HydroRIVERS shapefile via geopandas
    import geopandas as gpd
    shp = gpd.read_file(
        os.path.join(wd, "HydroATLAS", "HydroRIVERS", "Extracted_Rivers",
                     river_name, f"{river_name}_reaches.shp")
    )
    gdf_wgs84 = shp.to_crs(epsg=4326)
    centroid = gdf_wgs84.geometry.unary_union.centroid
    lon = centroid.x
    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
    shp = shp.to_crs(epsg=epsg_code)
    shp["real_reach_length_m"] = shp.geometry.length
    rl = shp[["ds_order", "real_reach_length_m"]].copy()

    # Merge all reach-level data
    merged = rl.merge(hg, on="ds_order")
    merged = merged.merge(mob, on="ds_order")
    merged = merged.merge(wbmsed, on="ds_order")
    merged = merged.merge(cb[["ds_order", "area_sq_km"]], on="ds_order")

    # Tstor distributions — one DataFrame per reach
    tstor_dir = os.path.join(wd, "RiverMapping", "Mobility", river_name,
                             "Tstor_distributions")
    tstor = {}
    for f in os.listdir(tstor_dir):
        if f.startswith("Reach_") and f.endswith("_Tstor_distribution.csv"):
            rnum = int(f.split("_")[1])
            tstor[rnum] = pd.read_csv(os.path.join(tstor_dir, f))

    return {
        "merged": merged,
        "tstor": tstor,
        "wd": wd,
    }

# =============================================================================
# HELPER: compute xtran and n_stor for a given washload fraction
# =============================================================================

def compute_xtran(merged: pd.DataFrame, wash_proportion: float,
                  bulk_density: float) -> pd.DataFrame:
    """Return a copy of merged with x_tran_m, n_stor, sediment_flux_m3_yr added."""
    df = merged.copy()
    seconds_per_year = 365.25 * 24 * 60 * 60

    bedload_kg_yr     = df["mean_BedloadFlux_kg_s"]        * seconds_per_year
    suspended_kg_yr   = df["mean_SuspendedBedFlux_kg_s"]   * seconds_per_year
    washload_kg_yr    = df["mean_WashloadFlux_kg_s"]        * seconds_per_year * wash_proportion

    df["sediment_flux_m3_yr"] = (bedload_kg_yr + suspended_kg_yr + washload_kg_yr) / bulk_density

    df["x_tran_m"] = (
        df["sediment_flux_m3_yr"] * df["Tw_yr"] /
        (df["depth_for_calcs_m"] * df["median_width_m"])
    )
    df["n_stor"] = df["real_reach_length_m"] / df["x_tran_m"]

    return df

# =============================================================================
# HELPER: compute ttot Monte Carlo for a given river + n_stor values
# =============================================================================

def compute_ttot(df_xtran: pd.DataFrame, tstor: dict,
                 reach_start: int, num_reaches: int,
                 num_iterations: int) -> np.ndarray:
    """
    Returns array of ttot (yr) of length num_iterations.
    For each iteration, sums sampled treach values across all reaches.
    treach for each reach = sum of floor(n_stor) Tstor samples + optional fractional sample.
    """
    reach_end = reach_start + num_reaches - 1
    results = np.zeros(num_iterations)

    for reach_num in range(reach_start, reach_end + 1):
        if reach_num not in tstor:
            raise FileNotFoundError(
                f"Tstor distribution missing for reach {reach_num}")

        tstor_vals = tstor[reach_num].iloc[:, 0].dropna().values
        if len(tstor_vals) == 0:
            raise ValueError(f"Empty Tstor distribution for reach {reach_num}")

        row = df_xtran.loc[df_xtran["ds_order"] == reach_num]
        if row.empty:
            raise ValueError(f"Reach {reach_num} not found in xtran DataFrame")

        n = float(row["n_stor"].values[0])
        int_part  = int(np.floor(n))
        frac_part = n - int_part

        # Sample treach for this reach across all iterations
        if int_part > 0:
            samples = np.random.choice(tstor_vals, size=(num_iterations, int_part), replace=True)
            treach = samples.sum(axis=1)
        else:
            treach = np.zeros(num_iterations)

        # Fractional part
        mask = np.random.rand(num_iterations) < frac_part
        extra = np.random.choice(tstor_vals, size=num_iterations, replace=True)
        treach[mask] += extra[mask]

        results += treach

    return results

# =============================================================================
# MAIN SENSITIVITY LOOP
# =============================================================================

def run_sensitivity(river_name: str, cfg: dict) -> dict:
    """
    Runs the sensitivity analysis for one river across all washload fractions.
    Returns a dict of results for plotting.
    Saves per-fraction CSVs to WashloadSensitivity/ subfolder.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {river_name}")
    print(f"{'='*60}")

    data = load_river_data(river_name, cfg)
    merged   = data["merged"]
    tstor    = data["tstor"]
    wd       = data["wd"]
    bulk_density = cfg["bulk_density"]
    reach_start  = cfg["reach_start"]
    num_reaches  = cfg["num_reaches"]

    # Compute total real river length (km) — sum of all reach lengths
    river_length_km = merged["real_reach_length_m"].sum() / 1000.0
    print(f"  Total river length: {river_length_km:.1f} km over {num_reaches} reaches")

    # Output directory
    sens_dir = os.path.join(wd, "RiverMapping", "Mobility", river_name,
                            "WashloadSensitivity")
    os.makedirs(sens_dir, exist_ok=True)

    results = {
        "washload_pct":       [],
        "total_flux_kg_s":    [],   # total sediment flux kg/s (median across reaches)
        "bed_material_frac":  [],   # bed material fraction (median across reaches)
        "xtran_median_km":    [],   # median x_tran across reaches (km)
        "ttot_distributions": [],   # list of np arrays (one per fraction)
        "utot_distributions": [],   # list of np arrays (km/yr)
    }

    seconds_per_year = 365.25 * 24 * 60 * 60

    for wp in WASHLOAD_FRACTIONS:
        pct = int(round(wp * 100))
        print(f"  Washload fraction: {pct}%")

        df = compute_xtran(merged, wp, bulk_density)

        # ---- Scalar metrics (median across reaches) ----
        # Total flux per reach in kg/s
        bedload_kgs     = df["mean_BedloadFlux_kg_s"]
        suspended_kgs   = df["mean_SuspendedBedFlux_kg_s"]
        washload_kgs    = df["mean_WashloadFlux_kg_s"] * wp

        total_flux_kgs_per_reach  = bedload_kgs + suspended_kgs + washload_kgs
        bed_mat_kgs_per_reach     = bedload_kgs + suspended_kgs
        bed_frac_per_reach        = bed_mat_kgs_per_reach / total_flux_kgs_per_reach.replace(0, np.nan)

        results["washload_pct"].append(pct)
        results["total_flux_kg_s"].append(float(total_flux_kgs_per_reach.median()))
        results["bed_material_frac"].append(float(bed_frac_per_reach.median()))
        results["xtran_median_km"].append(float(df["x_tran_m"].median() / 1000.0))

        # ---- Save per-fraction xtran CSV ----
        xtran_out = os.path.join(
            sens_dir, f"{river_name}_xtran_washload{pct:03d}pct.csv")
        df[["ds_order", "x_tran_m", "n_stor", "sediment_flux_m3_yr"]].to_csv(
            xtran_out, index=False)

        # ---- Monte Carlo ttot ----
        ttot = compute_ttot(df, tstor, reach_start, num_reaches, NUM_ITERATIONS)
        results["ttot_distributions"].append(ttot)

        # ---- utot = river_length_km / ttot (km/yr) ----
        # Guard against zero ttot
        utot = np.where(ttot > 0, river_length_km / ttot, np.nan)
        results["utot_distributions"].append(utot)

        # ---- Save ttot and utot distributions ----
        dist_out = os.path.join(
            sens_dir, f"{river_name}_ttot_utot_washload{pct:03d}pct.csv")
        pd.DataFrame({"ttot_yr": ttot, "utot_km_yr": utot}).to_csv(
            dist_out, index=False)

        print(f"    x_tran median = {results['xtran_median_km'][-1]:.1f} km  |  "
              f"ttot median = {np.median(ttot):.0f} yr  |  "
              f"utot median = {np.nanmedian(utot):.3f} km/yr")

    return results, sens_dir, river_length_km

# =============================================================================
# PLOTTING
# =============================================================================

def make_figure(all_results: dict) -> plt.Figure:
    """
    Build the 5-row × 2-column figure.
    Rows: total flux | bed material fraction | x_tran | ttot | utot
    Columns: Beni | Bermejo
    """
    river_names = ["Beni", "Bermejo"]
    fig, axes = plt.subplots(5, 2, figsize=(6.5, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.38)

    panel_labels = list("abcdefghij")

    for col, river in enumerate(river_names):
        res = all_results[river]["results"]
        color = COLORS[river]
        wl_pcts = res["washload_pct"]
        x = np.array(wl_pcts)

        # ------------------------------------------------------------------
        # Row 0: Total sediment flux (kg/s)
        # ------------------------------------------------------------------
        ax = axes[0, col]
        ax.plot(x, res["total_flux_kg_s"], color=color, marker="o",
                linewidth=1.5, markersize=5)
        ax.set_ylabel("Total sediment flux\n(kg/s)", fontsize=8)
        ax.set_xlabel("Washload fraction (%)", fontsize=8)
        ax.set_xlim(-2, 102)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(river, fontweight="bold", fontsize=10)
        ax.text(-0.12, 1.05, panel_labels[col] + ")", transform=ax.transAxes,
                fontsize=10, fontweight="bold")

        # ------------------------------------------------------------------
        # Row 1: Bed material fraction
        # ------------------------------------------------------------------
        ax = axes[1, col]
        ax.plot(x, res["bed_material_frac"], color=color, marker="o",
                linewidth=1.5, markersize=5)
        ax.set_ylabel("Bed material fraction", fontsize=8)
        ax.set_xlabel("Washload fraction (%)", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-2, 102)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(-0.12, 1.05, panel_labels[col + 2] + ")", transform=ax.transAxes,
                fontsize=10, fontweight="bold")

        # ------------------------------------------------------------------
        # Row 2: Transit length x_tran (km)
        # ------------------------------------------------------------------
        ax = axes[2, col]
        ax.plot(x, res["xtran_median_km"], color=color, marker="o",
                linewidth=1.5, markersize=5)
        ax.set_ylabel("Transit length,\n" + r"$x_{tran}$ (km)", fontsize=8)
        ax.set_xlabel("Washload fraction (%)", fontsize=8)
        ax.set_xlim(-2, 102)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.text(-0.12, 1.05, panel_labels[col + 4] + ")", transform=ax.transAxes,
                fontsize=10, fontweight="bold")

        # ------------------------------------------------------------------
        # Row 3: ttot distributions (box-and-whisker)
        # ------------------------------------------------------------------
        ax = axes[3, col]
        bp_data = res["ttot_distributions"]
        bp = ax.boxplot(
            bp_data,
            positions=x,
            widths=7,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.7),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        ax.set_ylabel("Transit time,\n" + r"$t_{tot}$ (yr)", fontsize=8)
        ax.set_xlabel("Washload fraction (%)", fontsize=8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.set_xlim(-5, 105)
        ax.set_xticks(x)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.text(-0.12, 1.05, panel_labels[col + 6] + ")", transform=ax.transAxes,
                fontsize=10, fontweight="bold")

        # ------------------------------------------------------------------
        # Row 4: utot distributions (box-and-whisker)
        # ------------------------------------------------------------------
        ax = axes[4, col]
        utot_data = [arr[~np.isnan(arr)] for arr in res["utot_distributions"]]
        ax.boxplot(
            utot_data,
            positions=x,
            widths=7,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(facecolor=color, alpha=0.7),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
        )
        ax.set_ylabel("Transit velocity,\n" + r"$u_{tran}$ (km/yr)", fontsize=8)
        ax.set_xlabel("Washload fraction (%)", fontsize=8)
        ax.set_xlim(-5, 105)
        ax.set_xticks(x)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.text(-0.12, 1.05, panel_labels[col + 8] + ")", transform=ax.transAxes,
                fontsize=10, fontweight="bold")

    return fig

# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    all_results = {}

    for river_name, cfg in RIVERS.items():
        res, sens_dir, river_length_km = run_sensitivity(river_name, cfg)
        all_results[river_name] = {
            "results": res,
            "sens_dir": sens_dir,
            "river_length_km": river_length_km,
        }

    # ---- Compile summary statistics CSV ----
    summary_rows = []
    for river_name, entry in all_results.items():
        res = entry["results"]
        for i, pct in enumerate(res["washload_pct"]):
            ttot = res["ttot_distributions"][i]
            utot = res["utot_distributions"][i]
            utot_clean = utot[~np.isnan(utot)]
            row = {
                "river":              river_name,
                "washload_pct":       pct,
                "total_flux_kg_s":    res["total_flux_kg_s"][i],
                "bed_material_frac":  res["bed_material_frac"][i],
                "xtran_median_km":    res["xtran_median_km"][i],
                "ttot_min":           np.min(ttot),
                "ttot_Q1":            np.percentile(ttot, 25),
                "ttot_median":        np.median(ttot),
                "ttot_Q3":            np.percentile(ttot, 75),
                "ttot_max":           np.max(ttot),
                "ttot_mean":          np.mean(ttot),
                "ttot_std":           np.std(ttot),
                "utot_min":           np.nanmin(utot_clean) if len(utot_clean) else np.nan,
                "utot_Q1":            np.percentile(utot_clean, 25) if len(utot_clean) else np.nan,
                "utot_median":        np.nanmedian(utot_clean) if len(utot_clean) else np.nan,
                "utot_Q3":            np.percentile(utot_clean, 75) if len(utot_clean) else np.nan,
                "utot_max":           np.nanmax(utot_clean) if len(utot_clean) else np.nan,
                "utot_mean":          np.nanmean(utot_clean) if len(utot_clean) else np.nan,
                "utot_std":           np.nanstd(utot_clean) if len(utot_clean) else np.nan,
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(all_results["Beni"]["sens_dir"],
                                "washload_sensitivity_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary stats: {summary_path}")

    print("\nGenerating figure...")
    fig = make_figure(all_results)

    # Save figure alongside the data — use Beni's sens_dir as a proxy,
    # or save to the Beni working directory root
    fig_dir = all_results["Beni"]["sens_dir"]
    for ext in ("pdf", "png"):
        fig_path = os.path.join(fig_dir, f"washload_sensitivity_figure.{ext}")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"  Saved figure: {fig_path}")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
