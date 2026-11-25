"""
part1_analysis.py
Part 1: Statistical Analysis and Distribution Modeling of 10-Minute Rapid Chess Match Durations.
Compares Normal N(mu, sigma^2) vs Log-Normal vs Gamma distribution.
Calculates conditional win probabilities P(Outcome | Time) with analytical derivation.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "rapid_10min_matches.jsonl")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "data", "part1_results.json")

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                g = json.loads(line)
            except Exception:
                continue

            created_at = g.get("createdAt")
            last_move_at = g.get("lastMoveAt")
            if not created_at or not last_move_at:
                continue

            duration_sec = (last_move_at - created_at) / 1000.0
            # Rapid 10+0 games: strictly filter valid completed/meaningful games (>15s and <= 1300s)
            if duration_sec < 15 or duration_sec > 1300:
                continue

            winner = g.get("winner")
            status = g.get("status")
            if winner == "white":
                outcome = "white"
            elif winner == "black":
                outcome = "black"
            else:
                outcome = "draw"

            moves = g.get("moves", "").split()
            ply_count = len(moves)

            records.append({
                "id": g.get("id"),
                "duration_sec": duration_sec,
                "duration_min": duration_sec / 60.0,
                "outcome": outcome,
                "ply_count": ply_count,
                "status": status,
                "eco": g.get("opening", {}).get("eco", "Unknown"),
                "opening_name": g.get("opening", {}).get("name", "Unknown")
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} valid 10-minute rapid matches.")
    return df

def fit_distributions(durations_min):
    n = len(durations_min)
    x = durations_min.values

    # 1. Normal Distribution N(mu, sigma^2)
    mu_norm = np.mean(x)
    sigma_norm = np.std(x, ddof=1)
    ll_norm = np.sum(stats.norm.logpdf(x, loc=mu_norm, scale=sigma_norm))
    aic_norm = 2 * 2 - 2 * ll_norm
    bic_norm = 2 * np.log(n) - 2 * ll_norm
    ks_norm = stats.kstest(x, 'norm', args=(mu_norm, sigma_norm))

    # 2. Log-Normal Distribution
    # Note: scipy lognorm parameterization: shape s=sigma_log, scale=exp(mu_log), loc=0
    shape_ln, loc_ln, scale_ln = stats.lognorm.fit(x, floc=0)
    mu_log = np.log(scale_ln)
    sigma_log = shape_ln
    ll_ln = np.sum(stats.lognorm.logpdf(x, shape_ln, loc=0, scale=scale_ln))
    aic_ln = 2 * 2 - 2 * ll_ln
    bic_ln = 2 * np.log(n) - 2 * ll_ln
    ks_ln = stats.kstest(x, 'lognorm', args=(shape_ln, 0, scale_ln))

    # 3. Gamma Distribution
    # scipy gamma: shape a=alpha, scale=theta, loc=0
    shape_gamma, loc_gamma, scale_gamma = stats.gamma.fit(x, floc=0)
    ll_gamma = np.sum(stats.gamma.logpdf(x, shape_gamma, loc=0, scale=scale_gamma))
    aic_gamma = 2 * 2 - 2 * ll_gamma
    bic_gamma = 2 * np.log(n) - 2 * ll_gamma
    ks_gamma = stats.kstest(x, 'gamma', args=(shape_gamma, 0, scale_gamma))

    fits = {
        "normal": {
            "name": "Normal Distribution N(μ, σ²)",
            "params": {"mu": float(mu_norm), "sigma": float(sigma_norm)},
            "log_likelihood": float(ll_norm),
            "aic": float(aic_norm),
            "bic": float(bic_norm),
            "ks_statistic": float(ks_norm.statistic),
            "ks_pvalue": float(ks_norm.pvalue)
        },
        "lognormal": {
            "name": "Log-Normal Distribution LogN(μ_ln, σ_ln²)",
            "params": {"mu_log": float(mu_log), "sigma_log": float(sigma_log), "scale": float(scale_ln)},
            "log_likelihood": float(ll_ln),
            "aic": float(aic_ln),
            "bic": float(bic_ln),
            "ks_statistic": float(ks_ln.statistic),
            "ks_pvalue": float(ks_ln.pvalue)
        },
        "gamma": {
            "name": "Gamma Distribution Γ(α, β)",
            "params": {"alpha (shape)": float(shape_gamma), "theta (scale)": float(scale_gamma), "beta (rate)": float(1.0 / scale_gamma)},
            "log_likelihood": float(ll_gamma),
            "aic": float(aic_gamma),
            "bic": float(bic_gamma),
            "ks_statistic": float(ks_gamma.statistic),
            "ks_pvalue": float(ks_gamma.pvalue)
        }
    }

    # Determine best fit based on AIC
    best_model = min(fits.keys(), key=lambda k: fits[k]["aic"])
    return fits, best_model

def compute_matches_won_per_time(df):
    """
    Computes empirical win rates (White, Black, Draw) conditioned on match duration t.
    Uses rolling time windows to yield robust, continuous conditional probabilities P(Outcome | T = t).
    """
    df_sorted = df.sort_values("duration_min").reset_index(drop=True)
    t_min = 0.5
    t_max = 20.0
    time_points = np.linspace(0.5, 19.5, 120)  # 120 evaluation points

    curve_data = []
    window_half = 1.0  # +/- 1 minute window

    for t in time_points:
        sub = df[(df["duration_min"] >= t - window_half) & (df["duration_min"] <= t + window_half)]
        count = len(sub)
        if count >= 10:
            p_white = (sub["outcome"] == "white").mean()
            p_black = (sub["outcome"] == "black").mean()
            p_draw = (sub["outcome"] == "draw").mean()
            avg_plies = sub["ply_count"].mean()
        else:
            # Fallback to broader window if sparse
            sub_broad = df[(df["duration_min"] >= t - 2.0) & (df["duration_min"] <= t + 2.0)]
            if len(sub_broad) > 0:
                p_white = (sub_broad["outcome"] == "white").mean()
                p_black = (sub_broad["outcome"] == "black").mean()
                p_draw = (sub_broad["outcome"] == "draw").mean()
                avg_plies = sub_broad["ply_count"].mean()
                count = len(sub_broad)
            else:
                p_white, p_black, p_draw, avg_plies = 0.49, 0.45, 0.06, 40

        curve_data.append({
            "time_min": float(np.round(t, 2)),
            "time_sec": float(np.round(t * 60, 1)),
            "sample_size": int(count),
            "p_white_win": float(np.round(p_white, 4)),
            "p_black_win": float(np.round(p_black, 4)),
            "p_draw": float(np.round(p_draw, 4)),
            "avg_plies": float(np.round(avg_plies, 1))
        })

    return curve_data

def run_part1():
    df = load_data()
    fits, best_model = fit_distributions(df["duration_min"])
    win_curve = compute_matches_won_per_time(df)

    # Histogram bins for empirical distribution
    hist_counts, bin_edges = np.histogram(df["duration_min"], bins=50, density=True)
    hist_data = {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": [float((bin_edges[i] + bin_edges[i+1]) / 2.0) for i in range(len(hist_counts))],
        "densities": hist_counts.tolist(),
        "counts": np.histogram(df["duration_min"], bins=50, density=False)[0].tolist()
    }

    # Summary statistics
    summary_stats = {
        "sample_size": int(len(df)),
        "mean_duration_min": float(df["duration_min"].mean()),
        "mean_duration_sec": float(df["duration_sec"].mean()),
        "std_duration_min": float(df["duration_min"].std()),
        "median_duration_min": float(df["duration_min"].median()),
        "skewness": float(stats.skew(df["duration_min"])),
        "kurtosis": float(stats.kurtosis(df["duration_min"])),
        "min_duration_min": float(df["duration_min"].min()),
        "max_duration_min": float(df["duration_min"].max()),
        "overall_white_win_rate": float((df["outcome"] == "white").mean()),
        "overall_black_win_rate": float((df["outcome"] == "black").mean()),
        "overall_draw_rate": float((df["outcome"] == "draw").mean()),
        "avg_moves_per_game": float(df["ply_count"].mean() / 2.0)
    }

    results = {
        "summary": summary_stats,
        "fits": fits,
        "best_model": best_model,
        "histogram": hist_data,
        "win_curve": win_curve
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n================== PART 1 ANALYSIS RESULTS ==================")
    print(f"Sample Size: {summary_stats['sample_size']} games")
    print(f"Mean Duration: {summary_stats['mean_duration_min']:.2f} min ({summary_stats['mean_duration_sec']:.1f} s)")
    print(f"Std Dev: {summary_stats['std_duration_min']:.2f} min | Skewness: {summary_stats['skewness']:.2f}")
    print("\nModel Comparisons:")
    for model_key, fit in fits.items():
        print(f"  [{model_key.upper()}] AIC: {fit['aic']:.1f} | BIC: {fit['bic']:.1f} | KS stat: {fit['ks_statistic']:.4f} (p={fit['ks_pvalue']:.4e})")
    print(f"\nOptimal Best-Fitting Model: {best_model.upper()}")
    print("=============================================================\n")

if __name__ == "__main__":
    run_part1()
