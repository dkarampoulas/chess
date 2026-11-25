"""
generate_figures.py
Generates high-resolution publication-quality PNG figures for CV portfolio and reports.
"""

import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

PART1_FILE = os.path.join(DATA_DIR, "part1_results.json")
PART2_FILE = os.path.join(DATA_DIR, "part2_results.json")

# Publication style
plt.style.use('dark_background')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.edgecolor'] = '#444c56'
matplotlib.rcParams['axes.linewidth'] = 1.0

def make_figure_part1(part1_data):
    summary = part1_data["summary"]
    fits = part1_data["fits"]
    hist = part1_data["histogram"]
    win_curve = part1_data["win_curve"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), gridspec_kw={'height_ratios': [1.8, 1.2]}, dpi=300)

    # Top Plot: Distribution and Model Fits
    x_dense = np.linspace(0.1, 20.0, 500)
    
    # 1. Histogram
    ax1.bar(hist["bin_centers"], hist["densities"], width=(hist["bin_edges"][1] - hist["bin_edges"][0]),
            color="#d29922", alpha=0.35, edgecolor="#d29922", linewidth=0.8, label="Empirical Density (2,500 games)")

    # 2. Normal fit
    mu_n = fits["normal"]["params"]["mu"]
    sig_n = fits["normal"]["params"]["sigma"]
    pdf_norm = stats.norm.pdf(x_dense, loc=mu_n, scale=sig_n)
    ax1.plot(x_dense, pdf_norm, color="#58a6ff", linewidth=2.8, label=f"Normal $\\mathcal{{N}}(\\mu={mu_n:.1f}, \\sigma={sig_n:.1f})$ [Base Model]")

    # 3. Gamma fit
    alpha_g = fits["gamma"]["params"]["alpha (shape)"]
    scale_g = fits["gamma"]["params"]["theta (scale)"]
    pdf_gamma = stats.gamma.pdf(x_dense, alpha_g, loc=0, scale=scale_g)
    ax1.plot(x_dense, pdf_gamma, color="#3fb950", linewidth=2.0, linestyle="--", label=f"Gamma $\\Gamma(\\alpha={alpha_g:.2f}, \\theta={scale_g:.2f})$")

    # 4. LogNormal fit
    s_ln = fits["lognormal"]["params"]["sigma_log"]
    scale_ln = fits["lognormal"]["params"]["scale"]
    pdf_ln = stats.lognorm.pdf(x_dense, s_ln, loc=0, scale=scale_ln)
    ax1.plot(x_dense, pdf_ln, color="#bc8cff", linewidth=2.0, linestyle=":", label=f"Log-Normal $(\\mu={np.log(scale_ln):.2f}, \\sigma={s_ln:.2f})$")

    ax1.axvline(mu_n, color="#58a6ff", linestyle="--", alpha=0.6, label=f"Mean Duration: {mu_n:.2f} min ({mu_n*60:.0f}s)")
    ax1.set_xlim(0, 20)
    ax1.set_ylabel("Probability Density $f(t)$", fontsize=11, color="#c9d1d9")
    ax1.set_title("10-Minute Rapid Chess Match Duration: Parametric Distribution Modeling", fontsize=14, fontweight="bold", pad=12, color="#ffffff")
    ax1.legend(loc="upper left", framealpha=0.8, facecolor="#161b22", edgecolor="#30363d", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.3, color="#8b949e")

    # Bottom Plot: Conditional Matches Won Per Time P(Outcome | Time)
    t_vals = [w["time_min"] for w in win_curve]
    p_white = [w["p_white_win"] * 100 for w in win_curve]
    p_black = [w["p_black_win"] * 100 for w in win_curve]
    p_draw = [w["p_draw"] * 100 for w in win_curve]

    ax2.plot(t_vals, p_white, color="#ffffff", linewidth=2.4, label="White Win % (First-Move Advantage)")
    ax2.plot(t_vals, p_black, color="#79c0ff", linewidth=2.4, label="Black Win % (Counterplay & Flags)")
    ax2.plot(t_vals, p_draw, color="#d29922", linewidth=2.4, linestyle="-.", label="Draw % (Endgame Simplification)")

    # Fill regions for visual clarity
    ax2.fill_between(t_vals, p_white, alpha=0.15, color="#ffffff")
    ax2.fill_between(t_vals, p_black, alpha=0.15, color="#79c0ff")
    ax2.fill_between(t_vals, p_draw, alpha=0.2, color="#d29922")

    ax2.set_xlim(0, 20)
    ax2.set_ylim(0, 75)
    ax2.set_xlabel("Match Duration $t$ (Minutes)", fontsize=11, color="#c9d1d9")
    ax2.set_ylabel("Conditional Win Probability (%)", fontsize=11, color="#c9d1d9")
    ax2.set_title("Matches Won Conditioned on Duration: $P(\\text{Outcome} \\mid T = t)$", fontsize=13, fontweight="bold", pad=10, color="#ffffff")
    ax2.legend(loc="upper right", framealpha=0.8, facecolor="#161b22", edgecolor="#30363d", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.3, color="#8b949e")

    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "part1_match_duration_distribution.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

def make_figure_part2(part2_data):
    all_squares = part2_data["all_squares"]
    deepdive = part2_data["deadliest_square_deepdive"]
    metadata = part2_data["metadata"]

    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    ax_board = fig.add_subplot(gs[0])
    ax_dossier = fig.add_subplot(gs[1])

    # Board grid 8x8
    files = "abcdefgh"
    ranks = "12345678"
    matrix = np.zeros((8, 8))
    sq_map = {s["square"]: s for s in all_squares}

    for r_idx, r in enumerate(ranks):
        for f_idx, f in enumerate(files):
            sq_name = f + r
            matrix[7 - r_idx, f_idx] = sq_map.get(sq_name, {}).get("deaths", 0)

    # Custom Blood Red Colormap
    blood_cmap = LinearSegmentedColormap.from_list("BloodCarnage", [
        "#161b22", "#4a1010", "#8b1818", "#d11f1f", "#ff3333"
    ])

    im = ax_board.imshow(matrix, cmap=blood_cmap, interpolation="nearest")

    # Annotate top squares
    for r_idx, r in enumerate(ranks):
        for f_idx, f in enumerate(files):
            sq_name = f + r
            sq_info = sq_map.get(sq_name, {})
            val = sq_info.get("deaths", 0)
            p_pct = sq_info.get("p_emp_percentage", 0)
            
            # If high carnage, show coordinate and deaths
            if val >= 1000:
                ax_board.text(f_idx, 7 - r_idx, f"{sq_name.upper()}\n{val}\n({p_pct:.1f}%)",
                             ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=7.5)
            elif val >= 300:
                ax_board.text(f_idx, 7 - r_idx, f"{sq_name}\n{val}",
                             ha="center", va="center", color="#dddddd", fontsize=6.5)

    ax_board.set_xticks(range(8))
    ax_board.set_xticklabels(list("abcdefgh"), fontsize=11, fontweight="bold", color="#c9d1d9")
    ax_board.set_yticks(range(8))
    ax_board.set_yticklabels(list("87654321"), fontsize=11, fontweight="bold", color="#c9d1d9")
    ax_board.set_title("Chessboard Carnage Heatmap: Piece Deaths $p_{\\text{emp}}(i)$", fontsize=13, fontweight="bold", pad=12, color="#ff4d4d")
    
    cbar = plt.colorbar(im, ax=ax_board, fraction=0.046, pad=0.04)
    cbar.set_label("Total Piece Deaths (Captures)", color="#c9d1d9", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#c9d1d9")

    # Right Plot: Piece Demographics & Brawl Decisiveness on #1 Deadliest Square (D4)
    ax_dossier.axis("off")

    # Sub-axes within right panel
    sub_gs = gs[1].subgridspec(2, 1, hspace=0.35)
    ax_pieces = fig.add_subplot(sub_gs[0])
    ax_brawl = fig.add_subplot(sub_gs[1])

    # Piece Sacrifice Distribution
    pieces = list(deepdive["piece_breakdown"].keys())
    counts = [deepdive["piece_breakdown"][p]["count"] for p in pieces]
    pcts = [deepdive["piece_breakdown"][p]["percentage"] for p in pieces]
    colors = ["#f0883e", "#58a6ff", "#bc8cff", "#3fb950", "#f85149"]

    bars = ax_pieces.bar(pieces, counts, color=colors, alpha=0.85, edgecolor="#ffffff", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        yval = bar.get_height()
        ax_pieces.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{pct:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#ffffff")

    ax_pieces.set_title(f"Square {deepdive['square'].upper()}: Victim Piece Demographics (Total: {deepdive['deaths']:,})", fontsize=11, fontweight="bold", color="#ffffff")
    ax_pieces.set_ylabel("Captures Count", fontsize=9, color="#c9d1d9")
    ax_pieces.grid(True, linestyle=":", alpha=0.25, color="#8b949e", axis="y")

    # Brawl Decisiveness Bars
    brawl_stats = deepdive["brawl_statistics"]
    categories = ["White Wins Brawl", "Black Wins Brawl"]
    win_pcts = [brawl_stats["p_game_win_given_brawl_win_white"] * 100, brawl_stats["p_game_win_given_brawl_win_black"] * 100]

    b_bars = ax_brawl.barh(categories, win_pcts, color=["#ffffff", "#58a6ff"], alpha=0.85, edgecolor="#444c56", height=0.55)
    for bar, val in zip(b_bars, win_pcts):
        xval = bar.get_width()
        ax_brawl.text(xval + 1.5, bar.get_y() + bar.get_height()/2.0, f"P(Win Match) = {val:.1f}%", ha='left', va='center', fontsize=9.5, fontweight='bold', color="#ffffff")

    ax_brawl.set_xlim(0, 75)
    ax_brawl.set_xlabel("Probability of Winning Match After Winning Brawl (%)", fontsize=9.5, color="#c9d1d9")
    ax_brawl.set_title(f"Decisiveness of Tactical Exchanges on {deepdive['square'].upper()}", fontsize=11, fontweight="bold", color="#ffffff")
    ax_brawl.grid(True, linestyle=":", alpha=0.25, color="#8b949e", axis="x")

    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, "part2_chessboard_death_heatmap.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_file}")

def main():
    with open(PART1_FILE, "r", encoding="utf-8") as f:
        p1 = json.load(f)
    with open(PART2_FILE, "r", encoding="utf-8") as f:
        p2 = json.load(f)

    make_figure_part1(p1)
    make_figure_part2(p2)
    print("Static high-resolution publication figures generated successfully!")

if __name__ == "__main__":
    main()
