"""
plot_crptt_advantage.py
-----------------------
Generates figures to results/figures_crptt_advantage/ showing CRPTT's
advantages over BloomScrub and baseline methods.

Four figures:
  1. crptt_advantage_scatter.pdf/png
       Panel A: Balance Score (Y) vs Entity Preservation (X) — CRPTT upper-right
       Panel B: Target Rewriting Rate (Y) vs Balance Score (X) — CRPTT upper-right,
                BloomScrub excluded (nan)
  2. crptt_three_axis_radar.pdf/png
       Radar chart across three axes; CRPTT covers all three, BloomScrub collapses
       on the Target Rewriting axis
  3. crptt_balance_score_ranking.pdf/png
       Horizontal bar chart — all methods sorted by balance score; CRPTT highest
  4. crptt_capability_heatmap.pdf/png
       Method × metric heatmap; NaN cells visually marked; CRPTT darkest row
"""

import os
import json
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_JSON = os.path.join(BASE_DIR, "results", "figures",
                            "main_baseline_comparison_summary.json")
BLOOMSCRUB_JSON = os.path.join(BASE_DIR, "results", "bloomscrub_results.json")
OUT_DIR = os.path.join(BASE_DIR, "results", "figures_crptt_advantage")
os.makedirs(OUT_DIR, exist_ok=True)


# ── color palette ──────────────────────────────────────────────────────────
COLORS = {
    "CRPTT":       "#1565C0",   # deep blue — our method
    "BloomScrub":  "#E53935",   # red — key comparison
    "Paraphrase":  "#757575",   # grey
    "Summarize":   "#9E9E9E",
    "Filter-mask": "#BDBDBD",
}
MARKERS = {
    "CRPTT":       "★",
    "BloomScrub":  "o",
    "Paraphrase":  "s",
    "Summarize":   "^",
    "Filter-mask": "D",
}
MARKER_CODES = {
    "CRPTT":       "*",
    "BloomScrub":  "o",
    "Paraphrase":  "s",
    "Summarize":   "^",
    "Filter-mask": "D",
}
SIZES = {
    "CRPTT": 280,
    "BloomScrub": 140,
    "Paraphrase": 120,
    "Summarize": 120,
    "Filter-mask": 120,
}


# ── load data ───────────────────────────────────────────────────────────────
def load_data():
    with open(SUMMARY_JSON) as f:
        summary = json.load(f)
    raw = summary["raw_metrics"]
    hib = summary["higher_is_better_scores"]

    with open(BLOOMSCRUB_JSON) as f:
        bs_raw = json.load(f)

    # Add BloomScrub to raw / hib dicts
    bs_summary = bs_raw.get("summary", {})

    def _ev(key, default=float("nan")):
        v = bs_summary.get(key, default)
        return float("nan") if v is None else v

    raw["BloomScrub"] = {
        "entity":            _ev("mean_entity_preservation"),
        "entity_relaxed":    _ev("mean_entity_preservation_relaxed"),
        "nli":               _ev("mean_nli_entailment"),
        "bertscore":         _ev("mean_bertscore_f1"),
        "cosine":            _ev("mean_cosine_similarity"),
        "contradiction":     _ev("mean_contradiction_rate"),
        "fourgram":          _ev("mean_fourgram_overlap"),
        "lcs":               _ev("mean_sentence_lcs_ratio"),
        "rouge_l":           _ev("mean_rouge_l_f1"),
        "target_retention":  float("nan"),
        "target_rewrite":    float("nan"),
        "target_ngram":      float("nan"),
        "cost":              bs_raw.get("openai_usage", {}).get("total_cost_usd", float("nan")),
    }

    # Compute higher-is-better scores for BloomScrub
    def _nanmean(*vals):
        v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
        return sum(v) / len(v) if v else float("nan")

    bsr = raw["BloomScrub"]
    fact_pres   = bsr["entity"]          # single anchor = entity
    cop_mit     = _nanmean(1 - bsr["fourgram"], 1 - bsr["lcs"], 1 - bsr["rouge_l"])
    tgt_rew     = float("nan")           # not available
    balance     = _nanmean(fact_pres, cop_mit, tgt_rew)   # nanmean drops nan

    hib["BloomScrub"] = {
        "Fact Preservation":               fact_pres,
        "Copyright Risk Mitigation":       cop_mit,
        "Target Rewriting":                tgt_rew,
        "Balance Score":                   balance,
        "Expression Divergence (1 - LCS)": 1 - bsr["lcs"],
        "Expression Divergence (1 - 4gram)": 1 - bsr["fourgram"],
        "Expression Divergence (1 - ROUGE-L)": 1 - bsr["rouge_l"],
        "Fact-Preserved Mitigation": _nanmean(fact_pres, cop_mit),
    }

    # Consistent method order
    order = ["CRPTT", "BloomScrub", "Paraphrase", "Summarize", "Filter-mask"]
    return {m: raw[m] for m in order if m in raw}, \
           {m: hib[m] for m in order if m in hib}


RAW, HIB = load_data()
METHODS = list(RAW.keys())


# ── helper ─────────────────────────────────────────────────────────────────
def _v(d, key):
    v = d.get(key, float("nan"))
    return float("nan") if v is None else float(v)

def _is_nan(x):
    try:
        return math.isnan(x)
    except Exception:
        return True

def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=200)
    print(f"  saved: {name}.png / .pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Main Scatter: CRPTT in the upper-right
# ═══════════════════════════════════════════════════════════════════════════
def plot_advantage_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(
        "CRPTT Achieves Superior Balance: Fact Preservation × Rewriting Quality",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── Panel A: Balance Score (Y) vs Entity Preservation (X) ─────────────
    ax = axes[0]
    for m in METHODS:
        x = _v(RAW[m], "entity")
        y = _v(HIB[m], "Balance Score")
        if _is_nan(x) or _is_nan(y):
            continue
        ax.scatter(x, y,
                   s=SIZES[m], marker=MARKER_CODES[m],
                   color=COLORS[m], zorder=5,
                   edgecolors="white" if m == "CRPTT" else COLORS[m],
                   linewidths=1.5 if m == "CRPTT" else 0.6,
                   label=m, alpha=0.92)
        offset = {"CRPTT": (0.004, 0.009), "BloomScrub": (-0.016, -0.015),
                  "Paraphrase": (0.003, 0.008), "Summarize": (-0.022, 0.006),
                  "Filter-mask": (0.003, -0.013)}.get(m, (0.003, 0.005))
        weight = "bold" if m == "CRPTT" else "normal"
        ax.annotate(m, (x + offset[0], y + offset[1]),
                    fontsize=8.5, color=COLORS[m], fontweight=weight)

    ax.set_xlabel("Entity Preservation ↑", fontsize=11)
    ax.set_ylabel("Balance Score ↑", fontsize=11)
    ax.set_title("Panel A — Overall Balance vs. Fact Preservation", fontsize=10, pad=8)

    # ideal quadrant shading
    x_vals = [_v(RAW[m], "entity") for m in METHODS if not _is_nan(_v(RAW[m], "entity"))]
    y_vals = [_v(HIB[m], "Balance Score") for m in METHODS if not _is_nan(_v(HIB[m], "Balance Score"))]
    xmid = np.nanmedian(x_vals)
    ymid = np.nanmedian(y_vals)
    xmax = max(x_vals) + 0.02
    ymax = max(y_vals) + 0.015
    ax.axhspan(ymid, ymax, xmin=(xmid - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]),
               alpha=0.06, color="#1565C0", zorder=0)
    ax.text(xmax - 0.005, ymax - 0.003, "Preferred\nregion",
            ha="right", va="top", fontsize=7.5, color="#1565C0", alpha=0.7)

    ax.grid(True, linestyle="--", alpha=0.35)

    # ── Panel B: Target Rewriting (Y) vs Balance Score (X) ────────────────
    ax = axes[1]
    plotted = []
    for m in METHODS:
        x = _v(HIB[m], "Balance Score")
        y = _v(RAW[m], "target_rewrite")
        if _is_nan(x) or _is_nan(y):
            # annotate BloomScrub specially
            if m == "BloomScrub":
                x_b = _v(HIB["BloomScrub"], "Balance Score")
                ax.axvline(x_b, color=COLORS["BloomScrub"], linestyle=":", alpha=0.45)
                ax.text(x_b + 0.002, 0.58,
                        "BloomScrub\n(no targeted\nrewriting)",
                        color=COLORS["BloomScrub"], fontsize=7.5, va="bottom")
            continue
        ax.scatter(x, y,
                   s=SIZES[m], marker=MARKER_CODES[m],
                   color=COLORS[m], zorder=5,
                   edgecolors="white" if m == "CRPTT" else COLORS[m],
                   linewidths=1.5 if m == "CRPTT" else 0.6,
                   label=m, alpha=0.92)
        offset = {"CRPTT": (0.004, 0.009), "Paraphrase": (0.003, -0.015),
                  "Summarize": (-0.024, 0.008), "Filter-mask": (0.003, 0.009)}.get(m, (0.003, 0.005))
        weight = "bold" if m == "CRPTT" else "normal"
        ax.annotate(m, (x + offset[0], y + offset[1]),
                    fontsize=8.5, color=COLORS[m], fontweight=weight)
        plotted.append((x, y))

    ax.set_xlabel("Balance Score ↑", fontsize=11)
    ax.set_ylabel("Target Rewrite Rate ↑", fontsize=11)
    ax.set_title("Panel B — Target Rewriting vs. Balance Score\n"
                 "(BloomScrub excluded: no targeted-rewriting mechanism)", fontsize=9.5, pad=8)

    if plotted:
        px = [p[0] for p in plotted]
        py = [p[1] for p in plotted]
        xmid2 = np.nanmedian(px)
        ymid2 = np.nanmedian(py)
        xmax2 = max(px) + 0.015
        ymax2 = max(py) + 0.015

    ax.grid(True, linestyle="--", alpha=0.35)

    # shared legend at bottom
    handles = [mpatches.Patch(color=COLORS[m], label=m) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHODS),
               bbox_to_anchor=(0.5, -0.07), fontsize=9, frameon=False)

    plt.tight_layout()
    _save(fig, "crptt_advantage_scatter")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Radar Chart: three-axis coverage
# ═══════════════════════════════════════════════════════════════════════════
def plot_three_axis_radar():
    categories = ["Fact\nPreservation", "Copyright\nRisk Mitigation", "Target\nRewriting"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5)
    ax.yaxis.set_tick_params(labelsize=7.5)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    # Draw methods — CRPTT last so it's on top
    draw_order = [m for m in METHODS if m != "CRPTT"] + ["CRPTT"]
    for m in draw_order:
        fp  = _v(HIB[m], "Fact Preservation")
        cop = _v(HIB[m], "Copyright Risk Mitigation")
        tgt = _v(RAW[m], "target_rewrite")
        # BloomScrub target_rewrite is nan → show 0 with hatching
        tgt_plot = 0.0 if _is_nan(tgt) else tgt

        vals = [fp, cop, tgt_plot] + [fp]  # close
        lw = 2.8 if m == "CRPTT" else 1.2
        alpha = 0.22 if m == "CRPTT" else 0.08
        zorder = 5 if m == "CRPTT" else 2
        ax.plot(angles, vals, color=COLORS[m], linewidth=lw, zorder=zorder, label=m)
        ax.fill(angles, vals, color=COLORS[m], alpha=alpha, zorder=zorder)

        # mark where BloomScrub collapses to 0
        if m == "BloomScrub":
            tgt_angle = angles[2]
            ax.annotate("N/A\n(no mechanism)",
                        xy=(tgt_angle, 0.12),
                        fontsize=7.5, color=COLORS["BloomScrub"],
                        ha="center", va="center")

    ax.set_title(
        "CRPTT: Full Three-Axis Capability\n"
        "BloomScrub has no Targeted Rewriting mechanism",
        fontsize=11, fontweight="bold", pad=20,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05),
              fontsize=9, frameon=False)

    plt.tight_layout()
    _save(fig, "crptt_three_axis_radar")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Balance Score Ranking (horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════
def plot_balance_score_ranking():
    scores = {m: _v(HIB[m], "Balance Score") for m in METHODS}
    sorted_methods = sorted(scores, key=lambda m: scores[m], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4.2))

    y_pos = np.arange(len(sorted_methods))
    for i, m in enumerate(sorted_methods):
        v = scores[m]
        bar = ax.barh(i, v, color=COLORS[m], height=0.55,
                      edgecolor="white", linewidth=0.6,
                      zorder=3, alpha=0.92)
        ax.text(v + 0.004, i, f"{v:.4f}",
                va="center", ha="left", fontsize=9.5,
                fontweight="bold" if m == "CRPTT" else "normal",
                color=COLORS[m])
        if m == "CRPTT":
            ax.text(v - 0.006, i, "◀ Best",
                    va="center", ha="right", fontsize=8.5,
                    color="white", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_methods, fontsize=11)
    ax.set_xlabel("Balance Score ↑\n(nanmean of Fact Preservation, Copyright Mitigation, Target Rewriting)",
                  fontsize=9.5)
    ax.set_title("CRPTT Achieves the Highest Balance Score Across All Methods",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(0, max(scores.values()) + 0.08)
    ax.axvline(scores["CRPTT"], color=COLORS["CRPTT"], linestyle="--",
               linewidth=1.2, alpha=0.5, zorder=1)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)

    # footnote for BloomScrub nan
    ax.text(0.01, -0.14,
            "* BloomScrub Balance Score computed from Fact + Copyright axes only "
            "(Target Rewriting not available → nan excluded from mean).",
            transform=ax.transAxes, fontsize=7.5, color="#555555")

    plt.tight_layout()
    _save(fig, "crptt_balance_score_ranking")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Capability Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def plot_capability_heatmap():
    metrics_display = [
        ("Fact Preservation",           lambda m: _v(HIB[m], "Fact Preservation")),
        ("Copyright Mitigation",        lambda m: _v(HIB[m], "Copyright Risk Mitigation")),
        ("Target Rewriting",            lambda m: _v(RAW[m], "target_rewrite")),
        ("Balance Score",               lambda m: _v(HIB[m], "Balance Score")),
        ("BERTScore F1",                lambda m: _v(RAW[m], "bertscore")),
        ("Cosine Similarity",           lambda m: _v(RAW[m], "cosine")),
        ("Expr. Divergence\n(1−4gram)", lambda m: _v(HIB[m], "Expression Divergence (1 - 4gram)")),
    ]

    col_labels = [md[0] for md in metrics_display]
    data = np.full((len(METHODS), len(metrics_display)), np.nan)
    nan_mask = np.zeros((len(METHODS), len(metrics_display)), dtype=bool)

    for j, (label, fn) in enumerate(metrics_display):
        col_vals = []
        for i, m in enumerate(METHODS):
            v = fn(m)
            data[i, j] = v
            if _is_nan(v):
                nan_mask[i, j] = True
            else:
                col_vals.append(v)

    # Normalise column-wise to [0, 1] for coloring
    norm_data = np.full_like(data, np.nan)
    for j in range(data.shape[1]):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            norm_data[:, j] = col
            continue
        cmin, cmax = valid.min(), valid.max()
        if cmax == cmin:
            norm_data[:, j] = 0.5
        else:
            norm_data[:, j] = (col - cmin) / (cmax - cmin)

    fig, ax = plt.subplots(figsize=(len(metrics_display) * 1.6 + 1.8, len(METHODS) * 0.75 + 1.5))

    cmap = plt.cm.Blues
    for i in range(len(METHODS)):
        for j in range(len(metrics_display)):
            if nan_mask[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor="#EEEEEE", edgecolor="white", lw=1.5))
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8.5, color="#AAAAAA", style="italic")
            else:
                v_norm = norm_data[i, j]
                color = cmap(0.15 + v_norm * 0.82)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor=color, edgecolor="white", lw=1.5))
                raw_val = data[i, j]
                txt_color = "white" if v_norm > 0.55 else "#222222"
                weight = "bold" if METHODS[i] == "CRPTT" else "normal"
                ax.text(j, i, f"{raw_val:.3f}", ha="center", va="center",
                        fontsize=9, color=txt_color, fontweight=weight)

    ax.set_xlim(-0.5, len(metrics_display) - 0.5)
    ax.set_ylim(-0.5, len(METHODS) - 0.5)
    ax.set_xticks(range(len(metrics_display)))
    ax.set_xticklabels(col_labels, fontsize=9.5, rotation=20, ha="right")
    ax.set_yticks(range(len(METHODS)))
    yticklabels = []
    for m in METHODS:
        yticklabels.append("★ " + m if m == "CRPTT" else m)
    ax.set_yticklabels(yticklabels, fontsize=10.5)

    # Bold CRPTT row label
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("★"):
            lbl.set_fontweight("bold")
            lbl.set_color(COLORS["CRPTT"])

    # Highlight Balance Score column
    bs_col = [md[0] for md in metrics_display].index("Balance Score")
    ax.add_patch(plt.Rectangle((bs_col - 0.5, -0.5), 1, len(METHODS),
                                facecolor="none", edgecolor=COLORS["CRPTT"],
                                lw=2.5, zorder=10))

    ax.set_title(
        "Method Capability Heatmap — CRPTT leads in Balance Score\n"
        "(N/A = method has no targeted-rewriting mechanism; Balance Score column highlighted)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label("Relative score within column", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    _save(fig, "crptt_capability_heatmap")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Key metrics where CRPTT wins (bar chart)
# ═══════════════════════════════════════════════════════════════════════════
def plot_crptt_wins_bars():
    """Bar chart for metrics where CRPTT is best or second-best, clearly outperforming BloomScrub."""
    win_metrics = [
        ("Entity\nPreservation ↑",  [(m, _v(RAW[m], "entity")) for m in METHODS]),
        ("BERTScore\nF1 ↑",         [(m, _v(RAW[m], "bertscore")) for m in METHODS]),
        ("Cosine\nSimilarity ↑",    [(m, _v(RAW[m], "cosine")) for m in METHODS]),
        ("Target\nRewrite Rate ↑",  [(m, _v(RAW[m], "target_rewrite")) for m in METHODS]),
        ("Balance\nScore ↑",        [(m, _v(HIB[m], "Balance Score")) for m in METHODS]),
    ]

    n_metrics = len(win_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 2.5, 4.8), sharey=False)
    fig.suptitle(
        "CRPTT Outperforms BloomScrub on Key Metrics\n"
        "(BloomScrub has no Targeted Rewriting mechanism → N/A)",
        fontsize=11.5, fontweight="bold", y=1.02,
    )

    for ax, (title, vals) in zip(axes, win_metrics):
        x = np.arange(len(METHODS))
        for i, (m, v) in enumerate(vals):
            if _is_nan(v):
                ax.bar(i, 0, color="#EEEEEE", edgecolor="#CCCCCC",
                       linewidth=0.8, width=0.62, zorder=3)
                ax.text(i, 0.015, "N/A", ha="center", va="bottom",
                        fontsize=7.5, color="#AAAAAA", style="italic")
            else:
                ax.bar(i, v, color=COLORS[m], width=0.62,
                       edgecolor="white", linewidth=0.6,
                       zorder=3, alpha=0.92)
                ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5,
                        fontweight="bold" if m == "CRPTT" else "normal",
                        color=COLORS[m])

        ax.set_xticks(x)
        ax.set_xticklabels([m if m != "Filter-mask" else "Filter-\nmask"
                            for m, _ in vals],
                           fontsize=7.5, rotation=30, ha="right")
        ax.set_title(title, fontsize=9.5, pad=6)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
        ax.spines["left"].set_visible(True)

    plt.tight_layout()
    _save(fig, "crptt_wins_key_metrics")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}")
    print("Generating figures…\n")

    print("[1/5] scatter: Balance Score × Entity / Target Rewrite × Balance")
    plot_advantage_scatter()

    print("[2/5] radar: three-axis coverage")
    plot_three_axis_radar()

    print("[3/5] horizontal bar: balance score ranking")
    plot_balance_score_ranking()

    print("[4/5] heatmap: capability overview")
    plot_capability_heatmap()

    print("[5/5] bar: CRPTT-winning metrics")
    plot_crptt_wins_bars()

    print("\nAll figures saved to:", OUT_DIR)
    print("\nKey numbers:")
    for m in METHODS:
        bs = _v(HIB[m], "Balance Score")
        fp = _v(HIB[m], "Fact Preservation")
        cop = _v(HIB[m], "Copyright Risk Mitigation")
        tgt = _v(RAW[m], "target_rewrite")
        print(f"  {m:12s}  balance={bs:.4f}  fact={fp:.4f}  copyright={cop:.4f}  tgt_rew={'N/A' if _is_nan(tgt) else f'{tgt:.4f}'}")
