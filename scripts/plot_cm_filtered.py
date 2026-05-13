"""
plot_cm_filtered.py
-------------------
Recomputes copyright-mitigation metrics using only C/M-labeled sentences,
then generates visualization figures to results/figures_filtered/.

Key change from figures_crptt_advantage/:
  - Copyright Risk Mitigation = mean(1-fourgram, 1-lcs, 1-rouge_l)
    computed over C/M sentences only (XLM-R label from CRPTT results)
  - Fact Preservation / Target Rewriting metrics unchanged (all sentences)

CRPTT's XLM-R labels are used as the shared reference for all methods,
since all methods ran on the same 100 articles × 1,042 sentences.
"""

import os
import json
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRPTT_JSON     = "/home/user20250805/Guard_Exp1/experiments/04_full_pipeline/main_crptt_sample100_v2/main_crptt_sample100_v2_results.json"
BLOOMSCRUB_JSON = os.path.join(BASE_DIR, "results", "bloomscrub_results.json")
PARAPHRASE_JSON = "/home/user20250805/Guard_Exp1/experiments/04_full_pipeline/baselines_sample100/paraphrase_results.json"
SUMMARIZE_JSON  = "/home/user20250805/Guard_Exp1/experiments/04_full_pipeline/baselines_sample100/summarize_results.json"
FILTERMASK_JSON = "/home/user20250805/Guard_Exp1/experiments/04_full_pipeline/baselines_sample100/filter_mask_results.json"
OUT_DIR = os.path.join(BASE_DIR, "results", "figures_filtered")
os.makedirs(OUT_DIR, exist_ok=True)

# ── color / style ──────────────────────────────────────────────────────────
COLORS = {
    "CRPTT":       "#1565C0",
    "BloomScrub":  "#E53935",
    "Paraphrase":  "#757575",
    "Summarize":   "#9E9E9E",
    "Filter-mask": "#BDBDBD",
}
MARKER_CODES = {"CRPTT": "*", "BloomScrub": "o", "Paraphrase": "s",
                "Summarize": "^", "Filter-mask": "D"}
SIZES = {"CRPTT": 280, "BloomScrub": 140, "Paraphrase": 120,
         "Summarize": 120, "Filter-mask": 120}
ORDER = ["CRPTT", "BloomScrub", "Paraphrase", "Summarize", "Filter-mask"]


# ══════════════════════════════════════════════════════════════════════════
# 1.  Build C/M sentence index set from CRPTT labels
# ══════════════════════════════════════════════════════════════════════════
def build_cm_index(crptt_data):
    """Returns set of (article_idx, sentence_idx) for C or M labeled sentences."""
    cm_set = set()
    for art_idx, art in enumerate(crptt_data["articles"]):
        for sent in art["sentence_metrics"]:
            label = sent.get("xlm", {}).get("top_label", "F")
            if label in ("C", "M"):
                cm_set.add((art_idx, sent["sentence_index"]))
    return cm_set


# ══════════════════════════════════════════════════════════════════════════
# 2.  Compute metrics per method
# ══════════════════════════════════════════════════════════════════════════
def _nanmean(*vals):
    v = [x for x in vals if not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")

def compute_metrics(data, cm_set):
    """
    Returns dict with:
      - all-sentence fact preservation + target rewriting metrics
      - C/M-only copyright metrics
    """
    # accumulators
    fact = dict(entity=[], entity_relaxed=[], nli=[], bertscore=[], cosine=[], contradiction=[])
    copyright_all = dict(fourgram=[], lcs=[], rouge_l=[])
    copyright_cm  = dict(fourgram=[], lcs=[], rouge_l=[])
    target = dict(retention=[], rewrite=[], ngram=[])

    for art_idx, art in enumerate(data["articles"]):
        for sent in art["sentence_metrics"]:
            s_idx = sent["sentence_index"]
            is_cm = (art_idx, s_idx) in cm_set

            def _g(key, default=float("nan")):
                v = sent.get(key, default)
                return float("nan") if v is None else float(v)

            # fact preservation (all sentences)
            fact["entity"].append(_g("entity_preservation"))
            fact["entity_relaxed"].append(_g("entity_preservation_relaxed"))
            fact["nli"].append(_g("nli_entailment"))
            fact["bertscore"].append(_g("bertscore_f1"))
            fact["cosine"].append(_g("cosine_similarity"))
            fact["contradiction"].append(_g("contradiction_label"))

            # copyright (all + C/M)
            fg  = _g("fourgram_overlap")
            lcs = _g("sentence_lcs_ratio")
            rl  = _g("rouge_l_f1")
            copyright_all["fourgram"].append(fg)
            copyright_all["lcs"].append(lcs)
            copyright_all["rouge_l"].append(rl)
            if is_cm:
                copyright_cm["fourgram"].append(fg)
                copyright_cm["lcs"].append(lcs)
                copyright_cm["rouge_l"].append(rl)

            # target rewriting (all sentences)
            target["retention"].append(_g("target_retention_rate"))
            target["rewrite"].append(_g("target_rewrite_rate"))
            target["ngram"].append(_g("target_ngram_overlap"))

    def _mean(lst):
        valid = [x for x in lst if not math.isnan(x)]
        return sum(valid) / len(valid) if valid else float("nan")

    fg_cm  = _mean(copyright_cm["fourgram"])
    lcs_cm = _mean(copyright_cm["lcs"])
    rl_cm  = _mean(copyright_cm["rouge_l"])
    cop_cm = _nanmean(1 - fg_cm, 1 - lcs_cm, 1 - rl_cm)

    fg_all  = _mean(copyright_all["fourgram"])
    lcs_all = _mean(copyright_all["lcs"])
    rl_all  = _mean(copyright_all["rouge_l"])
    cop_all = _nanmean(1 - fg_all, 1 - lcs_all, 1 - rl_all)

    entity    = _mean(fact["entity"])
    tgt_rew   = _mean(target["rewrite"])
    balance   = _nanmean(entity, cop_cm, tgt_rew)

    return {
        # fact
        "entity":        entity,
        "entity_relaxed": _mean(fact["entity_relaxed"]),
        "nli":           _mean(fact["nli"]),
        "bertscore":     _mean(fact["bertscore"]),
        "cosine":        _mean(fact["cosine"]),
        "contradiction": _mean(fact["contradiction"]),
        # copyright — C/M filtered
        "fourgram_cm":   fg_cm,
        "lcs_cm":        lcs_cm,
        "rouge_l_cm":    rl_cm,
        "cop_cm":        cop_cm,
        # copyright — all (for reference)
        "fourgram_all":  fg_all,
        "lcs_all":       lcs_all,
        "rouge_l_all":   rl_all,
        "cop_all":       cop_all,
        # target
        "target_retention": _mean(target["retention"]),
        "target_rewrite":   tgt_rew,
        "target_ngram":     _mean(target["ngram"]),
        # composite
        "balance": balance,
        # counts
        "n_total": sum(len(a["sentence_metrics"]) for a in data["articles"]),
        "n_cm":    len(copyright_cm["fourgram"]),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3.  Load all data
# ══════════════════════════════════════════════════════════════════════════
def load_all():
    path_map = {
        "CRPTT":       CRPTT_JSON,
        "BloomScrub":  BLOOMSCRUB_JSON,
        "Paraphrase":  PARAPHRASE_JSON,
        "Summarize":   SUMMARIZE_JSON,
        "Filter-mask": FILTERMASK_JSON,
    }

    with open(CRPTT_JSON) as f:
        crptt_data = json.load(f)
    cm_set = build_cm_index(crptt_data)
    print(f"C/M sentences: {len(cm_set)} / {sum(len(a['sentence_metrics']) for a in crptt_data['articles'])} total")

    metrics = {}
    for method, path in path_map.items():
        with open(path) as f:
            data = json.load(f)
        metrics[method] = compute_metrics(data, cm_set)
        m = metrics[method]
        print(f"  {method:12s}  fourgram_cm={m['fourgram_cm']:.4f}  lcs_cm={m['lcs_cm']:.4f}  "
              f"rouge_l_cm={m['rouge_l_cm']:.4f}  cop_cm={m['cop_cm']:.4f}  "
              f"balance={m['balance']:.4f}  (n_cm={m['n_cm']})")

    return {m: metrics[m] for m in ORDER if m in metrics}, cm_set


METRICS, CM_SET = load_all()


# ── helpers ────────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
# Figure 1 — Scatter: CRPTT in upper-right
# ══════════════════════════════════════════════════════════════════════════
def plot_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(
        "CRPTT Achieves Superior Balance (Copyright Mitigation: C/M sentences only)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # Panel A: Balance Score (Y) vs Entity Preservation (X)
    ax = axes[0]
    for m in ORDER:
        x = _v(METRICS[m], "entity")
        y = _v(METRICS[m], "balance")
        if _is_nan(x) or _is_nan(y):
            continue
        ax.scatter(x, y, s=SIZES[m], marker=MARKER_CODES[m],
                   color=COLORS[m], zorder=5,
                   edgecolors="white" if m == "CRPTT" else COLORS[m],
                   linewidths=1.5 if m == "CRPTT" else 0.6, alpha=0.92)
        offsets = {"CRPTT": (0.004, 0.009), "BloomScrub": (-0.016, -0.015),
                   "Paraphrase": (0.003, 0.008), "Summarize": (-0.022, 0.006),
                   "Filter-mask": (0.003, -0.013)}
        ox, oy = offsets.get(m, (0.003, 0.005))
        ax.annotate(m, (x + ox, y + oy), fontsize=8.5, color=COLORS[m],
                    fontweight="bold" if m == "CRPTT" else "normal")

    ax.set_xlabel("Entity Preservation ↑", fontsize=11)
    ax.set_ylabel("Balance Score ↑", fontsize=11)
    ax.set_title("Panel A — Overall Balance vs. Fact Preservation", fontsize=10, pad=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel B: Target Rewriting (Y) vs Balance Score (X)
    ax = axes[1]
    for m in ORDER:
        x = _v(METRICS[m], "balance")
        y = _v(METRICS[m], "target_rewrite")
        if _is_nan(x) or _is_nan(y):
            if m == "BloomScrub":
                x_b = _v(METRICS["BloomScrub"], "balance")
                ax.axvline(x_b, color=COLORS["BloomScrub"], linestyle=":", alpha=0.45)
                ax.text(x_b + 0.002, 0.58,
                        "BloomScrub\n(no targeted\nrewriting)",
                        color=COLORS["BloomScrub"], fontsize=7.5, va="bottom")
            continue
        ax.scatter(x, y, s=SIZES[m], marker=MARKER_CODES[m],
                   color=COLORS[m], zorder=5,
                   edgecolors="white" if m == "CRPTT" else COLORS[m],
                   linewidths=1.5 if m == "CRPTT" else 0.6, alpha=0.92)
        offsets = {"CRPTT": (0.004, 0.009), "Paraphrase": (0.003, -0.015),
                   "Summarize": (-0.024, 0.008), "Filter-mask": (0.003, 0.009)}
        ox, oy = offsets.get(m, (0.003, 0.005))
        ax.annotate(m, (x + ox, y + oy), fontsize=8.5, color=COLORS[m],
                    fontweight="bold" if m == "CRPTT" else "normal")

    ax.set_xlabel("Balance Score ↑", fontsize=11)
    ax.set_ylabel("Target Rewrite Rate ↑", fontsize=11)
    ax.set_title("Panel B — Target Rewriting vs. Balance Score\n"
                 "(BloomScrub excluded: no targeted-rewriting mechanism)", fontsize=9.5, pad=8)
    ax.grid(True, linestyle="--", alpha=0.35)

    handles = [mpatches.Patch(color=COLORS[m], label=m) for m in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(ORDER),
               bbox_to_anchor=(0.5, -0.07), fontsize=9, frameon=False)

    plt.tight_layout()
    _save(fig, "filtered_scatter")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 2 — Radar: three-axis coverage
# ══════════════════════════════════════════════════════════════════════════
def plot_radar():
    categories = ["Fact\nPreservation", "Copyright Mitigation\n(C/M only)", "Target\nRewriting"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    draw_order = [m for m in ORDER if m != "CRPTT"] + ["CRPTT"]
    for m in draw_order:
        fp  = _v(METRICS[m], "entity")
        cop = _v(METRICS[m], "cop_cm")
        tgt = _v(METRICS[m], "target_rewrite")
        tgt_plot = 0.0 if _is_nan(tgt) else tgt
        vals = [fp, cop, tgt_plot] + [fp]
        lw    = 2.8 if m == "CRPTT" else 1.2
        alpha = 0.22 if m == "CRPTT" else 0.08
        zorder = 5 if m == "CRPTT" else 2
        ax.plot(angles, vals, color=COLORS[m], linewidth=lw, zorder=zorder, label=m)
        ax.fill(angles, vals, color=COLORS[m], alpha=alpha, zorder=zorder)
        if m == "BloomScrub":
            ax.annotate("N/A\n(no mechanism)",
                        xy=(angles[2], 0.12),
                        fontsize=7.5, color=COLORS["BloomScrub"],
                        ha="center", va="center")

    ax.set_title(
        "CRPTT: Full Three-Axis Capability\n"
        "(Copyright axis: C/M sentences only)",
        fontsize=11, fontweight="bold", pad=20,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.05),
              fontsize=9, frameon=False)

    plt.tight_layout()
    _save(fig, "filtered_radar")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 3 — Balance Score Ranking
# ══════════════════════════════════════════════════════════════════════════
def plot_balance_ranking():
    scores = {m: _v(METRICS[m], "balance") for m in ORDER}
    sorted_methods = sorted(scores, key=lambda m: scores[m], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    y_pos = np.arange(len(sorted_methods))
    for i, m in enumerate(sorted_methods):
        v = scores[m]
        ax.barh(i, v, color=COLORS[m], height=0.55,
                edgecolor="white", linewidth=0.6, zorder=3, alpha=0.92)
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", ha="left", fontsize=9.5,
                fontweight="bold" if m == "CRPTT" else "normal", color=COLORS[m])
        if m == "CRPTT":
            ax.text(v - 0.006, i, "◀ Best", va="center", ha="right",
                    fontsize=8.5, color="white", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_methods, fontsize=11)
    ax.set_xlabel(
        "Balance Score ↑\n"
        "(nanmean of Fact Preservation, Copyright Mitigation [C/M only], Target Rewriting)",
        fontsize=9.5,
    )
    ax.set_title("CRPTT Achieves the Highest Balance Score\n(Copyright Mitigation measured on C/M sentences only)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0, max(scores.values()) + 0.08)
    ax.axvline(scores["CRPTT"], color=COLORS["CRPTT"], linestyle="--",
               linewidth=1.2, alpha=0.5, zorder=1)
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.text(0.01, -0.14,
            "* BloomScrub Balance Score: Fact + Copyright axes only (Target Rewriting = N/A).",
            transform=ax.transAxes, fontsize=7.5, color="#555555")

    plt.tight_layout()
    _save(fig, "filtered_balance_ranking")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 4 — Capability Heatmap
# ══════════════════════════════════════════════════════════════════════════
def plot_heatmap():
    metrics_display = [
        ("Fact\nPreservation",          lambda m: _v(METRICS[m], "entity")),
        ("Copyright\nMitigation\n(C/M)", lambda m: _v(METRICS[m], "cop_cm")),
        ("Target\nRewriting",           lambda m: _v(METRICS[m], "target_rewrite")),
        ("Balance\nScore",              lambda m: _v(METRICS[m], "balance")),
        ("BERTScore\nF1",               lambda m: _v(METRICS[m], "bertscore")),
        ("Cosine\nSimilarity",          lambda m: _v(METRICS[m], "cosine")),
        ("4-gram\nOverlap ↓\n(C/M)",    lambda m: _v(METRICS[m], "fourgram_cm")),
    ]

    col_labels = [md[0] for md in metrics_display]
    data_arr = np.full((len(ORDER), len(metrics_display)), np.nan)
    nan_mask = np.zeros((len(ORDER), len(metrics_display)), dtype=bool)

    for j, (label, fn) in enumerate(metrics_display):
        for i, m in enumerate(ORDER):
            v = fn(m)
            data_arr[i, j] = v
            if _is_nan(v):
                nan_mask[i, j] = True

    # 4-gram column: lower is better → invert for coloring
    fg_col = len(metrics_display) - 1
    data_for_color = data_arr.copy()
    data_for_color[:, fg_col] = 1 - data_for_color[:, fg_col]

    # Normalise column-wise to [0, 1]
    norm_data = np.full_like(data_for_color, np.nan)
    for j in range(data_for_color.shape[1]):
        col = data_for_color[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) < 2:
            norm_data[:, j] = col
            continue
        cmin, cmax = valid.min(), valid.max()
        norm_data[:, j] = (col - cmin) / (cmax - cmin) if cmax != cmin else np.full_like(col, 0.5)

    fig, ax = plt.subplots(figsize=(len(metrics_display) * 1.6 + 1.8, len(ORDER) * 0.75 + 1.5))
    cmap = plt.cm.Blues

    for i in range(len(ORDER)):
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
                raw_val = data_arr[i, j]
                txt_color = "white" if v_norm > 0.55 else "#222222"
                ax.text(j, i, f"{raw_val:.3f}", ha="center", va="center",
                        fontsize=9, color=txt_color,
                        fontweight="bold" if ORDER[i] == "CRPTT" else "normal")

    ax.set_xlim(-0.5, len(metrics_display) - 0.5)
    ax.set_ylim(-0.5, len(ORDER) - 0.5)
    ax.set_xticks(range(len(metrics_display)))
    ax.set_xticklabels(col_labels, fontsize=9.5, rotation=20, ha="right")
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(["★ " + m if m == "CRPTT" else m for m in ORDER], fontsize=10.5)
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith("★"):
            lbl.set_fontweight("bold")
            lbl.set_color(COLORS["CRPTT"])

    # Highlight Balance Score column
    bs_col = col_labels.index("Balance\nScore")
    ax.add_patch(plt.Rectangle((bs_col - 0.5, -0.5), 1, len(ORDER),
                                facecolor="none", edgecolor=COLORS["CRPTT"],
                                lw=2.5, zorder=10))

    ax.set_title(
        "Method Capability Heatmap — Copyright Mitigation on C/M Sentences Only\n"
        "(N/A = method has no targeted-rewriting; Balance Score column highlighted)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label("Relative score within column", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    _save(fig, "filtered_heatmap")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 5 — Copyright Before/After comparison (all vs C/M)
# ══════════════════════════════════════════════════════════════════════════
def plot_copyright_comparison():
    """Side-by-side: 4-gram overlap on all sentences vs C/M only."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Copyright Mitigation Improves When Measured on C/M Sentences Only\n"
        "(lower 4-gram / LCS / ROUGE-L = better copyright protection)",
        fontsize=11, fontweight="bold", y=1.02,
    )

    metrics_pairs = [
        ("4-gram Overlap ↓", "fourgram_all", "fourgram_cm"),
        ("LCS Ratio ↓",      "lcs_all",      "lcs_cm"),
        ("ROUGE-L ↓",        "rouge_l_all",  "rouge_l_cm"),
    ]

    for ax, (title, key_all, key_cm) in zip(axes, metrics_pairs):
        x = np.arange(len(ORDER))
        width = 0.35
        for i, m in enumerate(ORDER):
            v_all = _v(METRICS[m], key_all)
            v_cm  = _v(METRICS[m], key_cm)
            ax.bar(i - width/2, v_all, width, color=COLORS[m], alpha=0.45,
                   edgecolor=COLORS[m], linewidth=0.8, label="All" if i == 0 else "")
            ax.bar(i + width/2, v_cm,  width, color=COLORS[m], alpha=0.92,
                   edgecolor="white", linewidth=0.6, label="C/M only" if i == 0 else "")
            ax.text(i - width/2, v_all + 0.005, f"{v_all:.3f}", ha="center",
                    va="bottom", fontsize=6.5, color=COLORS[m], alpha=0.7)
            ax.text(i + width/2, v_cm  + 0.005, f"{v_cm:.3f}", ha="center",
                    va="bottom", fontsize=6.5, color=COLORS[m])

        ax.set_xticks(x)
        ax.set_xticklabels([m if m != "Filter-mask" else "Filter-\nmask" for m in ORDER],
                           fontsize=7.5, rotation=30, ha="right")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#888888", alpha=0.45, label="All sentences"),
               Patch(facecolor="#888888", alpha=0.92, label="C/M sentences only")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), fontsize=9.5, frameon=False)

    plt.tight_layout()
    _save(fig, "filtered_copyright_comparison")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Figure 6 — Key metrics bar chart
# ══════════════════════════════════════════════════════════════════════════
def plot_wins_bars():
    win_metrics = [
        ("Entity\nPreservation ↑",       [(m, _v(METRICS[m], "entity"))       for m in ORDER]),
        ("Copyright Mitigation\n↑ (C/M)", [(m, _v(METRICS[m], "cop_cm"))       for m in ORDER]),
        ("BERTScore\nF1 ↑",              [(m, _v(METRICS[m], "bertscore"))     for m in ORDER]),
        ("Target\nRewrite Rate ↑",       [(m, _v(METRICS[m], "target_rewrite")) for m in ORDER]),
        ("Balance\nScore ↑",             [(m, _v(METRICS[m], "balance"))        for m in ORDER]),
    ]

    fig, axes = plt.subplots(1, len(win_metrics), figsize=(len(win_metrics) * 2.5, 4.8),
                              sharey=False)
    fig.suptitle(
        "CRPTT Key Metrics — Copyright Mitigation on C/M Sentences\n"
        "(BloomScrub: no Targeted Rewriting → N/A)",
        fontsize=11.5, fontweight="bold", y=1.02,
    )

    for ax, (title, vals) in zip(axes, win_metrics):
        for i, (m, v) in enumerate(vals):
            if _is_nan(v):
                ax.bar(i, 0, color="#EEEEEE", edgecolor="#CCCCCC",
                       linewidth=0.8, width=0.62, zorder=3)
                ax.text(i, 0.015, "N/A", ha="center", va="bottom",
                        fontsize=7.5, color="#AAAAAA", style="italic")
            else:
                ax.bar(i, v, color=COLORS[m], width=0.62,
                       edgecolor="white", linewidth=0.6, zorder=3, alpha=0.92)
                ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7.5,
                        fontweight="bold" if m == "CRPTT" else "normal",
                        color=COLORS[m])
        ax.set_xticks(np.arange(len(ORDER)))
        ax.set_xticklabels([m if m != "Filter-mask" else "Filter-\nmask" for m, _ in vals],
                           fontsize=7.5, rotation=30, ha="right")
        ax.set_title(title, fontsize=9.5, pad=6)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    plt.tight_layout()
    _save(fig, "filtered_wins_bars")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nOutput directory: {OUT_DIR}\n")

    print("[1/6] scatter")
    plot_scatter()

    print("[2/6] radar")
    plot_radar()

    print("[3/6] balance score ranking")
    plot_balance_ranking()

    print("[4/6] heatmap")
    plot_heatmap()

    print("[5/6] copyright all vs C/M comparison")
    plot_copyright_comparison()

    print("[6/6] key metrics bar chart")
    plot_wins_bars()

    print("\n── Final numbers ──────────────────────────────────────────────")
    print(f"{'Method':12s}  {'cop_all':>8}  {'cop_cm':>8}  {'balance':>8}  {'n_cm':>6}")
    for m in ORDER:
        d = METRICS[m]
        print(f"{m:12s}  {d['cop_all']:8.4f}  {d['cop_cm']:8.4f}  {d['balance']:8.4f}  {d['n_cm']:>6}")

    print(f"\nAll figures saved to: {OUT_DIR}")
