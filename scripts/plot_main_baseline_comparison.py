import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


_ALL_METHODS = ["CRPTT", "BloomScrub", "Paraphrase", "Summarize", "Filter-mask"]
METHOD_COLORS = {
    "CRPTT": "#2563EB",
    "BloomScrub": "#8B5CF6",
    "Paraphrase": "#F59E0B",
    "Summarize": "#10B981",
    "Filter-mask": "#EF4444",
}
METHOD_MARKERS = {
    "CRPTT": "o",
    "BloomScrub": "D",
    "Paraphrase": "o",
    "Summarize": "o",
    "Filter-mask": "o",
}
METHOD_LABELS = {
    "CRPTT": "CRPTT\nbalanced",
    "BloomScrub": "BloomScrub\niterative",
    "Paraphrase": "Paraphrase\nmore copying",
    "Summarize": "Summarize\nfact loss",
    "Filter-mask": "Filter-mask\nhigh overlap",
}

RAW_METRICS = {
    "entity": "mean_entity_preservation",
    "entity_relaxed": "mean_entity_preservation_relaxed",
    "nli": "mean_nli_entailment",
    "bertscore": "mean_bertscore_f1",
    "cosine": "mean_cosine_similarity",
    "contradiction": "mean_contradiction_rate",
    "fourgram": "mean_fourgram_overlap",
    "lcs": "mean_sentence_lcs_ratio",
    "rouge_l": "mean_rouge_l_f1",
    "target_retention": "mean_target_retention_rate",
    "target_rewrite": "mean_target_rewrite_rate",
    "target_ngram": "mean_target_ngram_overlap",
    "cost": "openai_cost_usd",
}


def read_json(path: str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_summary(data: Dict[str, object]) -> Dict[str, object]:
    if isinstance(data.get("summary"), dict):
        return data["summary"]
    return {k: v for k, v in data.items() if k not in {"articles", "openai_usage"}}


def as_float(value, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_method_summaries(args) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Returns (summaries, method_order) where method_order excludes missing optional methods."""
    paths = {
        "CRPTT": args.crptt_json,
        "BloomScrub": getattr(args, "bloomscrub_json", None),
        "Paraphrase": args.paraphrase_json,
        "Summarize": args.summarize_json,
        "Filter-mask": args.filter_mask_json,
    }
    summaries: Dict[str, Dict[str, float]] = {}
    method_order: List[str] = []
    for method in _ALL_METHODS:
        path = paths[method]
        if not path:
            continue
        data = read_json(path)
        summary = extract_summary(data)
        summaries[method] = {
            metric_name: as_float(summary.get(json_key))
            for metric_name, json_key in RAW_METRICS.items()
        }
        summaries[method]["num_articles"] = as_float(summary.get("num_articles", len(data.get("articles", []))))
        summaries[method]["num_sentences"] = as_float(summary.get("num_sentences", sum(len(a.get("sentence_metrics", [])) for a in data.get("articles", []))))
        method_order.append(method)
    return summaries, method_order


def compute_paper_scores(summaries: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for method, s in summaries.items():
        fact_preservation = s["entity"]
        copyright_mitigation = np.nanmean([
            1.0 - s["fourgram"],
            1.0 - s["lcs"],
            1.0 - s["rouge_l"],
        ])
        residual_expression_overlap = np.nanmean([
            s["fourgram"],
            s["lcs"],
            s["rouge_l"],
        ])
        target_rewriting = s["target_rewrite"]
        residual_target_reduction = np.nanmean([
            1.0 - s["target_retention"],
            1.0 - s["target_ngram"],
        ])
        rewrite_quality = np.nanmean([target_rewriting, residual_target_reduction])
        fact_preserved_mitigation = fact_preservation * copyright_mitigation
        fact_preserving_rewrite = fact_preservation * target_rewriting
        copyright_mitigating_rewrite = copyright_mitigation * target_rewriting
        fact_preserved_mitigating_rewrite = fact_preservation * copyright_mitigation * target_rewriting
        balance_score = np.nanmean([fact_preservation, copyright_mitigation, target_rewriting])
        scores[method] = {
            "Fact Preservation": fact_preservation,
            "Copyright Risk Mitigation": copyright_mitigation,
            "Residual Expression Overlap": residual_expression_overlap,
            "Fact-Preserved Mitigation": fact_preserved_mitigation,
            "Target Rewriting": target_rewriting,
            "Residual Target Reduction": residual_target_reduction,
            "Rewrite Quality": rewrite_quality,
            "Fact-preserving Rewrite": fact_preserving_rewrite,
            "Copyright-mitigating Rewrite": copyright_mitigating_rewrite,
            "Fact-preserved Mitigating Rewrite": fact_preserved_mitigating_rewrite,
            "Balance Score": balance_score,
            "Expression Divergence (1 - LCS)": 1.0 - s["lcs"],
            "Expression Divergence (1 - 4gram)": 1.0 - s["fourgram"],
            "Expression Divergence (1 - ROUGE-L)": 1.0 - s["rouge_l"],
            "Target Residual Reduction (1 - Target n-gram)": 1.0 - s["target_ngram"],
            "Target Residual Reduction (1 - Target Retention)": 1.0 - s["target_retention"],
        }
    return scores


def annotate_bars(ax, bars, fmt="{:.3f}", y_pad=0.01, fontsize=9):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + span * y_pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def style_axis(ax):
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def method_edge(method: str) -> str:
    return "#111827" if method == "CRPTT" else "white"


def target_rewrite_bubble_size(target_rewrite_rate: float) -> float:
    clipped = max(0.0, min(1.0, target_rewrite_rate))
    return 180 + 980 * clipped


def add_target_rewrite_size_legend(ax, anchor=(0.03, 0.05)) -> None:
    legend_rates = [0.60, 0.75, 0.90]
    handles = [
        ax.scatter(
            [],
            [],
            s=target_rewrite_bubble_size(rate),
            color="#CBD5E1",
            edgecolor="#475569",
            linewidth=0.8,
            alpha=0.75,
            label=f"TR={rate:.2f}",
        )
        for rate in legend_rates
    ]
    legend = ax.legend(
        handles=handles,
        title="Bubble size:\nTarget Rewrite ↑",
        loc="lower left",
        bbox_to_anchor=anchor,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=8,
        title_fontsize=8,
        labelspacing=1.0,
        borderpad=0.7,
    )
    ax.add_artist(legend)


def plot_main_paper_figure(
    summaries: Dict[str, Dict[str, float]],
    scores: Dict[str, Dict[str, float]],
    out_dir: Path,
    dpi: int,
    method_order: List[str] = None,
) -> None:
    if method_order is None:
        method_order = _ALL_METHODS
    fig = plt.figure(figsize=(18.4, 5.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.08])
    ax_fact = fig.add_subplot(gs[0, 0])
    ax_mitig = fig.add_subplot(gs[0, 1])
    ax_advantage = fig.add_subplot(gs[0, 2])

    fig.suptitle(
        "Main Results: Targeted Rewriting Preserves Facts and Reduces Copyright-Relevant Overlap",
        fontsize=17,
        y=1.04,
        weight="bold",
    )

    def draw_rewrite_map(
        ax,
        y_metric: str,
        title: str,
        y_label: str,
        desired_label: str,
        label_offsets: Dict[str, Tuple[int, int]],
        desired_xytext: Tuple[float, float] = (0.47, 0.84),
        x_pad_right: float = 0.045,
    ) -> None:
        for method in method_order:
            x = scores[method]["Rewrite Quality"]
            y = scores[method][y_metric]
            marker_size = 290
            ax.scatter(
                x,
                y,
                s=marker_size,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                edgecolor=method_edge(method),
                linewidth=1.8 if method == "CRPTT" else 1.15,
                alpha=0.96,
                zorder=4 if method == "CRPTT" else 3,
            )

        x_values = [scores[m]["Rewrite Quality"] for m in method_order]
        y_values = [scores[m][y_metric] for m in method_order]
        ax.set_xlim(max(0.50, min(x_values) - 0.055), min(0.96, max(x_values) + x_pad_right))
        ax.set_ylim(max(0.25, min(y_values) - 0.055), min(1.0, max(y_values) + 0.045))
        ax.axvspan(scores["CRPTT"]["Rewrite Quality"], ax.get_xlim()[1], color="#DBEAFE", alpha=0.25, zorder=0)
        ax.axhspan(scores["CRPTT"][y_metric], ax.get_ylim()[1], color="#DCFCE7", alpha=0.22, zorder=0)
        ax.annotate(
            desired_label,
            xy=(ax.get_xlim()[1] - 0.006, ax.get_ylim()[1] - 0.004),
            xytext=desired_xytext,
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.25, color="#111827"),
            fontsize=9.2,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.30", fc="#F8FAFC", ec="#CBD5E1", alpha=0.95),
        )
        ax.set_title(title, fontsize=13.5, weight="bold", loc="left")
        ax.set_xlabel("Rewrite Quality ↑\nmean(Target Rewrite, 1 - Target Retention, 1 - Target n-gram)", fontsize=10.2)
        ax.set_ylabel(y_label, fontsize=10.5)
        style_axis(ax)

    draw_rewrite_map(
        ax_fact,
        "Fact Preservation",
        "A) Does Rewriting Preserve Key Facts?",
        "Fact Preservation ↑\nEntity Preservation",
        "Desired:\nhigh rewrite quality\nwithout fact loss",
        {
            "CRPTT": (14, 10),
            "Paraphrase": (-102, -34),
            "Summarize": (12, -18),
            "Filter-mask": (-118, -12),
        },
        desired_xytext=(0.48, 0.84),
    )

    draw_rewrite_map(
        ax_mitig,
        "Fact-Preserved Mitigation",
        "B) Does Rewriting Reduce Overlap and Keep Facts?",
        "Fact-preserved Mitigation ↑\nEntity × Low Source Overlap",
        "Desired:\nhigh rewrite quality\nwith lower source overlap",
        {
            "CRPTT": (18, 2),
            "Paraphrase": (36, -38),
            "Summarize": (-96, -28),
            "Filter-mask": (12, -12),
        },
        desired_xytext=(0.06, 0.84),
        x_pad_right=0.070,
    )

    # Panel C: Metrics that couple target rewriting with fact preservation and source-overlap reduction.
    advantage_metrics = [
        ("Target Rewriting", "Target\nRewrite"),
        ("Rewrite Quality", "Rewrite\nQuality"),
        ("Fact-preserving Rewrite", "Fact-preserving\nRewrite"),
        ("Copyright-mitigating Rewrite", "Overlap-mitigating\nRewrite"),
        ("Fact-preserved Mitigating Rewrite", "Fact-preserved\nMitigating Rewrite"),
    ]
    x = np.arange(len(advantage_metrics))
    bar_w = 0.85 / len(method_order)
    offsets = np.linspace(-(len(method_order) - 1) / 2, (len(method_order) - 1) / 2, len(method_order)) * bar_w
    for method_idx, method in enumerate(method_order):
        values = [scores[method][metric_key] for metric_key, _ in advantage_metrics]
        bars = ax_advantage.bar(
            x + offsets[method_idx],
            values,
            width=bar_w,
            color=METHOD_COLORS[method],
            alpha=0.95,
            label=method,
            edgecolor="#111827" if method == "CRPTT" else "white",
            linewidth=0.9 if method == "CRPTT" else 0.35,
        )
        if method == "CRPTT":
            for bar, value in zip(bars, values):
                ax_advantage.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.014,
                    f"{value:.2f}",
                    va="bottom",
                    ha="center",
                    fontsize=8.8,
                    weight="bold",
                    color="#111827",
                    rotation=0,
                )

    ax_advantage.set_title("C) Which Method Rewrites Risky Expression Best?", fontsize=13.5, weight="bold", loc="left")
    ax_advantage.set_ylabel("Score ↑", fontsize=10.5)
    ax_advantage.set_xticks(x)
    ax_advantage.set_xticklabels([label for _, label in advantage_metrics], fontsize=8.8, rotation=18, ha="right")
    ax_advantage.set_ylim(0.0, 0.98)
    ax_advantage.grid(axis="y", linestyle="--", alpha=0.25)
    ax_advantage.spines["top"].set_visible(False)
    ax_advantage.spines["right"].set_visible(False)
    ax_advantage.legend(loc="upper center", bbox_to_anchor=(0.52, 1.16), ncol=len(method_order), frameon=False, fontsize=8.8)

    fig.text(
        0.5,
        -0.04,
        "Reading guide: right/up is better in Panels A–B.  X-axis = Rewrite Quality: mean(Target Rewrite Rate, 1−Target Retention, 1−Target n-gram).  "
        "Panel A Y-axis = Entity Preservation (NER-based fact recall).  Panel B Y-axis = Entity × Copyright Risk Mitigation (balanced fact + overlap reduction).  "
        "CRPTT explicitly separates factual content from expressive targets before rewriting, achieving the best balance across all three evaluation axes.",
        ha="center",
        fontsize=10.0,
        color="#111827",
    )

    fig.savefig(out_dir / "figure_0_main_baseline_comparison_paper.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figure_0_main_baseline_comparison_paper.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summaries: Dict[str, Dict[str, float]], out_dir: Path, dpi: int, method_order: List[str] = None) -> None:
    if method_order is None:
        method_order = _ALL_METHODS
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    fig.suptitle("Fact Preservation vs. Residual Expression Overlap", fontsize=17, y=1.04)

    panels = [
        ("Sentence 4-gram Overlap ↓", "fourgram"),
        ("Sentence LCS Ratio ↓", "lcs"),
    ]
    for ax, (xlabel, metric_key) in zip(axes, panels):
        for method in method_order:
            s = summaries[method]
            x = s[metric_key]
            y = s["entity"]
            ax.scatter(
                x,
                y,
                s=360 if method == "CRPTT" else 230,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                edgecolor=method_edge(method),
                linewidth=1.8,
                alpha=0.92,
                label=method,
                zorder=3,
            )
            ax.annotate(
                method,
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                weight="bold" if method == "CRPTT" else "normal",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#E5E7EB", alpha=0.9),
            )
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Entity Preservation ↑", fontsize=12)
        ax.set_xlim(left=max(0.0, min(s[metric_key] for s in summaries.values()) - 0.05))
        ax.set_ylim(bottom=max(0.65, min(s["entity"] for s in summaries.values()) - 0.05), top=0.96)
        style_axis(ax)
        ax.annotate(
            "Better region",
            xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
            xytext=(0.18, 0.91),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"),
            fontsize=10,
            color="#333333",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(method_order), frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.savefig(out_dir / "figure_1_fact_vs_overlap_tradeoff.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figure_1_fact_vs_overlap_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_three_axis_scores(scores: Dict[str, Dict[str, float]], out_dir: Path, dpi: int, method_order: List[str] = None) -> None:
    if method_order is None:
        method_order = _ALL_METHODS
    metrics = ["Fact Preservation", "Copyright Risk Mitigation", "Target Rewriting", "Balance Score"]
    x = np.arange(len(metrics))
    width = 0.75 / len(method_order)

    fig, ax = plt.subplots(figsize=(13.5, 5.6), constrained_layout=True)
    offsets = np.linspace(-(len(method_order) - 1) / 2, (len(method_order) - 1) / 2, len(method_order)) * width
    for i, method in enumerate(method_order):
        values = [scores[method][metric] for metric in metrics]
        bars = ax.bar(
            x + offsets[i],
            values,
            width,
            label=method,
            color=METHOD_COLORS[method],
            alpha=0.9,
            edgecolor="#222222",
            linewidth=0.35,
        )
        annotate_bars(ax, bars, fmt="{:.2f}", y_pad=0.008, fontsize=8)

    ax.set_title("Main Comparison on Three Evaluation Axes (Higher is Better)", fontsize=17, pad=18)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(method_order), frameon=False)
    style_axis(ax)
    fig.savefig(out_dir / "figure_2_three_axis_balance_scores.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figure_2_three_axis_balance_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_raw_key_metrics(summaries: Dict[str, Dict[str, float]], out_dir: Path, dpi: int, method_order: List[str] = None) -> None:
    if method_order is None:
        method_order = _ALL_METHODS
    panels: List[Tuple[str, List[Tuple[str, str, str]]]] = [
        ("Fact Preservation / Loss", [
            ("Entity ↑", "entity", "up"),
            ("NLI ↑", "nli", "up"),
            ("BERTScore F1 ↑", "bertscore", "up"),
            ("Cosine Sim ↑", "cosine", "up"),
            ("Contradiction ↓", "contradiction", "down"),
        ]),
        ("Copyright Risk Mitigation", [
            ("4-gram ↓", "fourgram", "down"),
            ("LCS ↓", "lcs", "down"),
            ("ROUGE-L ↓", "rouge_l", "down"),
        ]),
        ("Target-level Expression Rewriting", [
            ("Retention ↓", "target_retention", "down"),
            ("Rewrite ↑", "target_rewrite", "up"),
            ("Target n-gram ↓", "target_ngram", "down"),
        ]),
    ]

    fig = plt.figure(figsize=(24, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.65, 1.0, 1.0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    fig.suptitle("Raw Evaluation Metrics by Method", fontsize=17, y=1.05)

    for ax, (title, metrics) in zip(axes, panels):
        x = np.arange(len(metrics))
        width = 0.75 / len(method_order)
        offsets = np.linspace(-(len(method_order) - 1) / 2, (len(method_order) - 1) / 2, len(method_order)) * width
        for i, method in enumerate(method_order):
            values = [summaries[method][key] for _, key, _ in metrics]
            bars = ax.bar(
                x + offsets[i],
                values,
                width,
                color=METHOD_COLORS[method],
                alpha=0.9,
                label=method,
                edgecolor="#222222",
                linewidth=0.3,
            )
            annotate_bars(ax, bars, fmt="{:.2f}", y_pad=0.01, fontsize=7)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in metrics], fontsize=10)
        ax.set_ylim(0, min(1.05, max(max(summaries[m][key] for m in method_order) for _, key, _ in metrics) + 0.15))
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(method_order), frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(out_dir / "figure_3_raw_key_metrics.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figure_3_raw_key_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_score_heatmap(scores: Dict[str, Dict[str, float]], out_dir: Path, dpi: int, method_order: List[str] = None) -> None:
    if method_order is None:
        method_order = _ALL_METHODS
    metrics = [
        "Fact Preservation",
        "Copyright Risk Mitigation",
        "Fact-Preserved Mitigation",
        "Expression Divergence (1 - 4gram)",
        "Expression Divergence (1 - LCS)",
        "Target Rewriting",
        "Balance Score",
    ]
    matrix = np.array([[scores[method][metric] for metric in metrics] for method in method_order])

    fig_h = max(4.8, 0.7 * len(method_order) + 2.5)
    fig_w = 2.0 * len(metrics) + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Higher-is-Better Metric Matrix", fontsize=16, pad=16)
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=11)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right", fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color="#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Score ↑", rotation=90)
    fig.savefig(out_dir / "figure_4_higher_is_better_heatmap.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "figure_4_higher_is_better_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def write_tables(
    summaries: Dict[str, Dict[str, float]],
    scores: Dict[str, Dict[str, float]],
    out_dir: Path,
    method_order: List[str] = None,
) -> None:
    if method_order is None:
        method_order = _ALL_METHODS
    raw_header = [
        "Method",
        "Entity ↑",
        "Entity (rlx) ↑",
        "NLI ↑",
        "BERTScore F1 ↑",
        "Cosine Sim ↑",
        "Contradiction ↓",
        "4-gram ↓",
        "LCS ↓",
        "ROUGE-L ↓",
        "Target Retention ↓",
        "Target Rewrite ↑",
        "Target n-gram ↓",
        "Cost",
    ]
    raw_rows = []
    for method in method_order:
        s = summaries[method]
        raw_rows.append([
            method,
            s["entity"],
            s["entity_relaxed"],
            s["nli"],
            s["bertscore"],
            s["cosine"],
            s["contradiction"],
            s["fourgram"],
            s["lcs"],
            s["rouge_l"],
            s["target_retention"],
            s["target_rewrite"],
            s["target_ngram"],
            s["cost"],
        ])

    score_header = ["Method", "Fact Preservation ↑", "Copyright Mitigation ↑", "Target Rewriting ↑", "Balance Score ↑"]
    score_rows = [
        [
            method,
            scores[method]["Fact Preservation"],
            scores[method]["Copyright Risk Mitigation"],
            scores[method]["Target Rewriting"],
            scores[method]["Balance Score"],
        ]
        for method in method_order
    ]

    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return ""
            return f"{v:.4f}"
        return str(v)

    with (out_dir / "main_baseline_raw_metrics.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(raw_header) + "\n")
        for row in raw_rows:
            f.write(",".join(fmt(v) for v in row) + "\n")

    with (out_dir / "main_baseline_balance_scores.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(score_header) + "\n")
        for row in score_rows:
            f.write(",".join(fmt(v) for v in row) + "\n")

    with (out_dir / "main_baseline_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"raw_metrics": summaries, "higher_is_better_scores": scores}, f, ensure_ascii=False, indent=2)

    def markdown_table(header, rows):
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
        for row in rows:
            lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
        return "\n".join(lines)

    report = f"""# Main Baseline Comparison

## Raw Metrics
{markdown_table(raw_header, raw_rows)}

## Higher-is-Better Axis Scores
{markdown_table(score_header, score_rows)}

## Score Definitions
- Fact Preservation = Entity Preservation (NER-based strict recall).
- Copyright Risk Mitigation = mean(1 - 4-gram overlap, 1 - LCS ratio, 1 - ROUGE-L).
- Fact-Preserved Mitigation = Fact Preservation × Copyright Risk Mitigation.
- Target Rewriting = Target Rewrite Rate.
- Balance Score = mean(Fact Preservation, Copyright Risk Mitigation, Target Rewriting).
- Additional fact preservation metrics reported in raw table: Entity (relaxed), NLI Entailment, BERTScore F1, Cosine Similarity.

## Figure Files
- `figure_0_main_baseline_comparison_paper.png`: paper-ready main figure showing the key trade-off and supporting metrics.
- `figure_1_fact_vs_overlap_tradeoff.png`: trade-off between fact preservation and expression overlap.
- `figure_2_three_axis_balance_scores.png`: three-axis comparison with all scores converted to higher-is-better.
- `figure_3_raw_key_metrics.png`: raw metrics grouped by evaluation axis.
- `figure_4_higher_is_better_heatmap.png`: compact higher-is-better matrix.
"""
    (out_dir / "main_baseline_comparison_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crptt-json", required=True)
    parser.add_argument("--bloomscrub-json", default=None, help="Optional BloomScrub results JSON")
    parser.add_argument("--paraphrase-json", required=True)
    parser.add_argument("--summarize-json", required=True)
    parser.add_argument("--filter-mask-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, method_order = load_method_summaries(args)
    scores = compute_paper_scores(summaries)

    plot_main_paper_figure(summaries, scores, out_dir, args.dpi, method_order=method_order)
    plot_tradeoff(summaries, out_dir, args.dpi, method_order=method_order)
    plot_three_axis_scores(scores, out_dir, args.dpi, method_order=method_order)
    plot_raw_key_metrics(summaries, out_dir, args.dpi, method_order=method_order)
    plot_score_heatmap(scores, out_dir, args.dpi, method_order=method_order)
    write_tables(summaries, scores, out_dir, method_order=method_order)

    print(f"Saved figures and tables to: {out_dir}")
    for name in [
        "figure_0_main_baseline_comparison_paper.png",
        "figure_1_fact_vs_overlap_tradeoff.png",
        "figure_2_three_axis_balance_scores.png",
        "figure_3_raw_key_metrics.png",
        "figure_4_higher_is_better_heatmap.png",
        "main_baseline_comparison_report.md",
    ]:
        print(f"- {out_dir / name}")


if __name__ == "__main__":
    main()
