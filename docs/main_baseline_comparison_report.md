# Main Baseline Comparison

## Raw Metrics
| Method | Entity ↑ | NLI ↑ | Contradiction ↓ | 4-gram ↓ | LCS ↓ | ROUGE-L ↓ | Target Retention ↓ | Target Rewrite ↑ | Target n-gram ↓ | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CRPTT | 0.8624 | 0.9907 | 0.0030 | 0.1624 | 0.5261 | 0.5383 | 0.1023 | 0.8977 | 0.3683 | 7.6147 |
| Paraphrase | 0.8631 | 0.9911 | 0.0039 | 0.1816 | 0.5773 | 0.5812 | 0.1665 | 0.8335 | 0.4468 | 0.9894 |
| Summarize | 0.7739 | 0.9926 | 0.0040 | 0.1171 | 0.4532 | 0.5319 | 0.2656 | 0.7344 | 0.4333 | 0.8273 |
| Filter-mask | 0.9143 | 0.9959 | 0.0009 | 0.4512 | 0.7139 | 0.7711 | 0.4213 | 0.5787 | 0.5755 | 0.9552 |

## Higher-is-Better Axis Scores
| Method | Fact Preservation ↑ | Copyright Mitigation ↑ | Target Rewriting ↑ | Balance Score ↑ |
|---|---:|---:|---:|---:|
| CRPTT | 0.8624 | 0.5911 | 0.8977 | 0.7837 |
| Paraphrase | 0.8631 | 0.5533 | 0.8335 | 0.7499 |
| Summarize | 0.7739 | 0.6326 | 0.7344 | 0.7136 |
| Filter-mask | 0.9143 | 0.3546 | 0.5787 | 0.6159 |

## Score Definitions
- Fact Preservation = Entity Preservation.
- Copyright Risk Mitigation = mean(1 - 4-gram overlap, 1 - LCS ratio, 1 - ROUGE-L).
- Target Rewriting = Target Rewrite Rate.
- Balance Score = mean(Fact Preservation, Copyright Risk Mitigation, Target Rewriting).

## Figure Files
- `figure_0_main_baseline_comparison_paper.png`: paper-ready main figure showing the key trade-off and supporting metrics.
- `figure_1_fact_vs_overlap_tradeoff.png`: trade-off between fact preservation and expression overlap.
- `figure_2_three_axis_balance_scores.png`: three-axis comparison with all scores converted to higher-is-better.
- `figure_3_raw_key_metrics.png`: raw metrics grouped by evaluation axis.
- `figure_4_higher_is_better_heatmap.png`: compact higher-is-better matrix.
