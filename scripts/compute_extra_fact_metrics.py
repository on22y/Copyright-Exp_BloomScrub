"""
compute_extra_fact_metrics.py
-----------------------------
Post-processing script: computes two additional fact-preservation metrics
on already-completed experiment result JSONs (no API calls needed).

Metrics:
  - Numeric/Date Match ↑  : ratio of numbers/dates in original that are
                             preserved in the rewritten sentence
  - Quote Preservation ↑  : ratio of direct-quote content (인용구) in original
                             that is preserved in the rewritten sentence

Usage:
  python3 scripts/compute_extra_fact_metrics.py

Outputs:
  results/extra_fact_metrics_summary.json   (machine-readable)
  results/extra_fact_metrics_report.md      (human-readable)
"""

import json
import re
import os
import math
from collections import defaultdict

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCES = {
    "CRPTT": os.path.join(
        "/home/user20250805/Guard_Exp1/experiments/04_full_pipeline",
        "main_crptt_sample100_v2/main_crptt_sample100_v2_results.json",
    ),
    "BloomScrub": os.path.join(BASE, "results/bloomscrub_results.json"),
}
OUT_JSON = os.path.join(BASE, "results/extra_fact_metrics_summary.json")
OUT_MD   = os.path.join(BASE, "results/extra_fact_metrics_report.md")


# ══════════════════════════════════════════════════════════════════════════
# 1.  Numeric / Date extraction
# ══════════════════════════════════════════════════════════════════════════
_NUM_PATTERNS = [
    # full Korean date
    r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일",
    r"\d{4}년\s*\d{1,2}월",
    r"\d{1,2}월\s*\d{1,2}일",
    r"\d{4}년",
    r"\d{1,2}월",
    r"\d{1,2}일(?!\w)",          # "일" not followed by word char (avoid 일반 etc.)
    # monetary amounts
    r"\d[\d,]*억\s*\d*만?\s*원?",
    r"\d[\d,]*만\s*원?",
    r"\d[\d,]*원(?!\w)",
    # percentage / ratio
    r"\d+\.?\d*\s*%",
    r"\d+\.?\d*\s*퍼센트",
    # counts with units
    r"\d+\s*명",
    r"\d+\s*개",
    r"\d+\s*건",
    r"\d+\s*곳",
    r"\d+\s*억",
    r"\d+\s*만",
    # plain integers / decimals (catch-all, lower priority)
    r"\d[\d,]*\.?\d*",
]

def extract_numbers(text: str) -> list:
    """Return list of normalised numeric/date tokens found in text."""
    found = {}
    for pat in _NUM_PATTERNS:
        for m in re.finditer(pat, text):
            tok = re.sub(r"\s+", "", m.group()).strip()
            if tok and tok not in found:
                found[m.start()] = tok
    # sort by position, deduplicate overlapping spans
    result, prev_end = [], -1
    for start in sorted(found):
        tok = found[start]
        end = start + len(tok)
        if start >= prev_end:
            result.append(tok)
            prev_end = end
    return result


def numeric_match_rate(original: str, rewritten: str) -> float:
    """
    Fraction of numeric/date tokens in `original` that appear in `rewritten`.
    Returns nan if original has no numeric tokens.
    """
    orig_nums = extract_numbers(original)
    if not orig_nums:
        return float("nan")
    matched = sum(1 for n in orig_nums if n in rewritten)
    return matched / len(orig_nums)


# ══════════════════════════════════════════════════════════════════════════
# 2.  Quote extraction & preservation
# ══════════════════════════════════════════════════════════════════════════
_QUOTE_PATTERNS = [
    r'"([^"]{4,})"',        # ASCII "…"
    r'"([^"]{4,})"',        # curly "…"
    r"'([^']{4,})'",        # curly '…'
    r"'([^']{4,})'",        # ASCII '…'  (single, 4+ chars)
    r"「([^」]{4,})」",      # 「…」
    r"『([^』]{4,})』",      # 『…』
    r"〔([^〕]{4,})〕",      # 〔…〕
]

def extract_quotes(text: str) -> list:
    """Return deduplicated list of quoted strings (content only, ≥ 6 chars, ≥ 2 words)."""
    seen, quotes = set(), []
    for pat in _QUOTE_PATTERNS:
        for m in re.finditer(pat, text):
            q = m.group(1).strip()
            # require ≥ 6 chars AND ≥ 2 space-separated tokens to skip
            # fragments like '이며,' captured at quote boundaries
            if len(q) >= 6 and len(q.split()) >= 2 and q not in seen:
                seen.add(q)
                quotes.append(q)
    return quotes


def _char_bigram_f1(ref: str, hyp: str) -> float:
    """
    Character-bigram F1 — robust to Korean morphological variation
    (e.g. '마케팅' vs '마케팅을').
    """
    def bigrams(s):
        s = re.sub(r"\s+", "", s)   # remove whitespace
        return [s[i:i+2] for i in range(len(s) - 1)]

    ref_bg = bigrams(ref)
    hyp_bg = bigrams(hyp)
    if not ref_bg:
        return 1.0
    if not hyp_bg:
        return 0.0

    ref_cnt = defaultdict(int)
    hyp_cnt = defaultdict(int)
    for bg in ref_bg:
        ref_cnt[bg] += 1
    for bg in hyp_bg:
        hyp_cnt[bg] += 1

    common = sum(min(ref_cnt[bg], hyp_cnt[bg]) for bg in ref_cnt)
    p = common / len(hyp_bg)
    r = common / len(ref_bg)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def quote_preservation_rate(original: str, rewritten: str,
                            threshold: float = 0.5) -> float:
    """
    For each direct quote found in `original`:
      - exact match in rewritten → preserved
      - else character-bigram F1 between quote and rewritten text ≥ threshold → preserved
    Returns nan if original contains no qualifying quotes.
    """
    quotes = extract_quotes(original)
    if not quotes:
        return float("nan")

    preserved = 0
    for q in quotes:
        if q in rewritten:
            preserved += 1
        else:
            f1 = _char_bigram_f1(q, rewritten)
            if f1 >= threshold:
                preserved += 1

    return preserved / len(quotes)


# ══════════════════════════════════════════════════════════════════════════
# 3.  Main computation
# ══════════════════════════════════════════════════════════════════════════
def compute_for_source(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])

    all_numeric, all_quote = [], []
    n_sents_with_nums, n_sents_with_quotes = 0, 0

    for art in articles:
        for sent in art.get("sentence_metrics", []):
            orig = sent.get("original", "")
            rew  = sent.get("rewritten", "")
            if not orig or not rew:
                continue

            nm = numeric_match_rate(orig, rew)
            qp = quote_preservation_rate(orig, rew)

            if not math.isnan(nm):
                all_numeric.append(nm)
                n_sents_with_nums += 1
            if not math.isnan(qp):
                all_quote.append(qp)
                n_sents_with_quotes += 1

    total = sum(len(art.get("sentence_metrics", [])) for art in articles)

    return {
        "total_sentences":          total,
        "sents_with_numbers":       n_sents_with_nums,
        "sents_with_quotes":        n_sents_with_quotes,
        "mean_numeric_match":       sum(all_numeric) / len(all_numeric) if all_numeric else float("nan"),
        "mean_quote_preservation":  sum(all_quote)   / len(all_quote)   if all_quote   else float("nan"),
        "numeric_coverage_pct":     100 * n_sents_with_nums  / total if total else 0,
        "quote_coverage_pct":       100 * n_sents_with_quotes / total if total else 0,
    }


def main():
    results = {}
    for method, path in SOURCES.items():
        print(f"Processing {method} …")
        results[method] = compute_for_source(path)

    # ── save JSON ──────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_JSON}")

    # ── print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"{'Metric':<35} {'CRPTT':>10} {'BloomScrub':>12}")
    print("=" * 62)

    def _fmt(v):
        return f"{v:.4f}" if not math.isnan(v) else "  N/A "

    cr = results.get("CRPTT", {})
    bs = results.get("BloomScrub", {})

    rows = [
        ("Numeric/Date Match ↑",         "mean_numeric_match"),
        ("Quote Preservation ↑",         "mean_quote_preservation"),
        ("  └ sents w/ numbers (%)",     "numeric_coverage_pct"),
        ("  └ sents w/ quotes  (%)",     "quote_coverage_pct"),
        ("Total sentences",              "total_sentences"),
    ]
    for label, key in rows:
        cv = cr.get(key, float("nan"))
        bv = bs.get(key, float("nan"))
        if isinstance(cv, float):
            print(f"{label:<35} {_fmt(cv):>10} {_fmt(bv):>12}")
        else:
            print(f"{label:<35} {cv:>10} {bv:>12}")

    print("=" * 62)

    # ── save markdown report ───────────────────────────────────────────────
    nm_cr = cr.get("mean_numeric_match", float("nan"))
    nm_bs = bs.get("mean_numeric_match", float("nan"))
    qp_cr = cr.get("mean_quote_preservation", float("nan"))
    qp_bs = bs.get("mean_quote_preservation", float("nan"))

    def _winner(a, b, higher_better=True):
        if math.isnan(a) or math.isnan(b):
            return "-"
        if higher_better:
            return "CRPTT 우세" if a > b else ("BloomScrub 우세" if b > a else "동일")
        else:
            return "CRPTT 우세" if a < b else ("BloomScrub 우세" if b < a else "동일")

    md = f"""## 추가 사실 보존 지표 계산 결과

> 기존 실험 결과 JSON을 후처리로 계산 (API 재호출 없음)

### 계산 방법

| 지표 | 계산 방식 |
|---|---|
| **Numeric/Date Match ↑** | 원문의 숫자·날짜·금액 토큰 추출 → 재작성문에 동일 토큰이 몇 개 남았는지 비율 계산 |
| **Quote Preservation ↑** | 원문의 직접 인용구(`"..."`, `'...'`, `「」` 등) 추출 → 재작성문에서 단어 수준 F1 ≥ 0.5이면 보존으로 판정 |

### 결과

| 지표 | CRPTT | BloomScrub | 비교 |
|---|---|---|---|
| **Numeric/Date Match ↑** | **{nm_cr:.3f}** | {nm_bs:.3f} | {_winner(nm_cr, nm_bs)} |
| **Quote Preservation ↑** | **{qp_cr:.3f}** | {qp_bs:.3f} | {_winner(qp_cr, qp_bs)} |

> 커버리지: 숫자 포함 문장 {cr.get('numeric_coverage_pct',0):.1f}% / 인용구 포함 문장 {cr.get('quote_coverage_pct',0):.1f}% (CRPTT 기준, 1,042문장)
"""

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
