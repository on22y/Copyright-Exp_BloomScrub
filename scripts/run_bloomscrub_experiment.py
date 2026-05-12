"""
BloomScrub-adapted experiment script.

Adapts BloomScrub's iterative detect-and-rewrite strategy (EMNLP 2025) to the
sentence-level copyright-risk evaluation framework used in CRPTT.

Core adaptation from BloomScrub:
- Detection: n-gram overlap between rewritten text and original sentence
  (replaces BloomScrub's QUIP Bloom-filter service, which is unavailable here)
- Rewriting: GPT-4o with BloomScrub-style prompts
  - Round 0: general paraphrase (same as BloomScrub's first-round strategy)
  - Round 1+: targeted removal of the longest detected overlapping segment
- Stopping: when longest overlapping segment <= overlap_threshold tokens,
  or max_rewrites is reached (analogous to BloomScrub's --dynamic_rewrite + -r)
- Evaluation: reuses the full CRPTT evaluation pipeline for fair comparison
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_SCRIPT_PATH = str(_REPO_ROOT / "scripts" / "crptt_policy_guided_rewrite_experiment.py")
DEFAULT_DATASET_PATH = str(_REPO_ROOT / "data" / "final_dataset_kor.json")
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OUTPUT_PATH = str(_REPO_ROOT / "results" / "bloomscrub_results.json")
DEFAULT_MAX_REWRITES = 3       # analogous to BloomScrub's -r (rewrite_times)
DEFAULT_OVERLAP_THRESHOLD = 3  # stop when longest matching segment <= N tokens
DEFAULT_MIN_NGRAM = 4          # minimum n-gram size to consider a "detected overlap"
GPT_TEMPERATURE = 0


# ---------------------------------------------------------------------------
# Module loader (reuses CRPTT evaluation infrastructure)
# ---------------------------------------------------------------------------

def load_base_module(path: str):
    spec = importlib.util.spec_from_file_location("crptt_policy_guided_rewrite_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_openai_client(api_key: Optional[str] = None) -> OpenAI:
    load_dotenv()
    import os
    key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY가 필요합니다. .env 또는 환경변수에 설정하세요.")
    return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# N-gram overlap detection  (replaces BloomScrub's QUIP API)
# ---------------------------------------------------------------------------

def find_overlapping_segments(
    original: str,
    candidate: str,
    min_ngram: int = DEFAULT_MIN_NGRAM,
) -> List[str]:
    """
    Find verbatim word n-gram segments shared between original and candidate.
    Returns segments sorted longest-first (analogous to QUIP's 'quoted_segments').

    BloomScrub uses a Redis-backed Bloom-filter service (QUIP) to find exact
    verbatim matches against a large copyrighted corpus.  Here we compare
    directly against the original source sentence, which is the relevant
    reference for copyright overlap in this sentence-level framework.
    """
    orig_tokens = original.lower().split()
    cand_tokens = candidate.lower().split()
    cand_original = candidate.split()  # preserve case for readable output

    segments: List[str] = []
    seen: set = set()

    for n in range(len(cand_tokens), min_ngram - 1, -1):
        for i in range(len(cand_tokens) - n + 1):
            ngram = tuple(cand_tokens[i : i + n])
            if ngram in seen:
                continue
            for j in range(len(orig_tokens) - n + 1):
                if tuple(orig_tokens[j : j + n]) == ngram:
                    segments.append(" ".join(cand_original[i : i + n]))
                    seen.add(ngram)
                    break

    return sorted(segments, key=len, reverse=True)


# ---------------------------------------------------------------------------
# BloomScrub-style prompt construction
# ---------------------------------------------------------------------------

def build_bloomscrub_prompt(
    sentence: str,
    detected_segments: List[str],
    round_num: int,
) -> Tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) for client.responses.create().

    Mirrors BloomScrub's two-phase strategy (lib/decoding_intervention.py):
      Round 0  -> general paraphrase instruction (first-round strategy)
      Round 1+ -> targeted removal of the longest detected verbatim segment
    """
    if round_num == 0:
        system_prompt = (
            "You are a multilingual news rewriting assistant. "
            "Paraphrase the provided sentence to reduce verbatim overlap with the source "
            "while preserving all factual information (names, dates, numbers, locations, events). "
            "Output only the paraphrased sentence with no additional text or explanation."
        )
        user_prompt = (
            f"{sentence}\n\n"
            "Paraphrase the provided text. Your response should be a paraphrase of the original "
            "sentence. Your response should contain no text other than the paraphrase."
        )
        return system_prompt, user_prompt

    # Round 1+: show the longest verbatim segment (BloomScrub's targeted strategy)
    longest = detected_segments[0] if detected_segments else sentence
    system_prompt = (
        "You will be given a sentence which contains a verbatim overlapping segment with a "
        "copyrighted source. You will also be given the overlapping segment. "
        "Your goal is to rewrite the sentence to remove or rephrase this verbatim overlap "
        "while preserving all factual meaning. "
        "Your response should contain only the rewritten sentence and nothing else."
    )
    user_prompt = (
        f"Sentence: {sentence}\n"
        f'Verbatim overlapping segment: "{longest}"\n\n'
        "Please rewrite the sentence to reduce verbatim overlap with the segment above."
    )
    return system_prompt, user_prompt


def clean_rewritten_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    return text.strip().strip('"').strip("'")


# ---------------------------------------------------------------------------
# Iterative rewriting loop  (core BloomScrub logic)
# ---------------------------------------------------------------------------

def rewrite_bloomscrub_sentence(
    module,
    client: OpenAI,
    model: str,
    sentence: str,
    usage_tracker,
    max_rewrites: int = DEFAULT_MAX_REWRITES,
    overlap_threshold: int = DEFAULT_OVERLAP_THRESHOLD,
    min_ngram: int = DEFAULT_MIN_NGRAM,
) -> Tuple[str, int]:
    """
    Iteratively rewrites a sentence using BloomScrub's detect-and-rewrite loop.

    Loop:
      1. Detect overlapping n-gram segments between original and current output
      2. If longest segment <= threshold, stop (analogous to --dynamic_rewrite)
      3. Otherwise, call GPT-4o with BloomScrub-style prompt
      4. Repeat up to max_rewrites times (analogous to -r / --rewrite_times)

    Returns (rewritten_sentence, num_rewrites_performed).
    """
    current = sentence
    num_rewrites = 0

    for round_num in range(max_rewrites):
        segments = find_overlapping_segments(sentence, current, min_ngram=min_ngram)
        longest_len = len(segments[0].split()) if segments else 0

        if longest_len <= overlap_threshold:
            break

        system_prompt, user_prompt = build_bloomscrub_prompt(current, segments, round_num)
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=250,
            temperature=GPT_TEMPERATURE,
        )
        usage_tracker.record(
            "bloomscrub",
            model,
            module.usage_to_dict(getattr(response, "usage", None)),
        )
        rewritten = clean_rewritten_output(response.output_text)

        if not rewritten or rewritten == current:
            break

        current = rewritten
        num_rewrites += 1

    return current, num_rewrites


# ---------------------------------------------------------------------------
# Article-level processing  (mirrors run_baseline_rewrite_experiment.py)
# ---------------------------------------------------------------------------

def load_crptt_sentence_lookup(path: Optional[str]) -> Dict[Tuple[int, int], Dict]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup: Dict[Tuple[int, int], Dict] = {}
    for article in data.get("articles", []):
        dataset_index = int(article.get("dataset_index", article.get("article_index", -1)))
        for sent in article.get("sentence_metrics", []):
            sentence_index = int(sent.get("sentence_index", -1))
            scoring = sent.get("scoring")
            if isinstance(scoring, dict):
                lookup[(dataset_index, sentence_index)] = scoring
    return lookup


def fallback_scoring_for_eval(module, original: str) -> Dict:
    return {
        "factuality_score": math.nan,
        "expressive_risk_score": math.nan,
        "rewrite_freedom_score": math.nan,
        "rewrite_targets": [],
        "fact_locks": module.extract_fallback_fact_locks(original),
        "baseline_eval_scoring_fallback": True,
    }


def process_article_bloomscrub(
    module,
    client: OpenAI,
    model: str,
    article_record: Dict,
    scoring_lookup: Dict[Tuple[int, int], Dict],
    usage_tracker,
    max_rewrites: int,
    overlap_threshold: int,
    min_ngram: int,
) -> List[Dict]:
    rows: List[Dict] = []
    dataset_index = int(article_record["dataset_index"])
    clean_sentence_index = 0

    for raw_sentence in module.split_sentences(str(article_record["text"])):
        sentence = module.clean_sentence_for_processing(raw_sentence)
        if module.is_noisy_sentence(sentence):
            continue

        rewritten, num_rewrites = rewrite_bloomscrub_sentence(
            module,
            client,
            model,
            sentence,
            usage_tracker,
            max_rewrites=max_rewrites,
            overlap_threshold=overlap_threshold,
            min_ngram=min_ngram,
        )

        scoring = scoring_lookup.get((dataset_index, clean_sentence_index))
        if not isinstance(scoring, dict):
            scoring = fallback_scoring_for_eval(module, sentence)

        rows.append(
            {
                "original": sentence,
                "rewritten": rewritten,
                "xlm": {},
                "scoring": scoring,
                "baseline_mode": "bloomscrub",
                "bloomscrub_num_rewrites": num_rewrites,
            }
        )
        clean_sentence_index += 1

    return rows


def summarize_with_usage(
    module,
    article_results: List[Dict],
    sample_source: str,
    sample_size: int,
    max_rewrites: int,
    overlap_threshold: int,
    usage_tracker,
) -> Dict:
    summary = module.summarize_results(article_results)
    summary["framework_mode"] = "bloomscrub"
    summary["baseline_mode"] = "bloomscrub"
    summary["sample_source"] = sample_source
    summary["sample_size_requested"] = sample_size
    summary["bloomscrub_max_rewrites"] = max_rewrites
    summary["bloomscrub_overlap_threshold"] = overlap_threshold
    summary["gpt_temperature"] = GPT_TEMPERATURE
    usage_snapshot = usage_tracker.snapshot()
    total_usage = usage_snapshot["total"]
    summary["openai_api_calls"] = total_usage["api_calls"]
    summary["openai_input_tokens"] = total_usage["input_tokens"]
    summary["openai_cached_input_tokens"] = total_usage["cached_input_tokens"]
    summary["openai_output_tokens"] = total_usage["output_tokens"]
    summary["openai_total_tokens"] = total_usage["total_tokens"]
    summary["openai_cost_usd"] = total_usage["cost_usd"]
    summary["_openai_usage"] = usage_snapshot
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BloomScrub-adapted iterative rewrite experiment"
    )
    parser.add_argument("--base-script-path", default=DEFAULT_BASE_SCRIPT_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-source", choices=["start", "end"], default="start")
    parser.add_argument(
        "--max-rewrites",
        type=int,
        default=DEFAULT_MAX_REWRITES,
        help="Max iterative rewrite rounds per sentence (BloomScrub -r)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=int,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help="Stop rewriting when longest overlap <= N tokens (BloomScrub --dynamic_rewrite)",
    )
    parser.add_argument(
        "--min-ngram",
        type=int,
        default=DEFAULT_MIN_NGRAM,
        help="Minimum n-gram length to count as detected overlap",
    )
    parser.add_argument("--crptt-results-path", default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--bertscore-model", default=None)
    args = parser.parse_args()

    module = load_base_module(args.base_script_path)
    article_records = module.load_article_records(
        args.dataset_path, args.sample_size, args.sample_source
    )
    if not article_records:
        raise ValueError("평가할 기사를 불러오지 못했습니다.")

    client = load_openai_client(args.openai_api_key)
    usage_tracker = module.UsageTracker()
    scoring_lookup = load_crptt_sentence_lookup(args.crptt_results_path)
    eval_resources = module.load_eval_resources(
        bertscore_model_name=args.bertscore_model or module.DEFAULT_BERTSCORE_MODEL
    )

    article_results: List[Dict] = []
    for article_idx, article_record in enumerate(article_records):
        sentence_rows = process_article_bloomscrub(
            module,
            client,
            args.openai_model,
            article_record,
            scoring_lookup,
            usage_tracker,
            max_rewrites=args.max_rewrites,
            overlap_threshold=args.overlap_threshold,
            min_ngram=args.min_ngram,
        )
        if not sentence_rows:
            print(f"[Article {article_idx}] skipped: no clean sentences after filtering")
            continue

        rewritten_article = module.reconstruct_article(sentence_rows)
        article_result = module.evaluate_article(
            str(article_record["text"]),
            rewritten_article,
            sentence_rows,
            eval_resources,
        )
        article_result["article_index"] = article_idx
        article_result["sample_index"] = article_idx
        article_result["dataset_index"] = int(article_record["dataset_index"])
        article_result["article_hash"] = str(article_record["article_hash"])
        article_result["sample_source"] = args.sample_source
        article_result["framework_mode"] = "bloomscrub"
        article_result["baseline_mode"] = "bloomscrub"
        article_result["avg_bloomscrub_rewrites"] = (
            sum(r.get("bloomscrub_num_rewrites", 0) for r in sentence_rows) / len(sentence_rows)
            if sentence_rows else 0.0
        )
        article_results.append(article_result)

        summary = summarize_with_usage(
            module,
            article_results,
            args.sample_source,
            args.sample_size,
            args.max_rewrites,
            args.overlap_threshold,
            usage_tracker,
        )
        module.save_results(args.output_path, summary, article_results)

        print(
            f"[BloomScrub Article {article_idx}] "
            f"dataset_index={article_record['dataset_index']} | "
            f"entity={article_result['avg_entity_preservation']:.4f} | "
            f"nli={article_result['avg_nli_entailment']:.4f} | "
            f"contradiction_rate={article_result['contradiction_rate']:.4f} | "
            f"rougeL={article_result['avg_rouge_l_f1']:.4f} | "
            f"lcs_ratio={article_result['avg_sentence_lcs_ratio']:.4f} | "
            f"4gram_overlap={article_result['avg_fourgram_overlap']:.4f} | "
            f"target_retention={article_result['avg_target_retention_rate']:.4f} | "
            f"target_rewrite={article_result['avg_target_rewrite_rate']:.4f} | "
            f"target_ngram={article_result['avg_target_ngram_overlap']:.4f} | "
            f"avg_rewrites={article_result['avg_bloomscrub_rewrites']:.2f} | "
            f"openai_cost=${usage_tracker.aggregate()['cost_usd']:.4f}"
        )

    summary = summarize_with_usage(
        module,
        article_results,
        args.sample_source,
        args.sample_size,
        args.max_rewrites,
        args.overlap_threshold,
        usage_tracker,
    )
    module.save_results(args.output_path, summary, article_results)

    print("\n===== Summary =====")
    for key, value in summary.items():
        if key == "_openai_usage":
            continue
        print(f"{key}: {value}")
    print(f"\nSaved results to: {args.output_path}")


if __name__ == "__main__":
    main()
