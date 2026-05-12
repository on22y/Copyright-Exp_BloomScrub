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
DEFAULT_OUTPUT_PATH = str(_REPO_ROOT / "results" / "baseline_results.json")
GPT_TEMPERATURE = 0


def load_base_module(path: str):
    spec = importlib.util.spec_from_file_location("risk_scored_rewrite_experiment_v2", path)
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


def build_baseline_prompt(mode: str, sentence: str) -> Tuple[str, str]:
    if mode == "paraphrase":
        system_prompt = (
            "You are a multilingual news rewriting assistant. Rewrite the input sentence in the same language as the input. "
            "Preserve factual meaning, names, numbers, dates, organizations, and events. "
            "Do not add facts. Do not output explanations."
        )
        user_prompt = f"""
Rewrite the following news sentence by paraphrasing it naturally.

Sentence:
{sentence}

Rules:
1. Output exactly one sentence in the same language as the input.
2. Preserve all factual information.
3. Preserve names, dates, numbers, organizations, locations, and event details.
4. Avoid copying the original wording when possible.
5. Do not add or remove facts.
""".strip()
        return system_prompt, user_prompt

    if mode == "summarize":
        system_prompt = (
            "You are a multilingual news compression assistant. Rewrite the input as a shorter news sentence "
            "in the same language as the input. "
            "Preserve the central factual content and avoid adding facts. Do not output explanations."
        )
        user_prompt = f"""
Summarize the following news sentence into one concise news sentence.

Sentence:
{sentence}

Rules:
1. Output exactly one sentence in the same language as the input.
2. Keep the central factual content.
3. Preserve names, dates, numbers, organizations, locations, and event details when they are central to the sentence.
4. Remove non-essential wording and details if needed.
5. Do not add facts.
""".strip()
        return system_prompt, user_prompt

    if mode == "filter_mask":
        system_prompt = (
            "You are a multilingual news safety-filtering assistant. Produce one news sentence in the same language as the input by preserving factual "
            "information while removing, generalizing, or masking expressive, evaluative, or stylistically distinctive wording. "
            "Do not output explanations."
        )
        user_prompt = f"""
Filter the following news sentence.

Sentence:
{sentence}

Rules:
1. Output exactly one sentence in the same language as the input.
2. Preserve names, dates, numbers, organizations, and core facts.
3. Remove or generalize evaluative, promotional, rhetorical, or stylistically distinctive expressions.
4. If an expression is risky but not factual, replace it with neutral wording or a short language-appropriate deletion marker.
5. Do not add facts.
""".strip()
        return system_prompt, user_prompt

    raise ValueError(f"Unsupported baseline mode: {mode}")


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


def rewrite_baseline_sentence(
    module,
    client: OpenAI,
    model: str,
    mode: str,
    sentence: str,
    usage_tracker,
) -> str:
    system_prompt, user_prompt = build_baseline_prompt(mode, sentence)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=250,
        temperature=GPT_TEMPERATURE,
    )
    usage_tracker.record(mode, model, module.usage_to_dict(getattr(response, "usage", None)))
    return clean_rewritten_output(response.output_text)


def load_crptt_sentence_lookup(path: Optional[str]) -> Dict[Tuple[int, int], Dict[str, object]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup: Dict[Tuple[int, int], Dict[str, object]] = {}
    for article in data.get("articles", []):
        dataset_index = int(article.get("dataset_index", article.get("article_index", -1)))
        for sent in article.get("sentence_metrics", []):
            sentence_index = int(sent.get("sentence_index", -1))
            scoring = sent.get("scoring")
            if isinstance(scoring, dict):
                lookup[(dataset_index, sentence_index)] = scoring
    return lookup


def fallback_scoring_for_eval(module, original: str) -> Dict[str, object]:
    return {
        "factuality_score": math.nan,
        "expressive_risk_score": math.nan,
        "rewrite_freedom_score": math.nan,
        "rewrite_targets": [],
        "fact_locks": module.extract_fallback_fact_locks(original),
        "baseline_eval_scoring_fallback": True,
    }


def process_article_baseline(
    module,
    client: OpenAI,
    model: str,
    mode: str,
    article_record: Dict[str, object],
    scoring_lookup: Dict[Tuple[int, int], Dict[str, object]],
    usage_tracker,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    dataset_index = int(article_record["dataset_index"])
    clean_sentence_index = 0
    for raw_sentence in module.split_sentences(str(article_record["text"])):
        sentence = module.clean_sentence_for_processing(raw_sentence)
        if module.is_noisy_sentence(sentence):
            continue
        rewritten = rewrite_baseline_sentence(module, client, model, mode, sentence, usage_tracker)
        scoring = scoring_lookup.get((dataset_index, clean_sentence_index))
        if not isinstance(scoring, dict):
            scoring = fallback_scoring_for_eval(module, sentence)
        rows.append(
            {
                "original": sentence,
                "rewritten": rewritten,
                "xlm": {},
                "scoring": scoring,
                "baseline_mode": mode,
            }
        )
        clean_sentence_index += 1
    return rows


def summarize_with_usage(module, article_results: List[Dict[str, object]], mode: str, sample_source: str, sample_size: int, usage_tracker) -> Dict[str, object]:
    summary = module.summarize_results(article_results)
    summary["framework_mode"] = f"baseline_{mode}"
    summary["baseline_mode"] = mode
    summary["sample_source"] = sample_source
    summary["sample_size_requested"] = sample_size
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script-path", default=DEFAULT_BASE_SCRIPT_PATH)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-source", choices=["start", "end"], default="start")
    parser.add_argument("--baseline-mode", choices=["paraphrase", "summarize", "filter_mask"], required=True)
    parser.add_argument("--crptt-results-path", default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--bertscore-model", default=None)
    args = parser.parse_args()

    module = load_base_module(args.base_script_path)
    article_records = module.load_article_records(args.dataset_path, args.sample_size, args.sample_source)
    if not article_records:
        raise ValueError("평가할 기사를 불러오지 못했습니다.")

    client = load_openai_client(args.openai_api_key)
    usage_tracker = module.UsageTracker()
    scoring_lookup = load_crptt_sentence_lookup(args.crptt_results_path)
    eval_resources = module.load_eval_resources(
        bertscore_model_name=args.bertscore_model or module.DEFAULT_BERTSCORE_MODEL
    )

    article_results: List[Dict[str, object]] = []
    for article_idx, article_record in enumerate(article_records):
        sentence_rows = process_article_baseline(
            module,
            client,
            args.openai_model,
            args.baseline_mode,
            article_record,
            scoring_lookup,
            usage_tracker,
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
        article_result["framework_mode"] = f"baseline_{args.baseline_mode}"
        article_result["baseline_mode"] = args.baseline_mode
        article_results.append(article_result)

        summary = summarize_with_usage(
            module,
            article_results,
            args.baseline_mode,
            args.sample_source,
            args.sample_size,
            usage_tracker,
        )
        module.save_results(args.output_path, summary, article_results)

        print(
            f"[{args.baseline_mode} Article {article_idx}] "
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
            f"openai_cost=${usage_tracker.aggregate()['cost_usd']:.4f}"
        )

    summary = summarize_with_usage(
        module,
        article_results,
        args.baseline_mode,
        args.sample_source,
        args.sample_size,
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
