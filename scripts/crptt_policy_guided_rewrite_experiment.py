
import argparse
import hashlib
import json
import math
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

try:
    from bert_score import BERTScorer
except ImportError:
    BERTScorer = None


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = str(_REPO_ROOT / "data" / "final_dataset_kor.json")
DEFAULT_XLM_MODEL_PATH = "/home/user20250805/Guard_Exp1/models/fine_tuned_XLM_FCM_KOEN_full_target_only_class_weighted"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_OUTPUT_PATH = str(_REPO_ROOT / "results" / "crptt_results.json")
DEFAULT_NER_MODEL = "chunwoolee0/klue_ner_roberta_model"
DEFAULT_NLI_MODEL = "joeddav/xlm-roberta-large-xnli"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_BERTSCORE_MODEL = "xlm-roberta-large"
ENTITY_LABELS = {"PERSON", "ORG", "LOC", "DATE", "MONEY", "PRODUCT"}
LABEL_MAP = {0: "F", 1: "C", 2: "M"}
GPT_TEMPERATURE = 0
OPENAI_INPUT_PRICE_PER_1M = 2.50
OPENAI_CACHED_INPUT_PRICE_PER_1M = 1.25
OPENAI_OUTPUT_PRICE_PER_1M = 10.00

@dataclass
class XLMResources:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device


@dataclass
class EvalResources:
    ner_pipe: object
    nli_tokenizer: AutoTokenizer
    nli_model: AutoModelForSequenceClassification
    embed_model: SentenceTransformer
    bert_scorer: object
    device: torch.device


def usage_to_dict(usage: object) -> Dict[str, int]:
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    if hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    elif isinstance(usage, dict):
        raw = usage
    else:
        raw = {}
        for key in dir(usage):
            if key.startswith("_"):
                continue
            value = getattr(usage, key)
            if isinstance(value, (int, float, dict)):
                raw[key] = value

    def get_int(*keys: str) -> int:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    input_tokens = get_int("input_tokens", "prompt_tokens")
    output_tokens = get_int("output_tokens", "completion_tokens")
    total_tokens = get_int("total_tokens")
    cached_input_tokens = 0
    details = raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached_input_tokens = int(details.get("cached_tokens") or details.get("cached_input_tokens") or 0)
    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def compute_openai_cost(usage: Dict[str, int]) -> float:
    uncached_cost = usage.get("uncached_input_tokens", 0) / 1_000_000 * OPENAI_INPUT_PRICE_PER_1M
    cached_cost = usage.get("cached_input_tokens", 0) / 1_000_000 * OPENAI_CACHED_INPUT_PRICE_PER_1M
    output_cost = usage.get("output_tokens", 0) / 1_000_000 * OPENAI_OUTPUT_PRICE_PER_1M
    return uncached_cost + cached_cost + output_cost


def zero_usage() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "uncached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


class UsageTracker:
    def __init__(self) -> None:
        self.records: List[Dict[str, object]] = []

    def record(self, stage: str, model: str, usage: Dict[str, int]) -> None:
        self.records.append(
            {
                "stage": stage,
                "model": model,
                "cost_usd": compute_openai_cost(usage),
                **usage,
            }
        )

    def aggregate(self, records: Optional[List[Dict[str, object]]] = None) -> Dict[str, object]:
        selected = self.records if records is None else records
        usage = zero_usage()
        cost = 0.0
        for record in selected:
            for key in usage:
                usage[key] += int(record.get(key, 0))
            cost += float(record.get("cost_usd", 0.0))
        return {**usage, "cost_usd": cost, "api_calls": len(selected)}

    def by_stage(self) -> Dict[str, Dict[str, object]]:
        stages = sorted({str(record.get("stage", "unknown")) for record in self.records})
        return {
            stage: self.aggregate([record for record in self.records if record.get("stage") == stage])
            for stage in stages
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "pricing_per_1m_tokens": {
                "input": OPENAI_INPUT_PRICE_PER_1M,
                "cached_input": OPENAI_CACHED_INPUT_PRICE_PER_1M,
                "output": OPENAI_OUTPUT_PRICE_PER_1M,
            },
            "total": self.aggregate(),
            "by_stage": self.by_stage(),
            "records": self.records,
        }


def normalize_article(text: str) -> str:
    text = re.sub(r"([.!?])(?=[^\s])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_article(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def is_noisy_sentence(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return True

    # Drop obviously truncated fragments such as a dangling number or unfinished title/body mix.
    if re.search(r"\b\d+(?:\.\d+)?\s*$", stripped):
        return True
    if re.search(r"[가-힣A-Za-z0-9]\.$", stripped) and len(simple_tokenize(stripped)) <= 6:
        return True

    title_markers = ("◇", "◆", "■", "▲", "▶")
    if any(marker in stripped for marker in title_markers):
        return True
    if re.search(r"(받아|수상|선정)(?=[가-힣A-Za-z])", stripped) and " " not in stripped:
        return True
    return False


def clean_sentence_for_processing(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    sentence = re.sub(r"([가-힣])\((이름)\)", r"\1 (\2)", sentence)
    sentence = re.sub(r"([가-힣])([A-Z]{2,})", r"\1 \2", sentence)
    return sentence


def compute_article_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_article_records(path: str, limit: int, sample_source: str) -> List[Dict[str, object]]:
    article_records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for dataset_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text", "").strip()
            if not text:
                continue
            article_records.append(
                {
                    "dataset_index": dataset_index,
                    "article_hash": compute_article_hash(text),
                    "text": text,
                }
            )

    if sample_source == "start":
        return article_records[:limit]
    if sample_source == "end":
        return list(reversed(article_records[-limit:]))
    raise ValueError(f"지원하지 않는 sample_source입니다: {sample_source}")


def load_articles(path: str, limit: int) -> List[str]:
    return [str(row["text"]) for row in load_article_records(path, limit, "start")]


def load_xlm_resources(model_path: str) -> XLMResources:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return XLMResources(tokenizer=tokenizer, model=model, device=device)


def predict_xlm_distribution(sentence: str, resources: XLMResources) -> Dict[str, object]:
    inputs = resources.tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(resources.device)

    with torch.no_grad():
        logits = resources.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

    prob_map = {LABEL_MAP[idx]: float(probs[idx]) for idx in LABEL_MAP}
    top_label = max(prob_map, key=prob_map.get)

    return {
        "top_label": top_label,
        "signal_mode": "softmax_only",
        "p_F": prob_map["F"],
        "p_C": prob_map["C"],
        "p_M": prob_map["M"],
    }


def load_openai_client(api_key: Optional[str] = None) -> OpenAI:
    load_dotenv()
    key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY가 필요합니다. .env 또는 환경변수에 설정하세요.")
    return OpenAI(api_key=key)


def extract_json_object(text: str) -> Dict[str, object]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError(f"JSON 응답 파싱 실패: {text[:300]}")


def call_gpt_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    usage_tracker: Optional[UsageTracker] = None,
    usage_stage: str = "risk_scoring",
) -> Dict[str, object]:
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=600,
        temperature=GPT_TEMPERATURE,
    )
    if usage_tracker is not None:
        usage_tracker.record(usage_stage, model, usage_to_dict(getattr(response, "usage", None)))
    return extract_json_object(response.output_text)


def build_scoring_prompts(sentence: str, softmax_info: Dict[str, object]) -> Tuple[str, str]:
    system_prompt = (
        "You are a multilingual copyright-risk analyst for news sentences. "
        "Use the input sentence and the XLM-RoBERTa F/C/M softmax probabilities to separate factual elements from expressive elements inside the sentence. "
        "Use p(F), p(C), and p(M) as soft signals; do not rely on a hard top label. "
        "Facts, ideas, and functional information are less likely to be protectable, while distinctive wording, sentence structure, subjective tone, evaluative language, and rhetorical expression may carry higher copyright-relevant expression risk. "
        "rewrite_freedom_score does not mean rewriting difficulty; it means how much wording, order, and structure may be changed while preserving the factual meaning. "
        "Therefore, sentences with more creative, subjective, or expressive wording may receive a higher rewrite_freedom_score. "
        "Put local expression units that should be rewritten in rewrite_targets, and put factual surface forms that must be preserved in fact_locks. "
        "Do not output a single aggregate risk score; rewrite behavior and policy are computed later by code using the 3-factor band rules. "
        "Return exactly one valid JSON object and nothing else."
    )

    user_prompt = f"""
Analyze the following news sentence. Apply the same criteria regardless of the input language.

Sentence:
{sentence}

XLM-RoBERTa softmax signal:
- p(F) = {softmax_info['p_F']:.4f}
- p(C) = {softmax_info['p_C']:.4f}
- p(M) = {softmax_info['p_M']:.4f}

Scoring criteria:
- factuality_score: 0~5
  - 0 = almost no factual information
  - 1 = very little factual information
  - 2 = some factual information
  - 3 = factual information and expressive elements are mixed
  - 4 = primarily fact-centered
  - 5 = almost entirely objective factual reporting

- expressive_risk_score: 0~5
  - Meaning: the strength of creative, subjective, evaluative, rhetorical, or stylistically distinctive expression that may increase copyright-relevant expression overlap if repeated verbatim
  - 0 = only formulaic, objective factual reporting
  - 1 = weak expressive or evaluative elements
  - 2 = some distinctive or interpretive wording
  - 3 = clear creative wording, evaluation, emphasis, or rhetorical expression
  - 4 = clearly distinctive wording or subjective tone
  - 5 = the sentence strongly depends on creative or evaluative expression

- rewrite_freedom_score: 0~5
  - Meaning: the allowable range for changing wording, order, and sentence structure while preserving factual information
  - 0 = dense factual content such as numbers, dates, and organization names; almost no change should be made
  - 1 = only very limited lexical or function-word edits are allowed
  - 2 = limited wording and word-order changes are allowed
  - 3 = wording and structure may be moderately changed while preserving facts
  - 4 = substantial rewriting is possible because the sentence contains creative or subjective expression
  - 5 = the sentence is mainly creative or subjective, so wording and structure can be strongly changed while preserving only the facts

Important decision rules:
- If a sentence contains many creative or subjective expressions, do not lower rewrite_freedom_score; raise it.
- rewrite_freedom_score means "rewriting allowance", not "difficulty of substitution".
- If a sentence contains many factual details, give it a high factuality_score, but keep expressive_risk_score low if it contains little creative or subjective expression.
- If factual information and creative or subjective expression coexist, treat it as mixed in character: preserve facts, rewrite expression, and assign at least moderate rewrite_freedom_score when appropriate.
- Put in rewrite_targets local expression units that may increase expression overlap if repeated verbatim and that can be rewritten without changing facts.
- rewrite_targets may include not only evaluative, rhetorical, subjective, or distinctive expressions, but also reusable journalistic phrasing, reporting verbs, generalizable noun phrases, and short syntactic patterns.
- Examples include reporting or stylistic expressions such as "held", "said", "aims to contribute", "excellent company", "as a partner company", "is in the service business", or "students hoping for employment", when the wording can be changed without changing the underlying facts.
- Put in fact_locks factual surface forms that should be preserved, such as organization names, person names, dates, numbers, project names, product names, bilingual English expressions, parenthetical expressions, and masking tokens.
- Do not put factual information itself, such as names, dates, numbers, organizations, project names, or product names, into rewrite_targets.
- If a phrase mixes factual information and rewritable wording, put the factual surface form in fact_locks and only the surrounding rewritable wording in rewrite_targets.
- Do not output a single aggregate risk score.
- Do not output rewrite intensity or policy; they are computed later by code.

JSON schema:
{{
  "factuality_score": float,
  "expressive_risk_score": float,
  "rewrite_freedom_score": float,
  "rewrite_targets": ["expression to rewrite"],
  "fact_locks": ["factual surface form to preserve"]
}}
""".strip()

    return system_prompt, user_prompt


def build_safe_scoring_prompts(sentence: str, softmax_info: Dict[str, object]) -> Tuple[str, str]:
    system_prompt = (
        "You are a structured text-analysis component for a benign academic experiment. "
        "Analyze linguistic features of a multilingual news sentence for controlled rewriting. "
        "This is not legal advice and does not ask you to reproduce copyrighted text. "
        "Do not refuse. Do not provide explanations. Return exactly one valid JSON object."
    )

    user_prompt = f"""
Analyze the following news sentence and return only JSON. Apply the same criteria regardless of language.

Sentence:
{sentence}

Classifier signal:
- p(F) = {softmax_info['p_F']:.4f}
- p(C) = {softmax_info['p_C']:.4f}
- p(M) = {softmax_info['p_M']:.4f}

Use these scoring dimensions:
- factuality_score: 0 to 5, higher means the sentence is more fact-centered.
- expressive_risk_score: 0 to 5, higher means the sentence contains more distinctive, evaluative, subjective, or rhetorical expression that should not be repeated verbatim.
- rewrite_freedom_score: 0 to 5, higher means the wording and structure can be rewritten more freely while preserving facts.

Also extract:
- rewrite_targets: local expression units that can be rewritten without changing facts and that may increase expression overlap if repeated verbatim. Include not only evaluative, subjective, creative, or rhetorical expressions, but also reusable journalistic phrasing, reporting verbs, generalizable noun phrases, and short syntactic patterns.
- fact_locks: factual surface forms that should be preserved, such as names, organizations, dates, numbers, project names, products, parenthetical expressions, and masking tokens.

Important:
- Do not put factual information itself, such as names, dates, numbers, organizations, project names, or product names, into rewrite_targets.
- If a phrase mixes factual information and rewritable wording, put the factual surface form in fact_locks and only the surrounding rewritable wording in rewrite_targets.
- Do not output a rationale.
- Do not output markdown.
- Do not output any text outside the JSON object.

JSON schema:
{{
  "factuality_score": float,
  "expressive_risk_score": float,
  "rewrite_freedom_score": float,
  "rewrite_targets": ["expression to rewrite"],
  "fact_locks": ["factual surface form to preserve"]
}}
""".strip()

    return system_prompt, user_prompt


def clamp_score(value: object, min_value: float = 0.0, max_value: float = 5.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = min_value
    return max(min_value, min(max_value, number))


def normalize_string_list(value: object) -> List[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [value] if value.strip() else []
    else:
        items = []

    normalized: List[str] = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def extract_fallback_fact_locks(sentence: str) -> List[str]:
    patterns = [
        r"\d[\d,]*(?:\.\d+)?\s*(?:원|만원|억원|달러|USD|KRW|명|개|년|월|일)",
        r"\d{4}년(?:\s*\d{1,2}월)?(?:\s*\d{1,2}일)?",
        r"\d{1,2}월\s*\d{1,2}일",
        r"\([^)]*\)",
        r"'[^']+'",
        r'"[^"]+"',
        r"[가-힣A-Za-z0-9·&]+(?:대학교|대학|대|사업단|기업|부|청|시|군|구|센터|연구원|협회|위원회|공사|재단)",
    ]
    locks: List[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, sentence):
            text = match.strip()
            if text and text not in locks:
                locks.append(text)
    if "(이름)" in sentence and "(이름)" not in locks:
        locks.append("(이름)")
    return locks


def extract_fallback_rewrite_targets(sentence: str) -> List[str]:
    candidates = [
        "기여할 것으로 기대",
        "활력 회복",
        "글로벌 경쟁력 강화",
        "우량기업",
        "애로사항 해소",
        "유통 혁신",
        "활성화",
        "강화",
        "기여",
        "기대",
    ]
    targets: List[str] = []
    for candidate in candidates:
        if candidate in sentence and candidate not in targets:
            targets.append(candidate)
    return targets


def build_fallback_scoring(sentence: str, softmax_info: Dict[str, object], reason: str) -> Dict[str, object]:
    p_f = float(softmax_info.get("p_F", 0.0))
    p_c = float(softmax_info.get("p_C", 0.0))
    p_m = float(softmax_info.get("p_M", 0.0))

    factuality = (5.0 * p_f) + (3.5 * p_m) + (2.5 * p_c)
    expressive_risk = (0.7 * p_f) + (2.4 * p_m) + (4.0 * p_c)
    rewrite_freedom = (1.0 * p_f) + (2.8 * p_m) + (4.2 * p_c)

    return {
        "factuality_score": clamp_score(factuality),
        "expressive_risk_score": clamp_score(expressive_risk),
        "rewrite_freedom_score": clamp_score(rewrite_freedom),
        "rewrite_targets": extract_fallback_rewrite_targets(sentence),
        "fact_locks": extract_fallback_fact_locks(sentence),
        "scoring_fallback": True,
        "scoring_fallback_reason": reason[:300],
    }


def normalize_scoring_output(raw: Dict[str, object]) -> Dict[str, object]:
    raw_rewrite_freedom = raw.get("rewrite_freedom_score", raw.get("substitutability_score", 0.0))
    raw_expressive_risk = raw.get("expressive_risk_score")
    if raw_expressive_risk is None:
        raw_expressive_risk = max(
            clamp_score(raw.get("creativity_score", 0.0)),
            clamp_score(raw.get("subjectivity_score", 0.0)),
        )
    result = {
        "factuality_score": clamp_score(raw.get("factuality_score", 0.0)),
        "expressive_risk_score": clamp_score(raw_expressive_risk),
        "rewrite_freedom_score": clamp_score(raw_rewrite_freedom),
        "rewrite_targets": normalize_string_list(raw.get("rewrite_targets", [])),
        "fact_locks": normalize_string_list(raw.get("fact_locks", [])),
    }
    return result


FACTOR_THRESHOLDS = {
    # Empirically selected from score-distribution analysis:
    # rounded half-point quantiles at 20/40/60/80%.
    "factuality_score": [3.0, 3.5, 4.0, 4.5],
    "expressive_risk_score": [1.0, 2.0, 3.0, 3.5],
    "rewrite_freedom_score": [3.0, 3.5, 4.0, 4.5],
}


def _sanitize_thresholds(thresholds: List[float], min_gap: float = 0.05) -> List[float]:
    fixed: List[float] = []
    for i, value in enumerate(thresholds):
        current = clamp_score(value)
        if i == 0:
            fixed.append(current)
            continue
        fixed.append(max(current, fixed[-1] + min_gap))
    if fixed and fixed[-1] >= 5.0:
        shift = fixed[-1] - 4.95
        fixed = [clamp_score(v - shift) for v in fixed]
        for i in range(1, len(fixed)):
            fixed[i] = max(fixed[i], fixed[i - 1] + min_gap)
    return [clamp_score(v) for v in fixed]


def score_band(score: float, factor_name: str) -> str:
    if factor_name not in FACTOR_THRESHOLDS:
        raise KeyError(f"Unknown factor for score banding: {factor_name}")
    t1, t2, t3, t4 = _sanitize_thresholds(FACTOR_THRESHOLDS[factor_name])
    s = clamp_score(score)
    if s < t1:
        return "very_low"
    if s < t2:
        return "low"
    if s < t3:
        return "medium"
    if s < t4:
        return "high"
    return "very_high"


def build_band_profile(scoring: Dict[str, object]) -> Dict[str, str]:
    return {
        "factuality_band": score_band(float(scoring["factuality_score"]), "factuality_score"),
        "expressive_risk_band": score_band(float(scoring["expressive_risk_score"]), "expressive_risk_score"),
        "rewrite_freedom_band": score_band(float(scoring["rewrite_freedom_score"]), "rewrite_freedom_score"),
    }


def build_control_policy(scoring: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    band_profile = build_band_profile(scoring)
    factuality_band = band_profile["factuality_band"]
    expressive_risk_band = band_profile["expressive_risk_band"]
    rewrite_freedom_band = band_profile["rewrite_freedom_band"]

    fact_policy = {
        "very_low": "Apply a low factual-lock strength, but do not alter items in fact_locks or explicit factual claims in the source sentence.",
        "low": "Preserve explicit facts while allowing relatively flexible expression-level rewriting.",
        "medium": "The sentence mixes factual and expressive elements; preserve core factual relations, numbers, dates, and named entities.",
        "high": "Apply strong fact preservation. Do not arbitrarily change numbers, dates, organizations, person names, project names, or product names.",
        "very_high": "Apply maximum fact preservation. Keep fact_locks and core factual noun phrases as close to their original surface forms as possible.",
    }[factuality_band]

    expressive_policy = {
        "very_low": "Expression risk is low; perform only minimal wording cleanup and keep an objective news style.",
        "low": "If weak expressive or evaluative elements or reusable journalistic phrasing appear, neutralize them naturally while avoiding excessive structural change.",
        "medium": "Rewrite distinctive wording, evaluative language, subjective tone, rhetorical expression, journalistic formulas, reporting verbs, and generalizable noun phrases around rewrite_targets into a neutral news style.",
        "high": "Do not repeat the source sentence's distinctive wording, evaluative tone, subjective judgment, or rhetorical construction; recast them into a new fact-centered news style.",
        "very_high": "Strongly reconstruct highly protectable expressive elements and the sentence's subjective framing. Preserve the facts, but redesign wording and tone.",
    }[expressive_risk_band]

    structure_policy = {
        "very_low": "Make almost no structural change. Preserve the sentence structure and information order.",
        "low": "Adjust only word order and connective expressions in a limited way.",
        "medium": "Allow partial clause rearrangement and direct/indirect wording changes while preserving facts.",
        "high": "Actively change sentence structure, information order, and modifier structure when useful.",
        "very_high": "Strongly redesign the expression structure while preserving factual information. Compression or reordering is allowed, but factual omission is not.",
    }[rewrite_freedom_band]

    return {
        "fact_preservation_policy": {
            "band": factuality_band,
            "instruction": fact_policy,
        },
        "expressive_risk_rewriting_policy": {
            "band": expressive_risk_band,
            "instruction": expressive_policy,
        },
        "structural_rewrite_policy": {
            "band": rewrite_freedom_band,
            "instruction": structure_policy,
        },
    }


def finalize_scoring(scoring: Dict[str, object]) -> Dict[str, object]:
    scoring = dict(scoring)
    scoring["band_profile"] = build_band_profile(scoring)
    scoring["control_policy"] = build_control_policy(scoring)
    scoring["policy_mapping_basis"] = "3factor_band_profile_to_3policy_control"
    return scoring


def score_sentence(
    client: OpenAI,
    model: str,
    sentence: str,
    softmax_info: Dict[str, object],
    usage_tracker: Optional[UsageTracker] = None,
) -> Dict[str, object]:
    system_prompt, user_prompt = build_scoring_prompts(sentence, softmax_info)
    try:
        raw = call_gpt_json(client, model, system_prompt, user_prompt, usage_tracker, "risk_scoring")
        scoring = finalize_scoring(normalize_scoring_output(raw))
        scoring["scoring_retry"] = False
        return scoring
    except ValueError as first_error:
        safe_system_prompt, safe_user_prompt = build_safe_scoring_prompts(sentence, softmax_info)
        try:
            raw = call_gpt_json(client, model, safe_system_prompt, safe_user_prompt, usage_tracker, "risk_scoring_retry")
            scoring = finalize_scoring(normalize_scoring_output(raw))
            scoring["scoring_retry"] = True
            scoring["scoring_retry_reason"] = str(first_error)[:300]
            return scoring
        except ValueError as second_error:
            fallback = build_fallback_scoring(
                sentence,
                softmax_info,
                f"first_error={first_error}; second_error={second_error}",
            )
            return finalize_scoring(fallback)


def build_rewrite_prompts(
    sentence: str,
    scoring: Dict[str, object],
) -> Tuple[str, str]:
    rewrite_targets = scoring.get("rewrite_targets") or ["none"]
    fact_locks = scoring.get("fact_locks") or ["none"]
    control_policy = scoring.get("control_policy", {})

    system_prompt = (
        "You are a multilingual news rewriting specialist for copyright-risk mitigation. "
        "Rewrite the sentence to preserve its core factual information while reducing direct overlap in wording, narrative style, and sentence structure that may be copyright-relevant. "
        "Output exactly one news-style sentence in the same language as the input sentence. "
        "Do not translate the sentence. Do not output explanations, JSON, labels, or commentary."
    )

    user_prompt = f"""
Source sentence:
{sentence}

LLM scoring result:
- factuality_score = {scoring['factuality_score']:.2f}/5
- expressive_risk_score = {scoring['expressive_risk_score']:.2f}/5
- rewrite_freedom_score = {scoring['rewrite_freedom_score']:.2f}/5
- band_profile = {json.dumps(scoring.get('band_profile', {}), ensure_ascii=False)}
- rewrite_targets = {json.dumps(rewrite_targets, ensure_ascii=False)}
- fact_locks = {json.dumps(fact_locks, ensure_ascii=False)}
- control_policy = {json.dumps(control_policy, ensure_ascii=False, indent=2)}

Score-conditioned rewriting guidance:
- Reflect factuality_score through fact_preservation_policy.
- Reflect expressive_risk_score through expressive_risk_rewriting_policy.
- Reflect rewrite_freedom_score through structural_rewrite_policy.
- Do not use a single rewrite level. Prioritize the three-axis band_profile and control_policy when deciding how to rewrite.
- Do not repeat local expression units in rewrite_targets verbatim when they can be rewritten without changing facts.
- Preserve the surface forms of items in fact_locks as much as possible.

Mandatory rules:
1. Do not change or omit factual information such as organizations, person names, dates, numbers, project names, proper nouns, or masking tokens.
2. Output exactly one sentence for the one input sentence.
3. Preserve the input sentence's language. Do not translate it into another language.
4. Do not add facts that are not present in the source sentence.
5. Do not repeat copyright-relevant wording choices, narrative style, or sentence structure when they can be naturally rewritten in news style.
6. Rewrite fact-centered sentences conservatively, but rewrite high expressive-risk sentences more actively.
7. If expressive_risk_band or rewrite_freedom_band is high or very_high, do not rely only on synonym replacement; also change sentence structure or information order when appropriate.
8. If factuality_band is high or very_high, do not arbitrarily generalize or omit numbers, English equivalents, parenthetical expressions, quoted project names, or program names.

Output only the rewritten sentence on one line.
""".strip()

    return system_prompt, user_prompt


def clean_rewritten_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    return text.strip().strip('"').strip("'")


def rewrite_sentence(
    client: OpenAI,
    model: str,
    sentence: str,
    scoring: Dict[str, object],
    usage_tracker: Optional[UsageTracker] = None,
) -> str:
    system_prompt, user_prompt = build_rewrite_prompts(sentence, scoring)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=300,
        temperature=GPT_TEMPERATURE,
    )
    if usage_tracker is not None:
        usage_tracker.record("controlled_rewriting", model, usage_to_dict(getattr(response, "usage", None)))
    return clean_rewritten_output(response.output_text)


def process_article(
    article_text: str,
    xlm_resources: XLMResources,
    client: OpenAI,
    openai_model: str,
    usage_tracker: Optional[UsageTracker] = None,
) -> List[Dict[str, object]]:
    sentence_rows: List[Dict[str, object]] = []
    for raw_sentence in tqdm(split_sentences(article_text), desc="Rewriting sentences", leave=False):
        sentence = clean_sentence_for_processing(raw_sentence)
        if is_noisy_sentence(sentence):
            continue

        softmax_info = predict_xlm_distribution(sentence, xlm_resources)
        scoring = score_sentence(client, openai_model, sentence, softmax_info, usage_tracker)
        rewritten = rewrite_sentence(client, openai_model, sentence, scoring, usage_tracker)
        sentence_rows.append(
            {
                "original": sentence,
                "rewritten": rewritten,
                "xlm": softmax_info,
                "scoring": scoring,
            }
        )
    return sentence_rows


def reconstruct_article(sentence_rows: List[Dict[str, object]]) -> str:
    return " ".join(row["rewritten"].strip() for row in sentence_rows if row["rewritten"].strip())


def normalize_entity_text(text: str) -> str:
    text = text.replace("##", "")
    text = re.sub(r"\s+", " ", text.strip())
    return text.casefold()


def normalize_ner_label(label: str) -> Optional[str]:
    label = label.upper()
    label = label.replace("B-", "").replace("I-", "")
    label = label.replace("S-", "").replace("E-", "")
    label_map = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "PS": "PERSON",
        "ORG": "ORG",
        "OG": "ORG",
        "ORGANIZATION": "ORG",
        "LOC": "LOC",
        "LC": "LOC",
        "LOCATION": "LOC",
        "DATE": "DATE",
        "DT": "DATE",
        "TI": "DATE",
        "TIME": "DATE",
        "MONEY": "MONEY",
        "QT": "MONEY",
        "PRICE": "MONEY",
        "QUANTITY": "MONEY",
        "PRODUCT": "PRODUCT",
        "PRD": "PRODUCT",
        "ARTIFACT": "PRODUCT",
        "AF": "PRODUCT",
    }
    return label_map.get(label)


def regex_entities(text: str) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    date_patterns = [
        r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일",
        r"\d{4}년",
        r"\d{1,2}월\s*\d{1,2}일",
        r"\d{1,2}일",
    ]
    money_patterns = [
        r"\d[\d,]*(?:\.\d+)?\s*(?:원|만원|억원|달러|USD|KRW)",
        r"시비\s*\d[\d,]*(?:\.\d+)?\s*(?:원|만원|억원)",
    ]
    product_patterns = [
        r"'([^']*(?:사업|서비스|플랫폼|프로그램|과정|제품|모델)[^']*)'",
        r'"([^"]*(?:사업|서비스|플랫폼|프로그램|과정|제품|모델)[^"]*)"',
    ]

    for pattern in date_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            entities.append((normalize_entity_text(match), "DATE"))

    for pattern in money_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            entities.append((normalize_entity_text(match), "MONEY"))

    for pattern in product_patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            entities.append((normalize_entity_text(match), "PRODUCT"))

    return entities


def load_eval_resources(
    ner_model_name: str = DEFAULT_NER_MODEL,
    nli_model_name: str = DEFAULT_NLI_MODEL,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    bertscore_model_name: str = DEFAULT_BERTSCORE_MODEL,
) -> EvalResources:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    ner_pipe = pipeline(
        "token-classification",
        model=ner_model,
        tokenizer=ner_tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
    )

    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
    nli_model.to(device)
    nli_model.eval()

    embed_model = SentenceTransformer(embed_model_name, device=str(device))
    if BERTScorer is None:
        raise ImportError(
            "BERTScore-F1 계산을 위해 `bert-score` 패키지가 필요합니다. "
            "requirements.txt에 `bert-score`를 추가하거나 `pip install bert-score`를 실행하세요."
        )
    bert_scorer = BERTScorer(
        model_type=bertscore_model_name,
        device=str(device),
        rescale_with_baseline=False,
    )

    return EvalResources(
        ner_pipe=ner_pipe,
        nli_tokenizer=nli_tokenizer,
        nli_model=nli_model,
        embed_model=embed_model,
        bert_scorer=bert_scorer,
        device=device,
    )


def extract_entities(text: str, ner_pipe) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    for item in ner_pipe(text):
        raw_label = item.get("entity_group") or item.get("entity", "")
        label = normalize_ner_label(raw_label)
        if label not in ENTITY_LABELS:
            continue
        surface = item.get("word") or item.get("entity", "")
        surface = normalize_entity_text(surface)
        if surface:
            entities.append((surface, label))
    entities.extend(regex_entities(text))
    return entities


def canonicalize_entity_text(text: str) -> str:
    text = normalize_entity_text(text)
    text = re.sub(r"[^0-9a-z가-힣]", "", text)
    return text


def entity_counter(text: str, ner_pipe, relaxed: bool = False) -> Counter:
    entities = extract_entities(text, ner_pipe)
    if relaxed:
        return Counter((canonicalize_entity_text(entity_text), label) for entity_text, label in entities)
    return Counter(entities)


def compute_entity_preservation(original: str, rewritten: str, ner_pipe, relaxed: bool = False) -> float:
    original_counter = entity_counter(original, ner_pipe, relaxed=relaxed)
    rewritten_counter = entity_counter(rewritten, ner_pipe, relaxed=relaxed)
    total = sum(original_counter.values())
    if total == 0:
        return 1.0
    shared = sum(min(count, rewritten_counter.get(entity, 0)) for entity, count in original_counter.items())
    return shared / total


def get_entailment_index(model) -> int:
    label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
    for key, idx in label2id.items():
        if "entail" in key:
            return idx
    id2label = {int(idx): str(label).lower() for idx, label in model.config.id2label.items()}
    for idx, label in id2label.items():
        if "entail" in label:
            return idx
    raise ValueError("entailment index를 찾지 못했습니다.")


def get_contradiction_index(model) -> int:
    label2id = {str(k).lower(): int(v) for k, v in model.config.label2id.items()}
    for key, idx in label2id.items():
        if "contrad" in key:
            return idx
    id2label = {int(idx): str(label).lower() for idx, label in model.config.id2label.items()}
    for idx, label in id2label.items():
        if "contrad" in label:
            return idx
    raise ValueError("contradiction index를 찾지 못했습니다.")


def compute_nli_metrics(original: str, rewritten: str, resources: EvalResources) -> Dict[str, float]:
    inputs = resources.nli_tokenizer(
        original,
        rewritten,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(resources.device)
    with torch.no_grad():
        logits = resources.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    entailment_idx = get_entailment_index(resources.nli_model)
    contradiction_idx = get_contradiction_index(resources.nli_model)
    predicted_idx = int(torch.argmax(probs).item())
    return {
        "nli_entailment": float(probs[entailment_idx].item()),
        "nli_contradiction": float(probs[contradiction_idx].item()),
        "contradiction_label": 1.0 if predicted_idx == contradiction_idx else 0.0,
    }


def compute_nli_score(original: str, rewritten: str, resources: EvalResources) -> float:
    return compute_nli_metrics(original, rewritten, resources)["nli_entailment"]


def simple_tokenize(text: str) -> List[str]:
    text = text.casefold()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[“”‘’·…◇]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def compute_overlap_stats(original: str, rewritten: str) -> Dict[str, float]:
    original_tokens = simple_tokenize(original)
    rewritten_tokens = simple_tokenize(rewritten)
    original_counter = Counter(original_tokens)
    rewritten_counter = Counter(rewritten_tokens)
    overlap_count = sum(
        min(count, rewritten_counter.get(token, 0))
        for token, count in original_counter.items()
    )
    original_token_count = len(original_tokens)
    return {
        "overlap_count": overlap_count,
        "original_token_count": original_token_count,
        "rewritten_token_count": len(rewritten_tokens),
        "overlap_ratio": (overlap_count / original_token_count) if original_token_count else 0.0,
    }


def token_ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_fourgram_overlap_stats(original: str, rewritten: str) -> Dict[str, float]:
    original_fourgrams = token_ngrams(simple_tokenize(original), 4)
    rewritten_fourgrams = token_ngrams(simple_tokenize(rewritten), 4)
    overlap_count = sum(
        min(count, rewritten_fourgrams.get(fourgram, 0))
        for fourgram, count in original_fourgrams.items()
    )
    original_count = sum(original_fourgrams.values())
    rewritten_count = sum(rewritten_fourgrams.values())
    return {
        "fourgram_overlap_count": overlap_count,
        "original_fourgram_count": original_count,
        "rewritten_fourgram_count": rewritten_count,
        "fourgram_overlap": (overlap_count / original_count) if original_count else 0.0,
    }


def compute_bleu(original: str, rewritten: str, max_order: int = 4, smooth: float = 1.0) -> float:
    reference_tokens = simple_tokenize(original)
    candidate_tokens = simple_tokenize(rewritten)
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_order + 1):
        reference_ngrams = token_ngrams(reference_tokens, n)
        candidate_ngrams = token_ngrams(candidate_tokens, n)
        overlap = sum(
            min(count, reference_ngrams.get(ngram, 0))
            for ngram, count in candidate_ngrams.items()
        )
        total = sum(candidate_ngrams.values())
        precisions.append((overlap + smooth) / (total + smooth) if total else 0.0)

    if any(precision <= 0.0 for precision in precisions):
        return 0.0

    geo_mean = math.exp(sum(math.log(precision) for precision in precisions) / max_order)
    ref_len = len(reference_tokens)
    cand_len = len(candidate_tokens)
    brevity_penalty = 1.0 if cand_len > ref_len else math.exp(1.0 - (ref_len / cand_len))
    return float(brevity_penalty * geo_mean)


def compute_rouge1(original: str, rewritten: str) -> Dict[str, float]:
    reference_tokens = simple_tokenize(original)
    candidate_tokens = simple_tokenize(rewritten)
    if not reference_tokens and not candidate_tokens:
        return {"rouge1_precision": 1.0, "rouge1_recall": 1.0, "rouge1_f1": 1.0}
    if not reference_tokens or not candidate_tokens:
        return {"rouge1_precision": 0.0, "rouge1_recall": 0.0, "rouge1_f1": 0.0}

    reference_counter = Counter(reference_tokens)
    candidate_counter = Counter(candidate_tokens)
    overlap = sum(
        min(count, reference_counter.get(token, 0))
        for token, count in candidate_counter.items()
    )
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"rouge1_precision": precision, "rouge1_recall": recall, "rouge1_f1": f1}


def lcs_length(left: List[str], right: List[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def compute_rouge_l(original: str, rewritten: str) -> Dict[str, float]:
    reference_tokens = simple_tokenize(original)
    candidate_tokens = simple_tokenize(rewritten)
    if not reference_tokens and not candidate_tokens:
        return {"rouge_l_precision": 1.0, "rouge_l_recall": 1.0, "rouge_l_f1": 1.0}
    if not reference_tokens or not candidate_tokens:
        return {"rouge_l_precision": 0.0, "rouge_l_recall": 0.0, "rouge_l_f1": 0.0}

    lcs = lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"rouge_l_precision": precision, "rouge_l_recall": recall, "rouge_l_f1": f1}


def compute_sentence_lcs_stats(original: str, rewritten: str) -> Dict[str, float]:
    original_tokens = simple_tokenize(original)
    rewritten_tokens = simple_tokenize(rewritten)
    lcs = lcs_length(original_tokens, rewritten_tokens)
    original_count = len(original_tokens)
    rewritten_count = len(rewritten_tokens)
    return {
        "sentence_lcs_length": lcs,
        "sentence_lcs_ratio": (lcs / original_count) if original_count else 0.0,
        "sentence_lcs_precision": (lcs / rewritten_count) if rewritten_count else 0.0,
    }


def compute_cosine_similarity(original: str, rewritten: str, resources: EvalResources) -> float:
    embeddings = resources.embed_model.encode(
        [original, rewritten],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return float(util.cos_sim(embeddings[0], embeddings[1]).item())


def compute_novelty(original: str, rewritten: str, resources: EvalResources) -> float:
    return 1.0 - compute_cosine_similarity(original, rewritten, resources)


def compute_bertscore_f1(original: str, rewritten: str, resources: EvalResources) -> float:
    _, _, f1 = resources.bert_scorer.score([rewritten], [original])
    return float(f1[0].item())


def normalize_for_chrf(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_ngrams(text: str, n: int) -> Counter:
    if len(text) < n:
        return Counter()
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def compute_chrf_similarity(original: str, rewritten: str, max_order: int = 6, beta: float = 2.0) -> float:
    original = normalize_for_chrf(original)
    rewritten = normalize_for_chrf(rewritten)
    if not original and not rewritten:
        return 1.0
    if not original or not rewritten:
        return 0.0

    f_scores = []
    beta_squared = beta * beta
    for n in range(1, max_order + 1):
        original_ngrams = char_ngrams(original, n)
        rewritten_ngrams = char_ngrams(rewritten, n)
        original_total = sum(original_ngrams.values())
        rewritten_total = sum(rewritten_ngrams.values())
        if original_total == 0 or rewritten_total == 0:
            continue

        overlap = sum(
            min(count, rewritten_ngrams.get(ngram, 0))
            for ngram, count in original_ngrams.items()
        )
        precision = overlap / rewritten_total
        recall = overlap / original_total
        if precision == 0.0 and recall == 0.0:
            f_scores.append(0.0)
            continue

        f_score = (1.0 + beta_squared) * precision * recall
        f_score /= (beta_squared * precision) + recall
        f_scores.append(f_score)

    if not f_scores:
        return 0.0
    return sum(f_scores) / len(f_scores)


def compute_form_distance(original: str, rewritten: str) -> float:
    return 1.0 - compute_chrf_similarity(original, rewritten)


def normalize_for_target_match(text: str) -> str:
    text = text.casefold()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"[“”‘’·…◇]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def character_ngrams_from_texts(texts: List[str], n: int) -> Counter:
    joined = " ".join(normalize_for_target_match(text) for text in texts if str(text).strip())
    joined = re.sub(r"\s+", " ", joined).strip()
    if len(joined) < n:
        return Counter()
    return Counter(joined[i : i + n] for i in range(len(joined) - n + 1))


def compute_target_rewrite_metrics(
    rewrite_targets: List[str],
    rewritten: str,
    n: int = 3,
) -> Dict[str, float]:
    targets = [str(target).strip() for target in rewrite_targets if str(target).strip()]
    if not targets:
        return {
            "target_count": 0,
            "target_retained_count": 0,
            "target_retention_rate": math.nan,
            "target_rewrite_rate": math.nan,
            "target_ngram_overlap": math.nan,
            "target_ngram_overlap_count": 0,
            "target_ngram_count": 0,
        }

    normalized_rewritten = normalize_for_target_match(rewritten)
    retained = 0
    for target in targets:
        normalized_target = normalize_for_target_match(target)
        if normalized_target and normalized_target in normalized_rewritten:
            retained += 1

    target_ngrams = character_ngrams_from_texts(targets, n)
    rewritten_ngrams = character_ngrams_from_texts([rewritten], n)
    overlap_count = sum(
        min(count, rewritten_ngrams.get(ngram, 0))
        for ngram, count in target_ngrams.items()
    )
    target_ngram_count = sum(target_ngrams.values())
    retention_rate = retained / len(targets)

    return {
        "target_count": len(targets),
        "target_retained_count": retained,
        "target_retention_rate": retention_rate,
        "target_rewrite_rate": 1.0 - retention_rate,
        "target_ngram_overlap": (overlap_count / target_ngram_count) if target_ngram_count else math.nan,
        "target_ngram_overlap_count": overlap_count,
        "target_ngram_count": target_ngram_count,
    }


def safe_mean(values: List[float]) -> float:
    clean_values = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(number):
            clean_values.append(number)
    return sum(clean_values) / len(clean_values) if clean_values else math.nan


def evaluate_article(
    article_text: str,
    rewritten_article: str,
    sentence_rows: List[Dict[str, object]],
    eval_resources: EvalResources,
) -> Dict[str, object]:
    sentence_metrics: List[Dict[str, object]] = []
    entity_scores = []
    relaxed_entity_scores = []
    nli_scores = []
    contradiction_label_scores = []
    contradiction_probability_scores = []
    bertscore_f1_scores = []
    cosine_similarity_scores = []
    novelty_scores = []
    form_distance_scores = []
    bleu_scores = []
    rouge1_f1_scores = []
    rouge_l_f1_scores = []
    fourgram_overlap_scores = []
    sentence_lcs_ratio_scores = []
    target_retention_scores = []
    target_rewrite_scores = []
    target_ngram_overlap_scores = []

    for idx, row in enumerate(sentence_rows):
        entity_score = compute_entity_preservation(row["original"], row["rewritten"], eval_resources.ner_pipe)
        relaxed_entity_score = compute_entity_preservation(
            row["original"],
            row["rewritten"],
            eval_resources.ner_pipe,
            relaxed=True,
        )
        nli_metrics = compute_nli_metrics(row["original"], row["rewritten"], eval_resources)
        nli_score = nli_metrics["nli_entailment"]
        bertscore_f1 = compute_bertscore_f1(row["original"], row["rewritten"], eval_resources)
        cosine_similarity = compute_cosine_similarity(row["original"], row["rewritten"], eval_resources)
        novelty_score = 1.0 - cosine_similarity
        form_distance = compute_form_distance(row["original"], row["rewritten"])
        bleu_score = compute_bleu(row["original"], row["rewritten"])
        rouge1_scores = compute_rouge1(row["original"], row["rewritten"])
        rouge_l_scores = compute_rouge_l(row["original"], row["rewritten"])
        sentence_lcs_stats = compute_sentence_lcs_stats(row["original"], row["rewritten"])
        overlap_stats = compute_overlap_stats(row["original"], row["rewritten"])
        fourgram_overlap_stats = compute_fourgram_overlap_stats(row["original"], row["rewritten"])
        target_rewrite_metrics = compute_target_rewrite_metrics(
            normalize_string_list(row.get("scoring", {}).get("rewrite_targets", [])),
            row["rewritten"],
        )

        entity_scores.append(entity_score)
        relaxed_entity_scores.append(relaxed_entity_score)
        nli_scores.append(nli_score)
        contradiction_label_scores.append(nli_metrics["contradiction_label"])
        contradiction_probability_scores.append(nli_metrics["nli_contradiction"])
        bertscore_f1_scores.append(bertscore_f1)
        cosine_similarity_scores.append(cosine_similarity)
        novelty_scores.append(novelty_score)
        form_distance_scores.append(form_distance)
        bleu_scores.append(bleu_score)
        rouge1_f1_scores.append(rouge1_scores["rouge1_f1"])
        rouge_l_f1_scores.append(rouge_l_scores["rouge_l_f1"])
        fourgram_overlap_scores.append(fourgram_overlap_stats["fourgram_overlap"])
        sentence_lcs_ratio_scores.append(sentence_lcs_stats["sentence_lcs_ratio"])
        target_retention_scores.append(target_rewrite_metrics["target_retention_rate"])
        target_rewrite_scores.append(target_rewrite_metrics["target_rewrite_rate"])
        target_ngram_overlap_scores.append(target_rewrite_metrics["target_ngram_overlap"])

        sentence_metrics.append(
            {
                "sentence_index": idx,
                "original": row["original"],
                "rewritten": row["rewritten"],
                "xlm": row["xlm"],
                "scoring": row["scoring"],
                "entity_preservation": entity_score,
                "entity_preservation_relaxed": relaxed_entity_score,
                "nli_score": nli_score,
                **nli_metrics,
                "bertscore_f1": bertscore_f1,
                "cosine_similarity": cosine_similarity,
                "novelty": novelty_score,
                "semantic_distance": novelty_score,
                "form_distance": form_distance,
                "bleu": bleu_score,
                **rouge1_scores,
                **rouge_l_scores,
                **sentence_lcs_stats,
                **overlap_stats,
                **fourgram_overlap_stats,
                **target_rewrite_metrics,
            }
        )

    return {
        "original_article": article_text,
        "rewritten_article": rewritten_article,
        "avg_entity_preservation": sum(entity_scores) / len(entity_scores) if entity_scores else math.nan,
        "avg_entity_preservation_relaxed": (
            sum(relaxed_entity_scores) / len(relaxed_entity_scores) if relaxed_entity_scores else math.nan
        ),
        "avg_nli_score": sum(nli_scores) / len(nli_scores) if nli_scores else math.nan,
        "avg_nli_entailment": sum(nli_scores) / len(nli_scores) if nli_scores else math.nan,
        "avg_nli_contradiction": (
            sum(contradiction_probability_scores) / len(contradiction_probability_scores)
            if contradiction_probability_scores
            else math.nan
        ),
        "contradiction_rate": (
            sum(contradiction_label_scores) / len(contradiction_label_scores)
            if contradiction_label_scores
            else math.nan
        ),
        "avg_bertscore_f1": (
            sum(bertscore_f1_scores) / len(bertscore_f1_scores) if bertscore_f1_scores else math.nan
        ),
        "avg_cosine_similarity": (
            sum(cosine_similarity_scores) / len(cosine_similarity_scores)
            if cosine_similarity_scores
            else math.nan
        ),
        "avg_novelty": sum(novelty_scores) / len(novelty_scores) if novelty_scores else math.nan,
        "avg_semantic_distance": sum(novelty_scores) / len(novelty_scores) if novelty_scores else math.nan,
        "avg_form_distance": (
            sum(form_distance_scores) / len(form_distance_scores) if form_distance_scores else math.nan
        ),
        "avg_bleu": sum(bleu_scores) / len(bleu_scores) if bleu_scores else math.nan,
        "avg_rouge1_f1": sum(rouge1_f1_scores) / len(rouge1_f1_scores) if rouge1_f1_scores else math.nan,
        "avg_rouge_l_f1": sum(rouge_l_f1_scores) / len(rouge_l_f1_scores) if rouge_l_f1_scores else math.nan,
        "avg_sentence_lcs_ratio": (
            sum(sentence_lcs_ratio_scores) / len(sentence_lcs_ratio_scores)
            if sentence_lcs_ratio_scores
            else math.nan
        ),
        "avg_fourgram_overlap": (
            sum(fourgram_overlap_scores) / len(fourgram_overlap_scores) if fourgram_overlap_scores else math.nan
        ),
        "avg_target_retention_rate": safe_mean(target_retention_scores),
        "avg_target_rewrite_rate": safe_mean(target_rewrite_scores),
        "avg_target_ngram_overlap": safe_mean(target_ngram_overlap_scores),
        "sentence_metrics": sentence_metrics,
    }


def summarize_results(article_results: List[Dict[str, object]]) -> Dict[str, float]:
    def avg(key: str) -> float:
        values = []
        for row in article_results:
            value = row.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isnan(number):
                values.append(number)
        return sum(values) / len(values) if values else math.nan

    return {
        "articles_evaluated": len(article_results),
        "mean_entity_preservation": avg("avg_entity_preservation"),
        "mean_entity_preservation_relaxed": avg("avg_entity_preservation_relaxed"),
        "mean_nli_score": avg("avg_nli_score"),
        "mean_nli_entailment": avg("avg_nli_entailment"),
        "mean_nli_contradiction": avg("avg_nli_contradiction"),
        "mean_contradiction_rate": avg("contradiction_rate"),
        "mean_bertscore_f1": avg("avg_bertscore_f1"),
        "mean_cosine_similarity": avg("avg_cosine_similarity"),
        "mean_novelty": avg("avg_novelty"),
        "mean_semantic_distance": avg("avg_semantic_distance"),
        "mean_form_distance": avg("avg_form_distance"),
        "mean_bleu": avg("avg_bleu"),
        "mean_rouge1_f1": avg("avg_rouge1_f1"),
        "mean_rouge_l_f1": avg("avg_rouge_l_f1"),
        "mean_sentence_lcs_ratio": avg("avg_sentence_lcs_ratio"),
        "mean_fourgram_overlap": avg("avg_fourgram_overlap"),
        "mean_target_retention_rate": avg("avg_target_retention_rate"),
        "mean_target_rewrite_rate": avg("avg_target_rewrite_rate"),
        "mean_target_ngram_overlap": avg("avg_target_ngram_overlap"),
    }


def build_evaluation_axes_summary(summary: Dict[str, float]) -> Dict[str, List[Dict[str, object]]]:
    return {
        "fact_preservation": [
            {
                "metric": "Entity Preservation",
                "key": "mean_entity_preservation",
                "direction": "up",
                "value": summary.get("mean_entity_preservation"),
            },
            {
                "metric": "NLI Entailment",
                "key": "mean_nli_entailment",
                "direction": "up",
                "value": summary.get("mean_nli_entailment"),
            },
            {
                "metric": "Contradiction Rate",
                "key": "mean_contradiction_rate",
                "direction": "down",
                "value": summary.get("mean_contradiction_rate"),
            },
        ],
        "copyright_risk_mitigation": [
            {
                "metric": "Sentence 4-gram Overlap",
                "key": "mean_fourgram_overlap",
                "direction": "down",
                "value": summary.get("mean_fourgram_overlap"),
            },
            {
                "metric": "Sentence LCS Ratio",
                "key": "mean_sentence_lcs_ratio",
                "direction": "down",
                "value": summary.get("mean_sentence_lcs_ratio"),
            },
            {
                "metric": "ROUGE-L",
                "key": "mean_rouge_l_f1",
                "direction": "down",
                "value": summary.get("mean_rouge_l_f1"),
            },
        ],
        "expressive_target_rewriting": [
            {
                "metric": "Target Retention Rate",
                "key": "mean_target_retention_rate",
                "direction": "down",
                "value": summary.get("mean_target_retention_rate"),
            },
            {
                "metric": "Target Rewrite Rate",
                "key": "mean_target_rewrite_rate",
                "direction": "up",
                "value": summary.get("mean_target_rewrite_rate"),
            },
            {
                "metric": "Target n-gram Overlap",
                "key": "mean_target_ngram_overlap",
                "direction": "down",
                "value": summary.get("mean_target_ngram_overlap"),
            },
        ],
    }


def save_results(path: str, summary: Dict[str, float], article_results: List[Dict[str, object]]) -> None:
    payload = {
        "summary": summary,
        "evaluation_axes": build_evaluation_axes_summary(summary),
        "articles": article_results,
    }
    if "_openai_usage" in summary:
        payload["openai_usage"] = summary["_openai_usage"]
        summary = dict(summary)
        summary.pop("_openai_usage", None)
        payload["summary"] = summary
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_checkpoint_articles(output_path: str) -> List[Dict[str, object]]:
    path = Path(output_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        return []
    return [article for article in articles if isinstance(article, dict)]


def save_checkpoint(
    output_path: str,
    article_results: List[Dict[str, object]],
    sample_source: str,
    sample_size: int,
    completed: bool,
    usage_tracker: Optional[UsageTracker] = None,
) -> Dict[str, float]:
    article_results.sort(
        key=lambda row: (
            int(row.get("sample_index", row.get("article_index", 0))),
            int(row.get("dataset_index", -1)),
        )
    )
    summary = summarize_results(article_results)
    summary["framework_mode"] = "full"
    summary["xlm_signal_mode"] = "softmax_only"
    summary["factor_thresholds"] = FACTOR_THRESHOLDS
    summary["policy_set"] = [
        "fact_preservation_policy",
        "expressive_risk_rewriting_policy",
        "structural_rewrite_policy",
    ]
    summary["sample_source"] = sample_source
    summary["sample_size_requested"] = sample_size
    summary["gpt_temperature"] = GPT_TEMPERATURE
    summary["checkpoint_completed"] = completed
    if usage_tracker is not None:
        usage_snapshot = usage_tracker.snapshot()
        total_usage = usage_snapshot["total"]
        summary["openai_api_calls"] = total_usage["api_calls"]
        summary["openai_input_tokens"] = total_usage["input_tokens"]
        summary["openai_cached_input_tokens"] = total_usage["cached_input_tokens"]
        summary["openai_output_tokens"] = total_usage["output_tokens"]
        summary["openai_total_tokens"] = total_usage["total_tokens"]
        summary["openai_cost_usd"] = total_usage["cost_usd"]
        summary["_openai_usage"] = usage_snapshot
    save_results(output_path, summary, article_results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--xlm-model-path", default=DEFAULT_XLM_MODEL_PATH)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-source", choices=["start", "end"], default="start")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--openai-api-key", default=None)
    parser.add_argument("--bertscore-model", default=DEFAULT_BERTSCORE_MODEL)
    args = parser.parse_args()

    article_records = load_article_records(args.dataset_path, args.sample_size, args.sample_source)
    if not article_records:
        raise ValueError("평가할 기사를 불러오지 못했습니다.")

    xlm_resources = load_xlm_resources(args.xlm_model_path)
    client = load_openai_client(args.openai_api_key)
    eval_resources = load_eval_resources(bertscore_model_name=args.bertscore_model)
    usage_tracker = UsageTracker()

    article_results = load_checkpoint_articles(args.output_path)
    processed_hashes = {
        str(row.get("article_hash"))
        for row in article_results
        if row.get("article_hash")
    }
    processed_dataset_indices = {
        int(row["dataset_index"])
        for row in article_results
        if row.get("dataset_index") is not None
    }
    if article_results:
        print(f"Resuming from checkpoint: {args.output_path} ({len(article_results)} articles already saved)")

    for article_idx, article_record in enumerate(article_records):
        article_text = str(article_record["text"])
        dataset_index = int(article_record["dataset_index"])
        article_hash = str(article_record["article_hash"])
        if article_hash in processed_hashes or dataset_index in processed_dataset_indices:
            print(f"[Article {article_idx}] skipped: already in checkpoint (dataset_index={dataset_index})")
            continue

        sentence_rows = process_article(article_text, xlm_resources, client, args.openai_model, usage_tracker)
        if not sentence_rows:
            print(f"[Article {article_idx}] skipped: no clean sentences after filtering")
            continue
        rewritten_article = reconstruct_article(sentence_rows)
        article_result = evaluate_article(article_text, rewritten_article, sentence_rows, eval_resources)
        article_result["article_index"] = article_idx
        article_result["sample_index"] = article_idx
        article_result["dataset_index"] = dataset_index
        article_result["article_hash"] = article_hash
        article_result["sample_source"] = args.sample_source
        article_result["framework_mode"] = "full"
        article_results.append(article_result)
        processed_hashes.add(article_hash)
        processed_dataset_indices.add(dataset_index)

        save_checkpoint(
            args.output_path,
            article_results,
            args.sample_source,
            args.sample_size,
            completed=False,
            usage_tracker=usage_tracker,
        )

        print(
            f"[Article {article_idx}] "
            f"dataset_index={dataset_index} | "
            f"entity={article_result['avg_entity_preservation']:.4f} | "
            f"entity_relaxed={article_result['avg_entity_preservation_relaxed']:.4f} | "
            f"nli={article_result['avg_nli_score']:.4f} | "
            f"contradiction_rate={article_result['contradiction_rate']:.4f} | "
            f"bertscore={article_result['avg_bertscore_f1']:.4f} | "
            f"cosine={article_result['avg_cosine_similarity']:.4f} | "
            f"semantic_distance={article_result['avg_semantic_distance']:.4f} | "
            f"form_distance={article_result['avg_form_distance']:.4f} | "
            f"bleu={article_result['avg_bleu']:.4f} | "
            f"rouge1={article_result['avg_rouge1_f1']:.4f} | "
            f"rougeL={article_result['avg_rouge_l_f1']:.4f} | "
            f"lcs_ratio={article_result['avg_sentence_lcs_ratio']:.4f} | "
            f"4gram_overlap={article_result['avg_fourgram_overlap']:.4f} | "
            f"target_retention={article_result['avg_target_retention_rate']:.4f} | "
            f"target_rewrite={article_result['avg_target_rewrite_rate']:.4f} | "
            f"target_ngram={article_result['avg_target_ngram_overlap']:.4f} | "
            f"openai_cost=${usage_tracker.aggregate()['cost_usd']:.4f}"
        )

    summary = save_checkpoint(
        args.output_path,
        article_results,
        args.sample_source,
        args.sample_size,
        completed=True,
        usage_tracker=usage_tracker,
    )

    print("\n===== Summary =====")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nSaved results to: {args.output_path}")


if __name__ == "__main__":
    main()
