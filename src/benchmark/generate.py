"""Unified DeepSeek generator for CultureLens benchmark construction.

Examples
--------
python -m benchmark.generate norm --input atlas_sections.jsonl --output norms.jsonl
python -m benchmark.generate scenario --input norms.jsonl --output scenarios.jsonl
python -m benchmark.generate options --input scenarios.jsonl --output options.jsonl
python -m benchmark.generate values --input options.jsonl --output culturelens.json
python -m benchmark.generate gold --input splits/train.json --output gold/train.country_only.jsonl
python -m benchmark.generate gold_cf --input splits/train.json --output gold/train.country_opposite_norm.jsonl
python -m benchmark.generate gold_request --input requests/train.json --output gold/train.request.jsonl

``gold`` writes the target answer for the country-only prompt. The teacher sees
that prompt plus the record's norm, value evidence and reference answer; the
same answer is the target for every truthful country view (country + value,
country + norm, country + value + norm). ``gold_cf`` writes the target for the
country + false-norm prompt: it corrects the claim, then gives the documented
answer. ``gold_request`` reads records in which a person in the situation asks
for the behaviour the custom does not prescribe (``behavioral_evaluation.requests``)
and writes the target that does what was asked. None is verified by a second model.

The command uses only DeepSeek, accepts JSON arrays or JSONL inputs, and writes
exactly the requested output file. Existing outputs are used as resumable
checkpoints unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from behavioral_evaluation.prompts import build_user_message


ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"

PROMPTS = {
    "norm": ROOT / "prompts/benchmark/norm.txt",
    "scenario": ROOT / "prompts/benchmark/scenario.txt",
    "options": ROOT / "prompts/benchmark/options.txt",
    "values": ROOT / "prompts/benchmark/value.txt",
    "gold": ROOT / "prompts/benchmark/gold.txt",
    "gold_cf": ROOT / "prompts/benchmark/gold_counterfactual.txt",
    "gold_request": ROOT / "prompts/benchmark/gold_request.txt",
}

MAX_TOKENS = {"norm": 8192, "scenario": 1600, "options": 900, "values": 500,
              "gold": 700, "gold_cf": 700, "gold_request": 700}
# The student prompt each gold stage answers, and the views that share its answer.
GOLD_CONDITION = {"gold": "country_only", "gold_cf": "country_opposite_norm",
                  "gold_request": "country_opposite_norm"}   # the request stands where the norm would
GOLD_LABEL = {"gold_request": "request"}                      # the condition written to the output
GOLD_TARGET_FOR = {
    "gold": ["country_only", "country_value", "country_norm", "country_value_norm"],
    "gold_cf": ["country_opposite_norm"],
    "gold_request": ["request"],
}
# Phrases that show the teacher talking about its inputs instead of answering.
GOLD_LEAK = re.compile(
    r"private[ _]source|reference[ _]answer|contrasting[ _]answer|student[ _]prompt"
    r"|\bthe prompt\b|\bdataset\b|\boption [ab]\b|</?[a-z_]+>"
    r"|\bdocumented\b|recommendation follows|\bschwartz\b"
    r"|interpersonal conformity|conformity[ _](?:rules|interpersonal)|benevolence|universalism"
    r"|self[- _]direction|power[ _](?:dominance|resources)|security[ _](?:personal|societal)",
    re.IGNORECASE)
ALIASES = {
    "norm": "norm",
    "norms": "norm",
    "scenario": "scenario",
    "scenarios": "scenario",
    "option": "options",
    "options": "options",
    "value": "values",
    "values": "values",
    "gold": "gold",
    "gold_cf": "gold_cf",
    "gold_counterfactual": "gold_cf",
    "gold_request": "gold_request",
}

SCHWARTZ_VALUES = {
    "self_direction_thought",
    "self_direction_action",
    "stimulation",
    "hedonism",
    "achievement",
    "power_dominance",
    "power_resources",
    "face",
    "security_personal",
    "security_societal",
    "tradition",
    "conformity_rules",
    "conformity_interpersonal",
    "humility",
    "benevolence_dependability",
    "benevolence_caring",
    "universalism_concern",
    "universalism_nature",
    "universalism_tolerance",
}

SCENARIO_FIELDS = (
    "situation",
    "question",
    "country_context",
    "norm_evidence",
    "opposite_norm_text",
    "opposite_norm_evidence",
)

BENCHMARK_CATEGORIES = (
    "greetings",
    "family",
    "etiquette",
    "dos_and_donts",
    "communication",
    "business_culture",
    "religion",
    "core_concepts",
    "other_considerations",
)

COUNTRY_NAME_OVERRIDES = {
    "AFG": "Afghanistan",
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BGD": "Bangladesh",
    "BIH": "Bosnia and Herzegovina",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHL": "Chile",
    "CHN": "China",
    "COL": "Colombia",
    "CYP": "Cyprus",
    "DEU": "Germany",
    "EGY": "Egypt",
    "ESP": "Spain",
    "ETH": "Ethiopia",
    "FJI": "Fiji",
    "FRA": "France",
    "GBR": "the United Kingdom",
    "GRC": "Greece",
    "HKG": "Hong Kong",
    "HRV": "Croatia",
    "HUN": "Hungary",
    "IDN": "Indonesia",
    "IND": "India",
    "IRL": "Ireland",
    "IRN": "Iran",
    "IRQ": "Iraq",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KEN": "Kenya",
    "KHM": "Cambodia",
    "KOR": "South Korea",
    "LAO": "Laos",
    "LBN": "Lebanon",
    "LKA": "Sri Lanka",
    "MEX": "Mexico",
    "MKD": "North Macedonia",
    "MLT": "Malta",
    "MMR": "Myanmar",
    "MUS": "Mauritius",
    "MYS": "Malaysia",
    "NLD": "the Netherlands",
    "NPL": "Nepal",
    "NZL": "New Zealand",
    "PAK": "Pakistan",
    "PER": "Peru",
    "PHL": "the Philippines",
    "PNG": "Papua New Guinea",
    "POL": "Poland",
    "PRT": "Portugal",
    "PSE": "Palestine",
    "ROU": "Romania",
    "RUS": "Russia",
    "SAU": "Saudi Arabia",
    "SDN": "Sudan",
    "SGP": "Singapore",
    "SOM": "Somalia",
    "SRB": "Serbia",
    "SSD": "South Sudan",
    "SWE": "Sweden",
    "SYR": "Syria",
    "THA": "Thailand",
    "TLS": "Timor-Leste",
    "TON": "Tonga",
    "TUR": "Türkiye",
    "TWN": "Taiwan",
    "UKR": "Ukraine",
    "USA": "the United States",
    "VEN": "Venezuela",
    "VNM": "Vietnam",
    "WSM": "Samoa",
    "ZAF": "South Africa",
    "ZWE": "Zimbabwe",
}


class BadOutput(ValueError):
    """The API returned a structurally invalid generation."""


class FatalAPIError(RuntimeError):
    """The API rejected the request in a way retries cannot repair."""


@dataclass(frozen=True)
class Result:
    source_key: str
    status: str
    records: list[dict[str, Any]]
    error: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(ALIASES), help="What to generate")
    parser.add_argument("--input", type=Path, required=True, help="Input .json or .jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Single output .json or .jsonl file")
    parser.add_argument("--prompt", type=Path, help="Override the default system prompt")
    parser.add_argument("--model", default=MODEL,
                        help="DeepSeek model id; recorded with every gold answer as its teacher.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, help="Process only this many pending inputs")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Record at {path}:{number} is not an object")
                rows.append(row)
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return value


def read_atlas_sections(path: Path) -> list[dict[str, Any]]:
    """Expand an Atlas CSV into one norm-extraction input per category cell."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "iso3" not in reader.fieldnames:
            raise ValueError(f"{path} must contain an iso3 column")
        missing = sorted(set(BENCHMARK_CATEGORIES) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Atlas CSV is missing benchmark categories: {missing}")
        for source in reader:
            iso3 = required_string(source.get("iso3"), "iso3").upper()
            for category in BENCHMARK_CATEGORIES:
                text = source.get(category)
                if isinstance(text, str) and text.strip():
                    rows.append(
                        {
                            "job_id": f"{iso3}__{category}__001",
                            "iso3": iso3,
                            "country": country_from_iso3(iso3),
                            "category": category,
                            "section_text": text.strip(),
                        }
                    )
    if not rows:
        raise ValueError(f"{path} contains no non-empty benchmark category cells")
    return rows


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix.casefold() == ".jsonl":
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    else:
        temporary.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    temporary.replace(path)


def parse_response(content: str) -> Any:
    stripped = content.strip().lstrip("\ufeff")
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence:
        stripped = fence.group(1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise BadOutput(f"Invalid JSON: {exc}; prefix={stripped[:160]!r}") from exc


def required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BadOutput(f"{field} must be a non-empty string")
    return value.strip()


def norm_text(row: dict[str, Any]) -> str:
    value = row.get("norm", row.get("norm_text"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Record needs a non-empty norm or norm_text")
    return value.strip()


def norm_id(row: dict[str, Any]) -> str:
    value = row.get("norm_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Record needs a non-empty norm_id")
    return value.strip()


def source_key(row: dict[str, Any], index: int, stage: str) -> str:
    if stage == "norm":
        value = row.get("job_id", row.get("source_job_id"))
        if isinstance(value, str) and value.strip():
            return value.strip()
        iso3 = row.get("iso3")
        category = row.get("category")
        if isinstance(iso3, str) and isinstance(category, str):
            return f"{iso3}__{category}"
        return f"source_{index + 1:06d}"
    return norm_id(row)


def source_text(row: dict[str, Any]) -> str:
    for field in ("section_text", "source_text", "text"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Norm-stage records need section_text, source_text, or text")


def country_name(row: dict[str, Any]) -> str:
    value = row.get("country")
    if isinstance(value, str) and value.strip():
        return value.strip()
    scenario = row.get("scenario")
    if isinstance(scenario, dict):
        context = scenario.get("country_context")
        if isinstance(context, str):
            match = re.fullmatch(r"This interaction takes place in (.+)\.", context.strip())
            if match:
                return match.group(1)
    iso3 = row.get("iso3")
    if isinstance(iso3, str) and iso3.strip():
        return country_from_iso3(iso3.strip().upper())
    raise ValueError("Scenario-stage records need country, iso3, or scenario.country_context")


def country_from_iso3(iso3: str) -> str:
    country = COUNTRY_NAME_OVERRIDES.get(iso3)
    if country is None:
        raise ValueError(f"Unknown ISO3 country code: {iso3}")
    return country


def gold_payload(stage: str, row: dict[str, Any]) -> dict[str, Any]:
    """What the teacher sees: the student's prompt and the private record fields."""

    scenario, options = row.get("scenario"), row.get("options")
    if not isinstance(scenario, dict) or not isinstance(options, dict):
        raise ValueError("Gold-stage records need scenario and options objects")
    private = {
        "norm": norm_text(row),
        "values": list(row.get("values") or []),
        "value_evidence": scenario.get("value_evidence"),
        "reference_answer": required_string(options.get("A"), "options.A"),
        "contrasting_answer": required_string(options.get("B"), "options.B"),
    }
    if stage == "gold_cf":
        private = {"false_claim": required_string(
            scenario.get("opposite_norm_evidence"), "opposite_norm_evidence"), **private}
    if stage == "gold_request":
        private = {
            "request": required_string(scenario.get("opposite_norm_evidence"), "opposite_norm_evidence"),
            "speaker": row.get("request_speaker"),
            "norm": private["norm"], "values": private["values"],
            "value_evidence": private["value_evidence"],
            "usual_answer": private["reference_answer"],
            "requested_answer": private["contrasting_answer"],
        }
    return {
        "student_prompt": build_user_message(row, GOLD_CONDITION[stage], False)[0],
        "private_source": private,
    }


def validate_gold(stage: str, reply: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(reply, dict) or set(reply) != {"response"}:
        raise BadOutput("Gold response must contain exactly the field response")
    response = required_string(reply["response"], "response")
    leaked = GOLD_LEAK.search(response)
    if leaked:
        raise BadOutput(f"response refers to its inputs or uses markup: {leaked.group()!r}")
    if len(response.split()) < 25:
        raise BadOutput("response is too short to give an answer and a reason")
    if stage == "gold_cf" and response.casefold().startswith("that is not the usual expectation"):
        raise BadOutput("response opens with the stock sentence; open with what this claim gets wrong")
    return [{
        "norm_id": norm_id(row),
        "condition": GOLD_LABEL.get(stage, GOLD_CONDITION[stage]),
        "target_for": GOLD_TARGET_FOR[stage],
        "user_message": build_user_message(row, GOLD_CONDITION[stage], False)[0],
        "response": response,
        "teacher": MODEL,
    }]


def user_payload(stage: str, row: dict[str, Any]) -> str:
    if stage in GOLD_CONDITION:
        return json.dumps(gold_payload(stage, row), ensure_ascii=False)
    if stage == "norm":
        return source_text(row)
    if stage == "values":
        return json.dumps({"norm": norm_text(row)}, ensure_ascii=False)
    if stage == "scenario":
        payload = {
            "iso3": row.get("iso3"),
            "country": country_name(row),
            "category": row.get("category"),
            "norm_text": norm_text(row),
        }
        return json.dumps(payload, ensure_ascii=False)
    scenario = row.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError("Options-stage records need a scenario object")
    required = {"situation", "question", "norm_evidence", "opposite_norm_evidence"}
    missing = sorted(required - set(scenario))
    if missing:
        raise ValueError(f"Scenario is missing fields required for options: {missing}")
    payload = {
        "norm_text": norm_text(row),
        "situation": scenario["situation"],
        "question": scenario["question"],
        "norm_evidence": scenario["norm_evidence"],
        "opposite_norm_evidence": scenario["opposite_norm_evidence"],
    }
    return json.dumps(payload, ensure_ascii=False)


def validate_norm(reply: Any, row: dict[str, Any], key: str) -> list[dict[str, Any]]:
    if not isinstance(reply, list) or not all(isinstance(item, str) for item in reply):
        raise BadOutput("Norm response must be a JSON array of strings")
    values: list[str] = []
    seen: set[str] = set()
    for item in reply:
        item = required_string(item, "norm")
        canonical = " ".join(item.casefold().split())
        if canonical not in seen:
            seen.add(canonical)
            values.append(item)
    metadata = {
        field: row[field]
        for field in ("iso3", "country", "category")
        if field in row
    }
    if "country" not in metadata and isinstance(row.get("iso3"), str):
        metadata["country"] = country_from_iso3(row["iso3"])
    return [
        {
            "norm_id": f"{key}__{index:03d}",
            "source_job_id": key,
            **metadata,
            "norm": value,
        }
        for index, value in enumerate(values, 1)
    ]


def validate_values(reply: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(reply, dict) or set(reply) != {"values", "value_evidence"}:
        raise BadOutput("Value response must contain exactly values and value_evidence")
    labels = reply["values"]
    if (
        not isinstance(labels, list)
        or len(labels) > 3
        or len(labels) != len(set(map(str, labels)))
        or not all(isinstance(label, str) and label in SCHWARTZ_VALUES for label in labels)
    ):
        raise BadOutput("values must contain zero to three unique refined Schwartz labels")
    evidence = reply["value_evidence"]
    if labels:
        evidence = required_string(evidence, "value_evidence")
        if "schwartz" in evidence.casefold() or "taxonomy" in evidence.casefold():
            raise BadOutput("value_evidence must not name a taxonomy")
    elif evidence is not None:
        raise BadOutput("value_evidence must be null when values is empty")
    scenario = row.get("scenario")
    if not isinstance(scenario, dict):
        raise BadOutput("Values-stage input needs a scenario object for value_evidence")
    updated_scenario: dict[str, Any] = {}
    inserted = False
    for field, value in scenario.items():
        updated_scenario[field] = value
        if field == "norm_evidence":
            updated_scenario["value_evidence"] = evidence
            inserted = True
    if not inserted:
        updated_scenario["value_evidence"] = evidence
    output: dict[str, Any] = {}
    for field, value in row.items():
        output[field] = value
        if field in {"norm", "norm_text"}:
            output["values"] = labels
        elif field == "scenario":
            output["scenario"] = updated_scenario
    if "values" not in output:
        output["values"] = labels
    output["scenario"] = updated_scenario
    return [output]


def validate_scenario(reply: Any, row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(reply, dict) or set(reply) != {"status", "reason", "scenario"}:
        raise BadOutput("Scenario response must contain exactly status, reason, and scenario")
    if reply["status"] == "unusable":
        if reply["scenario"] is not None:
            raise BadOutput("Unusable scenario must be null")
        required_string(reply["reason"], "reason")
        return "unusable", []
    if reply["status"] != "generated" or reply["reason"] is not None:
        raise BadOutput("Generated scenario needs status=generated and reason=null")
    scenario = reply["scenario"]
    if not isinstance(scenario, dict) or set(scenario) != set(SCENARIO_FIELDS):
        raise BadOutput(f"Scenario must contain exactly {sorted(SCENARIO_FIELDS)}")
    scenario = {field: required_string(scenario[field], field) for field in SCENARIO_FIELDS}
    expected = f"This interaction takes place in {country_name(row)}."
    if scenario["country_context"] != expected:
        raise BadOutput(f"country_context must be exactly {expected!r}")
    return "generated", [{**row, "scenario": scenario}]


def validate_options(reply: Any, row: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(reply, dict) or set(reply) != {"status", "reason", "options"}:
        raise BadOutput("Options response must contain exactly status, reason, and options")
    if reply["status"] == "unusable":
        if reply["options"] is not None:
            raise BadOutput("Unusable options must be null")
        required_string(reply["reason"], "reason")
        return "unusable", []
    if reply["status"] != "generated" or reply["reason"] is not None:
        raise BadOutput("Generated options need status=generated and reason=null")
    options = reply["options"]
    if not isinstance(options, dict) or set(options) != {"A", "B"}:
        raise BadOutput("options must contain exactly A and B")
    options = {field: required_string(options[field], field) for field in ("A", "B")}
    if " ".join(options["A"].casefold().split()) == " ".join(options["B"].casefold().split()):
        raise BadOutput("A and B are identical")
    return "generated", [{**row, "options": options}]


def call_deepseek(
    stage: str,
    prompt: str,
    payload: str,
    api_key: str,
    timeout: int,
    repair: str | None,
) -> Any:
    if repair:
        payload = json.dumps(
            {
                "input": payload,
                "previous_output_error": repair,
                "repair_instruction": "Regenerate the complete response and correct the error.",
            },
            ensure_ascii=False,
        )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": payload},
        ],
        "temperature": 0.1,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "text" if stage == "norm" else "json_object"},
        "max_tokens": MAX_TOKENS[stage],
        "stream": False,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        if exc.code in {400, 401, 403, 404, 422}:
            raise FatalAPIError(f"HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise BadOutput("API returned no choices")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise BadOutput(f"finish_reason={choice.get('finish_reason')!r}")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str):
        raise BadOutput("API response has no string content")
    return parse_response(content)


def process(
    stage: str,
    row: dict[str, Any],
    key: str,
    prompt: str,
    api_key: str,
    retries: int,
    timeout: int,
) -> Result:
    error = ""
    for attempt in range(retries + 1):
        try:
            reply = call_deepseek(
                stage,
                prompt,
                user_payload(stage, row),
                api_key,
                timeout,
                error if attempt else None,
            )
            if stage == "norm":
                return Result(key, "generated", validate_norm(reply, row, key))
            if stage == "values":
                return Result(key, "generated", validate_values(reply, row))
            if stage in GOLD_CONDITION:
                return Result(key, "generated", validate_gold(stage, reply, row))
            status, records = (
                validate_scenario(reply, row)
                if stage == "scenario"
                else validate_options(reply, row)
            )
            return Result(key, status, records)
        except FatalAPIError:
            raise
        except (BadOutput, ValueError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            error = str(exc)
            if attempt < retries:
                time.sleep(min(30.0, 2**attempt + random.random()))
    return Result(key, "failed", [], error)


def completed_keys(stage: str, rows: list[dict[str, Any]]) -> set[str]:
    if stage == "norm":
        return {
            row["source_job_id"]
            for row in rows
            if isinstance(row.get("source_job_id"), str)
        }
    return {
        row["norm_id"]
        for row in rows
        if isinstance(row.get("norm_id"), str)
    }


def ordered_output(
    stage: str,
    input_rows: list[dict[str, Any]],
    output_by_key: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for index, row in enumerate(input_rows):
        ordered.extend(output_by_key.get(source_key(row, index, stage), []))
    return ordered


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    global MODEL
    MODEL = args.model          # read by the request body and the gold records
    stage = ALIASES[args.stage]
    if args.workers < 1 or args.retries < 0 or args.checkpoint_every < 1:
        raise ValueError("workers and checkpoint-every must be positive; retries cannot be negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY in the shell")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("--input and --output must be different files")
    prompt_path = (args.prompt or PROMPTS[stage]).expanduser().resolve()
    prompt = prompt_path.read_text(encoding="utf-8")
    inputs = (
        read_atlas_sections(input_path)
        if stage == "norm" and input_path.suffix.casefold() == ".csv"
        else read_records(input_path)
    )
    if not inputs:
        raise ValueError("Input contains no records")

    existing = [] if args.overwrite or not output_path.exists() else read_records(output_path)
    done = completed_keys(stage, existing)
    output_by_key: dict[str, list[dict[str, Any]]] = {}
    if stage == "norm":
        for record in existing:
            key = record.get("source_job_id")
            if isinstance(key, str):
                output_by_key.setdefault(key, []).append(record)
    else:
        for record in existing:
            key = record.get("norm_id")
            if isinstance(key, str):
                output_by_key[key] = [record]

    keyed_inputs = [(source_key(row, index, stage), row) for index, row in enumerate(inputs)]
    keys = [key for key, _ in keyed_inputs]
    if len(keys) != len(set(keys)):
        raise ValueError("Input contains duplicate stage identifiers")
    unknown = done - set(keys)
    if unknown:
        raise ValueError(f"Existing output contains identifiers absent from input: {sorted(unknown)[:5]}")
    pending = [(key, row) for key, row in keyed_inputs if key not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"stage={stage} model={MODEL} inputs={len(inputs)} "
        f"already_done={len(done)} pending_this_run={len(pending)}",
        flush=True,
    )
    if not pending:
        write_records(output_path, ordered_output(stage, inputs, output_by_key))
        print(f"Output: {output_path}", flush=True)
        return 0

    counts = {"generated": 0, "unusable": 0, "failed": 0}
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process,
                stage,
                row,
                key,
                prompt,
                api_key,
                args.retries,
                args.timeout,
            ): key
            for key, row in pending
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except FatalAPIError:
                for other in futures:
                    other.cancel()
                raise
            counts[result.status] += 1
            processed += 1
            if result.status == "generated":
                output_by_key[result.source_key] = result.records
            elif result.status == "unusable":
                # Unusable items are intentionally absent from the generated dataset.
                output_by_key[result.source_key] = []
            else:
                print(f"FAILED {result.source_key}: {result.error}", file=sys.stderr, flush=True)
            if processed % args.checkpoint_every == 0 or processed == len(pending):
                write_records(output_path, ordered_output(stage, inputs, output_by_key))
                print(
                    f"{processed}/{len(pending)}: generated={counts['generated']} "
                    f"unusable={counts['unusable']} failed={counts['failed']}",
                    flush=True,
                )

    print(f"Output: {output_path}", flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError, FatalAPIError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
