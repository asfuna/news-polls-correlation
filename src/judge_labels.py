"""Build judge questions and persist classifications in the categorization work dir."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_DIR / "prompts" / "categorize_posts.md"
TAXONOMY_PATH = PROJECT_DIR / "prompts" / "taxonomy.json"
OUT_DIR = PROJECT_DIR / "data" / "llm-processed"
WORK_DIR = OUT_DIR / "work"
BATCH_DIR = WORK_DIR / "batches"
LABEL_DIR = WORK_DIR / "labels"
PROBS_DIR = WORK_DIR / "probs"
UNITS_PATH = WORK_DIR / "units.json"

_RULES_START = "## Classification rules"
_RULES_END = "### Confidence"
_CODE_RE = re.compile(r"^\s*(\d+)")
_CODE_INSTRUCTIONS = (
    "Classify the news post in `caption`, published by `outlet`, into exactly one "
    "taxonomy tag by its main subject."
)
_SV_INSTRUCTIONS = (
    "Is `caption` about violence by settlers or Jews in the West Bank: assaults, "
    "arson, nationalist crime, or arrests over them?"
)


def load_taxonomy() -> list[dict[str, Any]]:
    """Return taxonomy tags in code order; code = position + 1."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    tags = [
        {
            "topic": domain["domain"],
            "field": tag["tag"],
            "theme": tag["theme"],
            "sv_theme": tag.get("theme_if_settler_violence"),
            "desc": tag.get("desc"),
        }
        for domain in taxonomy["domains"]
        for tag in domain["tags"]
    ]
    excluded = taxonomy["excluded"]
    tags.extend(
        {
            "topic": excluded["domain"],
            "field": tag,
            "theme": excluded["theme"],
            "sv_theme": None,
            "desc": None,
        }
        for tag in excluded["tags"]
    )
    return tags


def questions() -> dict[str, dict[str, Any]]:
    """Build judge questions from the taxonomy and current prompt rules."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    start = prompt.find(_RULES_START)
    if start < 0:
        raise ValueError(f"Missing {_RULES_START!r} in {PROMPT_PATH}")
    end = prompt.find(_RULES_END, start + len(_RULES_START))
    if end < 0:
        raise ValueError(f"Missing {_RULES_END!r} after {_RULES_START!r} in {PROMPT_PATH}")
    rules = prompt[start:end].strip()
    criteria = {
        f"{code} {tag['field']}": tag["desc"]
        for code, tag in enumerate(load_taxonomy(), 1)
    }
    return {
        "code": {
            "type": "choice",
            "instructions": f"{_CODE_INSTRUCTIONS}\n\n{rules}",
            "criteria": criteria,
        },
        "sv": {"type": "bool", "instructions": _SV_INSTRUCTIONS},
    }


def pending_states() -> dict[str, dict[str, str]]:
    """Return unfilled units as judge states, preserving their batch order."""
    if not BATCH_DIR.is_dir() or not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"Run `uv run src/categorize.py prep` first ({WORK_DIR}).")

    pending: dict[str, dict[str, str]] = {}
    for batch_path in sorted(BATCH_DIR.glob("B*.txt")):
        states: dict[str, dict[str, str]] = {}
        for line_number, line in enumerate(batch_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            parts = line.split(" | ", 2)
            if len(parts) != 3:
                raise ValueError(f"Malformed batch line {batch_path}:{line_number}")
            unit_id, outlet, caption = parts
            unit_id = unit_id.strip()
            if not unit_id:
                raise ValueError(f"Missing unit_id at {batch_path}:{line_number}")
            states[unit_id] = {"outlet": outlet.strip(), "caption": caption.strip()}

        label_path = LABEL_DIR / f"{batch_path.stem}.tsv"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            columns = line.split("\t")
            if len(columns) < 3:
                continue
            unit_id = columns[0].strip()
            if unit_id in states and not columns[1].strip() and not columns[2].strip():
                pending[unit_id] = states[unit_id]
    return pending


def _answer_record(answers: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
    try:
        code_answer = answers["code"]
        choice = code_answer["choice"]
        probabilities = code_answer["probabilities"]
        match = _CODE_RE.match(choice)
        if match is None:
            raise ValueError(f"choice does not start with a code: {choice!r}")
        code = int(match.group(1))
        chosen_probability = float(probabilities[choice])
        sv_probability = float(answers["sv"]["bool"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed judge answers: {answers!r}") from error

    if not math.isfinite(chosen_probability) or not 0 <= chosen_probability <= 1:
        raise ValueError(f"Invalid chosen-label probability: {chosen_probability!r}")
    if not math.isfinite(sv_probability) or not 0 <= sv_probability <= 1:
        raise ValueError(f"Invalid settler-violence probability: {sv_probability!r}")
    if not isinstance(probabilities, dict):
        raise ValueError("Judge code probabilities must be a mapping")

    confidence = "h" if chosen_probability >= 0.5 else "m" if chosen_probability >= 0.25 else "l"
    if code == 62:
        confidence = "l"
    sv_flag = "sv" if code == 5 and sv_probability >= 0.5 else ""
    stored = {
        "probabilities": {str(label): float(probability) for label, probability in probabilities.items()},
        "sv_probability": sv_probability,
        "code": code,
        "sv_flag": sv_flag,
    }
    return code, confidence, {"sv_flag": sv_flag, "stored": stored}


def _split_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1:]
    return line, ""


def write_results(results: dict[str, dict[str, Any]]) -> dict[str, dict[Any, int]]:
    """Fill only blank label rows and save the full judge answers per batch."""
    label_updates: dict[Path, str] = {}
    probability_updates: dict[str, dict[str, dict[str, Any]]] = {}
    confidence_counts: Counter[str] = Counter()
    code_counts: Counter[int] = Counter()

    for label_path in sorted(LABEL_DIR.glob("B*.tsv")):
        original_lines = label_path.read_text(encoding="utf-8").splitlines(keepends=True)
        updated_lines: list[str] = []
        changed = False
        batch_probabilities: dict[str, dict[str, Any]] = {}
        for line in original_lines:
            body, ending = _split_ending(line)
            columns = body.split("\t")
            if len(columns) != 5:
                updated_lines.append(line)
                continue
            unit_id = columns[0].strip()
            if unit_id not in results or columns[1].strip() or columns[2].strip():
                updated_lines.append(line)
                continue

            code, confidence, record = _answer_record(results[unit_id])
            columns[1] = str(code)
            columns[2] = confidence
            columns[3] = record["sv_flag"]
            updated_lines.append("\t".join(columns) + ending)
            batch_probabilities[unit_id] = record["stored"]
            confidence_counts[confidence] += 1
            code_counts[code] += 1
            changed = True

        if changed:
            label_updates[label_path] = "".join(updated_lines)
            probability_updates[label_path.stem] = batch_probabilities

    for batch, updates in probability_updates.items():
        path = PROBS_DIR / f"{batch}.json"
        existing: dict[str, dict[str, Any]] = {}
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected a unit mapping in {path}")
            existing = loaded
        existing.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for path, text in label_updates.items():
        path.write_text(text, encoding="utf-8")

    confidence_summary = {level: confidence_counts[level] for level in ("h", "m", "l")}
    code_summary = dict(sorted(code_counts.items()))
    print("Judge results written:")
    print("  confidence: " + ", ".join(f"{level}={count}" for level, count in confidence_summary.items()))
    print("  code: " + (", ".join(f"{code}={count}" for code, count in code_summary.items()) or "none"))
    return {"confidence": confidence_summary, "code": code_summary}
