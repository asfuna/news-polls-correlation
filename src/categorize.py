"""Prepare, check, and finalize judge-assisted classification and topic weights.

    uv run src/categorize.py prep       # validate source, build batches + label templates
    uv run src/categorize.py codes      # numbered taxonomy codes
    uv run src/categorize.py check      # validate filled label files, list unfinished batches
    uv run src/categorize.py finalize   # expand labels and weights, write both CSVs, delete work

The omp judge reads `work/batches/B###.txt` and fills `work/labels/B###.tsv`;
`work/probs/` retains its probability distributions until finalization.
Empty captions, captions that are only links/mentions, and exact duplicates are
handled here, so the model sees each distinct caption once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from judge_labels import (
    BATCH_DIR,
    LABEL_DIR,
    OUT_DIR,
    PROBS_DIR,
    UNITS_PATH,
    WORK_DIR,
    load_taxonomy,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_DIR / "data" / "raw" / "news_coverage.xlsx"
OUTPUT_PATH = OUT_DIR / "news_posts_categorized.csv"
WEIGHTS_PATH = OUT_DIR / "news_posts_topic_weights.csv"
PROBABILITY_CUTOFF = 0.01
WEIGHT_COLUMNS = ["item_id", "topic", "field", "theme", "weight"]

SOURCE_COLUMNS = [
    "item_id", "week_id", "week_dates", "post_time_il", "outlet", "outlet_type", "platform",
    "account", "text", "has_text", "post_url", "likes", "comments", "shares",
]
LABEL_COLUMNS = ["topic", "field", "theme", "confidence"]
EXPECTED_ROWS = 14_218
TEXT_LIMIT = 400
SNIPPET_LEN = 40
CONFIDENCE = {"h": "high", "m": "medium", "l": "low"}
UNCLASSIFIABLE = "לא ניתן לסיווג"

URL_RE = re.compile(r"https?://\S+|\b(?:bit\.ly|did\.li|n12\.pro|calcal\.ist|tinyurl\.com)/\S+")
MENTION_RE = re.compile(r"@[\w.]+")
INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\u2800\ufeff]")



def load_source() -> pd.DataFrame:
    posts = pd.read_excel(SOURCE_PATH, sheet_name="Posts", engine="openpyxl")
    problems = []
    if len(posts) != EXPECTED_ROWS:
        problems.append(f"{len(posts)} rows, expected {EXPECTED_ROWS}")
    if list(posts.columns) != SOURCE_COLUMNS:
        problems.append(f"columns {list(posts.columns)}")
    elif posts["item_id"].isna().any() or not posts["item_id"].is_unique:
        problems.append("item_id has nulls or duplicates")
    if problems:
        sys.exit("Source mismatch: " + "; ".join(problems))
    return posts




def clean_caption(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = INVISIBLE_RE.sub("", text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    return " ".join(text.split())


def snippet(text: str) -> str:
    return text[:SNIPPET_LEN].replace("\t", " ")


def source_digest(posts: pd.DataFrame) -> str:
    payload = posts[["item_id", "text"]].to_json(orient="values", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_units() -> dict:
    if not UNITS_PATH.exists():
        sys.exit("Run `uv run src/categorize.py prep` first.")
    return json.loads(UNITS_PATH.read_text(encoding="utf-8"))


def cmd_prep(args: argparse.Namespace) -> None:
    posts = load_source()
    digest = source_digest(posts)
    if UNITS_PATH.exists():
        if load_units()["source_digest"] == digest:
            print(f"Work dir is current: {WORK_DIR}. Filled labels are kept.")
            return
        sys.exit(f"{WORK_DIR} was built from a different source; delete it to start over.")

    cleaned = posts["text"].map(clean_caption)
    units: list[dict] = []
    by_text: dict[str, dict] = {}
    for item_id, text, outlet in zip(posts["item_id"], cleaned, posts["outlet"]):
        if not text:
            continue  # labelled לא ניתן לסיווג in finalize
        unit = by_text.get(text)
        if unit is None:
            unit = {"unit_id": f"U{len(units) + 1:05d}", "outlet": outlet, "text": text, "item_ids": []}
            by_text[text] = unit
            units.append(unit)
        unit["item_ids"].append(item_id)

    BATCH_DIR.mkdir(parents=True)
    LABEL_DIR.mkdir(parents=True)
    batches = []
    for start in range(0, len(units), args.batch_size):
        name = f"B{len(batches) + 1:03d}"
        chunk = units[start:start + args.batch_size]
        batches.append({"batch": name, "units": [u["unit_id"] for u in chunk]})
        view = []
        for u in chunk:
            text = u["text"] if len(u["text"]) <= TEXT_LIMIT else u["text"][:TEXT_LIMIT] + "…"
            view.append(f"{u['unit_id']} | {u['outlet']} | {text}")
        (BATCH_DIR / f"{name}.txt").write_text("\n".join(view) + "\n", encoding="utf-8")
        template = [f"{u['unit_id']}\t\t\t\t{snippet(u['text'])}" for u in chunk]
        (LABEL_DIR / f"{name}.tsv").write_text("\n".join(template) + "\n", encoding="utf-8")

    UNITS_PATH.write_text(json.dumps({
        "source_digest": digest,
        "units": {u["unit_id"]: {"item_ids": u["item_ids"], "snippet": snippet(u["text"])} for u in units},
        "batches": batches,
    }, ensure_ascii=False), encoding="utf-8")
    empty = int((cleaned == "").sum())
    print(f"{len(posts)} posts: {empty} with no usable caption (auto {UNCLASSIFIABLE}), "
          f"{len(posts) - empty} captioned -> {len(units)} distinct captions in {len(batches)} batches")


def cmd_codes(_: argparse.Namespace) -> None:
    for code, tag in enumerate(load_taxonomy(), 1):
        print(f"{code}\t{tag['field']}\t[{tag['topic']}]" + ("\t(sv allowed)" if tag["sv_theme"] else ""))


def parse_batch(batch: dict, units: dict, tags: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Return (labels by unit_id, errors). Unfilled rows are reported as errors."""
    path = LABEL_DIR / f"{batch['batch']}.tsv"
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels, errors = {}, []
    ids = [l.split("\t", 1)[0].strip() for l in lines]
    if ids != batch["units"]:
        return {}, ["unit_id column was changed: lines must stay in template order, one per unit"]
    for line in lines:
        cols = line.split("\t")
        unit_id = cols[0].strip()
        if len(cols) != 5:
            errors.append(f"{unit_id}: expected 5 tab-separated columns, got {len(cols)}")
            continue
        _, code, conf, flag, snip = (c.strip() for c in cols)
        if not code and not conf:
            errors.append(f"{unit_id}: unfilled")
            continue
        if snip != units[unit_id]["snippet"].strip():
            errors.append(f"{unit_id}: snippet column was changed")
            continue
        if not code.isdigit() or not 1 <= int(code) <= len(tags):
            errors.append(f"{unit_id}: bad code {code!r}")
            continue
        if conf not in CONFIDENCE:
            errors.append(f"{unit_id}: bad confidence {conf!r}")
            continue
        tag = tags[int(code) - 1]
        if flag not in ("", "sv") or (flag == "sv" and not tag["sv_theme"]):
            errors.append(f"{unit_id}: sv flag only allowed on code {[i for i, t in enumerate(tags, 1) if t['sv_theme']]}")
            continue
        if tag["field"] == UNCLASSIFIABLE and conf != "l":
            errors.append(f"{unit_id}: {UNCLASSIFIABLE} requires confidence l")
            continue
        labels[unit_id] = {
            "topic": tag["topic"], "field": tag["field"],
            "theme": tag["sv_theme"] if flag == "sv" else tag["theme"],
            "confidence": CONFIDENCE[conf], "code": int(code), "sv_flag": flag,
        }
    return labels, errors


def collect(quiet: bool = False) -> tuple[dict[str, dict], bool]:
    meta, tags = load_units(), load_taxonomy()
    all_labels, pending = {}, []
    for batch in meta["batches"]:
        labels, errors = parse_batch(batch, meta["units"], tags)
        all_labels.update(labels)
        if errors:
            pending.append(batch["batch"])
            problems = [e for e in errors if not e.endswith(": unfilled")]
            if not quiet and problems:
                shown = "; ".join(problems[:5]) + (f"; … {len(problems) - 5} more" if len(problems) > 5 else "")
                print(f"{batch['batch']}: {shown}")
    if not quiet:
        done = len(meta["batches"]) - len(pending)
        print(f"{done}/{len(meta['batches'])} batches complete, {len(all_labels)}/{len(meta['units'])} captions labeled"
              + (f"; unfinished: {' '.join(pending)}" if pending else ""))
    return all_labels, not pending


def cmd_check(_: argparse.Namespace) -> None:
    collect()


def load_judge_records(batches: list[dict]) -> dict[str, dict]:
    """Load saved judge distributions for the current batches."""
    records = {}
    for batch in batches:
        path = PROBS_DIR / f"{batch['batch']}.json"
        if not path.exists():
            continue
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            sys.exit(f"Cannot read judge probabilities from {path}: {error}")
        if not isinstance(saved, dict):
            sys.exit(f"Expected unit probability records in {path}")
        for unit_id, record in saved.items():
            if not isinstance(record, dict) or unit_id in records:
                sys.exit(f"Invalid or duplicate judge probability record for {unit_id}")
            records[unit_id] = record
    return records


def unit_topic_weights(unit_id: str, label: dict, judge_record: dict | None, tags: list[dict]) -> dict:
    """Return positive weights keyed by (topic, field, theme) for one unit."""
    one_hot = {(label["topic"], label["field"], label["theme"]): 1.0}
    if judge_record is None:
        return one_hot
    try:
        judge_code = int(judge_record["code"])
        judge_sv_flag = judge_record["sv_flag"]
    except (KeyError, TypeError, ValueError) as error:
        sys.exit(f"Invalid judge label record for {unit_id}: {error}")
    if judge_code != label["code"] or judge_sv_flag != label["sv_flag"]:
        return one_hot

    probabilities = judge_record.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        sys.exit(f"Missing judge distribution for {unit_id}")
    retained = []
    for choice, raw_probability in probabilities.items():
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError):
            sys.exit(f"Invalid probability for {unit_id}, {choice!r}: {raw_probability!r}")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            sys.exit(f"Invalid probability for {unit_id}, {choice!r}: {probability!r}")
        if probability < PROBABILITY_CUTOFF:
            continue
        match = re.match(r"^\s*(\d+)", str(choice))
        if match is None:
            sys.exit(f"Invalid judge choice for {unit_id}: {choice!r}")
        code = int(match.group(1))
        if not 1 <= code <= len(tags):
            sys.exit(f"Invalid judge code for {unit_id}: {code}")
        retained.append((code, probability))

    retained_total = math.fsum(probability for _, probability in retained)
    if not math.isfinite(retained_total) or retained_total <= 0:
        sys.exit(f"No judge probability remains after pruning for {unit_id}")
    try:
        sv_probability = float(judge_record["sv_probability"])
    except (KeyError, TypeError, ValueError):
        sys.exit(f"Missing settler-violence probability for {unit_id}")
    if not math.isfinite(sv_probability) or not 0 <= sv_probability <= 1:
        sys.exit(f"Invalid settler-violence probability for {unit_id}: {sv_probability!r}")

    weights = {}
    for code, probability in retained:
        mass = probability / retained_total
        tag = tags[code - 1]
        if code == 5:
            if not tag["sv_theme"]:
                sys.exit("Taxonomy code 5 must define a settler-violence theme")
            plain_key = (tag["topic"], tag["field"], tag["theme"])
            sv_key = (tag["topic"], tag["field"], tag["sv_theme"])
            plain_weight = mass * (1 - sv_probability)
            sv_weight = mass * sv_probability
            if plain_weight > 0:
                weights[plain_key] = weights.get(plain_key, 0.0) + plain_weight
            if sv_weight > 0:
                weights[sv_key] = weights.get(sv_key, 0.0) + sv_weight
        else:
            key = (tag["topic"], tag["field"], tag["theme"])
            weights[key] = weights.get(key, 0.0) + mass
    if not math.isclose(math.fsum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        sys.exit(f"Judge weights for {unit_id} do not sum to 1")
    return weights

def cmd_finalize(_: argparse.Namespace) -> None:
    posts = load_source()
    meta = load_units()
    if meta["source_digest"] != source_digest(posts):
        sys.exit("Source changed since prep; rebuild the work dir.")
    unit_labels, complete = collect()
    if not complete:
        sys.exit("Fix or finish the batches above before finalizing.")

    tags = load_taxonomy()
    judge_records = load_judge_records(meta["batches"])
    by_item, weights_by_item = {}, {}
    for unit_id, unit in meta["units"].items():
        label = unit_labels[unit_id]
        weights = unit_topic_weights(unit_id, label, judge_records.get(unit_id), tags)
        for item_id in unit["item_ids"]:
            if item_id in by_item:
                sys.exit(f"item_id {item_id} appears in more than one unit")
            by_item[item_id] = label
            weights_by_item[item_id] = weights

    unclassifiable = next(t for t in tags if t["field"] == UNCLASSIFIABLE)
    empty = {
        "topic": unclassifiable["topic"],
        "field": UNCLASSIFIABLE,
        "theme": unclassifiable["theme"],
        "confidence": "low",
    }
    empty_weights = {(empty["topic"], empty["field"], empty["theme"]): 1.0}
    item_ids = list(posts["item_id"])
    item_id_set = set(item_ids)
    expected_captioned = sum(len(unit["item_ids"]) for unit in meta["units"].values())
    if len(item_ids) != len(item_id_set) or not set(by_item) <= item_id_set \
            or len(by_item) != expected_captioned:
        sys.exit("Expanded unit labels do not match the source item_ids.")
    rows = [by_item.get(item_id, empty) for item_id in item_ids]

    # Validate labels and every weighted taxonomy tuple after expansion.
    allowed = {
        (tag["topic"], tag["field"]): {tag["theme"]}
        | ({tag["sv_theme"]} if tag["sv_theme"] else set())
        for tag in tags
    }
    allowed_triples = {
        (tag["topic"], tag["field"], theme)
        for tag in tags
        for theme in ({tag["theme"], tag["sv_theme"]} if tag["sv_theme"] else {tag["theme"]})
    }
    taxonomy_order = {}
    for code, tag in enumerate(tags, 1):
        taxonomy_order[(tag["topic"], tag["field"], tag["theme"])] = code * 2
        if tag["sv_theme"]:
            taxonomy_order[(tag["topic"], tag["field"], tag["sv_theme"])] = code * 2 + 1
    for item_id, label in zip(item_ids, rows):
        themes = allowed.get((label["topic"], label["field"]))
        if themes is None or label["theme"] not in themes \
                or label["confidence"] not in CONFIDENCE.values():
            sys.exit(f"Invalid label for {item_id}: {label}")

    weight_rows = []
    spread_posts = 0
    for item_id in item_ids:
        weights = weights_by_item.get(item_id, empty_weights)
        total = math.fsum(weights.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            sys.exit(f"Topic weights for {item_id} sum to {total}, not 1")
        if not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
            sys.exit(f"Topic weights for {item_id} must be positive finite values")
        for topic_field_theme, weight in weights.items():
            if topic_field_theme not in allowed_triples:
                sys.exit(f"Invalid weighted taxonomy tuple for {item_id}: {topic_field_theme}")
        if len(weights) > 1:
            spread_posts += 1
        for (topic, field, theme), weight in sorted(
            weights.items(), key=lambda entry: taxonomy_order[entry[0]]
        ):
            weight_rows.append({
                "item_id": item_id,
                "topic": topic,
                "field": field,
                "theme": theme,
                "weight": weight,
            })

    out = posts.copy()
    for col in LABEL_COLUMNS:
        out[col] = [label[col] for label in rows]
    out["week_id"] = pd.to_datetime(out["week_id"]).dt.strftime("%Y-%m-%d")
    weight_out = pd.DataFrame(weight_rows, columns=WEIGHT_COLUMNS)
    if list(weight_out.columns) != WEIGHT_COLUMNS or set(weight_out["item_id"]) != item_id_set \
            or weight_out.duplicated(["item_id", "topic", "field", "theme"]).any():
        sys.exit(f"Topic weights failed validation before writing: {WEIGHTS_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    weight_out.to_csv(WEIGHTS_PATH, index=False, encoding="utf-8")

    written = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
    if list(written.columns) != SOURCE_COLUMNS + LABEL_COLUMNS or len(written) != EXPECTED_ROWS \
            or list(written["item_id"]) != item_ids:
        sys.exit(f"Written CSV failed verification: {OUTPUT_PATH}")
    shutil.rmtree(WORK_DIR)
    spread_share = spread_posts / len(item_ids) if item_ids else 0.0
    print(f"Wrote {OUTPUT_PATH} ({len(written)} rows) and {WEIGHTS_PATH} "
          f"({len(weight_out)} rows); removed {WORK_DIR}")
    print(written["confidence"].value_counts().to_string())
    print(f"Topic weights: {spread_posts}/{len(item_ids)} posts have a spread "
          f"({spread_share:.1%}); {len(item_ids) - spread_posts} are one-hot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("--batch-size", type=int, default=150)
    for name in ("codes", "check", "finalize"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"prep": cmd_prep, "codes": cmd_codes, "check": cmd_check,
     "finalize": cmd_finalize}[args.command](args)


if __name__ == "__main__":
    main()
