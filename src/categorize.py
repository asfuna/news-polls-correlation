"""Prepare, check, and finalize hand classification of the Posts sheet.

    uv run src/categorize.py prep       # validate source, build batches + label templates
    uv run src/categorize.py codes      # numbered taxonomy codes
    uv run src/categorize.py check      # validate filled label files, list unfinished batches
    uv run src/categorize.py review     # labels worth a second look
    uv run src/categorize.py finalize   # expand, validate, write the CSV, delete the work dir

The model only reads `work/batches/B###.txt` and fills `work/labels/B###.tsv`.
Empty captions, captions that are only links/mentions, and exact duplicates are
handled here, so the model sees each distinct caption once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_DIR / "data" / "raw" / "news_coverage.xlsx"
TAXONOMY_PATH = PROJECT_DIR / "prompts" / "taxonomy.json"
OUT_DIR = PROJECT_DIR / "data" / "llm-processed"
WORK_DIR = OUT_DIR / "work"
BATCH_DIR = WORK_DIR / "batches"
LABEL_DIR = WORK_DIR / "labels"
UNITS_PATH = WORK_DIR / "units.json"
OUTPUT_PATH = OUT_DIR / "news_posts_categorized.csv"

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


def load_taxonomy() -> list[dict]:
    """Tags in taxonomy.json order; code = index + 1."""
    taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    tags = [
        {"topic": d["domain"], "field": t["tag"], "theme": t["theme"],
         "sv_theme": t.get("theme_if_settler_violence")}
        for d in taxonomy["domains"] for t in d["tags"]
    ]
    excluded = taxonomy["excluded"]
    tags += [{"topic": excluded["domain"], "field": t, "theme": excluded["theme"], "sv_theme": None}
             for t in excluded["tags"]]
    return tags


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
            "confidence": CONFIDENCE[conf], "code": int(code),
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


def cmd_review(_: argparse.Namespace) -> None:
    """Medium-confidence labels and labels that differ from both neighbours in a related run."""
    meta = load_units()
    labels, _ = collect(quiet=True)
    order = [u for b in meta["batches"] for u in b["units"] if u in labels]
    for i, unit_id in enumerate(order):
        code = labels[unit_id]["code"]
        prev_code = labels[order[i - 1]]["code"] if i else None
        next_code = labels[order[i + 1]]["code"] if i + 1 < len(order) else None
        isolated = prev_code is not None and prev_code == next_code and prev_code != code
        if labels[unit_id]["confidence"] == "medium" or isolated:
            reason = "medium" if labels[unit_id]["confidence"] == "medium" else f"neighbours {prev_code}"
            print(f"{unit_id}\t{code}\t{reason}\t{meta['units'][unit_id]['snippet']}")


def cmd_finalize(_: argparse.Namespace) -> None:
    posts = load_source()
    meta = load_units()
    if meta["source_digest"] != source_digest(posts):
        sys.exit("Source changed since prep; rebuild the work dir.")
    unit_labels, complete = collect()
    if not complete:
        sys.exit("Fix or finish the batches above before finalizing.")

    tags = load_taxonomy()
    unclassifiable = next(t for t in tags if t["field"] == UNCLASSIFIABLE)
    by_item = {}
    for unit_id, unit in meta["units"].items():
        for item_id in unit["item_ids"]:
            by_item[item_id] = unit_labels[unit_id]
    empty = {"topic": unclassifiable["topic"], "field": UNCLASSIFIABLE,
             "theme": unclassifiable["theme"], "confidence": "low"}
    rows = [by_item.get(item_id, empty) for item_id in posts["item_id"]]

    # Taxonomy validation of the expanded labels.
    allowed = {(t["topic"], t["field"]): {t["theme"]} | ({t["sv_theme"]} if t["sv_theme"] else set())
               for t in tags}
    assert set(by_item) <= set(posts["item_id"]) and len(by_item) == sum(len(u["item_ids"]) for u in meta["units"].values())
    for item_id, label in zip(posts["item_id"], rows):
        themes = allowed.get((label["topic"], label["field"]))
        if themes is None or label["theme"] not in themes or label["confidence"] not in CONFIDENCE.values():
            sys.exit(f"Invalid label for {item_id}: {label}")

    out = posts.copy()
    for col in LABEL_COLUMNS:
        out[col] = [label[col] for label in rows]
    out["week_id"] = pd.to_datetime(out["week_id"]).dt.strftime("%Y-%m-%d")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    written = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
    if list(written.columns) != SOURCE_COLUMNS + LABEL_COLUMNS or len(written) != EXPECTED_ROWS \
            or list(written["item_id"]) != list(posts["item_id"]):
        sys.exit(f"Written CSV failed verification: {OUTPUT_PATH}")
    shutil.rmtree(WORK_DIR)
    print(f"Wrote {OUTPUT_PATH} ({len(written)} rows); removed {WORK_DIR}")
    print(written["confidence"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("--batch-size", type=int, default=150)
    for name in ("codes", "check", "review", "finalize"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"prep": cmd_prep, "codes": cmd_codes, "check": cmd_check,
     "review": cmd_review, "finalize": cmd_finalize}[args.command](args)


if __name__ == "__main__":
    main()
