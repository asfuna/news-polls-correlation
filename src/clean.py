"""Clean weekly news coverage and coalition-poll data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

NEWS_COVERAGE_PATH = RAW_DATA_DIR / "news_coverage.xlsx"
POLLS_PATH = RAW_DATA_DIR / "israel_2026_seat_polls.csv"
NEWS_COVERAGE_OUTPUT_PATH = PROCESSED_DATA_DIR / "news_coverage.csv"
POLLS_WEEKLY_OUTPUT_PATH = PROCESSED_DATA_DIR / "polls_weekly.csv"

POSTS_COLUMNS = ["week_id", "outlet", "outlet_type", "platform"]
NEWS_OUTPUT_COLUMNS = ["week", "outlet", "outlet_type", "platforms", "posts"]
POLL_OUTPUT_COLUMNS = ["week", "gov"]
EXCLUDED_PUBLISHERS = {"Channel 14", "i24 News"}
POLL_YEAR = 2026


def clean_news_coverage(posts: pd.DataFrame) -> pd.DataFrame:
    """Count posts per outlet and Sunday week; weeks with no posts get no row.

    The workbook's Coverage sheet holds these counts as uncached formulas over
    the Posts sheet, so they are computed from Posts directly.
    """
    platforms = (
        posts.groupby("outlet")["platform"]
        .agg(lambda values: ", ".join(sorted(values.unique())))
        .rename("platforms")
    )
    news = (
        posts.groupby(["week_id", "outlet", "outlet_type"])
        .size()
        .rename("posts")
        .reset_index()
        .join(platforms, on="outlet")
        .rename(columns={"week_id": "week"})
    )
    news["week"] = pd.to_datetime(news["week"]).dt.strftime("%Y-%m-%d")
    news["posts"] = news["posts"].astype("int64")
    return (
        news.sort_values(["week", "outlet"], kind="stable")
        .reindex(columns=NEWS_OUTPUT_COLUMNS)
        .reset_index(drop=True)
    )


def clean_weekly_polls(
    polls: pd.DataFrame, *, exclude_ch14_i24: bool = False
) -> pd.DataFrame:
    """Average numeric government-seat polls by the Sunday of their final fieldwork date."""
    if exclude_ch14_i24:
        polls = polls.loc[~polls["Publisher"].isin(EXCLUDED_PUBLISHERS)]

    weekly = polls[["Fieldwork date", "Gov."]].copy()
    weekly["Gov."] = pd.to_numeric(weekly["Gov."], errors="coerce")
    weekly = weekly.dropna(subset=["Gov."])

    fieldwork_end = (
        weekly["Fieldwork date"]
        .astype("string")
        .str.rsplit("–", n=1)
        .str[-1]
        .str.strip()
    )
    weekly["fieldwork_date"] = pd.to_datetime(
        fieldwork_end + f" {POLL_YEAR}",
        format="%d %b %Y",
        errors="coerce",
    )
    weekly = weekly.dropna(subset=["fieldwork_date"])

    fieldwork_date = weekly["fieldwork_date"]
    sunday = fieldwork_date - pd.to_timedelta(
        (fieldwork_date.dt.dayofweek + 1) % 7,
        unit="D",
    )
    weekly["week"] = sunday.dt.strftime("%Y-%m-%d")
    return (
        weekly.groupby("week", as_index=False)["Gov."]
        .mean()
        .rename(columns={"Gov.": "gov"})
        .sort_values("week", kind="stable")
        .reindex(columns=POLL_OUTPUT_COLUMNS)
        .reset_index(drop=True)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse cleaner command-line options."""
    parser = argparse.ArgumentParser(
        description="Clean weekly news coverage and poll data."
    )
    parser.add_argument(
        "--exclude-ch14-i24",
        action="store_true",
        help="Exclude Channel 14 and i24 News polls (default: include all).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    posts = pd.read_excel(NEWS_COVERAGE_PATH, sheet_name="Posts", usecols=POSTS_COLUMNS)
    polls = pd.read_csv(POLLS_PATH)

    news_coverage = clean_news_coverage(posts)
    polls_weekly = clean_weekly_polls(
        polls,
        exclude_ch14_i24=args.exclude_ch14_i24,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    news_coverage.to_csv(NEWS_COVERAGE_OUTPUT_PATH, index=False)
    polls_weekly.to_csv(POLLS_WEEKLY_OUTPUT_PATH, index=False)
    print(
        f"Wrote {len(news_coverage)} news coverage rows and "
        f"{len(polls_weekly)} weekly poll rows to {PROCESSED_DATA_DIR}."
    )


if __name__ == "__main__":
    main()
