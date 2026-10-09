"""Correlate weekly news-topic activity with changes in coalition polling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
LLM_PROCESSED_DIR = DATA_DIR / "llm-processed"
PROCESSED_DIR = DATA_DIR / "processed"

CATEGORIZED_NEWS_PATH = LLM_PROCESSED_DIR / "news_posts_categorized.csv"
POLLS_WEEKLY_PATH = PROCESSED_DIR / "polls_weekly.csv"
CORRELATIONS_PATH = PROCESSED_DIR / "correlations.csv"
NEWS_ZSCORES_POLLS_PATH = PROCESSED_DIR / "news_zscores_polls.csv"

GROUP_COLUMNS = ["topic", "field", "theme"]
SCORE_COLUMNS = ["week", *GROUP_COLUMNS, "total_posts", "z_score"]
JOINED_COLUMNS = [*SCORE_COLUMNS, "poll_change"]
CORRELATION_COLUMNS = [*GROUP_COLUMNS, "n_weeks", "correlation"]
OUTSIDE_ANALYSIS_TOPIC = "מחוץ לניתוח"


def weekly_post_counts(news_posts: pd.DataFrame) -> pd.DataFrame:
    """Count each topic group for every week represented in the news input."""
    included_posts = news_posts.loc[
        news_posts["topic"].ne(OUTSIDE_ANALYSIS_TOPIC)
    ].copy()
    included_posts["week"] = pd.to_datetime(included_posts["week_id"])
    weeks = pd.to_datetime(news_posts["week_id"]).drop_duplicates().sort_values()
    groups = included_posts[GROUP_COLUMNS].drop_duplicates()
    counts = (
        included_posts.groupby(["week", *GROUP_COLUMNS], dropna=False)
        .size()
        .rename("total_posts")
        .reset_index()
    )

    return (
        pd.DataFrame({"week": weeks})
        .merge(groups, how="cross")
        .merge(counts, on=["week", *GROUP_COLUMNS], how="left")
        .assign(total_posts=lambda frame: frame["total_posts"].fillna(0).astype("int64"))
    )


def news_z_scores(weekly_counts: pd.DataFrame) -> pd.DataFrame:
    """Calculate sample-standard-deviation z-scores and drop undefined rows."""
    scores = weekly_counts.copy()
    counts_by_group = scores.groupby(GROUP_COLUMNS, dropna=False)["total_posts"]
    means = counts_by_group.transform("mean")
    standard_deviations = counts_by_group.transform("std")
    scores["z_score"] = (scores["total_posts"] - means) / standard_deviations
    return scores.dropna(subset=["z_score"]).reindex(columns=SCORE_COLUMNS)


def weekly_poll_changes(polls_weekly: pd.DataFrame) -> pd.DataFrame:
    """Calculate each week's government-seat change across its two neighbors."""
    polls = polls_weekly[["week", "gov"]].copy()
    polls["week"] = pd.to_datetime(polls["week"])
    polls["gov"] = pd.to_numeric(polls["gov"], errors="coerce")
    gov_by_week = polls.set_index("week")["gov"]
    weeks = gov_by_week.index
    changes = (
        gov_by_week.reindex(weeks + pd.Timedelta(days=7)).to_numpy()
        - gov_by_week.reindex(weeks - pd.Timedelta(days=7)).to_numpy()
    )
    return pd.DataFrame({"week": weeks, "poll_change": changes})


def join_weekly_data(
    z_scores: pd.DataFrame, poll_changes: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join news z-scores to weekly poll changes."""
    return (
        z_scores.merge(poll_changes, on="week", how="inner")
        .reindex(columns=JOINED_COLUMNS)
        .sort_values(["week", *GROUP_COLUMNS], kind="mergesort")
        .reset_index(drop=True)
    )


def summarize_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    """Correlate complete weekly pairs and report the number actually used."""
    rows = []
    for keys, group in joined.groupby(GROUP_COLUMNS, sort=False, dropna=False):
        paired = group.dropna(subset=["z_score", "poll_change"])
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, keys)),
                "n_weeks": len(paired),
                "correlation": (
                    paired["z_score"].corr(paired["poll_change"])
                    if len(paired) >= 2
                    else float("nan")
                ),
            }
        )

    return (
        pd.DataFrame(rows, columns=CORRELATION_COLUMNS)
        .sort_values("correlation", ascending=False, na_position="last", kind="mergesort")
        .reset_index(drop=True)
    )


def main() -> None:
    if not CATEGORIZED_NEWS_PATH.is_file():
        raise FileNotFoundError(
            f"Categorized news input not found: {CATEGORIZED_NEWS_PATH}. "
            "Run the prompt pipeline first to create news_posts_categorized.csv."
        )

    news_posts = pd.read_csv(CATEGORIZED_NEWS_PATH)
    polls_weekly = pd.read_csv(POLLS_WEEKLY_PATH)
    counts = weekly_post_counts(news_posts)
    z_scores = news_z_scores(counts)
    poll_changes = weekly_poll_changes(polls_weekly)
    joined = join_weekly_data(z_scores, poll_changes)
    correlations = summarize_correlations(joined)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(CORRELATIONS_PATH, index=False)
    joined.to_csv(NEWS_ZSCORES_POLLS_PATH, index=False)
    print(f"Wrote {len(correlations)} correlations to {CORRELATIONS_PATH}.")
    print(f"Wrote {len(joined)} joined weekly rows to {NEWS_ZSCORES_POLLS_PATH}.")


if __name__ == "__main__":
    main()
