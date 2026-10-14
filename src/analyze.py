"""Correlate weekly expected news-topic activity with changes in coalition polling."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
LLM_PROCESSED_DIR = DATA_DIR / "llm-processed"
PROCESSED_DIR = DATA_DIR / "processed"

CATEGORIZED_NEWS_PATH = LLM_PROCESSED_DIR / "news_posts_categorized.csv"
TOPIC_WEIGHTS_PATH = LLM_PROCESSED_DIR / "news_posts_topic_weights.csv"
POLLS_WEEKLY_PATH = PROCESSED_DIR / "polls_weekly.csv"
CORRELATIONS_PATH = PROCESSED_DIR / "correlations.csv"
NEWS_ZSCORES_POLLS_PATH = PROCESSED_DIR / "news_zscores_polls.csv"

GROUP_COLUMNS = ["topic", "field", "theme"]
SCORE_COLUMNS = ["week", *GROUP_COLUMNS, "total_posts", "z_score"]
JOINED_COLUMNS = [*SCORE_COLUMNS, "poll_change"]
CORRELATION_COLUMNS = [*GROUP_COLUMNS, "n_weeks", "correlation", "p_value", "p_value_adjusted"]
OUTSIDE_ANALYSIS_TOPIC = "מחוץ לניתוח"
MAX_EXACT_PERMUTATIONS = 100_000
RANDOM_PERMUTATION_SEED = 0


def weekly_post_counts(
    news_posts: pd.DataFrame, topic_weights: pd.DataFrame
) -> pd.DataFrame:
    """Sum topic weights for every week represented in the categorized news input."""
    weeks = pd.to_datetime(news_posts["week_id"]).drop_duplicates().sort_values()
    included_weights = topic_weights.loc[
        topic_weights["topic"].ne(OUTSIDE_ANALYSIS_TOPIC)
    ].copy()
    post_weeks = news_posts[["item_id", "week_id"]].copy()
    post_weeks["week"] = pd.to_datetime(post_weeks["week_id"])
    groups = included_weights[GROUP_COLUMNS].drop_duplicates()
    counts = (
        included_weights.merge(post_weeks[["item_id", "week"]], on="item_id")
        .groupby(["week", *GROUP_COLUMNS], dropna=False)["weight"]
        .sum()
        .rename("total_posts")
        .reset_index()
    )

    return (
        pd.DataFrame({"week": weeks})
        .merge(groups, how="cross")
        .merge(counts, on=["week", *GROUP_COLUMNS], how="left")
        .assign(total_posts=lambda frame: frame["total_posts"].fillna(0))
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


def poll_change_orders(n_weeks: int) -> tuple[np.ndarray, bool]:
    """Return every ordering of the weeks, or a seeded random sample if there are too many."""
    if math.factorial(n_weeks) <= MAX_EXACT_PERMUTATIONS:
        return np.array(list(itertools.permutations(range(n_weeks))), dtype=np.intp), True
    rng = np.random.default_rng(RANDOM_PERMUTATION_SEED)
    base = np.tile(np.arange(n_weeks), (MAX_EXACT_PERMUTATIONS, 1))
    return rng.permuted(base, axis=1), False


def row_correlations(x: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Pearson correlation of ``x`` with each row of ``ys``; NaN where undefined."""
    x = x - x.mean()
    ys = ys - ys.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ys @ x) / np.sqrt((x @ x) * np.einsum("ij,ij->i", ys, ys))


def summarize_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    """Correlate complete weekly pairs and attach two-sided permutation p-values.

    Each permutation reassigns the weekly poll changes to weeks and applies that
    same reassignment to every group. ``p_value`` compares a group's |r| with
    its own permuted |r| values; ``p_value_adjusted`` compares it with the
    largest |r| across all groups under each permutation (single-step max-T),
    which controls the family-wise error rate over every group tested.
    """
    paired = joined.dropna(subset=["z_score", "poll_change"])
    poll_by_week = paired.groupby("week")["poll_change"].first()
    week_positions = pd.Series(np.arange(len(poll_by_week)), index=poll_by_week.index)
    polls = poll_by_week.to_numpy()
    orders, exact = poll_change_orders(len(polls))
    shuffled_polls = polls[orders]

    grouped = joined.groupby(GROUP_COLUMNS, sort=False, dropna=False)
    shuffled_abs_r = np.full((len(orders), grouped.ngroups), np.nan)
    rows = []
    for index, (keys, all_weeks) in enumerate(grouped):
        group = all_weeks.dropna(subset=["z_score", "poll_change"])
        correlation = float("nan")
        if len(group) >= 2:
            positions = week_positions[group["week"]].to_numpy()
            z_scores = group["z_score"].to_numpy()
            correlation = row_correlations(z_scores, polls[positions][np.newaxis, :])[0]
            shuffled_abs_r[:, index] = np.abs(
                row_correlations(z_scores, shuffled_polls[:, positions])
            )
        rows.append(
            {
                **dict(zip(GROUP_COLUMNS, keys)),
                "n_weeks": len(group),
                "correlation": correlation,
            }
        )

    observed_abs_r = np.abs(np.array([row["correlation"] for row in rows]))
    # Tolerance keeps the identity ordering and exact ties from being missed.
    thresholds = observed_abs_r - 1e-12
    max_abs_r = np.where(np.isnan(shuffled_abs_r), -np.inf, shuffled_abs_r).max(
        axis=1, initial=-np.inf
    )
    hits = (shuffled_abs_r >= thresholds).sum(axis=0)
    max_hits = (max_abs_r[:, np.newaxis] >= thresholds).sum(axis=0)
    if exact:
        p_values, adjusted = hits / len(orders), max_hits / len(orders)
    else:
        p_values = (hits + 1) / (len(orders) + 1)
        adjusted = (max_hits + 1) / (len(orders) + 1)
    undefined = np.isnan(observed_abs_r)
    for row, p_value, p_adjusted, missing in zip(rows, p_values, adjusted, undefined):
        row["p_value"] = float("nan") if missing else p_value
        row["p_value_adjusted"] = float("nan") if missing else p_adjusted

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
    if not TOPIC_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Topic weights input not found: {TOPIC_WEIGHTS_PATH}. "
            "Run the prompt pipeline first to create news_posts_topic_weights.csv."
        )

    news_posts = pd.read_csv(
        CATEGORIZED_NEWS_PATH, usecols=["item_id", "week_id"]
    )
    topic_weights = pd.read_csv(TOPIC_WEIGHTS_PATH)
    polls_weekly = pd.read_csv(POLLS_WEEKLY_PATH)
    counts = weekly_post_counts(news_posts, topic_weights)
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
