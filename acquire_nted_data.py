"""Acquire and prepare NFL player-game data for the NTED research question.

The output contains regular-season TE, WR, and RB player-game observations for
2019--2025. Source data are loaded from nflverse with nflreadpy.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "nted_player_games_2019_2025.parquet"
VALIDATION_PATH = DATA_DIR / "nted_validation_2019_2025.json"
SUMMARY_PATH = DATA_DIR / "nted_position_summary_2019_2025.csv"

SEASONS = list(range(2019, 2026))
POSITIONS = {"TE", "WR", "RB"}
COUNTING_STATS = [
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "carries",
    "rushing_yards",
    "rushing_tds",
]
BASELINE_STATS = [
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "carries",
]
PLAYER_REQUIRED = {
    "player_id",
    "player_display_name",
    "position",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
    *COUNTING_STATS,
}
SCHEDULE_REQUIRED = {
    "game_id",
    "season",
    "week",
    "game_type",
    "gameday",
    "weekday",
    "away_team",
    "home_team",
}


def fourth_sunday_in_october(year: int) -> date:
    """Return the fourth Sunday in October for a calendar year."""
    first = date(year, 10, 1)
    days_until_sunday = (6 - first.weekday()) % 7
    return date(year, 10, 1 + days_until_sunday + 21)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def load_sources(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download weekly player statistics and schedules with nflreadpy."""
    import nflreadpy as nfl

    player_stats = nfl.load_player_stats(
        seasons, summary_level="week"
    ).to_pandas()
    schedules = nfl.load_schedules(seasons).to_pandas()
    return player_stats, schedules


def join_and_prepare(
    player_stats: pd.DataFrame, schedules: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate, join, filter, and feature-engineer the source tables."""
    _require_columns(player_stats, PLAYER_REQUIRED, "player statistics")
    _require_columns(schedules, SCHEDULE_REQUIRED, "schedules")

    duplicate_schedule_ids = int(schedules["game_id"].duplicated().sum())
    if duplicate_schedule_ids:
        raise ValueError(
            f"Schedule game_id must be unique; found {duplicate_schedule_ids} duplicates."
        )

    schedule_for_join = schedules[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "gameday",
            "weekday",
            "away_team",
            "home_team",
        ]
    ].rename(
        columns={
            "season": "schedule_season",
            "week": "schedule_week",
            "gameday": "game_date",
            "weekday": "schedule_weekday",
        }
    )

    joined = player_stats.merge(
        schedule_for_join,
        on="game_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    unmatched_rows = int((joined["_merge"] != "both").sum())
    joined = joined.drop(columns="_merge")

    matched = joined["schedule_season"].notna()
    season_mismatches = int(
        (
            matched
            & joined["season"].ne(joined["schedule_season"])
        ).sum()
    )
    week_mismatches = int(
        (matched & joined["week"].ne(joined["schedule_week"])).sum()
    )
    if unmatched_rows or season_mismatches or week_mismatches:
        raise ValueError(
            "Join validation failed: "
            f"unmatched={unmatched_rows}, season mismatches={season_mismatches}, "
            f"week mismatches={week_mismatches}."
        )

    joined["game_date"] = pd.to_datetime(
        joined["game_date"], errors="raise"
    ).dt.normalize()
    joined["season"] = joined["season"].astype("int16")
    joined["week"] = joined["week"].astype("int8")

    regular = joined.loc[
        joined["season_type"].eq("REG") & joined["game_type"].eq("REG")
    ].copy()

    # A player appearance row with a missing box-score counting value is treated
    # as zero. The number of replacements is retained in the validation report.
    missing_counting_values = {
        column: int(regular[column].isna().sum()) for column in COUNTING_STATS
    }
    regular[COUNTING_STATS] = (
        regular[COUNTING_STATS].apply(pd.to_numeric, errors="raise").fillna(0)
    )

    # Calculate shares before the position filter so that the denominator is all
    # recorded team targets, including targets to positions outside TE/WR/RB.
    team_targets = (
        regular.groupby(["game_id", "team"], as_index=False)
        .agg(team_targets=("targets", "sum"))
    )
    te_targets = (
        regular.assign(
            te_targets=regular["targets"].where(
                regular["position"].eq("TE"), 0
            )
        )
        .groupby(["game_id", "team"], as_index=False)
        .agg(team_te_targets=("te_targets", "sum"))
    )
    team_targets = team_targets.merge(
        te_targets,
        on=["game_id", "team"],
        how="left",
        validate="one_to_one",
    )
    team_targets["team_te_target_share"] = np.where(
        team_targets["team_targets"].gt(0),
        team_targets["team_te_targets"] / team_targets["team_targets"],
        np.nan,
    )
    zero_target_team_games = int(team_targets["team_targets"].eq(0).sum())

    analysis = regular.loc[regular["position"].isin(POSITIONS)].copy()
    analysis = analysis.merge(
        team_targets,
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    analysis["nted_date"] = pd.to_datetime(
        analysis["season"].map(
            lambda year: fourth_sunday_in_october(int(year))
        )
    )
    analysis["is_sunday"] = analysis["game_date"].dt.day_of_week.eq(6)
    analysis["is_nted"] = analysis["game_date"].eq(analysis["nted_date"])
    analysis["is_other_sunday"] = analysis["is_sunday"] & ~analysis["is_nted"]

    nted_week_by_season = (
        analysis.loc[analysis["is_nted"], ["season", "week"]]
        .drop_duplicates()
        .set_index("season")["week"]
    )
    if not nted_week_by_season.index.is_unique:
        raise ValueError("A season has more than one NFL week on NTED.")
    missing_nted_seasons = sorted(set(SEASONS).difference(nted_week_by_season.index))
    if missing_nted_seasons:
        raise ValueError(f"No NTED player rows for seasons: {missing_nted_seasons}")
    analysis["nted_week"] = analysis["season"].map(nted_week_by_season).astype("int8")
    analysis["week_relative_to_nted"] = (
        analysis["week"] - analysis["nted_week"]
    ).astype("int8")

    analysis = analysis.sort_values(
        ["season", "player_id", "game_date", "game_id", "team"]
    ).reset_index(drop=True)
    grouped_players = analysis.groupby(["season", "player_id"], sort=False)
    analysis["prior_player_games"] = grouped_players.cumcount().astype("int8")
    for column in BASELINE_STATS:
        cumulative_before_game = grouped_players[column].cumsum() - analysis[column]
        analysis[f"prior_avg_{column}"] = (
            cumulative_before_game
            / analysis["prior_player_games"].replace(0, np.nan)
        )

    duplicate_player_games = int(
        analysis.duplicated(["game_id", "team", "player_id"]).sum()
    )
    if duplicate_player_games:
        raise ValueError(
            "Expected one row per player-team-game; found "
            f"{duplicate_player_games} duplicate rows."
        )

    key_columns = ["game_id", "player_id", "team", "position", "game_date"]
    missing_key_values = {
        column: int(analysis[column].isna().sum()) for column in key_columns
    }
    if any(missing_key_values.values()):
        raise ValueError(f"Missing final key values: {missing_key_values}")

    keep_columns = [
        "season",
        "week",
        "game_id",
        "game_date",
        "schedule_weekday",
        "away_team",
        "home_team",
        "player_id",
        "player_display_name",
        "position",
        "team",
        "opponent_team",
        *COUNTING_STATS,
        "is_sunday",
        "nted_date",
        "nted_week",
        "is_nted",
        "is_other_sunday",
        "week_relative_to_nted",
        "team_targets",
        "team_te_targets",
        "team_te_target_share",
        "prior_player_games",
        *[f"prior_avg_{column}" for column in BASELINE_STATS],
    ]
    analysis = analysis[keep_columns].sort_values(
        ["season", "week", "game_id", "team", "position", "player_id"]
    ).reset_index(drop=True)

    nted_dates = {
        str(year): fourth_sunday_in_october(year).isoformat()
        for year in SEASONS
    }
    validation: dict[str, object] = {
        "source_accessed_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": SEASONS,
        "source_player_rows": int(len(player_stats)),
        "source_player_columns": int(player_stats.shape[1]),
        "source_player_rows_by_season": {
            str(season): int(count)
            for season, count in player_stats.groupby("season").size().items()
        },
        "source_schedule_rows": int(len(schedules)),
        "source_schedule_columns": int(schedules.shape[1]),
        "source_schedule_rows_by_season": {
            str(season): int(count)
            for season, count in schedules.groupby("season").size().items()
        },
        "schedule_duplicate_game_ids": duplicate_schedule_ids,
        "unmatched_player_rows": unmatched_rows,
        "season_mismatches_after_join": season_mismatches,
        "week_mismatches_after_join": week_mismatches,
        "regular_season_rows_all_positions": int(len(regular)),
        "analysis_rows": int(len(analysis)),
        "analysis_columns": int(analysis.shape[1]),
        "duplicate_player_team_games": duplicate_player_games,
        "missing_final_key_values": missing_key_values,
        "missing_counting_values_replaced_with_zero": missing_counting_values,
        "team_games_with_zero_target_denominator": zero_target_team_games,
        "player_rows_with_missing_team_target_share": int(
            analysis["team_te_target_share"].isna().sum()
        ),
        "first_appearance_rows_with_no_prior_baseline": int(
            analysis["prior_player_games"].eq(0).sum()
        ),
        "nted_dates": nted_dates,
        "nted_games_by_season": {
            str(season): int(count)
            for season, count in (
                analysis.loc[analysis["is_nted"], ["season", "game_id"]]
                .drop_duplicates()
                .groupby("season")
                .size()
                .items()
            )
        },
        "analysis_rows_by_position": {
            str(position): int(count)
            for position, count in analysis.groupby("position").size().items()
        },
        "analysis_rows_by_season": {
            str(season): int(count)
            for season, count in analysis.groupby("season").size().items()
        },
    }
    return analysis, validation


def make_position_summary(analysis: pd.DataFrame) -> pd.DataFrame:
    """Create a small descriptive summary, not an inferential analysis."""
    sunday_rows = analysis.loc[
        analysis["is_nted"] | analysis["is_other_sunday"]
    ].copy()
    sunday_rows["sunday_type"] = np.where(
        sunday_rows["is_nted"], "NTED", "Other Sunday"
    )
    return (
        sunday_rows.groupby(["position", "sunday_type"], as_index=False)
        .agg(
            player_games=("player_id", "size"),
            games=("game_id", "nunique"),
            mean_targets=("targets", "mean"),
            mean_receptions=("receptions", "mean"),
            mean_receiving_yards=("receiving_yards", "mean"),
            mean_receiving_tds=("receiving_tds", "mean"),
            mean_carries=("carries", "mean"),
        )
        .sort_values(["position", "sunday_type"])
        .reset_index(drop=True)
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    player_stats, schedules = load_sources(SEASONS)
    analysis, validation = join_and_prepare(player_stats, schedules)
    summary = make_position_summary(analysis)

    analysis.to_parquet(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.4f")
    VALIDATION_PATH.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(validation, indent=2))
    print(f"\nSaved {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved {SUMMARY_PATH.relative_to(PROJECT_DIR)}")
    print(f"Saved {VALIDATION_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
