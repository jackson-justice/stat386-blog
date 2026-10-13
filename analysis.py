from pathlib import Path

import matplotlib
import pandas as pd


matplotlib.use("Agg")


PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "team_week_records.csv"
FIGURE_PATH = PROJECT_DIR / "figures" / "playoff_rate_after_six_games.png"
CHECKPOINTS = [2, 4, 6, 8]


# Load the cleaned Statigami data. Each row represents one team in one
# regular-season week.
games = pd.read_csv(DATA_PATH)

required_columns = {
    "season",
    "week",
    "team",
    "games_played",
    "wins",
    "losses",
    "ties",
    "record",
    "final_wins",
    "final_losses",
    "made_playoffs",
}
missing_columns = required_columns.difference(games.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


# Select the checkpoints and keep one row per team-season at each checkpoint.
# A bye can otherwise create a second row with the same games_played value.
checkpoint_rows = (
    games.loc[games["games_played"].isin(CHECKPOINTS)]
    .sort_values(["season", "team", "week"])
    .drop_duplicates(["season", "team", "games_played"])
    .copy()
)

team_seasons = games[["season", "team"]].drop_duplicates().shape[0]
rows_per_checkpoint = checkpoint_rows.groupby("games_played").size()
if not rows_per_checkpoint.eq(team_seasons).all():
    raise ValueError("A checkpoint does not have exactly one row per team-season.")


# Count all team-seasons and successful playoff teams for every record.
record_summary = (
    checkpoint_rows.groupby(
        ["games_played", "wins", "losses", "ties", "record"],
        as_index=False,
    )
    .agg(
        team_seasons=("team", "size"),
        playoff_teams=("made_playoffs", "sum"),
    )
)

record_summary["playoff_rate"] = (
    record_summary["playoff_teams"] / record_summary["team_seasons"]
)
record_summary["win_pct_at_checkpoint"] = (
    record_summary["wins"] + 0.5 * record_summary["ties"]
) / record_summary["games_played"]


# At each checkpoint, find the worst record from which at least one team
# recovered to make the playoffs.
rare_paths = (
    record_summary.loc[record_summary["playoff_teams"] > 0]
    .sort_values(
        ["games_played", "win_pct_at_checkpoint", "wins", "ties"]
    )
    .groupby("games_played", as_index=False)
    .head(1)
    .loc[
        :,
        [
            "games_played",
            "record",
            "team_seasons",
            "playoff_teams",
            "playoff_rate",
        ],
    ]
    .reset_index(drop=True)
)


# Trace the 1-5 grouped result back to the individual playoff teams.
one_five_playoff_teams = (
    checkpoint_rows.loc[
        (checkpoint_rows["games_played"] == 6)
        & (checkpoint_rows["record"] == "1-5")
        & checkpoint_rows["made_playoffs"],
        ["season", "team", "record", "final_wins", "final_losses"],
    ]
    .sort_values(["season", "team"])
    .reset_index(drop=True)
)


# Validate the two central tutorial results.
one_five_summary = record_summary.loc[
    (record_summary["games_played"] == 6)
    & (record_summary["record"] == "1-5")
].iloc[0]

assert int(one_five_summary["team_seasons"]) == 102
assert int(one_five_summary["playoff_teams"]) == 3
assert abs(one_five_summary["playoff_rate"] - 3 / 102) < 1e-12
assert set(
    one_five_playoff_teams[["season", "team"]].itertuples(
        index=False, name=None
    )
) == {(2015, "KC"), (2018, "IND"), (2020, "WAS")}


# Visualize tie-free records after six games. The 1-5 bar is highlighted.
six_game_summary = (
    record_summary.loc[
        (record_summary["games_played"] == 6)
        & (record_summary["ties"] == 0)
    ]
    .sort_values("wins")
    .copy()
)
six_game_summary["playoff_rate_pct"] = (
    six_game_summary["playoff_rate"] * 100
)

bar_colors = [
    "#E31837" if record == "1-5" else "#9AA0A6"
    for record in six_game_summary["record"]
]

ax = six_game_summary.plot.bar(
    x="record",
    y="playoff_rate_pct",
    color=bar_colors,
    legend=False,
    figsize=(9, 5),
    width=0.75,
)
ax.set_title("NFL Playoff Rate by Record After Six Games, 1999-2025")
ax.set_xlabel("Record after six games")
ax.set_ylabel("Teams that made the playoffs (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.25)

for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    if tick.get_text() == "1-5":
        tick.set_color("#E31837")
        tick.set_fontweight("bold")

for container in ax.containers:
    ax.bar_label(
        container,
        labels=[f"{value:.1f}%" for value in six_game_summary["playoff_rate_pct"]],
        padding=3,
        fontsize=9,
    )

figure = ax.get_figure()
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")


print("Rows per checkpoint:")
print(rows_per_checkpoint.to_string())
print("\nRarest successful path at each checkpoint:")
print(
    rare_paths.to_string(
        index=False,
        formatters={"playoff_rate": "{:.1%}".format},
    )
)
print("\nTeams that started 1-5 and made the playoffs:")
print(one_five_playoff_teams.to_string(index=False))
print(f"\nSaved chart to {FIGURE_PATH.relative_to(PROJECT_DIR)}")
