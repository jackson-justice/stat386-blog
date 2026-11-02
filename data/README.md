# Data files

The `nted_*_2019_2025` files were derived from the
[nflverse-data](https://github.com/nflverse/nflverse-data) weekly player
statistics and schedules using `nflreadpy`. The latest acquisition run was on
2026-11-02 Mountain Time and UTC.

The source tables were joined, restricted to 2019--2025 regular-season TE,
WR, and RB player-games, and modified with calendar, prior player-season, and
team target-share features. See `../acquire_nted_data.py` for the complete
transformation and `nted_validation_2019_2025.json` for validation results.

At the time of acquisition, nflverse identified these data under the
[Creative Commons Attribution 4.0 International
License](https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md).
Credit: nflverse. Check the applicable nflverse release and license again
before redistributing these files. The data are provided without warranty.

`team_week_records.csv` belongs to the existing tutorial and was not changed
by the NTED acquisition workflow.
