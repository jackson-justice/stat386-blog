import unittest
from datetime import date

import pandas as pd

from acquire_nted_data import fourth_sunday_in_october, join_and_prepare


class TestCalendarRule(unittest.TestCase):
    def test_fourth_sunday_dates(self):
        expected = {
            2019: date(2019, 10, 27),
            2020: date(2020, 10, 25),
            2021: date(2021, 10, 24),
            2022: date(2022, 10, 23),
            2023: date(2023, 10, 22),
            2024: date(2024, 10, 27),
            2025: date(2025, 10, 26),
        }
        self.assertEqual(
            {year: fourth_sunday_in_october(year) for year in expected},
            expected,
        )


class TestPreparationRules(unittest.TestCase):
    def setUp(self):
        rows = []
        schedules = []
        for season in range(2019, 2026):
            nted = fourth_sunday_in_october(season)
            game_id = f"{season}_08_A_B"
            schedules.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": 8,
                    "game_type": "REG",
                    "gameday": nted.isoformat(),
                    "weekday": "Sunday",
                    "away_team": "A",
                    "home_team": "B",
                }
            )
            for player_id, position, targets in [
                (f"te-{season}", "TE", 2),
                (f"wr-{season}", "WR", 6),
                (f"qb-{season}", "QB", 2),
            ]:
                rows.append(
                    {
                        "player_id": player_id,
                        "player_display_name": player_id,
                        "position": position,
                        "season": season,
                        "week": 8,
                        "season_type": "REG",
                        "game_id": game_id,
                        "team": "A",
                        "opponent_team": "B",
                        "targets": targets,
                        "receptions": targets,
                        "receiving_yards": targets * 10,
                        "receiving_tds": 0,
                        "carries": 0,
                        "rushing_yards": 0,
                        "rushing_tds": 0,
                    }
                )
        self.player_stats = pd.DataFrame(rows)
        self.schedules = pd.DataFrame(schedules)

    def test_nted_and_target_share_use_all_positions(self):
        analysis, validation = join_and_prepare(
            self.player_stats, self.schedules
        )
        self.assertTrue(analysis["is_nted"].all())
        self.assertEqual(set(analysis["position"]), {"TE", "WR"})
        self.assertTrue((analysis["team_te_target_share"] == 0.2).all())
        self.assertTrue(analysis["prior_avg_targets"].isna().all())
        self.assertEqual(validation["unmatched_player_rows"], 0)

    def test_duplicate_schedule_game_id_fails(self):
        duplicated = pd.concat(
            [self.schedules, self.schedules.iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "game_id must be unique"):
            join_and_prepare(self.player_stats, duplicated)


if __name__ == "__main__":
    unittest.main()
