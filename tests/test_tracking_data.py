"""Tests for tracking-data integration: new NBAApiClient methods,
FeatureEngine new features, and FormulaLayer tracking-aware formulas."""
from __future__ import annotations

import pytest

from src.features.feature_engine import FeatureEngine, _previous_season
from src.formula.formula_layer import FormulaLayer


# ---------------------------------------------------------------------------
# Helpers — mock NBA API clients
# ---------------------------------------------------------------------------

_BASE_STATS = {
    "gp": 70,
    "min": 34.0,
    "pts": 22.0,
    "fga": 17.0,
    "fgm": 8.0,
    "fg_pct": 0.471,
    "fg3a": 6.0,
    "fg3m": 2.5,
    "fg3_pct": 0.417,
    "fta": 4.5,
    "ftm": 3.8,
    "ft_pct": 0.844,
    "oreb": 0.8,
    "dreb": 4.5,
    "reb": 5.3,
    "ast": 7.1,
    "stl": 1.5,
    "blk": 0.4,
    "tov": 3.2,
    "pf": 2.1,
    "plus_minus": 6.5,
}

_BASE_INFO = {
    "position": "Guard",
    "height": "6-3",
    "weight": "195",
    "team_id": 1610612749,
    "team_abbreviation": "MIL",
}

_SHOT_CHART = [
    {
        "shot_zone_basic": "Restricted Area",
        "shot_zone_area": "Center(C)",
        "shot_zone_range": "Less Than 8 ft.",
        "shot_made_flag": 1,
        "loc_x": 5,
        "loc_y": 5,
        "shot_type": "2PT Field Goal",
        "action_type": "Driving Layup Shot",
    },
    {
        "shot_zone_basic": "Above the Break 3",
        "shot_zone_area": "Center(C)",
        "shot_zone_range": "24+ ft.",
        "shot_made_flag": 0,
        "loc_x": 10,
        "loc_y": 240,
        "shot_type": "3PT Field Goal",
        "action_type": "Jump Shot",
    },
]

_LEAGUE_AVERAGES = [
    {"PTS": 10.0, "AST": 3.0, "REB": 5.0, "STL": 1.0, "BLK": 0.5,
     "FG3A": 3.0, "FGA": 10.0, "FTA": 2.5, "TOV": 2.0},
    {"PTS": 20.0, "AST": 6.0, "REB": 7.0, "STL": 1.5, "BLK": 1.0,
     "FG3A": 5.0, "FGA": 15.0, "FTA": 4.0, "TOV": 3.0},
]

_PLAY_TYPES = {
    "iso_freq": 0.08,
    "pnr_ball_freq": 0.18,
    "pnr_roll_freq": 0.05,
    "post_up_freq": 0.03,
    "spot_up_freq": 0.20,
    "handoff_freq": 0.04,
    "cut_freq": 0.10,
    "off_screen_freq": 0.06,
    "transition_freq": 0.12,
    "putback_freq": 0.02,
}

_TRACKING_SHOTS = {
    "catch_shoot_fga": 120.0,
    "pull_up_fga": 280.0,
    "total_tracked_fga": 400.0,
    "avg_dribbles_before_shot": 2.5,
}

_HUSTLE = {
    "deflections": 2.1,
    "contested_shots_2pt": 1.8,
    "contested_shots_3pt": 1.2,
    "charges_drawn": 0.15,
    "loose_balls_recovered": 0.4,
    "screen_assists": 0.3,
    "gp": 70.0,
}

_PASSING = {
    "passes_made": 42.0,
    "potential_assists": 8.5,
    "ast_adjust": 5.2,
}


class MockClientWithTracking:
    """Full mock client that returns all tracking data."""

    def get_player_info(self, player_id: int) -> dict:
        return _BASE_INFO

    def get_player_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return _BASE_STATS

    def get_shot_chart(self, player_id: int, season: str = "2024-25") -> list:
        return _SHOT_CHART

    def get_league_averages(self, season: str = "2024-25") -> list:
        return _LEAGUE_AVERAGES

    def get_play_types(self, player_id: int, season: str = "2024-25") -> dict:
        return _PLAY_TYPES

    def get_tracking_shots(self, player_id: int, season: str = "2024-25") -> dict:
        return _TRACKING_SHOTS

    def get_hustle_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return _HUSTLE

    def get_passing_tracking(self, player_id: int, season: str = "2024-25") -> dict:
        return _PASSING


class MockClientNoTracking:
    """Mock client that returns empty dicts for all tracking calls (graceful degradation)."""

    def get_player_info(self, player_id: int) -> dict:
        return _BASE_INFO

    def get_player_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return _BASE_STATS

    def get_shot_chart(self, player_id: int, season: str = "2024-25") -> list:
        return _SHOT_CHART

    def get_league_averages(self, season: str = "2024-25") -> list:
        return _LEAGUE_AVERAGES

    def get_play_types(self, player_id: int, season: str = "2024-25") -> dict:
        return {}

    def get_tracking_shots(self, player_id: int, season: str = "2024-25") -> dict:
        return {}

    def get_hustle_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return {}

    def get_passing_tracking(self, player_id: int, season: str = "2024-25") -> dict:
        return {}


class MockClientTrackingRaises:
    """Mock client where all tracking calls raise exceptions."""

    def get_player_info(self, player_id: int) -> dict:
        return _BASE_INFO

    def get_player_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return _BASE_STATS

    def get_shot_chart(self, player_id: int, season: str = "2024-25") -> list:
        return _SHOT_CHART

    def get_league_averages(self, season: str = "2024-25") -> list:
        return _LEAGUE_AVERAGES

    def get_play_types(self, player_id: int, season: str = "2024-25") -> dict:
        raise RuntimeError("API unavailable")

    def get_tracking_shots(self, player_id: int, season: str = "2024-25") -> dict:
        raise RuntimeError("API unavailable")

    def get_hustle_stats(self, player_id: int, season: str = "2024-25") -> dict:
        raise RuntimeError("API unavailable")

    def get_passing_tracking(self, player_id: int, season: str = "2024-25") -> dict:
        raise RuntimeError("API unavailable")


# ---------------------------------------------------------------------------
# Tests: NBAApiClient new method return shapes (unit-tested via mock)
# ---------------------------------------------------------------------------


class TestNBAApiClientNewMethodShapes:
    """Validate the expected interface/shape of the new client methods."""

    def test_get_play_types_returns_dict_with_freq_keys(self):
        client = MockClientWithTracking()
        result = client.get_play_types(1)
        assert isinstance(result, dict)
        expected_keys = {
            "iso_freq", "pnr_ball_freq", "pnr_roll_freq", "post_up_freq",
            "spot_up_freq", "handoff_freq", "cut_freq", "off_screen_freq",
            "transition_freq", "putback_freq",
        }
        assert expected_keys.issubset(result.keys())

    def test_get_play_types_freq_values_in_0_1(self):
        client = MockClientWithTracking()
        result = client.get_play_types(1)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_get_tracking_shots_returns_expected_keys(self):
        client = MockClientWithTracking()
        result = client.get_tracking_shots(1)
        assert "catch_shoot_fga" in result
        assert "pull_up_fga" in result
        assert "total_tracked_fga" in result
        assert "avg_dribbles_before_shot" in result

    def test_get_hustle_stats_returns_expected_keys(self):
        client = MockClientWithTracking()
        result = client.get_hustle_stats(1)
        assert "deflections" in result
        assert "contested_shots_2pt" in result
        assert "contested_shots_3pt" in result
        assert "charges_drawn" in result
        assert "screen_assists" in result

    def test_get_passing_tracking_returns_expected_keys(self):
        client = MockClientWithTracking()
        result = client.get_passing_tracking(1)
        assert "passes_made" in result
        assert "potential_assists" in result
        assert "ast_adjust" in result

    def test_all_methods_return_empty_on_no_tracking(self):
        client = MockClientNoTracking()
        assert client.get_play_types(1) == {}
        assert client.get_tracking_shots(1) == {}
        assert client.get_hustle_stats(1) == {}
        assert client.get_passing_tracking(1) == {}


# ---------------------------------------------------------------------------
# Tests: FeatureEngine new features
# ---------------------------------------------------------------------------


class TestFeatureEngineTrackingFeatures:
    """Ensure FeatureEngine emits correct new features."""

    def _build(self, client) -> dict:
        engine = FeatureEngine(client)
        return engine.build_features(1, season="2024-25")

    def test_play_type_features_present_when_data_available(self):
        f = self._build(MockClientWithTracking())
        assert "playtype_iso_freq" in f
        assert "playtype_pnr_ball_freq" in f
        assert "playtype_pnr_roll_freq" in f
        assert "playtype_post_up_freq" in f
        assert "playtype_spot_up_freq" in f
        assert "playtype_cut_freq" in f
        assert "playtype_transition_freq" in f
        assert "playtype_handoff_freq" in f

    def test_play_type_features_have_real_values_when_data_available(self):
        f = self._build(MockClientWithTracking())
        assert f["playtype_iso_freq"] == pytest.approx(0.08)
        assert f["playtype_spot_up_freq"] == pytest.approx(0.20)
        assert f["playtype_pnr_ball_freq"] == pytest.approx(0.18)

    def test_play_type_features_sentinel_when_no_data(self):
        f = self._build(MockClientNoTracking())
        assert f["playtype_iso_freq"] == -1.0
        assert f["playtype_spot_up_freq"] == -1.0
        assert f["playtype_pnr_ball_freq"] == -1.0

    def test_tracking_shot_features_present(self):
        f = self._build(MockClientWithTracking())
        assert "tracking_catch_shoot_fga_pct" in f
        assert "tracking_pull_up_fga_pct" in f
        assert "tracking_avg_dribbles_before_shot" in f

    def test_tracking_shot_pcts_correct(self):
        f = self._build(MockClientWithTracking())
        # 120/400 = 0.30, 280/400 = 0.70
        assert f["tracking_catch_shoot_fga_pct"] == pytest.approx(0.30)
        assert f["tracking_pull_up_fga_pct"] == pytest.approx(0.70)
        assert f["tracking_avg_dribbles_before_shot"] == pytest.approx(2.5)

    def test_tracking_shot_features_sentinel_when_no_data(self):
        f = self._build(MockClientNoTracking())
        assert f["tracking_catch_shoot_fga_pct"] == -1.0
        assert f["tracking_pull_up_fga_pct"] == -1.0
        assert f["tracking_avg_dribbles_before_shot"] == -1.0

    def test_hustle_features_present(self):
        f = self._build(MockClientWithTracking())
        assert "hustle_deflections_pg" in f
        assert "hustle_contested_shots_pg" in f
        assert "hustle_charges_drawn_pg" in f
        assert "hustle_screen_assists_pg" in f
        assert "hustle_loose_balls_pg" in f

    def test_hustle_features_correct_values(self):
        f = self._build(MockClientWithTracking())
        assert f["hustle_deflections_pg"] == pytest.approx(2.1)
        assert f["hustle_contested_shots_pg"] == pytest.approx(3.0)  # 1.8 + 1.2
        assert f["hustle_charges_drawn_pg"] == pytest.approx(0.15)

    def test_hustle_features_sentinel_when_no_data(self):
        f = self._build(MockClientNoTracking())
        assert f["hustle_deflections_pg"] == -1.0
        assert f["hustle_contested_shots_pg"] == -1.0
        assert f["hustle_charges_drawn_pg"] == -1.0

    def test_passing_features_present(self):
        f = self._build(MockClientWithTracking())
        assert "tracking_passes_made_pg" in f
        assert "tracking_potential_ast_pg" in f
        assert "tracking_ast_to_pass_pct" in f

    def test_passing_features_correct_values(self):
        f = self._build(MockClientWithTracking())
        assert f["tracking_passes_made_pg"] == pytest.approx(42.0)
        assert f["tracking_potential_ast_pg"] == pytest.approx(8.5)
        # ast_adjust / passes_made = 5.2 / 42.0
        assert f["tracking_ast_to_pass_pct"] == pytest.approx(5.2 / 42.0)

    def test_passing_features_sentinel_when_no_data(self):
        f = self._build(MockClientNoTracking())
        assert f["tracking_passes_made_pg"] == -1.0
        assert f["tracking_potential_ast_pg"] == -1.0
        assert f["tracking_ast_to_pass_pct"] == -1.0

    def test_graceful_degradation_when_tracking_raises(self):
        """build_features must succeed and emit -1 sentinels even when tracking APIs raise."""
        f = self._build(MockClientTrackingRaises())
        # Should not raise
        assert f["playtype_iso_freq"] == -1.0
        assert f["tracking_catch_shoot_fga_pct"] == -1.0
        assert f["hustle_deflections_pg"] == -1.0
        assert f["tracking_potential_ast_pg"] == -1.0

    def test_all_new_feature_keys_present_with_tracking(self):
        f = self._build(MockClientWithTracking())
        new_keys = [
            "playtype_iso_freq", "playtype_pnr_ball_freq", "playtype_pnr_roll_freq",
            "playtype_post_up_freq", "playtype_spot_up_freq", "playtype_cut_freq",
            "playtype_transition_freq", "playtype_handoff_freq",
            "tracking_catch_shoot_fga_pct", "tracking_pull_up_fga_pct",
            "tracking_avg_dribbles_before_shot",
            "hustle_deflections_pg", "hustle_contested_shots_pg",
            "hustle_charges_drawn_pg", "hustle_screen_assists_pg", "hustle_loose_balls_pg",
            "tracking_potential_ast_pg", "tracking_passes_made_pg",
            "tracking_ast_to_pass_pct",
        ]
        for key in new_keys:
            assert key in f, f"Missing feature key: {key}"

    def test_all_new_feature_keys_present_without_tracking(self):
        f = self._build(MockClientNoTracking())
        new_keys = [
            "playtype_iso_freq", "playtype_pnr_ball_freq", "playtype_pnr_roll_freq",
            "playtype_post_up_freq", "playtype_spot_up_freq", "playtype_cut_freq",
            "playtype_transition_freq", "playtype_handoff_freq",
            "tracking_catch_shoot_fga_pct", "tracking_pull_up_fga_pct",
            "tracking_avg_dribbles_before_shot",
            "hustle_deflections_pg", "hustle_contested_shots_pg",
            "hustle_charges_drawn_pg", "hustle_screen_assists_pg", "hustle_loose_balls_pg",
            "tracking_potential_ast_pg", "tracking_passes_made_pg",
            "tracking_ast_to_pass_pct",
        ]
        for key in new_keys:
            assert key in f, f"Missing feature key: {key}"


# ---------------------------------------------------------------------------
# Tests: FormulaLayer tracking-aware formulas
# ---------------------------------------------------------------------------


def _base_features(position: str = "PG") -> dict:
    """Minimal feature dict without tracking data."""
    return {
        "position": position,
        "usg_pct_proxy": 0.25,
        "fga_per36": 14.0,
        "fg3a_rate": 0.40,
        "fta_rate": 0.30,
        "ast_per36": 7.0,
        "pts_per36": 22.0,
        "stl_per36": 1.5,
        "blk_per36": 0.3,
        "pf_per36": 2.2,
        "oreb_pct_proxy": 0.08,
        "zone_fga_rate_ra": 0.22,
        "zone_fga_rate_paint": 0.12,
        "zone_fga_rate_mid_left": 0.06,
        "zone_fga_rate_mid_center": 0.05,
        "zone_fga_rate_mid_right": 0.06,
        "sub_zone_distribution_close": {"left": 30.0, "middle": 40.0, "right": 30.0},
        "sub_zone_distribution_mid": {
            "left": 20.0, "left_center": 20.0, "center": 20.0,
            "right_center": 20.0, "right": 20.0,
        },
        "sub_zone_distribution_three": {
            "left": 20.0, "left_center": 20.0, "center": 20.0,
            "right_center": 20.0, "right": 20.0,
        },
        # All tracking sentinels = -1 (no tracking data)
        "playtype_iso_freq": -1.0,
        "playtype_pnr_ball_freq": -1.0,
        "playtype_pnr_roll_freq": -1.0,
        "playtype_post_up_freq": -1.0,
        "playtype_spot_up_freq": -1.0,
        "playtype_cut_freq": -1.0,
        "playtype_transition_freq": -1.0,
        "playtype_handoff_freq": -1.0,
        "tracking_catch_shoot_fga_pct": -1.0,
        "tracking_pull_up_fga_pct": -1.0,
        "tracking_avg_dribbles_before_shot": -1.0,
        "hustle_deflections_pg": -1.0,
        "hustle_contested_shots_pg": -1.0,
        "hustle_charges_drawn_pg": -1.0,
        "hustle_screen_assists_pg": -1.0,
        "hustle_loose_balls_pg": -1.0,
        "tracking_potential_ast_pg": -1.0,
        "tracking_passes_made_pg": -1.0,
        "tracking_ast_to_pass_pct": -1.0,
    }


def _with_tracking(base: dict, **overrides) -> dict:
    """Return a copy of base with tracking values set (non-sentinel)."""
    f = dict(base)
    f.update(overrides)
    return f


class TestFormulaLayerHasTrackingFlag:
    """Verify has_tracking flag correctly switches between proxy and real-data paths."""

    def test_no_tracking_uses_proxy_iso(self):
        fl = FormulaLayer()
        f_no = _base_features("PG")  # all sentinels
        t_no = fl.generate(f_no)

        f_yes = _with_tracking(f_no, playtype_iso_freq=0.05)
        t_yes = fl.generate(f_yes)

        # Low iso_freq should yield lower iso values than USG-based proxy for a high-USG player
        # Just ensure the paths produce different results
        assert t_no["iso_vs_elite_defender"] != t_yes["iso_vs_elite_defender"]

    def test_high_iso_freq_raises_iso_tendencies(self):
        fl = FormulaLayer()
        f_low = _with_tracking(_base_features("PG"), playtype_iso_freq=0.02)
        f_high = _with_tracking(_base_features("PG"), playtype_iso_freq=0.15)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["iso_vs_poor_defender"] > t_low["iso_vs_poor_defender"]

    def test_spot_up_freq_affects_spot_up_tendencies(self):
        fl = FormulaLayer()
        # playtype_iso_freq=0.0 enables has_tracking; vary spot_up_freq
        f_low = _with_tracking(_base_features("SG"), playtype_iso_freq=0.0,
                                playtype_spot_up_freq=0.05)
        f_high = _with_tracking(_base_features("SG"), playtype_iso_freq=0.0,
                                 playtype_spot_up_freq=0.25)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["spot_up_shot_three"] > t_low["spot_up_shot_three"]
        assert t_high["spot_up_shot_mid_range"] > t_low["spot_up_shot_mid_range"]

    def test_post_up_freq_affects_post_tendencies(self):
        fl = FormulaLayer()
        f_no = _base_features("PF")
        # playtype_iso_freq=0.0 enables has_tracking; vary post_up_freq
        f_post = _with_tracking(f_no, playtype_iso_freq=0.0, playtype_post_up_freq=0.20)
        t_no = fl.generate(f_no)
        t_post = fl.generate(f_post)
        assert t_post["post_up"] > t_no["post_up"]

    def test_pnr_roll_freq_affects_roll_vs_pop_for_bigs(self):
        fl = FormulaLayer()
        f_no = _base_features("C")
        # playtype_iso_freq=0.0 enables has_tracking; vary pnr_roll_freq
        f_roll = _with_tracking(f_no, playtype_iso_freq=0.0, playtype_pnr_roll_freq=0.20)
        t_no = fl.generate(f_no)
        t_roll = fl.generate(f_roll)
        # More roll man freq = higher roll_vs_pop value (rolls more in PnR)
        assert t_roll["roll_vs_pop"] > t_no["roll_vs_pop"]

    def test_transition_freq_affects_transition_spot_up(self):
        fl = FormulaLayer()
        f_low = _with_tracking(_base_features("SG"), playtype_iso_freq=0.0,
                                playtype_transition_freq=0.05)
        f_high = _with_tracking(_base_features("SG"), playtype_iso_freq=0.0,
                                 playtype_transition_freq=0.25)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["transition_spot_up"] > t_low["transition_spot_up"]

    def test_pull_up_pct_affects_drive_pull_up_tendencies(self):
        fl = FormulaLayer()
        # tracking_catch_shoot_fga_pct=0.0 enables has_tracking_shots; vary pull_up_pct
        f_low = _with_tracking(_base_features("PG"), tracking_catch_shoot_fga_pct=0.0,
                                tracking_pull_up_fga_pct=0.10)
        f_high = _with_tracking(_base_features("PG"), tracking_catch_shoot_fga_pct=0.0,
                                 tracking_pull_up_fga_pct=0.60)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["drive_pull_up_mid_range"] > t_low["drive_pull_up_mid_range"]

    def test_avg_dribbles_affects_creation_score(self):
        fl = FormulaLayer()
        # tracking_catch_shoot_fga_pct=0.0 enables has_tracking_shots; vary avg dribbles
        f_low = _with_tracking(_base_features("PG"), tracking_catch_shoot_fga_pct=0.0,
                                tracking_avg_dribbles_before_shot=0.5)
        f_high = _with_tracking(_base_features("PG"), tracking_catch_shoot_fga_pct=0.0,
                                 tracking_avg_dribbles_before_shot=4.0)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        # High dribble count = higher driving crossover tendency
        assert t_high["driving_crossover"] > t_low["driving_crossover"]

    def test_contested_shots_affects_contest_shot(self):
        fl = FormulaLayer()
        # hustle_deflections_pg=0.0 enables has_hustle; vary contested_shots
        f_low = _with_tracking(_base_features("SF"), hustle_deflections_pg=0.0,
                                hustle_contested_shots_pg=0.5)
        f_high = _with_tracking(_base_features("SF"), hustle_deflections_pg=0.0,
                                 hustle_contested_shots_pg=5.0)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["contest_shot"] > t_low["contest_shot"]

    def test_charges_drawn_affects_take_charge(self):
        fl = FormulaLayer()
        # hustle_deflections_pg=0.0 enables has_hustle; vary charges_drawn
        f_low = _with_tracking(_base_features("SG"), hustle_deflections_pg=0.0,
                                hustle_charges_drawn_pg=0.0)
        f_high = _with_tracking(_base_features("SG"), hustle_deflections_pg=0.0,
                                 hustle_charges_drawn_pg=0.5)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["take_charge"] > t_low["take_charge"]

    def test_deflections_affects_pass_interception(self):
        fl = FormulaLayer()
        f_low = _with_tracking(_base_features("PG"), hustle_deflections_pg=0.5)
        f_high = _with_tracking(_base_features("PG"), hustle_deflections_pg=4.0)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["pass_interception"] > t_low["pass_interception"]

    def test_potential_ast_affects_flashy_pass(self):
        fl = FormulaLayer()
        f_low = _with_tracking(_base_features("PG"), tracking_potential_ast_pg=2.0,
                                tracking_ast_to_pass_pct=0.05)
        f_high = _with_tracking(_base_features("PG"), tracking_potential_ast_pg=15.0,
                                 tracking_ast_to_pass_pct=0.05)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["flashy_pass"] > t_low["flashy_pass"]

    def test_backward_compat_no_tracking_features_produce_valid_tendencies(self):
        """generate() must work with the old feature set (no tracking keys) unchanged."""
        fl = FormulaLayer()
        f_old = {
            "position": "SG",
            "usg_pct_proxy": 0.22,
            "fga_per36": 13.0,
            "fg3a_rate": 0.38,
            "fta_rate": 0.28,
            "ast_per36": 4.5,
            "pts_per36": 19.0,
            "stl_per36": 1.2,
            "blk_per36": 0.3,
            "pf_per36": 2.3,
            "oreb_pct_proxy": 0.09,
            "zone_fga_rate_ra": 0.18,
            "zone_fga_rate_paint": 0.10,
            "zone_fga_rate_mid_left": 0.07,
            "zone_fga_rate_mid_center": 0.05,
            "zone_fga_rate_mid_right": 0.07,
            "sub_zone_distribution_close": {"left": 30.0, "middle": 40.0, "right": 30.0},
            "sub_zone_distribution_mid": {
                "left": 20.0, "left_center": 20.0, "center": 20.0,
                "right_center": 20.0, "right": 20.0,
            },
            "sub_zone_distribution_three": {
                "left": 20.0, "left_center": 20.0, "center": 20.0,
                "right_center": 20.0, "right": 20.0,
            },
            # No tracking keys at all — backward compat
        }
        tendencies = fl.generate(f_old)
        # Should produce 99 tendencies without error
        assert len(tendencies) == 99
        for name, val in tendencies.items():
            assert isinstance(val, float), f"{name} is not float"
            assert val >= 0, f"{name}={val} is negative"

    def test_all_tendencies_produced_with_tracking(self):
        fl = FormulaLayer()
        f = _with_tracking(
            _base_features("PG"),
            playtype_iso_freq=0.10,
            playtype_spot_up_freq=0.18,
            playtype_pnr_ball_freq=0.20,
            playtype_pnr_roll_freq=0.08,
            playtype_post_up_freq=0.05,
            playtype_transition_freq=0.14,
            playtype_handoff_freq=0.03,
            tracking_catch_shoot_fga_pct=0.30,
            tracking_pull_up_fga_pct=0.55,
            tracking_avg_dribbles_before_shot=2.8,
            hustle_deflections_pg=2.0,
            hustle_contested_shots_pg=3.0,
            hustle_charges_drawn_pg=0.1,
            tracking_potential_ast_pg=9.0,
            tracking_ast_to_pass_pct=0.10,
        )
        tendencies = fl.generate(f)
        assert len(tendencies) == 99
        for name, val in tendencies.items():
            assert val >= 0, f"{name}={val} is negative"

    def test_handoff_freq_affects_off_screen_tendencies(self):
        fl = FormulaLayer()
        # playtype_iso_freq=0.0 enables has_tracking; vary handoff_freq
        f_low = _with_tracking(_base_features("SG"), playtype_iso_freq=0.0,
                                playtype_handoff_freq=0.01)
        f_high = _with_tracking(_base_features("SG"), playtype_iso_freq=0.0,
                                 playtype_handoff_freq=0.12)
        t_low = fl.generate(f_low)
        t_high = fl.generate(f_high)
        assert t_high["off_screen_shot_three"] > t_low["off_screen_shot_three"]
        assert t_high["off_screen_shot_mid_range"] > t_low["off_screen_shot_mid_range"]


# ---------------------------------------------------------------------------
# Tests: _previous_season helper
# ---------------------------------------------------------------------------


class TestPreviousSeason:
    """Validate the _previous_season() helper function."""

    def test_standard_season(self):
        assert _previous_season("2024-25") == "2023-24"

    def test_season_2023_24(self):
        assert _previous_season("2023-24") == "2022-23"

    def test_season_2022_23(self):
        assert _previous_season("2022-23") == "2021-22"

    def test_invalid_format_returns_fallback(self):
        assert _previous_season("invalid") == "2023-24"

    def test_empty_string_returns_fallback(self):
        assert _previous_season("") == "2023-24"

    def test_century_boundary(self):
        assert _previous_season("2000-01") == "1999-00"


# ---------------------------------------------------------------------------
# Tests: Season fallback logic in FeatureEngine.build_features()
# ---------------------------------------------------------------------------


class MockClientFallback:
    """Mock client: primary season returns empty for tracking, fallback season returns data."""

    def __init__(self, primary_season: str, fallback_season: str) -> None:
        self._primary_season = primary_season
        self._fallback_season = fallback_season

    def get_player_info(self, player_id: int) -> dict:
        return _BASE_INFO

    def get_player_stats(self, player_id: int, season: str = "2024-25") -> dict:
        return _BASE_STATS

    def get_shot_chart(self, player_id: int, season: str = "2024-25") -> list:
        return _SHOT_CHART

    def get_league_averages(self, season: str = "2024-25") -> list:
        return _LEAGUE_AVERAGES

    def get_play_types(self, player_id: int, season: str = "2024-25") -> dict:
        if season == self._fallback_season:
            return _PLAY_TYPES
        return {}

    def get_tracking_shots(self, player_id: int, season: str = "2024-25") -> dict:
        if season == self._fallback_season:
            return _TRACKING_SHOTS
        return {}

    def get_hustle_stats(self, player_id: int, season: str = "2024-25") -> dict:
        if season == self._fallback_season:
            return _HUSTLE
        return {}

    def get_passing_tracking(self, player_id: int, season: str = "2024-25") -> dict:
        if season == self._fallback_season:
            return _PASSING
        return {}


class TestSeasonFallback:
    """Validate that FeatureEngine falls back to previous season when primary is empty."""

    def test_fallback_provides_play_types_when_primary_empty(self):
        """When primary season returns empty play_types, fallback season is used."""
        client = MockClientFallback(primary_season="2024-25", fallback_season="2023-24")
        engine = FeatureEngine(client)
        features = engine.build_features(1, season="2024-25")
        # Tracking data from fallback season should produce real values (not sentinel -1)
        assert features["playtype_iso_freq"] >= 0.0, (
            "Expected real play_type data from fallback season"
        )

    def test_fallback_provides_tracking_shots_when_primary_empty(self):
        """When primary season returns empty tracking_shots, fallback season is used."""
        client = MockClientFallback(primary_season="2024-25", fallback_season="2023-24")
        engine = FeatureEngine(client)
        features = engine.build_features(1, season="2024-25")
        assert features["tracking_catch_shoot_fga_pct"] >= 0.0, (
            "Expected real tracking_shots data from fallback season"
        )

    def test_fallback_provides_hustle_when_primary_empty(self):
        """When primary season returns empty hustle, fallback season is used."""
        client = MockClientFallback(primary_season="2024-25", fallback_season="2023-24")
        engine = FeatureEngine(client)
        features = engine.build_features(1, season="2024-25")
        assert features["hustle_deflections_pg"] >= 0.0, (
            "Expected real hustle data from fallback season"
        )

    def test_fallback_provides_passing_when_primary_empty(self):
        """When primary season returns empty passing, fallback season is used."""
        client = MockClientFallback(primary_season="2024-25", fallback_season="2023-24")
        engine = FeatureEngine(client)
        features = engine.build_features(1, season="2024-25")
        assert features["tracking_potential_ast_pg"] >= 0.0, (
            "Expected real passing data from fallback season"
        )

    def test_no_fallback_when_primary_has_data(self):
        """When primary season returns data, fallback season is NOT consulted."""
        # MockClientWithTracking always returns data for any season
        client = MockClientWithTracking()
        engine = FeatureEngine(client)
        features = engine.build_features(1, season="2024-25")
        # All tracking features should be available (>= 0)
        assert features["playtype_iso_freq"] >= 0.0
        assert features["tracking_catch_shoot_fga_pct"] >= 0.0
        assert features["hustle_deflections_pg"] >= 0.0
        assert features["tracking_potential_ast_pg"] >= 0.0

    def test_sentinel_values_when_both_seasons_empty(self):
        """When both primary and fallback return empty, sentinel (-1) values are set."""
        client = MockClientNoTracking()
        engine = FeatureEngine(client)
        features = engine.build_features(1, season="2024-25")
        assert features["playtype_iso_freq"] == -1.0
        assert features["tracking_catch_shoot_fga_pct"] == -1.0
        assert features["hustle_deflections_pg"] == -1.0
        assert features["tracking_potential_ast_pg"] == -1.0
