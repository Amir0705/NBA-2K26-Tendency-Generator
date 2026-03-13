"""Tests for play-style priority scoring."""
from __future__ import annotations

from src.play_styles.scorer import PlayStyleScorer


class TestPlayStyleScorer:
    def test_usage_tiers_determine_style_count(self):
        scorer = PlayStyleScorer()
        assert scorer.score({"position": "PG", "usage_rate": 31.0}).target_count == 4
        assert scorer.score({"position": "PG", "usage_rate": 27.0}).target_count == 3
        assert scorer.score({"position": "PG", "usage_rate": 20.0}).target_count == 2
        assert scorer.score({"position": "PG", "usage_rate": 14.0}).target_count == 2

    def test_conflicting_pnr_roles_do_not_coexist(self):
        scorer = PlayStyleScorer()
        features = {
            "position": "PG",
            "usage_rate": 32.0,
            "assists": 9.5,
            "pick_and_roll_ball_handler_possessions": 14.0,
            "pick_and_roll_rollman_possessions": 12.0,
        }
        result = scorer.score(features)
        pnr_selected = [
            s for s in result.priorities
            if s in {
                "Pick and roll rollman",
                "Pick and roll wing",
                "Pick and roll point",
                "Pick and roll ball handler",
            }
        ]
        assert len(pnr_selected) <= 1

    def test_center_deprioritizes_isolation_when_iso_volume_low(self):
        scorer = PlayStyleScorer()
        features = {
            "position": "C",
            "usage_rate": 30.5,
            "assists": 8.5,
            "handoff_possessions": 5.4,
            "post_up_possessions": 5.1,
            "pick_and_roll_ball_handler_possessions": 3.2,
            "pick_and_roll_rollman_possessions": 4.8,
            "isolation_possessions": 1.2,
        }
        result = scorer.score(features)
        assert "Isolation" not in result.priorities

    def test_weights_sum_to_one_for_selected_styles(self):
        scorer = PlayStyleScorer()
        features = {
            "position": "SF",
            "usage_rate": 28.0,
            "fg3a_per_game": 7.2,
            "fg3_pct": 0.39,
            "midrange_attempts": 4.8,
            "isolation_possessions": 4.4,
            "pick_and_roll_ball_handler_possessions": 5.2,
        }
        result = scorer.score(features)
        assert len(result.priorities) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_creator_wing_profile_not_forced_into_three_point_priority(self):
        scorer = PlayStyleScorer()
        # LeBron-like profile: high usage + assists + creation load.
        features = {
            "position": "SF",
            "usage_rate": 29.5,
            "assists": 8.3,
            "fga_per_game": 19.2,
            "fg3a_per_game": 5.3,
            "fg3_pct": 0.39,
            "fg3a_rate": 0.28,
            "zone_fga_rate_ra": 0.26,
            "zone_fga_rate_paint": 0.22,
            "isolation_possessions": 3.5,
            "pick_and_roll_ball_handler_possessions": 6.2,
            "post_up_possessions": 2.4,
            "handoff_possessions": 2.2,
        }
        result = scorer.score(features)
        assert result.priorities[0] != "3pt"

    def test_tatum_like_wing_creator_not_three_point_primary(self):
        scorer = PlayStyleScorer()
        features = {
            "position": "SF",
            "usage_rate": 31.0,
            "assists": 5.0,
            "fga_per_game": 20.6,
            "fg3a_per_game": 10.1,
            "fg3_pct": 0.38,
            "fg3a_rate": 0.49,
            "zone_fga_rate_ra": 0.17,
            "zone_fga_rate_paint": 0.13,
            "isolation_possessions": 3.2,
            "pick_and_roll_ball_handler_possessions": 4.6,
            "handoff_possessions": 1.6,
        }
        result = scorer.score(features)
        assert result.priorities[0] != "3pt"
