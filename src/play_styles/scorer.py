"""Play-style priority scorer.

This module produces ordered play-style priorities intended for play-calling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PLAY_STYLE_NAMES: tuple[str, ...] = (
    "3pt",
    "Mid range",
    "Handoff pass",
    "Handoff receiver",
    "Cutter",
    "Guard post up",
    "Post up high",
    "Post up low",
    "Pick and roll rollman",
    "Pick and roll wing",
    "Pick and roll point",
    "Pick and roll ball handler",
    "Isolation wing",
    "Isolation point",
    "Isolation",
)

_EXCLUSIVITY_GROUP_BY_STYLE = {
    "Pick and roll rollman": "pnr_role",
    "Pick and roll wing": "pnr_role",
    "Pick and roll point": "pnr_role",
    "Pick and roll ball handler": "pnr_role",
    "Isolation wing": "iso_role",
    "Isolation point": "iso_role",
    "Isolation": "iso_role",
}


@dataclass(frozen=True)
class PlayStyleResult:
    """Selection payload for play-style priorities."""

    target_count: int
    priorities: list[str]
    weights: dict[str, float]
    scores: dict[str, float]
    usage_rate: float


class PlayStyleScorer:
    """Compute play-style priorities from player features."""

    def score(self, features: dict[str, Any], tendencies: dict[str, int] | None = None) -> PlayStyleResult:
        usage_rate = self._resolve_usage_rate(features)
        target_count = self._target_count_from_usage(usage_rate)

        position = str(features.get("position", "SF")).upper()
        ast_pg = self._to_float(features.get("ast_per_game", 0.0))
        fga_pg = self._to_float(features.get("fga_per_game", 0.0))
        fg3a_pg = self._to_float(features.get("fg3a_per_game", 0.0))
        fg3_pct = self._to_float(features.get("fg3_pct", 0.0))
        fg3a_rate = self._to_float(features.get("fg3a_rate", 0.0))
        height_inches = self._to_float(features.get("height_inches", 78.0))
        weight_lbs = self._to_float(features.get("weight_lbs", 220.0))

        ra_rate = self._to_float(features.get("zone_fga_rate_ra", 0.0))
        paint_rate = self._to_float(features.get("zone_fga_rate_paint", 0.0))
        mid_rate = (
            self._to_float(features.get("zone_fga_rate_mid_left", 0.0))
            + self._to_float(features.get("zone_fga_rate_mid_center", 0.0))
            + self._to_float(features.get("zone_fga_rate_mid_right", 0.0))
        )
        rim_pressure = ra_rate + paint_rate

        assists = self._to_float(features.get("assists", ast_pg))

        is_guard = 1.0 if position in {"PG", "SG"} else 0.0
        is_wing = 1.0 if position in {"SG", "SF"} else 0.0
        is_big = 1.0 if position in {"PF", "C"} else 0.0
        is_center = 1.0 if position == "C" else 0.0
        is_point = 1.0 if position == "PG" else 0.0

        size_big = max(
            is_big,
            0.60 * self._norm(height_inches, 79.0, 84.0)
            + 0.40 * self._norm(weight_lbs, 215.0, 265.0),
        )
        size_big = max(0.0, min(1.0, size_big))

        creator_proxy = (
            0.55 * self._norm(assists, 3.0, 10.0)
            + 0.45 * self._norm(usage_rate, 22.0, 35.0)
        )
        creator_proxy = max(0.0, min(1.0, creator_proxy))

        iso_fallback = max(
            0.0,
            (usage_rate - 22.0) * 0.14
            + 0.9 * self._norm(assists, 3.0, 10.0)
            + 0.6 * self._norm(mid_rate, 0.10, 0.30)
            - 0.7 * self._norm(fg3a_rate, 0.45, 0.70),
        )
        pnr_bh_fallback = max(
            0.0,
            (usage_rate - 20.0) * 0.18
            + 1.6 * self._norm(assists, 3.0, 10.0)
            + 1.4 * (1.0 - size_big)
            + (0.6 if position == "PG" else 0.0),
        )
        pnr_roll_fallback = max(
            0.0,
            rim_pressure * (3.0 + 3.0 * size_big)
            - 0.4 * self._norm(assists, 3.0, 10.0),
        )
        post_fallback = max(
            0.0,
            0.2
            + 2.8 * size_big
            + 0.05 * usage_rate
            - 1.2 * (1.0 - size_big) * self._norm(fg3a_rate, 0.35, 0.65),
        )
        cuts_fallback = max(
            0.0,
            rim_pressure * (2.0 + 2.0 * size_big)
            + 0.4 * self._norm(fg3a_rate, 0.28, 0.60)
            - 1.2 * creator_proxy,
        )
        handoff_fallback = max(
            0.0,
            0.3 + 0.18 * assists + 1.0 * size_big - 0.4 * self._norm(fg3a_rate, 0.45, 0.70),
        )

        mid_attempts = self._first_float(
            features,
            ("midrange_attempts", "mid_range_attempts", "midrange_fga"),
            fallback=mid_rate * fga_pg,
        )
        iso_pos = self._first_float(
            features,
            ("isolation_possessions", "iso_possessions", "isolation_poss"),
            fallback=iso_fallback,
        )
        pnr_bh_pos = self._first_float(
            features,
            (
                "pick_and_roll_ball_handler_possessions",
                "pick_and_roll_ball_handler_poss",
                "pnr_ball_handler_possessions",
                "pnr_bh_possessions",
            ),
            fallback=pnr_bh_fallback,
        )
        pnr_roll_pos = self._first_float(
            features,
            (
                "pick_and_roll_rollman_possessions",
                "pick_and_roll_rollman_poss",
                "pnr_rollman_possessions",
            ),
            fallback=pnr_roll_fallback,
        )
        post_pos = self._first_float(
            features,
            ("post_up_possessions", "post_ups", "post_possessions"),
            fallback=post_fallback,
        )
        cuts = self._first_float(
            features,
            ("cuts", "cut_possessions", "cut_poss"),
            fallback=cuts_fallback,
        )
        handoff_pos = self._first_float(
            features,
            ("handoff_possessions", "handoff_poss", "dhos"),
            fallback=handoff_fallback,
        )

        # Some feeds omit play-type possessions and surface zeros.
        # Treat non-positive values as missing and use proxy fallbacks.
        if iso_pos <= 0.0:
            iso_pos = iso_fallback
        if pnr_bh_pos <= 0.0:
            pnr_bh_pos = pnr_bh_fallback
        if pnr_roll_pos <= 0.0:
            pnr_roll_pos = pnr_roll_fallback
        if post_pos <= 0.0:
            post_pos = post_fallback
        if cuts <= 0.0:
            cuts = cuts_fallback
        if handoff_pos <= 0.0:
            handoff_pos = handoff_fallback
        if mid_attempts <= 0.0:
            mid_attempts = max(0.0, mid_rate * fga_pg)

        # Additional Synergy play-type signals
        spot_up_pos = self._to_float(features.get("spot_up_possessions", 0.0))
        off_screen_pos = self._to_float(features.get("off_screen_possessions", 0.0))
        transition_pos = self._to_float(features.get("transition_possessions", 0.0))

        # PPP efficiency data (0.0 = unavailable)
        iso_ppp = self._to_float(features.get("isolation_ppp", 0.0))
        pnr_bh_ppp = self._to_float(features.get("pick_and_roll_ball_handler_ppp", 0.0))
        pnr_roll_ppp = self._to_float(features.get("pick_and_roll_rollman_ppp", 0.0))
        post_ppp = self._to_float(features.get("post_up_ppp", 0.0))
        spot_up_ppp = self._to_float(features.get("spot_up_ppp", 0.0))

        # Mid-range FG% from zone data
        mid_fg_pct = (
            self._to_float(features.get("zone_fg_pct_mid_left", 0.0))
            + self._to_float(features.get("zone_fg_pct_mid_center", 0.0))
            + self._to_float(features.get("zone_fg_pct_mid_right", 0.0))
        ) / 3.0

        point_like = max(
            is_point,
            0.75 * self._norm(assists, 5.0, 11.0)
            + 0.25 * self._norm(usage_rate, 24.0, 36.0),
        )
        point_like = max(0.0, min(1.0, point_like))
        guard_like = max(is_guard, point_like * (1.0 - size_big))
        guard_like = max(0.0, min(1.0, guard_like))

        creator_load = (
            0.45 * self._norm(assists, 3.0, 10.0)
            + 0.35 * self._norm(usage_rate, 22.0, 35.0)
            + 0.20 * self._norm(rim_pressure, 0.22, 0.62)
        )
        creator_load = max(0.0, min(1.0, creator_load))
        off_ball_profile = (
            0.55 * self._norm(fg3a_rate, 0.20, 0.60)
            + 0.45 * self._norm(cuts, 0.4, 3.8)
        )
        off_ball_profile = max(0.0, min(1.0, off_ball_profile))
        movement_shooter = (
            0.70 * self._norm(fg3a_rate, 0.35, 0.70)
            + 0.30 * self._norm(fg3a_pg, 5.0, 12.0)
        ) * (1.0 - 0.55 * self._norm(rim_pressure, 0.35, 0.75))
        movement_shooter = max(0.0, min(1.0, movement_shooter))
        if off_screen_pos > 0.3:
            movement_shooter = min(1.0, movement_shooter + 0.12 * self._norm(off_screen_pos, 0.5, 4.0))
        creator_penalty = 1.0 - 0.22 * creator_load + 0.12 * movement_shooter

        scores: dict[str, float] = {
            "3pt": self._score(
                (
                    42.0 * self._norm(fg3a_pg, 1.2, 10.5)
                    + 24.0 * self._norm(fg3_pct, 0.31, 0.42)
                    + 22.0 * self._norm(fg3a_rate, 0.18, 0.60)
                        * min(1.0, self._norm(fg3a_pg, 2.5, 6.0))
                    + 8.0 * (is_guard + 0.5 * is_wing)
                )
                * creator_penalty
                + 14.0 * self._norm(spot_up_pos, 1.0, 8.0)
                + 12.0 * movement_shooter
                + 6.0 * off_ball_profile
            ),
            "Mid range": self._score(
                56.0 * self._norm(mid_attempts, 1.0, 7.5)
                + 24.0 * self._norm(mid_rate, 0.08, 0.33)
                + 20.0 * self._norm(usage_rate, 17.0, 33.0)
                + 8.0 * self._norm(fga_pg, 10.0, 24.0)
                + 10.0 * self._norm(mid_fg_pct, 0.36, 0.52)
            ),
            "Handoff pass": self._score(
                42.0 * self._norm(handoff_pos, 0.4, 5.0)
                + 34.0 * self._norm(assists, 2.0, 9.0)
                + 14.0 * self._norm(usage_rate, 18.0, 33.0)
                + 10.0 * (0.4 * is_wing + 0.8 * is_big + 0.6 * size_big)
                + 8.0 * creator_load
            ),
            "Handoff receiver": self._score(
                50.0 * self._norm(handoff_pos, 0.3, 4.0)
                + 24.0 * self._norm(fg3a_pg, 1.0, 9.5)
                + 16.0 * self._norm(cuts, 0.3, 3.2)
                + 10.0 * (is_guard + 0.8 * is_wing)
                + 8.0 * self._norm(off_screen_pos, 0.5, 4.0)
                - 12.0 * creator_load
                - 8.0 * size_big
            ),
            "Cutter": self._score(
                68.0 * self._norm(cuts, 0.4, 3.5)
                + 22.0 * self._norm(rim_pressure, 0.20, 0.65)
                + 10.0 * (0.7 * is_big + 0.3 * is_wing)
                + 12.0 * max(0.0, 1.0 - creator_load)
                    * max(0.0, 1.0 - self._norm(usage_rate, 16.0, 28.0))
                - 10.0 * creator_load
            ),
            "Guard post up": self._score(
                52.0 * self._norm(post_pos, 0.8, 4.5)
                + 20.0 * self._norm(usage_rate, 18.0, 32.0)
                + 28.0 * guard_like
                - 18.0 * size_big
            ),
            "Post up high": self._score(
                54.0 * self._norm(post_pos, 0.4, 6.0)
                + 30.0 * self._norm(assists, 1.5, 8.5)
                + 16.0 * (0.6 * is_wing + is_big)
                + 8.0 * size_big
            ),
            "Post up low": self._score(
                62.0 * self._norm(post_pos, 0.4, 6.2)
                + 22.0 * self._norm(rim_pressure, 0.18, 0.62)
                + 16.0 * is_big
                + 10.0 * size_big
            ),
            "Pick and roll rollman": self._score(
                68.0 * self._norm(pnr_roll_pos, 0.6, 6.5)
                + 18.0 * self._norm(rim_pressure, 0.20, 0.70)
                + 14.0 * is_big
                + 14.0 * size_big
                - 10.0 * creator_load
            ),
            "Pick and roll wing": self._score(
                30.0 * self._norm(pnr_bh_pos, 0.5, 6.5)
                + 30.0 * self._norm(fg3a_rate, 0.16, 0.55)
                + 16.0 * off_ball_profile
                + 24.0 * is_wing
                - 12.0 * self._norm(assists, 3.0, 10.0)
                - 14.0 * point_like
            ),
            "Pick and roll point": self._score(
                58.0 * self._norm(pnr_bh_pos, 0.8, 10.0)
                + 24.0 * self._norm(assists, 2.0, 11.0)
                + 18.0 * point_like
                + 10.0 * creator_load
                - 8.0 * size_big
            ),
            "Pick and roll ball handler": self._score(
                64.0 * self._norm(pnr_bh_pos, 0.8, 10.0)
                + 20.0 * self._norm(usage_rate, 18.0, 35.0)
                + 16.0 * self._norm(assists, 2.0, 10.0)
                + 12.0 * creator_load
                + 8.0 * point_like
                - 8.0 * size_big
            ),
            "Isolation wing": self._score(
                62.0 * self._norm(iso_pos, 0.4, 7.0)
                + 20.0 * self._norm(usage_rate, 20.0, 35.0)
                + 18.0 * is_wing
                + 8.0 * creator_load
                - 10.0 * point_like
            ),
            "Isolation point": self._score(
                62.0 * self._norm(iso_pos, 0.4, 7.0)
                + 22.0 * self._norm(assists, 2.5, 11.0)
                + 16.0 * point_like
                + 10.0 * creator_load
                - 8.0 * size_big
            ),
            "Isolation": self._score(
                70.0 * self._norm(iso_pos, 0.4, 7.0)
                + 30.0 * self._norm(usage_rate, 20.0, 35.0)
                + 10.0 * creator_load
                - 6.0 * movement_shooter
            ),
        }

        # Position-aware penalties to avoid unrealistic priorities.
        if is_center > 0:
            scores["Isolation"] *= 0.55
            scores["Isolation wing"] *= 0.45
            scores["Isolation point"] *= 0.40
            if iso_pos < 2.5:
                scores["Isolation"] *= 0.80

            scores["Pick and roll point"] *= 0.45
            scores["Pick and roll wing"] *= 0.55
            if pnr_bh_pos < 4.0:
                scores["Pick and roll ball handler"] *= 0.75

            scores["Guard post up"] *= 0.25

        if is_guard > 0:
            scores["Post up low"] *= 0.70
            scores["Pick and roll rollman"] *= 0.65
        if is_big > 0:
            scores["Guard post up"] *= 0.55
            scores["Pick and roll point"] *= 0.70
            scores["Isolation point"] *= 0.80
        if guard_like < 0.45:
            scores["Guard post up"] *= 0.35
        if post_pos < 1.2:
            scores["Guard post up"] *= 0.25
        if fg3a_rate > 0.45 and rim_pressure < 0.35:
            scores["Guard post up"] *= 0.45

        if movement_shooter > 0.65 and point_like > 0.55 and rim_pressure < 0.35:
            scores["Isolation"] *= 0.72
            scores["Isolation point"] *= 0.80
            scores["Handoff receiver"] *= 1.12
            scores["Pick and roll point"] *= 1.06

        # Wing creator correction: for high-usage on-ball wings, avoid over-calling
        # generic 3pt actions as the primary identity.
        if position in {"SG", "SF"} and usage_rate >= 27.0 and (iso_pos >= 2.0 or pnr_bh_pos >= 4.0):
            scores["3pt"] *= 0.82
            scores["Isolation wing"] *= 1.08
            scores["Pick and roll ball handler"] *= 1.08

        # Low-volume 3pt dampener: rate-inflated scores shouldn't dominate for
        # players who barely shoot threes.  Requires both low volume AND low
        # usage so that spot-up specialists with moderate volume keep credit.
        if fg3a_pg < 5.0 and usage_rate < 22.0:
            scores["3pt"] *= 0.45 + 0.55 * self._norm(fg3a_pg, 2.0, 5.0)

        # Efficiency multipliers: boost/dampen styles based on PPP
        scores["Isolation"] *= self._eff_mult(iso_ppp)
        scores["Isolation wing"] *= self._eff_mult(iso_ppp)
        scores["Isolation point"] *= self._eff_mult(iso_ppp)
        scores["Pick and roll ball handler"] *= self._eff_mult(pnr_bh_ppp)
        scores["Pick and roll point"] *= self._eff_mult(pnr_bh_ppp)
        scores["Pick and roll rollman"] *= self._eff_mult(pnr_roll_ppp)
        scores["Post up high"] *= self._eff_mult(post_ppp)
        scores["Post up low"] *= self._eff_mult(post_ppp)
        scores["3pt"] *= self._eff_mult(spot_up_ppp)

        # Hub-center: elite-passing big who initiates offense (Jokic/Bam type)
        if size_big > 0.7 and ast_pg >= 6.0:
            scores["Pick and roll ball handler"] *= 1.20
            scores["Handoff pass"] *= 1.12
            scores["Post up high"] *= 1.08

        # Transition-heavy players favor rim-attacking half-court styles
        if transition_pos > 1.5:
            t_boost = 0.08 * self._norm(transition_pos, 2.0, 8.0)
            scores["Cutter"] *= 1.0 + t_boost
            scores["Pick and roll rollman"] *= 1.0 + t_boost * 0.7

        # ------------------------------------------------------------------
        # Tendency-informed adjustments (second pass)
        # ------------------------------------------------------------------
        if tendencies:
            self._apply_tendency_adjustments(scores, tendencies)

        priorities = self._select_non_conflicting(scores, target_count)
        weights = self._weights_from_scores(priorities, scores)

        ordered_scores = {
            name: round(float(scores[name]), 2)
            for name in sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        }

        return PlayStyleResult(
            target_count=target_count,
            priorities=priorities,
            weights=weights,
            scores=ordered_scores,
            usage_rate=round(float(usage_rate), 2),
        )

    def _select_non_conflicting(self, scores: dict[str, float], target_count: int) -> list[str]:
        ranked = sorted(scores.keys(), key=lambda n: scores[n], reverse=True)
        selected: list[str] = []
        used_groups: set[str] = set()
        for name in ranked:
            group = _EXCLUSIVITY_GROUP_BY_STYLE.get(name)
            if group and group in used_groups:
                continue
            selected.append(name)
            if group:
                used_groups.add(group)
            if len(selected) >= target_count:
                break
        return selected

    def _apply_tendency_adjustments(self, scores: dict[str, float], t: dict[str, int]) -> None:
        """Adjust play-style scores using generated tendencies as a reality check."""
        drv_dunk = t.get("driving_dunk", 0)
        alley = t.get("alley_oop", 0)
        post_up = t.get("post_up", 0)
        shot_three = t.get("shot_three", 0)
        spot_three = t.get("spot_up_shot_three", 0)
        shot_mid = t.get("shot_mid_range", 0)
        iso_t = t.get("iso_vs_team", 50)
        dish = t.get("dish_to_open_man", 0)
        shot_close = t.get("shot_close", 0)
        drv_layup = t.get("driving_layup", 0)
        post_hook_l = t.get("post_hook_left", 0)
        post_hook_r = t.get("post_hook_right", 0)
        post_fade = t.get("post_fade", 0)
        post_spin = t.get("post_spin", 0)
        post_drive = t.get("post_drive", 0)

        # Rim-runner signal: high dunk/alley + low post = rollman/cutter, not post
        rim_runner = self._norm(drv_dunk, 30, 80) * 0.5 + self._norm(alley, 25, 70) * 0.5
        if rim_runner > 0.4 and post_up < 25:
            scores["Pick and roll rollman"] *= 1.0 + 0.20 * rim_runner
            scores["Cutter"] *= 1.0 + 0.18 * rim_runner
            scores["Post up low"] *= max(0.55, 1.0 - 0.35 * rim_runner)
            scores["Post up high"] *= max(0.60, 1.0 - 0.25 * rim_runner)

        # Strong post tendencies reinforce post styles
        post_moves = (post_hook_l + post_hook_r + post_fade + post_spin + post_drive) / 5.0
        if post_up >= 25 and post_moves >= 15:
            post_strength = self._norm(post_up, 25, 70) * 0.6 + self._norm(post_moves, 15, 50) * 0.4
            scores["Post up low"] *= 1.0 + 0.18 * post_strength
            scores["Post up high"] *= 1.0 + 0.12 * post_strength
            # Distinguish high vs low: fade/hook = high, drive/spin = low
            finesse = (post_fade + post_hook_l + post_hook_r) / 3.0
            power = (post_drive + post_spin) / 2.0
            if finesse > power + 10:
                scores["Post up high"] *= 1.06
            elif power > finesse + 10:
                scores["Post up low"] *= 1.06

        # 3pt tendency reinforcement
        three_signal = self._norm(shot_three, 20, 80) * 0.5 + self._norm(spot_three, 15, 70) * 0.5
        if three_signal > 0.3:
            scores["3pt"] *= 1.0 + 0.12 * three_signal
        elif shot_three < 15:
            scores["3pt"] *= 0.75

        # Mid-range tendency reinforcement
        if shot_mid >= 30:
            scores["Mid range"] *= 1.0 + 0.10 * self._norm(shot_mid, 30, 70)

        # Iso tendency: high iso_vs_team = prefers isolation plays
        if iso_t >= 60:
            iso_boost = 0.10 * self._norm(iso_t, 60, 85)
            scores["Isolation"] *= 1.0 + iso_boost
            scores["Isolation wing"] *= 1.0 + iso_boost
            scores["Isolation point"] *= 1.0 + iso_boost
        elif iso_t < 40:
            # Low iso = team player, favor pass-first styles
            scores["Handoff pass"] *= 1.0 + 0.08 * self._norm(40 - iso_t, 0, 25)
            scores["Pick and roll ball handler"] *= 1.0 + 0.06 * self._norm(40 - iso_t, 0, 25)

        # Dish-to-open-man = pass-first, boost handoff/PnR BH
        if dish >= 40:
            dish_boost = 0.08 * self._norm(dish, 40, 75)
            scores["Handoff pass"] *= 1.0 + dish_boost
            scores["Pick and roll ball handler"] *= 1.0 + dish_boost * 0.7

        # Interior finisher with low 3pt = penalize shooting styles
        if shot_close >= 50 and drv_layup >= 60 and shot_three < 15:
            scores["3pt"] *= 0.70
            scores["Pick and roll wing"] *= 0.75

    def _weights_from_scores(self, priorities: list[str], scores: dict[str, float]) -> dict[str, float]:
        if not priorities:
            return {}

        powered = [max(1.0, scores[name]) ** 1.35 for name in priorities]
        total = sum(powered)
        raw = [value / total for value in powered]

        rounded = [round(v, 3) for v in raw]
        diff = round(1.0 - sum(rounded), 3)
        rounded[0] = round(rounded[0] + diff, 3)
        return {name: rounded[i] for i, name in enumerate(priorities)}

    def _resolve_usage_rate(self, features: dict[str, Any]) -> float:
        if "usage_rate" in features:
            return self._to_float(features["usage_rate"])
        if "usage_rate_pct" in features:
            return self._to_float(features["usage_rate_pct"])
        if "usg_pct" in features:
            return self._to_float(features["usg_pct"])
        return self._to_float(features.get("usg_pct_proxy", 0.18)) * 100.0

    @staticmethod
    def _target_count_from_usage(usage_rate: float) -> int:
        if usage_rate > 30.0:
            return 4
        if usage_rate >= 24.0:
            return 3
        return 2

    @staticmethod
    def _score(value: float) -> float:
        return max(0.0, min(100.0, value))

    @staticmethod
    def _eff_mult(ppp: float) -> float:
        """Efficiency multiplier from points-per-possession (0.90--1.10).

        Returns 1.0 when no PPP data is available (ppp <= 0).
        """
        if ppp <= 0.0:
            return 1.0
        return 0.90 + 0.20 * max(0.0, min(1.0, (ppp - 0.70) / 0.50))

    @staticmethod
    def _norm(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return max(0.0, min(1.0, (value - low) / (high - low)))

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _first_float(
        self,
        features: dict[str, Any],
        keys: tuple[str, ...],
        fallback: float,
    ) -> float:
        for key in keys:
            if key in features:
                return self._to_float(features[key])
        return fallback
