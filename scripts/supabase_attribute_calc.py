from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attributes.calculator import ATTRIBUTE_CATEGORIES, ATTRIBUTE_LABELS, AttributeCalculator
from src.caps.cap_enforcer import CapEnforcer
from src.formula.formula_layer import FormulaLayer
from src.pipeline import (
    _apply_close_side_tiebreak,
    _clamp_mid_family_to_parent_band,
    _round_family_to_parent,
    _round_mid_family,
    _round_three_family,
    _round_to_5,
)
from src.play_styles.scorer import PlayStyleScorer
from src.validation.guardrails import Guardrails


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def env_value(name: str, fallback_env: dict[str, str]) -> str:
    return os.environ.get(name, "").strip() or fallback_env.get(name, "").strip()


def supabase_select(base_url: str, key: str, table: str, query: str) -> list[dict]:
    url = f"{base_url.rstrip('/')}/rest/v1/{table}{query}"
    req = urllib.request.Request(
        url,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def n(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_position(raw_position: str) -> str:
    pos = str(raw_position or "").upper().replace("_", "-").replace("/", "-")
    tokens = [p.strip() for p in pos.replace(" ", "-").split("-") if p.strip()]
    token_set = set(tokens)

    if "PG" in token_set or "POINT" in token_set:
        return "PG"
    if "SG" in token_set or "SHOOTING" in token_set:
        return "SG"
    if "SF" in token_set or "SMALL" in token_set:
        return "SF"
    if "PF" in token_set or "POWER" in token_set:
        return "PF"
    if "C" in token_set or "CENTER" in token_set:
        return "C"

    if "GUARD" in token_set:
        return "SG"
    if "FORWARD" in token_set:
        return "SF"

    return "SF"


def build_features(pbp: dict, src: dict | None) -> dict[str, float | str]:
    gp = n(pbp, "gp")
    min_pg = n(pbp, "min_pg")

    loose_ball_fouls = n(pbp, "loose_ball_fouls")
    loose_ball_fouls_drawn = n(pbp, "loose_ball_fouls_drawn")
    if gp > 0:
        if loose_ball_fouls > 1.0:
            loose_ball_fouls /= gp
        if loose_ball_fouls_drawn > 1.0:
            loose_ball_fouls_drawn /= gp

    fga_pg = n(pbp, "fga_pg")
    fg3a_pg = n(pbp, "fg3a_pg")
    fta_pg = n(pbp, "fta_pg")
    ast_pg = n(pbp, "ast_pg")
    tov_pg = n(pbp, "tov_pg")

    fga_per36 = (fga_pg * 36.0 / min_pg) if min_pg > 0 else 0.0
    fg3a_per36 = (fg3a_pg * 36.0 / min_pg) if min_pg > 0 else 0.0
    fta_per36 = (fta_pg * 36.0 / min_pg) if min_pg > 0 else 0.0
    tov_per36 = (tov_pg * 36.0 / min_pg) if min_pg > 0 else 0.0
    oreb_per36 = (n(pbp, "oreb_pg") * 36.0 / min_pg) if min_pg > 0 else 0.0
    dreb_per36 = (n(pbp, "dreb_pg") * 36.0 / min_pg) if min_pg > 0 else 0.0

    fg3a_rate = (fg3a_pg / fga_pg) if fga_pg > 0 else 0.0
    fta_rate = (fta_pg / fga_pg) if fga_pg > 0 else 0.0

    usage_raw = n(pbp, "pbp_usage_rate")
    usage_rate = usage_raw * 100.0 if usage_raw <= 1.0 else usage_raw
    usg_pct_proxy = max(0.05, min(0.45, usage_rate / 100.0))

    ast_to_tov = (ast_pg / tov_pg) if tov_pg > 0 else 1.0
    tov_pct_proxy = (tov_pg / (fga_pg + 0.44 * fta_pg + tov_pg)) if (fga_pg + 0.44 * fta_pg + tov_pg) > 0 else 0.12

    two_pa_pg = max(0.0, fga_pg - fg3a_pg)
    unassisted_two_rate = n(pbp, "unassisted_two_rate")
    creator_two_p36 = ((two_pa_pg * unassisted_two_rate) * 36.0 / min_pg) if min_pg > 0 else 0.0

    pull_up_three_rate = n(pbp, "pull_up_three_rate")
    iso_burden_p36 = creator_two_p36 * 0.45 + (fg3a_per36 * pull_up_three_rate) * 0.55

    isolation_possessions = n(pbp, "isolation_possessions")
    pnr_bh_possessions = n(pbp, "pick_and_roll_ball_handler_possessions")
    pnr_roll_possessions = n(pbp, "pick_and_roll_rollman_possessions")
    post_up_possessions = n(pbp, "post_up_possessions")
    cuts_possessions = n(pbp, "cuts")
    handoff_possessions = n(pbp, "handoff_possessions")
    spot_up_possessions = n(pbp, "spot_up_possessions")

    position = normalize_position(str((src or {}).get("position") or "SF"))
    age = n(src or {}, "age", 26.0)
    height_in = n(src or {}, "height_in", 78.0)
    weight_lbs = n(src or {}, "weight_lbs", 220.0)

    # When playtype possession fields are absent/zero, estimate from box-score shape.
    poss_used_pg = fga_pg + 0.44 * fta_pg + tov_pg
    creator_load = max(0.0, min(1.0, 0.45 * (usage_rate / 35.0) + 0.35 * (ast_pg / 10.0) + 0.20 * (n(pbp, "pts_pg") / 35.0)))
    rim_proxy = max(0.0, min(1.0, 0.42 * (fta_rate / 0.45) + 0.30 * (oreb_per36 / 4.5) + 0.18 * (n(pbp, "blk_pg") / 2.2) + 0.10 * (1.0 - min(1.0, fg3a_rate / 0.55))))

    if isolation_possessions <= 0.0:
        isolation_possessions = poss_used_pg * (0.06 + 0.22 * creator_load)
    if pnr_bh_possessions <= 0.0:
        pnr_bh_possessions = poss_used_pg * (0.10 + 0.25 * creator_load)
    if pnr_roll_possessions <= 0.0:
        pnr_roll_possessions = poss_used_pg * (0.04 + 0.20 * rim_proxy * (1.0 - creator_load))
    if post_up_possessions <= 0.0:
        post_up_possessions = poss_used_pg * (0.03 + 0.17 * max(0.0, min(1.0, (height_in - 75.0) / 9.0)))
    if cuts_possessions <= 0.0:
        cuts_possessions = poss_used_pg * (0.05 + 0.11 * rim_proxy * (1.0 - creator_load))
    if handoff_possessions <= 0.0:
        handoff_possessions = poss_used_pg * (0.02 + 0.09 * max(0.0, min(1.0, (fg3a_rate - 0.10) / 0.45)))
    if spot_up_possessions <= 0.0:
        spot_up_possessions = poss_used_pg * (0.04 + 0.24 * n(pbp, "catch_and_shoot_three_rate"))

    short_mid_frequency = n(pbp, "short_mid_frequency")
    long_mid_frequency = n(pbp, "long_mid_frequency")
    short_mid_accuracy = n(pbp, "short_mid_accuracy")
    long_mid_accuracy = n(pbp, "long_mid_accuracy")
    mid_rate = max(0.0, short_mid_frequency + long_mid_frequency)

    mid_weight = short_mid_frequency + long_mid_frequency
    mid_pct = (
        (short_mid_frequency * short_mid_accuracy + long_mid_frequency * long_mid_accuracy) / mid_weight
        if mid_weight > 0
        else 0.40
    )

    at_rim_frequency = n(pbp, "at_rim_frequency")
    at_rim_accuracy = n(pbp, "at_rim_accuracy")

    midrange_attempts = max(0.0, (short_mid_frequency + long_mid_frequency) * fga_pg)
    reb_pg = n(pbp, "reb_pg")
    oreb_pg = n(pbp, "oreb_pg")
    oreb_pct_proxy = (oreb_pg / reb_pg) if reb_pg > 0 else 0.10

    fg_pct = n(pbp, "fg_pct")
    fg3_pct = n(pbp, "fg3_pct")
    ft_pct = n(pbp, "ft_pct")

    fgm_pg = n(pbp, "fgm_pg")
    fg3m_pg = n(pbp, "fg3m_pg")
    pts_pg = n(pbp, "pts_pg")

    efg_pct = ((fgm_pg + 0.5 * fg3m_pg) / fga_pg) if fga_pg > 0 else fg_pct
    ts_denom = 2.0 * (fga_pg + 0.44 * fta_pg)
    ts_pct = (pts_pg / ts_denom) if ts_denom > 0 else 0.0

    assisted_2pt_pct = n(pbp, "assisted_2pt_pct")
    assisted_3pt_pct = n(pbp, "assisted_3pt_pct")
    if assisted_2pt_pct <= 1.0:
        assisted_2pt_pct *= 100.0
    if assisted_3pt_pct <= 1.0:
        assisted_3pt_pct *= 100.0

    return {
        "position": position,
        "age": age,
        "height_inches": height_in,
        "weight_lbs": weight_lbs,
        "gp": gp,
        "min_per_game": min_pg,
        "pts_per_game": pts_pg,
        "fga_per_game": fga_pg,
        "fta_per_game": fta_pg,
        "fg3a_per_game": fg3a_pg,
        "ast_per_game": ast_pg,
        "stl_per_game": n(pbp, "stl_pg"),
        "blk_per_game": n(pbp, "blk_pg"),
        "tov_per_game": tov_pg,
        "oreb_per_game": n(pbp, "oreb_pg"),
        "dreb_per_game": n(pbp, "dreb_pg"),
        "oreb_per36": oreb_per36,
        "dreb_per36": dreb_per36,
        "stl_per36": (n(pbp, "stl_pg") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "blk_per36": (n(pbp, "blk_pg") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "pf_per36": (n(pbp, "pf_pg") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "fg_pct": fg_pct,
        "fg3_pct": fg3_pct,
        "ft_pct": ft_pct,
        "efg_pct": efg_pct,
        "ts_pct": ts_pct,
        "fg3a_rate": fg3a_rate,
        "fta_rate": fta_rate,
        "fg3a_per36": fg3a_per36,
        "fta_per36": fta_per36,
        "fga_per36": fga_per36,
        "tov_per36": tov_per36,
        "pts_per36": (pts_pg * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "ast_per36": (ast_pg * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "reb_per36": (n(pbp, "reb_pg") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "usage_rate": usage_rate,
        "usg_pct_proxy": usg_pct_proxy,
        "ast_to_tov": ast_to_tov,
        "tov_pct_proxy": tov_pct_proxy,
        "oreb_pct_proxy": oreb_pct_proxy,
        "isolation_possessions": isolation_possessions,
        "pick_and_roll_ball_handler_possessions": pnr_bh_possessions,
        "pick_and_roll_rollman_possessions": pnr_roll_possessions,
        "post_up_possessions": post_up_possessions,
        "transition_possessions": n(pbp, "transition_possessions"),
        "cuts": cuts_possessions,
        "handoff_possessions": handoff_possessions,
        "spot_up_possessions": spot_up_possessions,
        "isolation_ppp": n(pbp, "isolation_ppp"),
        "pick_and_roll_ball_handler_ppp": n(pbp, "pick_and_roll_ball_handler_ppp"),
        "post_up_ppp": n(pbp, "post_up_ppp"),
        "spot_up_ppp": n(pbp, "spot_up_ppp"),
        "assisted_2pt_pct": assisted_2pt_pct,
        "assisted_3pt_pct": assisted_3pt_pct,
        "catch_and_shoot_three_rate": n(pbp, "catch_and_shoot_three_rate"),
        "pull_up_three_rate": pull_up_three_rate,
        "unassisted_two_rate": unassisted_two_rate,
        "putback_rate": n(pbp, "putback_rate"),
        "live_ball_turnover_pct": n(pbp, "live_ball_turnover_pct"),
        "shooting_fouls_drawn_pct": n(pbp, "shooting_fouls_drawn_pct"),
        "three_pt_fouls_drawn_pct": n(pbp, "three_pt_fouls_drawn_pct"),
        "seconds_per_poss_off": n(pbp, "seconds_per_poss_off"),
        "second_chance_off_poss_rate": n(pbp, "second_chance_off_poss_rate"),
        "midrange_attempts": midrange_attempts,
        "bad_pass_turnovers_per36": (n(pbp, "bad_pass_turnovers") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "lost_ball_turnovers_per36": (n(pbp, "lost_ball_turnovers") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "offensive_fouls_per36": (n(pbp, "offensive_fouls") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "offensive_fouls_drawn_per36": (n(pbp, "offensive_fouls_drawn") * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "loose_ball_fouls_per36": (loose_ball_fouls * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "loose_ball_fouls_drawn_per36": (loose_ball_fouls_drawn * 36.0 / min_pg) if min_pg > 0 else 0.0,
        "blocks_recovered_pct": n(pbp, "blocks_recovered_pct"),
        "pbp_data_available": 1.0,
        "zone_fga_rate_ra": at_rim_frequency,
        "zone_fga_rate_paint": 0.0,
        "zone_fg_pct_ra": at_rim_accuracy,
        "zone_fg_pct_paint": at_rim_accuracy,
        "zone_fg_pct_mid_left": mid_pct,
        "zone_fg_pct_mid_center": mid_pct,
        "zone_fg_pct_mid_right": mid_pct,
        "zone_fga_rate_mid_left": mid_rate / 3.0,
        "zone_fga_rate_mid_center": mid_rate / 3.0,
        "zone_fga_rate_mid_right": mid_rate / 3.0,
        "sub_zone_distribution_close": {"left": 32.0, "middle": 36.0, "right": 32.0},
        "sub_zone_distribution_mid": {
            "left": 20.0,
            "left_center": 20.0,
            "center": 20.0,
            "right_center": 20.0,
            "right": 20.0,
        },
        "sub_zone_distribution_three": {
            "left": 18.0,
            "left_center": 19.0,
            "center": 26.0,
            "right_center": 19.0,
            "right": 18.0,
        },
        "drive_right_bias": 50.0,
        "creator_two_p36": creator_two_p36,
        "iso_burden_p36": iso_burden_p36,
        "pnr_bh_burden_p36": 0.0,
    }


def group_attributes(attributes: dict[str, int]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for canonical, value in attributes.items():
        category = ATTRIBUTE_CATEGORIES.get(canonical, "other")
        label = ATTRIBUTE_LABELS.get(canonical, canonical)
        grouped.setdefault(category, {})[label] = int(value)
    return grouped


def group_tendencies(tendencies: dict[str, int], registry: list[dict]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    by_key = {str(row.get("canonical_name", "")): row for row in registry}
    for canonical, value in tendencies.items():
        row = by_key.get(canonical, {})
        category = str(row.get("category") or "other")
        label = str(row.get("primjer_label") or canonical)
        grouped.setdefault(category, {})[label] = int(value)
    return grouped


def calculate_tendencies(features: dict[str, float | str], registry_path: Path) -> tuple[dict[str, int], list[str]]:
    formula = FormulaLayer()
    guardrails = Guardrails()
    caps = CapEnforcer(str(registry_path))
    scorer = PlayStyleScorer()

    formula_raw = formula.generate(features)
    rounded = {k: _round_to_5(v) for k, v in formula_raw.items()}

    guardrail_input = dict(formula_raw)
    try:
        _ = guardrails.check(guardrail_input)
        for k, v in guardrail_input.items():
            rounded[k] = _round_to_5(v)

        _round_family_to_parent(
            guardrail_input,
            rounded,
            "shot_close",
            ["shot_close_left", "shot_close_middle", "shot_close_right"],
        )
        _round_mid_family(
            guardrail_input,
            rounded,
            "shot_mid_range",
            [
                "shot_mid_left",
                "shot_mid_left_center",
                "shot_mid_center",
                "shot_mid_right_center",
                "shot_mid_right",
            ],
        )
        _clamp_mid_family_to_parent_band(
            rounded,
            "shot_mid_range",
            [
                "shot_mid_left",
                "shot_mid_left_center",
                "shot_mid_center",
                "shot_mid_right_center",
                "shot_mid_right",
            ],
        )
        _round_three_family(
            guardrail_input,
            rounded,
            "shot_three",
            [
                "shot_three_left",
                "shot_three_left_center",
                "shot_three_center",
                "shot_three_right_center",
                "shot_three_right",
            ],
        )
        _apply_close_side_tiebreak(
            rounded,
            float(guardrail_input.get("drive_right", guardrail_input.get("drive_right_bias", 50.0))),
        )
    except Exception:
        # Keep rounded formula output if guardrail pass fails.
        pass

    # Parent-aware redistribution (close sub-zones only)
    close_parent = int(rounded.get("shot_close", 0))
    close_sum = int(rounded.get("shot_close_left", 0)) + int(rounded.get("shot_close_middle", 0)) + int(rounded.get("shot_close_right", 0))
    if close_sum != close_parent and close_sum > 0:
        diff = close_parent - close_sum
        largest_key = max(("shot_close_left", "shot_close_middle", "shot_close_right"), key=lambda x: int(rounded.get(x, 0)))
        rounded[largest_key] = int(rounded.get(largest_key, 0)) + diff

    # Tie-break close left/right by drive side if exact tie
    close_left = int(rounded.get("shot_close_left", 0))
    close_right = int(rounded.get("shot_close_right", 0))
    if close_left == close_right:
        drive_right = int(rounded.get("drive_right", 50))
        if drive_right > 50 and close_left >= 5:
            rounded["shot_close_right"] = close_right + 5
            rounded["shot_close_left"] = close_left - 5
        elif drive_right < 50 and close_right >= 5:
            rounded["shot_close_left"] = close_left + 5
            rounded["shot_close_right"] = close_right - 5

    capped, _audit = caps.enforce_all({k: int(v) for k, v in rounded.items()})

    priorities: list[str] = []
    try:
        style_result = scorer.score(features, tendencies=capped)
        priorities = list(style_result.priorities)
    except Exception:
        priorities = []

    return capped, priorities


def main() -> None:
    if len(sys.argv) < 3:
        raise RuntimeError("Usage: supabase_attribute_calc.py <player_id> <season_start>")

    player_id = int(sys.argv[1])
    season_start = int(sys.argv[2])
    season_label = f"{season_start}-{str(season_start + 1)[2:]}"

    frontend_env = load_env_file(REPO_ROOT / "frontend" / ".env.local")
    supabase_url = env_value("NEXT_PUBLIC_SUPABASE_URL", frontend_env).rstrip("/")
    supabase_key = env_value("NEXT_PUBLIC_SUPABASE_ANON_KEY", frontend_env)
    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing Supabase config for attribute calculation")

    encoded_season = urllib.parse.quote(season_label, safe="")
    pbp_query = f"?select=*&season=eq.{encoded_season}&player_id=eq.{player_id}&limit=1"
    src_query = (
        "?select=player_id,season_start,full_name,position,age,height_in,weight_lbs"
        f"&season_start=eq.{season_start}&player_id=eq.{player_id}&limit=1"
    )

    pbp_rows = supabase_select(supabase_url, supabase_key, "pbp_profiles", pbp_query)
    src_rows = supabase_select(supabase_url, supabase_key, "player_generation_source_v1", src_query)
    if not pbp_rows:
        pbp_fallback_query = f"?select=*&player_id=eq.{player_id}&order=season.desc&limit=1"
        pbp_rows = supabase_select(supabase_url, supabase_key, "pbp_profiles", pbp_fallback_query)
    if not src_rows:
        src_fallback_query = (
            "?select=player_id,season_start,full_name,position,age,height_in,weight_lbs"
            f"&player_id=eq.{player_id}&order=season_start.desc&limit=1"
        )
        src_rows = supabase_select(supabase_url, supabase_key, "player_generation_source_v1", src_fallback_query)
    if not pbp_rows:
        raise RuntimeError(f"No pbp_profiles row for player {player_id} season {season_label}")

    pbp = pbp_rows[0]
    src = src_rows[0] if src_rows else None

    features = build_features(pbp, src)
    calc = AttributeCalculator()
    registry_path = REPO_ROOT / "data" / "tendency_registry.json"
    registry: list[dict] = []
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = []

    tendencies, play_style_priorities = calculate_tendencies(features, registry_path)
    attrs = calc.calculate(features, tendencies={})

    payload = {
        "attributes": {k: int(v) for k, v in attrs.items()},
        "attributeGroups": group_attributes({k: int(v) for k, v in attrs.items()}),
        "tendencies": {k: int(v) for k, v in tendencies.items()},
        "tendencyGroups": group_tendencies({k: int(v) for k, v in tendencies.items()}, registry),
        "playStylePriorities": play_style_priorities,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
