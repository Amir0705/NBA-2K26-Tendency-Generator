import fs from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { NextRequest, NextResponse } from "next/server";
import { API_BASE_URL } from "@/lib/api/client";
import { generatePlayer } from "@/lib/generator/generate-player";
import { fetchPlayerStatsFromSupabase } from "@/lib/generator/supabase-source";
import type { GeneratedPlayerPayload, GeneratorRequest } from "@/lib/generator/types";

type TendencyRegistryRow = {
  canonical_name: string;
  primjer_key: string;
  primjer_label: string;
  category?: string;
};

type BackendTendencyEntry = {
  value: number;
  label?: string;
};

type BackendAttributeEntry = {
  value: number;
  label?: string;
  category?: string;
};

type BackendGenerateResponse = {
  player_name: string;
  player_id: number;
  position: string;
  team: string;
  season: string;
  play_style_priorities?: string[];
  tendencies: Record<string, BackendTendencyEntry>;
  attributes: Record<string, BackendAttributeEntry>;
};

let registryCache: TendencyRegistryRow[] | null = null;
const responseCache = new Map<string, { payload: GeneratedPlayerPayload; cachedAt: number }>();
const RESPONSE_CACHE_TTL_MS = 1000 * 60 * 30;
const STATS_CARD_SEASON = 2025;
const DISABLE_RESPONSE_CACHE = true;
const BACKEND_REQUEST_TIMEOUT_MS = 12_000;
const GENERATION_MODE = process.env.N2K_GENERATION_MODE ?? "supabase";
const ENABLE_LOCAL_FALLBACK = process.env.N2K_ALLOW_LOCAL_FALLBACK === "1";
const ALLOW_LEGACY_TS_ATTRIBUTE_FALLBACK = process.env.N2K_ALLOW_LEGACY_TS_ATTRIBUTE_FALLBACK !== "0";
const execFileAsync = promisify(execFile);

type PythonAttributePayload = {
  attributes: Record<string, number>;
  attributeGroups: Record<string, Record<string, number>>;
  tendencies?: Record<string, number>;
  tendencyGroups?: Record<string, Record<string, number>>;
  playStylePriorities?: string[];
};

function validateSeason(season: number): boolean {
  return Number.isInteger(season) && season >= 2000 && season <= 2025;
}

function seasonToLabel(season: number): string {
  return `${season}-${String(season + 1).slice(2)}`;
}

function loadRegistry(): TendencyRegistryRow[] {
  if (registryCache) return registryCache;

  const candidates = [
    path.resolve(process.cwd(), "..", "data", "tendency_registry.json"),
    path.resolve(process.cwd(), "data", "tendency_registry.json"),
  ];

  for (const filePath of candidates) {
    try {
      const raw = fs.readFileSync(filePath, "utf-8");
      const parsed = JSON.parse(raw) as TendencyRegistryRow[];
      if (Array.isArray(parsed) && parsed.length > 0) {
        registryCache = parsed;
        return registryCache;
      }
    } catch {
      // Try next candidate path.
    }
  }

  registryCache = [];
  return registryCache;
}

function canonicalizeTendencies(entries: Record<string, BackendTendencyEntry>): Record<string, number> {
  const registry = loadRegistry();
  const byPrimjerKey = new Map(registry.map((row) => [row.primjer_key, row.canonical_name]));
  const byLabel = new Map(registry.map((row) => [row.primjer_label.toLowerCase(), row.canonical_name]));

  const out: Record<string, number> = {};
  for (const [key, entry] of Object.entries(entries || {})) {
    const canonical = byPrimjerKey.get(key) || byLabel.get(String(entry?.label ?? "").toLowerCase()) || key;
    out[canonical] = Number(entry?.value ?? 0);
  }
  return out;
}

function groupTendencies(tendencies: Record<string, number>): Record<string, Record<string, number>> {
  const registry = loadRegistry();
  const byCanonical = new Map(registry.map((row) => [row.canonical_name, row]));

  const grouped: Record<string, Record<string, number>> = {};
  for (const [canonical, value] of Object.entries(tendencies)) {
    const row = byCanonical.get(canonical);
    const category = row?.category || "other";
    const label = row?.primjer_label || canonical;
    if (!grouped[category]) grouped[category] = {};
    grouped[category][label] = value;
  }
  return grouped;
}

function groupAttributes(attributes: Record<string, BackendAttributeEntry>): Record<string, Record<string, number>> {
  const grouped: Record<string, Record<string, number>> = {};
  for (const [canonical, entry] of Object.entries(attributes || {})) {
    const category = String(entry?.category || "other");
    const label = String(entry?.label || canonical);
    if (!grouped[category]) grouped[category] = {};
    grouped[category][label] = Number(entry?.value ?? 0);
  }
  return grouped;
}

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function avgByKeys(attributes: Record<string, number>, keys: string[]): number {
  return avg(keys.map((key) => Number(attributes[key] ?? 0)));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

type RoleProfile = "guard" | "wing" | "big";

function roleProfileFromPosition(position: string): RoleProfile {
  const pos = String(position || "").toUpperCase();
  const isGuard = pos.includes("PG") || pos.includes("SG") || pos === "G";
  const isCenter = pos === "C" || pos.includes("-C") || pos.includes("C-") || pos.includes("/C") || pos.includes("C/");
  if (isGuard) return "guard";
  if (isCenter || pos.includes("PF") || pos === "F-C" || pos === "C-F") return "big";
  return "wing";
}

function isCenterPosition(position: string): boolean {
  const pos = String(position || "").toUpperCase();
  return pos === "C" || pos.includes("-C") || pos.includes("C-") || pos.includes("/C") || pos.includes("C/");
}

function topNAvg(values: number[], n: number): number {
  return avg(
    values
      .slice()
      .sort((a, b) => b - a)
      .slice(0, Math.max(1, n)),
  );
}

function computeCategoryScores(attributes: Record<string, number>, position: string): { offense: number; defense: number; physical: number } {
  const role = roleProfileFromPosition(position);
  const isCenter = isCenterPosition(position);

  const scoringSkill = avgByKeys(attributes, [
    "close_shot",
    "mid_range_shot",
    "three_point_shot",
    "free_throw",
    "driving_layup",
  ]);

  const creationSkill = avgByKeys(attributes, [
    "ball_handle",
    "speed_with_ball",
    "pass_accuracy",
    "pass_vision",
    "shot_iq",
    "offensive_consistency",
  ]);

  const interiorPressure = avgByKeys(attributes, [
    "driving_dunk",
    "standing_dunk",
    "post_control",
    "post_hook",
    "post_fade",
    "offensive_rebound",
  ]);

  const rimPressureGuard = avgByKeys(attributes, [
    "driving_layup",
    "draw_foul",
    "close_shot",
    "speed_with_ball",
  ]);

  const rimPressureWing = avgByKeys(attributes, [
    "driving_layup",
    "driving_dunk",
    "draw_foul",
    "close_shot",
  ]);

  const wingPostTouch = avgByKeys(attributes, [
    "post_control",
    "post_fade",
  ]);

  const scoringSkillBig = avgByKeys(attributes, [
    "close_shot",
    "mid_range_shot",
    "three_point_shot",
    "free_throw",
  ]);

  const bigCreation = avgByKeys(attributes, [
    "pass_accuracy",
    "pass_vision",
    "shot_iq",
    "offensive_consistency",
  ]);

  const offensiveTop = topNAvg(
    [
      "close_shot",
      "mid_range_shot",
      "three_point_shot",
      "driving_layup",
      "driving_dunk",
      "post_control",
      "ball_handle",
      "speed_with_ball",
      "pass_accuracy",
      "pass_vision",
      "shot_iq",
      "offensive_consistency",
    ].map((key) => Number(attributes[key] ?? 0)),
    6,
  );

  let offense = 0;
  if (role === "guard") {
    offense = 0.4 * scoringSkill + 0.45 * creationSkill + 0.15 * rimPressureGuard;
  } else if (role === "big") {
    offense = isCenter
      ? 0.28 * scoringSkillBig + 0.14 * bigCreation + 0.58 * interiorPressure
      : 0.34 * scoringSkillBig + 0.22 * bigCreation + 0.44 * interiorPressure;
  } else {
    const wingPressure = 0.85 * rimPressureWing + 0.15 * wingPostTouch;
    offense = 0.4 * scoringSkill + 0.35 * creationSkill + 0.25 * wingPressure;
  }
  offense = clamp(0.88 * offense + 0.12 * offensiveTop, 25, 99);

  const perimeterDefense = avgByKeys(attributes, [
    "perimeter_defense",
    "steal",
    "pass_perception",
    "help_defense_iq",
  ]);
  const interiorDefense = avgByKeys(attributes, [
    "interior_defense",
    "block",
    "defensive_rebound",
    "strength",
  ]);
  const teamDefense = avgByKeys(attributes, [
    "help_defense_iq",
    "pass_perception",
    "defensive_consistency",
    "hustle",
  ]);

  let defense = 0;
  if (role === "guard") {
    defense = 0.5 * perimeterDefense + 0.19 * interiorDefense + 0.31 * teamDefense;
  } else if (role === "big") {
    defense = 0.23 * perimeterDefense + 0.52 * interiorDefense + 0.25 * teamDefense;
  } else {
    defense = 0.4 * perimeterDefense + 0.3 * interiorDefense + 0.3 * teamDefense;
  }

  const athleticTools = avgByKeys(attributes, ["speed", "agility", "vertical", "speed_with_ball"]);
  const forceTools = avgByKeys(attributes, ["strength", "stamina", "hustle"]);

  let physical = 0;
  if (role === "guard") {
    physical = 0.62 * athleticTools + 0.38 * forceTools;
  } else if (role === "big") {
    physical = 0.43 * athleticTools + 0.57 * forceTools;
  } else {
    physical = 0.54 * athleticTools + 0.46 * forceTools;
  }

  return {
    offense: clamp(offense, 25, 99),
    defense: clamp(defense, 25, 99),
    physical: clamp(physical, 25, 99),
  };
}

function computeOverall(attributes: Record<string, number>, position: string): number {
  const pos = String(position || "").toUpperCase();
  const isGuard = pos.includes("PG") || pos.includes("SG") || pos === "G";
  const isCenter = isCenterPosition(position);
  const isPowerForward = !isCenter && (pos.includes("PF") || pos === "F-C" || pos === "C-F");
  const { offense, defense, physical } = computeCategoryScores(attributes, position);

  const excluded = new Set(["intangibles", "overall_durability", "potential"]);
  const includedEntries = Object.entries(attributes).filter(([key]) => !excluded.has(key));

  const weightFor = (key: string): number => {
    // Base weight for all included attributes.
    let w = 1.0;

    if (isGuard) {
      if (["three_point_shot", "mid_range_shot", "free_throw", "driving_layup", "ball_handle", "speed_with_ball", "pass_accuracy", "pass_vision", "shot_iq", "offensive_consistency"].includes(key)) w *= 1.45;
      if (["perimeter_defense", "steal", "pass_perception", "help_defense_iq", "speed", "agility", "stamina"].includes(key)) w *= 1.2;
      if (["standing_dunk", "post_hook", "post_fade", "post_control", "offensive_rebound", "interior_defense", "block"].includes(key)) w *= 0.6;
    } else if (isCenter) {
      if (["close_shot", "driving_dunk", "standing_dunk", "post_hook", "post_fade", "post_control", "offensive_rebound", "defensive_rebound", "interior_defense", "block", "strength"].includes(key)) w *= 1.5;
      if (["help_defense_iq", "pass_perception", "defensive_consistency", "hustle", "stamina"].includes(key)) w *= 1.2;
      if (["ball_handle", "speed_with_ball", "perimeter_defense", "steal"].includes(key)) w *= 0.7;
    } else if (isPowerForward) {
      if (["close_shot", "driving_dunk", "standing_dunk", "post_hook", "post_fade", "post_control", "offensive_rebound", "defensive_rebound", "interior_defense", "block", "strength"].includes(key)) w *= 1.35;
      if (["mid_range_shot", "three_point_shot", "pass_accuracy", "pass_vision", "shot_iq", "offensive_consistency", "perimeter_defense", "help_defense_iq"].includes(key)) w *= 1.1;
      if (["ball_handle", "speed_with_ball"].includes(key)) w *= 0.8;
    } else {
      // Wings: balanced but still lean to scoring + perimeter two-way play.
      if (["three_point_shot", "mid_range_shot", "driving_layup", "ball_handle", "speed_with_ball", "pass_accuracy", "pass_vision", "shot_iq", "offensive_consistency"].includes(key)) w *= 1.28;
      if (["perimeter_defense", "steal", "help_defense_iq", "pass_perception", "defensive_consistency", "speed", "agility", "stamina"].includes(key)) w *= 1.18;
      if (["standing_dunk", "post_hook"].includes(key)) w *= 0.82;
    }

    return w;
  };

  const weightedPairs = includedEntries.map(([key, value]) => {
    const raw = Number(value ?? 0);
    const weight = weightFor(key);
    return { key, raw, weight, weighted: raw * weight };
  });

  const weightedSum = weightedPairs.reduce((sum, p) => sum + p.weighted, 0);
  const weightTotal = weightedPairs.reduce((sum, p) => sum + p.weight, 0);
  const weightedAvg = weightTotal > 0 ? weightedSum / weightTotal : 0;

  const allValues = weightedPairs.map((p) => p.raw);
  const topCore = topNAvg(allValues, 10);
  const upperCore = topNAvg(allValues, 5);
  const eliteCount = allValues.filter((value) => value >= 90).length;

  const base = 0.68 * weightedAvg + 0.22 * topCore + 0.1 * upperCore;
  const eliteBoost = 3.8 * clamp((topCore - 86) / 8, 0, 1) + 2.2 * clamp((eliteCount - 4) / 7, 0, 1);

  let positionBoost = 0;
  if (isGuard) {
    const creatorCore =
      0.24 * Number(attributes.ball_handle ?? 0) +
      0.22 * Number(attributes.speed_with_ball ?? 0) +
      0.16 * Number(attributes.pass_accuracy ?? 0) +
      0.22 * Number(attributes.pass_vision ?? 0) +
      0.16 * Number(attributes.shot_iq ?? 0);

    const scoringCore =
      0.35 * Number(attributes.three_point_shot ?? 0) +
      0.25 * Number(attributes.mid_range_shot ?? 0) +
      0.2 * Number(attributes.close_shot ?? 0) +
      0.2 * Number(attributes.driving_layup ?? 0);

    const offensiveConsistency = Number(attributes.offensive_consistency ?? 0);

    // Guards can carry overall more through offensive engine value.
    positionBoost += 11.0 * clamp((creatorCore - 84) / 8, 0, 1);
    positionBoost += 6.0 * clamp((scoringCore - 80) / 10, 0, 1);
    positionBoost += 3.0 * clamp((offensiveConsistency - 86) / 8, 0, 1);
    positionBoost += 5.0 * clamp((offense - 84) / 8, 0, 1);
  } else if (isCenter) {
    const interiorAnchor = avgByKeys(attributes, ["interior_defense", "block", "defensive_rebound", "strength", "help_defense_iq"]);
    const insideOffense = avgByKeys(attributes, ["close_shot", "standing_dunk", "post_hook", "post_fade", "post_control", "offensive_rebound"]);

    // Centers should lean into interior impact on both ends.
    positionBoost += 4.0 * clamp((interiorAnchor - 84) / 10, 0, 1);
    positionBoost += 2.5 * clamp((insideOffense - 82) / 10, 0, 1);
    positionBoost += 2.0 * clamp((physical - 83) / 10, 0, 1);
  }

  return clamp(Math.round(base + eliteBoost + positionBoost), 25, 99);
}

function roleFromPriorities(playStylePriorities?: string[]): string {
  const cleaned = (playStylePriorities || []).map((x) => String(x || "").trim()).filter(Boolean);
  if (cleaned.length >= 2) return `${cleaned[0]} / ${cleaned[1]}`;
  if (cleaned.length === 1) return cleaned[0];
  return "Balanced Creator";
}

function calibrateAttributes(
  attributes: Record<string, number>,
  position: string,
  currentStats: { PTS: number; REB: number; AST: number },
  age?: number,
): Record<string, number> {
  const out: Record<string, number> = { ...attributes };
  const pos = String(position || "").toUpperCase();
  const posNorm = pos.replace(/[-_/]/g, " ").replace(/\s+/g, " ").trim();
  const isCenter = isCenterPosition(position);
  const hasSG = /\bSG\b/.test(posNorm) || /SHOOTING\s+GUARD/.test(posNorm);
  const hasSF = /\bSF\b/.test(posNorm) || /SMALL\s+FORWARD/.test(posNorm);
  const hasPF = /\bPF\b/.test(posNorm) || /POWER\s+FORWARD/.test(posNorm);
  const hasForward = /\bF\b/.test(posNorm) || /FORWARD/.test(posNorm);
  const hasGuard = /\bG\b/.test(posNorm) || /GUARD/.test(posNorm);
  const isPointGuard = /\bPG\b/.test(posNorm) || /POINT\s+GUARD/.test(posNorm);
  const isWing = !isCenter && (hasSF || hasPF || hasSG || hasForward || (hasGuard && !isPointGuard));

  const ppg = Number(currentStats.PTS ?? 0);
  const apg = Number(currentStats.AST ?? 0);
  const rpg = Number(currentStats.REB ?? 0);
  const playerAge = Number(age ?? 0);

  const at = (key: string): number => Number(out[key] ?? 0);
  const raiseTo = (key: string, min: number) => {
    out[key] = Math.max(at(key), min);
  };

  const isPrimaryWingScorer = isWing && ppg >= 24;
  const isEliteWingScorer = isWing && ppg >= 27;
  const wingShotCreatorSignal = isWing
    ? (
      0.25 * at("shot_iq")
      + 0.2 * at("offensive_consistency")
      + 0.15 * at("ball_handle")
      + 0.12 * at("speed_with_ball")
      + 0.12 * at("driving_layup")
      + 0.1 * at("mid_range_shot")
      + 0.06 * at("three_point_shot")
    )
    : 0;
  const explosiveWingFinisherSignal = isWing
    ? (
      0.26 * at("driving_layup")
      + 0.22 * at("draw_foul")
      + 0.16 * at("speed_with_ball")
      + 0.15 * at("vertical")
      + 0.11 * at("strength")
      + 0.10 * at("close_shot")
    )
    : 0;
  const nonCenterAthleticFinisherSignal = !isCenter
    ? (
      0.28 * at("driving_layup")
      + 0.22 * at("speed_with_ball")
      + 0.20 * at("vertical")
      + 0.14 * at("draw_foul")
      + 0.10 * at("strength")
      + 0.06 * at("close_shot")
    )
    : 0;
  const isTrueExplosiveAthlete = !isCenter
    && at("driving_layup") >= 86
    && at("speed_with_ball") >= 84
    && at("vertical") >= 85
    && at("agility") >= 82;
  const isUltraExplosiveAthlete = !isCenter
    && at("driving_layup") >= 88
    && at("speed_with_ball") >= 86
    && at("vertical") >= 88
    && at("agility") >= 84;

  if (isPrimaryWingScorer) {
    raiseTo("close_shot", 82);
    raiseTo("driving_layup", 84);
    raiseTo("driving_dunk", 78);
    raiseTo("mid_range_shot", 83);
    raiseTo("three_point_shot", 80);
    raiseTo("ball_handle", 80);
    raiseTo("speed_with_ball", 80);
    raiseTo("pass_accuracy", 70);
    raiseTo("pass_vision", 68);
    raiseTo("draw_foul", 78);
    raiseTo("offensive_consistency", 85);
    raiseTo("shot_iq", 88);

    // High-minute scoring wings should not grade as weak perimeter defenders by default.
    raiseTo("perimeter_defense", 78);
    raiseTo("help_defense_iq", 70);
    raiseTo("pass_perception", 68);
    raiseTo("steal", 66);
    raiseTo("defensive_consistency", 77);
  }

  if (isEliteWingScorer) {
    raiseTo("close_shot", 84);
    raiseTo("driving_layup", 86);
    raiseTo("driving_dunk", 80);
    raiseTo("mid_range_shot", 85);
    raiseTo("three_point_shot", 82);
    raiseTo("ball_handle", 82);
    raiseTo("speed_with_ball", 82);
    raiseTo("offensive_consistency", 87);
  }

  // Protect elite power slashing wings from landing in unrealistic dunk bands.
  // This catches players whose shot profile is balanced enough that dunk can be
  // under-produced by upstream formulas despite clear rim-pressure tools.
  if (isWing && ppg >= 24 && explosiveWingFinisherSignal >= 80) {
    raiseTo("driving_dunk", 86);
    raiseTo("driving_layup", 86);
    raiseTo("close_shot", 84);
    raiseTo("draw_foul", 80);
    raiseTo("vertical", 84);
    raiseTo("strength", 72);
  }

  if (isWing && ppg >= 26 && explosiveWingFinisherSignal >= 83) {
    raiseTo("driving_dunk", 88);
    raiseTo("driving_layup", 88);
    raiseTo("close_shot", 85);
    raiseTo("draw_foul", 82);
    raiseTo("speed_with_ball", 82);
    raiseTo("offensive_consistency", 88);
  }

  // Position/source-noise fallback: if the profile is clearly an explosive
  // non-center finisher, force realistic dunk floors even when PTS or
  // position labels are inconsistent in upstream data.
  const nonCenterExplosiveFinisher = !isCenter
    && explosiveWingFinisherSignal >= 79
    && at("driving_layup") >= 82
    && at("speed_with_ball") >= 78
    && at("vertical") >= 78;

  if (nonCenterExplosiveFinisher) {
    raiseTo("driving_dunk", 86);
    raiseTo("driving_layup", 86);
    raiseTo("close_shot", 84);
    raiseTo("draw_foul", 80);
  }

  // Hard fallback for explosive non-centers: avoid unrealistic low dunk ratings
  // when upstream context (season stats/labels) is incomplete.
  if (!isCenter && nonCenterAthleticFinisherSignal >= 80 && isTrueExplosiveAthlete) {
    raiseTo("driving_dunk", 88);
  }

  if (
    !isCenter
    && nonCenterAthleticFinisherSignal >= 84
    && isUltraExplosiveAthlete
    && (playerAge <= 31 || playerAge === 0)
  ) {
    raiseTo("driving_dunk", 90);
    raiseTo("vertical", 86);
    raiseTo("draw_foul", 82);
  }

  // Aging curve guardrail: older non-centers should not be forced into
  // peak-explosiveness dunk floors by generic archetype logic.
  if (!isCenter && playerAge >= 34) {
    out["driving_dunk"] = Math.min(at("driving_dunk"), 86);
  }

  // Fallback path: when PTS-driven trigger misses (partial/noisy season),
  // still protect realistic shot ratings for clear wing creators/scorers.
  if (isWing && wingShotCreatorSignal >= 80) {
    raiseTo("mid_range_shot", 84);
    raiseTo("three_point_shot", 80);
    raiseTo("shot_iq", 87);
  }

  if (isWing && wingShotCreatorSignal >= 83) {
    raiseTo("mid_range_shot", 86);
    raiseTo("three_point_shot", 82);
    raiseTo("offensive_consistency", 86);
  }

  // Small bump for high-usage non-center creators to avoid low playmaking floors.
  if (!isCenter && ppg >= 23 && apg >= 4) {
    raiseTo("pass_accuracy", 72);
    raiseTo("pass_vision", 70);
  }

  // Rebounding wings/forwards should keep a sensible floor.
  if (!isCenter && rpg >= 7) {
    raiseTo("defensive_rebound", 80);
  }

  return out;
}

function deriveArchetypes(attributes: Record<string, number>, position: string, playStylePriorities: string[]): string[] {
  const pos = String(position || "").toUpperCase();
  const isGuard = pos.includes("PG") || pos.includes("SG") || pos === "G";
  const isCenter = pos === "C" || pos.includes("-C") || pos.includes("C-") || pos.includes("/C") || pos.includes("C/");
  const isBig = isCenter || pos.includes("PF") || pos === "F-C" || pos === "C-F";

  const styles = (playStylePriorities || []).map((x) => String(x || "").trim().toLowerCase());
  const hasStyle = (needle: string) => styles.some((x) => x.includes(needle));

  const get = (key: string) => Number(attributes[key] ?? 0);
  const shooting = avgByKeys(attributes, ["three_point_shot", "mid_range_shot", "free_throw"]);
  const creation = avgByKeys(attributes, ["ball_handle", "pass_accuracy", "pass_vision", "speed_with_ball"]);
  const finishing = avgByKeys(attributes, ["close_shot", "driving_layup", "driving_dunk", "standing_dunk"]);
  const perimeterDefense = avgByKeys(attributes, ["perimeter_defense", "steal", "pass_perception", "help_defense_iq"]);
  const interiorDefense = avgByKeys(attributes, ["interior_defense", "block", "defensive_rebound", "strength"]);
  const postScoring = avgByKeys(attributes, ["post_hook", "post_fade", "post_control"]);
  const physicalTools = avgByKeys(attributes, ["speed", "agility", "vertical", "stamina"]);

  const scores = new Map<string, number>();
  const add = (name: string, value: number) => {
    if (value <= 0) return;
    scores.set(name, (scores.get(name) || 0) + value);
  };

  if (hasStyle("3pt")) add("Sharpshooter", 3.2);
  if (hasStyle("mid range")) add("Mid-Range Specialist", 3.0);
  if (hasStyle("pick and roll ball handler") || hasStyle("pick and roll point")) add("Pick-and-Roll Engine", 3.2);
  if (hasStyle("pick and roll wing")) add("Off-Ball Wing Threat", 2.4);
  if (hasStyle("pick and roll rollman")) add("Rim-Running Roll Threat", 3.0);
  if (hasStyle("isolation point")) add("Point Isolation Creator", 3.0);
  if (hasStyle("isolation wing")) add("Wing Isolation Scorer", 3.0);
  if (hasStyle("isolation")) add("Shot-Creation Hub", 2.8);
  if (hasStyle("post up high")) add("High-Post Facilitator", 2.8);
  if (hasStyle("post up low")) add("Low-Post Power Scorer", 3.0);
  if (hasStyle("guard post up")) add("Mismatch Post Guard", 2.6);
  if (hasStyle("handoff pass")) add("DHO Playmaking Hub", 2.8);
  if (hasStyle("handoff receiver")) add("DHO Movement Receiver", 2.6);
  if (hasStyle("cutter")) add("Backdoor Cutting Threat", 2.6);

  if (shooting >= 86 && get("three_point_shot") >= 88) add("Elite Floor Spacer", 2.6);
  if (shooting >= 84 && get("mid_range_shot") >= 86) add("Three-Level Shotmaker", 2.0);
  if (creation >= 88 && (hasStyle("pick and roll") || hasStyle("isolation"))) add("Primary Offensive Engine", 3.0);
  if (creation >= 84 && get("pass_vision") >= 88) add(isBig ? "Point Big Creator" : "Playmaking Lead Guard", 2.8);
  if (finishing >= 86 && physicalTools >= 82) add("Rim Pressure Slasher", 2.6);
  if (postScoring >= 86) add(isCenter ? "Post Scoring Anchor" : "Face-Up Post Scorer", 2.6);
  if (perimeterDefense >= 85 && get("perimeter_defense") >= 87) add("Perimeter Lockdown Defender", 2.6);
  if (interiorDefense >= 86 && get("block") >= 84) add("Rim Protector Anchor", 2.8);
  if (perimeterDefense >= 84 && shooting >= 82 && !isCenter) add("Two-Way Wing", 2.4);
  if (isBig && creation >= 82 && get("pass_vision") >= 86) add("Point Forward/Center Hub", 2.4);
  if (get("offensive_rebound") >= 85 && finishing >= 82) add("Second-Chance Interior Finisher", 2.2);
  if (get("stamina") >= 90 && physicalTools >= 84) add("High-Motor Two-Way", 1.8);

  if (isGuard) {
    add("Perimeter Shot Creator", 1.2 * clamp((creation - 80) / 10, 0, 1));
  } else if (isCenter) {
    add("Interior Defensive Anchor", 1.4 * clamp((interiorDefense - 82) / 10, 0, 1));
  } else {
    add("Versatile Two-Way Forward", 1.2 * clamp((0.5 * perimeterDefense + 0.5 * finishing - 80) / 10, 0, 1));
  }

  const ranked = Array.from(scores.entries())
    .sort((a, b) => b[1] - a[1])
    .map(([name]) => name);

  if (ranked.length === 0) return ["Balanced Creator"];
  return ranked.slice(0, 4);
}

function resolveDisplayName(backendName: string | undefined, fallbackName: string, playerId: string): string {
  const candidate = String(backendName || "").trim();
  if (!candidate) return fallbackName;

  const fallback = String(fallbackName || "").trim();
  const idPattern = new RegExp(`^player\\s*${playerId}$`, "i");

  if (idPattern.test(candidate)) return fallback || candidate;
  if (/^player\s+\d+$/i.test(candidate)) return fallback || candidate;
  if (/^\d+$/.test(candidate)) return fallback || candidate;

  return candidate;
}

async function tryBackendPipeline(playerId: string, season: number, authorization?: string): Promise<GeneratedPlayerPayload | null> {
  if (!authorization) return null;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), BACKEND_REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/generate/id/${encodeURIComponent(playerId)}?season=${encodeURIComponent(seasonToLabel(season))}`, {
      method: "GET",
      headers: {
        Authorization: authorization,
      },
      cache: "no-store",
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Backend generation timed out after ${BACKEND_REQUEST_TIMEOUT_MS}ms.`);
    }
    throw new Error(`Backend service unavailable at ${API_BASE_URL}.`);
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      throw new Error("BACKEND_UNAUTHORIZED");
    }
    const detail = await response.text();
    throw new Error(detail || `BACKEND_FAILURE_${response.status}`);
  }

  const backend = (await response.json()) as BackendGenerateResponse;
  let infoAndStats = null;
  try {
    infoAndStats = await fetchPlayerStatsFromSupabase(playerId, STATS_CARD_SEASON);
  } catch {
    // Fallback keeps API resilient if a specific player is missing in 2025-26.
    infoAndStats = await fetchPlayerStatsFromSupabase(playerId, season);
  }
  const { info, stats } = infoAndStats;

  const rawAttributes = Object.fromEntries(
    Object.entries(backend.attributes || {}).map(([key, entry]) => [key, Number(entry?.value ?? 0)]),
  );
  const attributes = calibrateAttributes(rawAttributes, backend.position || info.position, stats.current, info.age);
  const adjustedBackendAttributes = Object.fromEntries(
    Object.entries(backend.attributes || {}).map(([key, entry]) => [
      key,
      {
        ...entry,
        value: Number(attributes[key] ?? 0),
      },
    ]),
  ) as Record<string, BackendAttributeEntry>;
  const tendencies = canonicalizeTendencies(backend.tendencies || {});

  const flatAttrs = Object.entries(adjustedBackendAttributes).map(([canonical, entry]) => ({
    label: entry?.label || canonical,
    value: Number(entry?.value ?? 0),
  }));

  const strengths = flatAttrs
    .slice()
    .sort((a, b) => b.value - a.value)
    .slice(0, 4)
    .map((x) => x.label);

  const weaknesses = flatAttrs
    .slice()
    .sort((a, b) => a.value - b.value)
    .slice(0, 4)
    .map((x) => x.label);

  const ovr = computeOverall(attributes, backend.position || info.position);
  const playStylePriorities = (backend.play_style_priorities || []).map((x) => String(x || "").trim()).filter(Boolean);
  const archetypes = deriveArchetypes(attributes, backend.position || info.position, playStylePriorities);

  return {
    info: {
      ...info,
      name: resolveDisplayName(backend.player_name, info.name, playerId),
      team: backend.team || info.team,
      position: backend.position || info.position,
    },
    stats,
    attributes,
    tendencies,
    attributeGroups: groupAttributes(adjustedBackendAttributes),
    tendencyGroups: groupTendencies(tendencies),
    archetype: archetypes[0] || "Balanced Creator",
    archetypes,
    strengths,
    weaknesses,
    role: roleFromPriorities(playStylePriorities),
    playStylePriorities,
    usage: playStylePriorities.slice(0, 4).map((x) => `Lean into ${String(x).toLowerCase()} actions.`),
    ovr,
  };
}

async function tryLocalPipeline(playerId: string, season: number): Promise<GeneratedPlayerPayload> {
  let infoAndStats = null;
  try {
    infoAndStats = await fetchPlayerStatsFromSupabase(playerId, STATS_CARD_SEASON);
  } catch {
    infoAndStats = await fetchPlayerStatsFromSupabase(playerId, season);
  }

  const { info, stats } = infoAndStats;
  const generated = generatePlayer(stats, { position: info.position });

  const pythonAttributes = await calculateSupabaseAttributesWithPython(playerId, season);
  if (!pythonAttributes && !ALLOW_LEGACY_TS_ATTRIBUTE_FALLBACK) {
    throw new Error("Python attribute calculator failed; legacy TS attribute fallback is disabled.");
  }
  const attributes = pythonAttributes?.attributes ?? generated.attributes;
  const attributeGroups = pythonAttributes?.attributeGroups ?? generated.attributeGroups;
  const tendencies = pythonAttributes?.tendencies ?? generated.tendencies;
  const tendencyGroups = pythonAttributes?.tendencyGroups ?? generated.tendencyGroups;
  const playStylePriorities = pythonAttributes?.playStylePriorities ?? generated.playStylePriorities;
  const role = playStylePriorities.length > 0 ? roleFromPriorities(playStylePriorities) : generated.role;
  const usage = playStylePriorities.length > 0
    ? playStylePriorities.slice(0, 4).map((x) => `Lean into ${String(x).toLowerCase()} actions.`)
    : generated.usage;

  const strengths = Object.entries(attributes)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([k]) => k);

  const weaknesses = Object.entries(attributes)
    .sort((a, b) => a[1] - b[1])
    .slice(0, 4)
    .map(([k]) => k);

  const ovr = computeOverall(attributes, info.position);

  return {
    info,
    stats,
    attributes,
    tendencies,
    attributeGroups,
    tendencyGroups,
    archetype: generated.archetype,
    archetypes: generated.archetypes,
    strengths,
    weaknesses,
    role,
    playStylePriorities,
    usage,
    ovr,
  };
}

async function calculateSupabaseAttributesWithPython(playerId: string, season: number): Promise<PythonAttributePayload | null> {
  const repoRoot = path.resolve(process.cwd(), "..");
  const scriptPath = path.join(repoRoot, "scripts", "supabase_attribute_calc.py");
  if (!fs.existsSync(scriptPath)) return null;

  const seasonStart = String(Number.isInteger(season) ? season : STATS_CARD_SEASON);
  const candidates = [
    process.env.N2K_PYTHON_EXECUTABLE,
    process.platform === "win32"
      ? path.join(repoRoot, ".venv", "Scripts", "python.exe")
      : path.join(repoRoot, ".venv", "bin", "python"),
    "python3",
    "python",
  ].filter((value): value is string => Boolean(value && value.trim()));

  for (const pythonExe of candidates) {
    try {
      const { stdout } = await execFileAsync(pythonExe, [scriptPath, String(playerId), seasonStart], {
        cwd: repoRoot,
        timeout: 15_000,
        windowsHide: true,
        maxBuffer: 1024 * 1024,
      });

      const parsed = JSON.parse(String(stdout || "{}")) as PythonAttributePayload;
      if (!parsed || !parsed.attributes || !parsed.attributeGroups) continue;
      return parsed;
    } catch {
      // Try next candidate executable.
    }
  }

  return null;
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as Partial<GeneratorRequest>;
    const playerId = String(body.playerId ?? "").trim();
    const season = Number(body.season);
    const forceRefresh = Boolean((body as { forceRefresh?: boolean }).forceRefresh);

    if (!playerId) {
      return NextResponse.json({ error: "playerId is required." }, { status: 400 });
    }

    if (!validateSeason(season)) {
      return NextResponse.json({ error: "season must be between 2000 and 2025." }, { status: 400 });
    }

    const cacheKey = `${playerId}:${season}`;
    const cached = responseCache.get(cacheKey);
    if (!DISABLE_RESPONSE_CACHE && !forceRefresh && cached && Date.now() - cached.cachedAt < RESPONSE_CACHE_TTL_MS) {
      return NextResponse.json({ ...cached.payload, generationSource: "cache" }, { status: 200 });
    }

    if (GENERATION_MODE !== "backend") {
      const localPayload = await tryLocalPipeline(playerId, season);
      if (!DISABLE_RESPONSE_CACHE) {
        responseCache.set(cacheKey, { payload: localPayload, cachedAt: Date.now() });
      }
      return NextResponse.json({ ...localPayload, generationSource: "supabase-local" }, { status: 200 });
    }

    const authHeader = request.headers.get("authorization") || undefined;

    let backendPayload: GeneratedPlayerPayload | null = null;
    try {
      backendPayload = await tryBackendPipeline(playerId, season, authHeader);
    } catch (error) {
      const code = error instanceof Error ? error.message : "";
      if (code === "BACKEND_UNAUTHORIZED") {
        return NextResponse.json(
          { error: "Backend auth failed. Please log out and log back in, then regenerate." },
          { status: 401 },
        );
      }
      if (!ENABLE_LOCAL_FALLBACK) {
        return NextResponse.json(
          { error: "Backend generation unavailable. Local fallback is disabled to preserve shot-based accuracy." },
          { status: 502 },
        );
      }
      backendPayload = null;
    }

    if (backendPayload) {
      if (!DISABLE_RESPONSE_CACHE) {
        responseCache.set(cacheKey, { payload: backendPayload, cachedAt: Date.now() });
      }
      return NextResponse.json({ ...backendPayload, generationSource: "backend" }, { status: 200 });
    }

    if (ENABLE_LOCAL_FALLBACK) {
      const localPayload = await tryLocalPipeline(playerId, season);
      if (!DISABLE_RESPONSE_CACHE) {
        responseCache.set(cacheKey, { payload: localPayload, cachedAt: Date.now() });
      }
      return NextResponse.json({ ...localPayload, generationSource: "local-fallback" }, { status: 200 });
    }

    return NextResponse.json(
      { error: "Backend generation unavailable. Local fallback is disabled to preserve shot-based accuracy." },
      { status: 502 },
    );

  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to generate player.";
    if (message === "BACKEND_UNAUTHORIZED") {
      return NextResponse.json(
        { error: "Backend auth failed. Please log out and log back in, then regenerate." },
        { status: 401 },
      );
    }
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
