import type { GeneratedOutput, GeneratorSourceStats, PlayerInfo } from "@/lib/generator/types";
import fs from "node:fs";
import path from "node:path";

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function safe(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function round(value: number): number {
  return Math.round(value);
}

type TendencyRegistryRow = {
  canonical_name: string;
  primjer_label?: string;
  category?: string;
  hard_cap?: number;
  parent_tendency?: string | null;
  is_sub_zone?: boolean;
};

const ATTRIBUTE_LABELS: Record<string, string> = {
  driving_layup: "Driving Layup",
  standing_dunk: "Standing Dunk",
  driving_dunk: "Driving Dunk",
  close_shot: "Close Shot",
  mid_range_shot: "Mid-Range Shot",
  three_point_shot: "Three-Point Shot",
  free_throw: "Free Throw",
  post_hook: "Post Hook",
  post_fade: "Post Fade",
  post_control: "Post Control",
  draw_foul: "Draw Foul",
  shot_iq: "Shot IQ",
  ball_handle: "Ball Handle",
  speed_with_ball: "Speed with Ball",
  hands: "Hands",
  pass_accuracy: "Pass Accuracy",
  pass_iq: "Pass IQ",
  pass_vision: "Pass Vision",
  offensive_consistency: "Offensive Consistency",
  interior_defense: "Interior Defense",
  perimeter_defense: "Perimeter Defense",
  steal: "Steal",
  block: "Block",
  offensive_rebound: "Offensive Rebound",
  defensive_rebound: "Defensive Rebound",
  help_defense_iq: "Help Defense IQ",
  pass_perception: "Pass Perception",
  defensive_consistency: "Defensive Consistency",
  speed: "Speed",
  agility: "Agility",
  strength: "Strength",
  vertical: "Vertical",
  stamina: "Stamina",
  intangibles: "Intangibles",
  hustle: "Hustle",
  overall_durability: "Overall Durability",
  potential: "Potential",
};

const ATTRIBUTE_CATEGORIES: Record<string, string> = {
  driving_layup: "finishing",
  standing_dunk: "finishing",
  driving_dunk: "finishing",
  close_shot: "finishing",
  mid_range_shot: "shooting",
  three_point_shot: "shooting",
  free_throw: "shooting",
  post_hook: "post_game",
  post_fade: "post_game",
  post_control: "post_game",
  draw_foul: "playmaking",
  shot_iq: "shooting",
  ball_handle: "playmaking",
  speed_with_ball: "playmaking",
  hands: "playmaking",
  pass_accuracy: "playmaking",
  pass_iq: "playmaking",
  pass_vision: "playmaking",
  offensive_consistency: "mental",
  interior_defense: "defense",
  perimeter_defense: "defense",
  steal: "defense",
  block: "defense",
  offensive_rebound: "rebounding",
  defensive_rebound: "rebounding",
  help_defense_iq: "defense",
  pass_perception: "defense",
  defensive_consistency: "mental",
  speed: "physical",
  agility: "physical",
  strength: "physical",
  vertical: "physical",
  stamina: "physical",
  intangibles: "meta",
  hustle: "meta",
  overall_durability: "meta",
  potential: "meta",
};

let tendencyRegistryCache: TendencyRegistryRow[] | null = null;

function loadTendencyRegistry(): TendencyRegistryRow[] {
  if (tendencyRegistryCache) return tendencyRegistryCache;

  const candidates = [
    path.resolve(process.cwd(), "..", "data", "tendency_registry.json"),
    path.resolve(process.cwd(), "data", "tendency_registry.json"),
  ];

  for (const filePath of candidates) {
    try {
      const raw = fs.readFileSync(filePath, "utf-8");
      const parsed = JSON.parse(raw) as TendencyRegistryRow[];
      if (Array.isArray(parsed) && parsed.length > 0) {
        tendencyRegistryCache = parsed.filter((row) => Boolean(row?.canonical_name));
        return tendencyRegistryCache;
      }
    } catch {
      // Continue to fallback candidate.
    }
  }

  tendencyRegistryCache = [];
  return tendencyRegistryCache;
}

function tendencyScore(row: TendencyRegistryRow, stats: GeneratorSourceStats, attributes: Record<string, number>): number {
  const canonical = row.canonical_name;
  const name = canonical.toLowerCase();
  const category = String(row.category || "").toLowerCase();
  const three = attributes.three_point_shot ?? 60;
  const mid = attributes.mid_range_shot ?? 60;
  const close = attributes.close_shot ?? 60;
  const layup = attributes.driving_layup ?? 60;
  const dunk = Math.max(attributes.driving_dunk ?? 60, attributes.standing_dunk ?? 60);
  const passing = Math.round(((attributes.pass_accuracy ?? 60) + (attributes.pass_vision ?? 60)) / 2);
  const handle = attributes.ball_handle ?? 60;
  const perimeter = attributes.perimeter_defense ?? 60;
  const interior = attributes.interior_defense ?? 60;
  const steal = attributes.steal ?? 60;
  const block = attributes.block ?? 60;

  if (name.includes("three") || name.includes("3pt")) return round(clamp(three - 5, 0, 100));
  if (name.includes("mid")) return round(clamp(mid - 5, 0, 100));
  if (name.includes("close") || name.includes("under") || name.includes("floater")) return round(clamp(close - 5, 0, 100));
  if (name.includes("layup") || name.includes("euro") || name.includes("hop") || name.includes("reverse")) return round(clamp(layup - 5, 0, 100));
  if (name.includes("dunk") || name.includes("putback") || name.includes("alley") || name.includes("lob")) return round(clamp(dunk - 6, 0, 100));
  if (name.includes("dribble") || name.includes("spin") || name.includes("hesitation") || name.includes("stepback")) return round(clamp(handle - 8, 0, 100));
  if (name.includes("pass")) return round(clamp(passing - 6, 0, 100));
  if (name.includes("steal")) return round(clamp(steal - 2, 0, 100));
  if (name.includes("block")) return round(clamp(block - 2, 0, 100));
  if (name.includes("foul") || name.includes("charge")) return round(clamp((attributes.draw_foul ?? 55) - 10, 0, 100));
  if (name.includes("contest") || name.includes("on_ball") || name.includes("pressure")) return round(clamp(perimeter - 8, 0, 100));
  if (name.includes("post")) return round(clamp(attributes.post_control ?? 55, 0, 100));
  if (name.includes("shot")) {
    const scoreBlend = currentScoringBlend(stats);
    return round(clamp(40 + scoreBlend * 1.4, 0, 100));
  }

  const defaultBlend = (stats.current.PTS + stats.current.AST + stats.current.REB) / 3;

  if (category === "shooting") {
    return round(clamp((three * 0.45 + mid * 0.4 + (attributes.shot_iq ?? 60) * 0.15) - 8, 0, 100));
  }
  if (category === "finishing") {
    return round(clamp((layup * 0.45 + close * 0.35 + dunk * 0.2) - 6, 0, 100));
  }
  if (category === "passing") {
    return round(clamp((passing * 0.7 + handle * 0.3) - 8, 0, 100));
  }
  if (category === "driving" || category === "dribble_moves" || category === "dribble_setup") {
    return round(clamp((handle * 0.6 + (attributes.speed_with_ball ?? 60) * 0.4) - 10, 0, 100));
  }
  if (category === "post") {
    return round(clamp(((attributes.post_control ?? 60) * 0.5 + (attributes.post_fade ?? 60) * 0.25 + (attributes.post_hook ?? 60) * 0.25) - 8, 0, 100));
  }
  if (category === "defense") {
    return round(clamp((perimeter * 0.45 + interior * 0.3 + steal * 0.15 + block * 0.1) - 12, 0, 100));
  }
  if (category === "physical") {
    return round(clamp((attributes.hustle ?? 60) * 0.45 + (attributes.strength ?? 60) * 0.25 + (attributes.stamina ?? 60) * 0.3 - 12, 0, 100));
  }
  if (category === "triple_threat" || category === "playstyle" || category === "isolation" || category === "core") {
    return round(clamp((handle * 0.35 + (attributes.shot_iq ?? 60) * 0.35 + (attributes.offensive_consistency ?? 60) * 0.3) - 12, 0, 100));
  }

  return round(clamp(30 + defaultBlend * 1.1 + interior * 0.08, 0, 100));
}

function currentScoringBlend(stats: GeneratorSourceStats): number {
  return safe(stats.current.PTS * 0.7 + stats.previous.PTS * 0.2 + stats.career.PTS * 0.1);
}

function groupAttributes(attributes: Record<string, number>): Record<string, Record<string, number>> {
  const grouped: Record<string, Record<string, number>> = {};
  for (const [canonical, value] of Object.entries(attributes)) {
    const category = ATTRIBUTE_CATEGORIES[canonical] ?? "other";
    const label = ATTRIBUTE_LABELS[canonical] ?? canonical;
    if (!grouped[category]) grouped[category] = {};
    grouped[category][label] = value;
  }
  return grouped;
}

function groupTendencies(tendencies: Record<string, number>, registry: TendencyRegistryRow[]): Record<string, Record<string, number>> {
  const grouped: Record<string, Record<string, number>> = {};
  for (const row of registry) {
    const canonical = row.canonical_name;
    if (!(canonical in tendencies)) continue;
    const category = row.category || "other";
    const label = row.primjer_label || canonical;
    if (!grouped[category]) grouped[category] = {};
    grouped[category][label] = tendencies[canonical];
  }

  if (Object.keys(grouped).length === 0) {
    grouped.general = {};
    for (const [canonical, value] of Object.entries(tendencies)) {
      grouped.general[canonical] = value;
    }
  }

  return grouped;
}

function roleFromArchetype(archetype: string): string {
  const lower = archetype.toLowerCase();
  if (lower.includes("sharpshooter")) return "Floor Spacer";
  if (lower.includes("slasher")) return "Pressure Rim Attacker";
  if (lower.includes("playmaker")) return "Primary Initiator";
  if (lower.includes("defender")) return "Two-Way Stopper";
  return "Balanced Creator";
}

function usageFromArchetype(archetype: string, stats: GeneratorSourceStats): string[] {
  const current = stats.current;
  const lower = archetype.toLowerCase();
  if (lower.includes("sharpshooter")) {
    return [
      "Run off-ball actions to free catch-and-shoot attempts.",
      "Space corners and wings to force hard closeouts.",
      "Use dribble handoffs into quick-trigger jumpers.",
    ];
  }
  if (lower.includes("slasher")) {
    return [
      "Attack paint touches early in the shot clock.",
      "Leverage downhill drives from high pick-and-roll.",
      "Pressure rim protectors to draw rotations and fouls.",
    ];
  }
  if (lower.includes("playmaker")) {
    return [
      "Initiate half-court offense through spread pick-and-roll.",
      "Manipulate weak-side tags to create assist windows.",
      "Push pace after stops to generate transition advantages.",
    ];
  }
  if (lower.includes("defender")) {
    return [
      "Take top perimeter assignment every possession.",
      "Disrupt passing lanes and trigger transition offense.",
      "Anchor help rotations while controlling defensive glass.",
    ];
  }

  return [
    `Operate as a balanced ${current.PTS >= 18 ? "scoring" : "support"} option within structure.`,
    "Blend on-ball reps with off-ball spacing and cuts.",
    "Scale usage by matchup and lineup context.",
  ];
}

export function generatePlayer(stats: GeneratorSourceStats, info: Pick<PlayerInfo, "position">): GeneratedOutput {
  const current = stats.current;
  const previous = stats.previous;
  const career = stats.career;

  const scoringBlend = safe(current.PTS * 0.6 + previous.PTS * 0.25 + career.PTS * 0.15);
  const playmakingBlend = safe(current.AST * 0.65 + previous.AST * 0.2 + career.AST * 0.15);
  const reboundingBlend = safe(current.REB * 0.65 + previous.REB * 0.2 + career.REB * 0.15);
  const stocksBlend = safe((current.STL + current.BLK) * 0.6 + (previous.STL + previous.BLK) * 0.25 + (career.STL + career.BLK) * 0.15);

  const perimeterBias = info.position.includes("G") ? 1.08 : 0.94;
  const interiorBias = info.position.includes("C") ? 1.12 : info.position.includes("F") ? 1.02 : 0.88;

  const attributes: Record<string, number> = {
    driving_layup: round(clamp(52 + scoringBlend * 1.5 + current.FG_PCT * 0.18, 30, 99)),
    standing_dunk: round(clamp(28 + reboundingBlend * 1.6 * interiorBias + current.BLK * 5.5, 25, 99)),
    driving_dunk: round(clamp(35 + scoringBlend * 0.9 * interiorBias + current.REB * 1.2, 25, 99)),
    close_shot: round(clamp(58 + scoringBlend * 1.4 + current.FG_PCT * 0.25, 35, 99)),
    mid_range_shot: round(clamp(45 + scoringBlend * 1.3 + current.FG_PCT * 0.22, 30, 99)),
    three_point_shot: round(clamp(30 + current.FG3_PCT * 1.2 * perimeterBias + current.PTS * 0.8, 25, 99)),
    free_throw: round(clamp(52 + current.PTS * 0.8 + previous.PTS * 0.25, 35, 98)),
    post_hook: round(clamp(30 + reboundingBlend * 1.1 * interiorBias + current.PTS * 0.35, 25, 99)),
    post_fade: round(clamp(32 + scoringBlend * 0.85 + current.FG_PCT * 0.12, 25, 99)),
    post_control: round(clamp(30 + reboundingBlend * 1.3 * interiorBias + current.PTS * 0.45, 25, 99)),
    draw_foul: round(clamp(38 + current.PTS * 1.05 + current.FG_PCT * 0.08, 25, 99)),
    shot_iq: round(clamp(52 + current.FG_PCT * 0.35 + current.FG3_PCT * 0.2, 35, 99)),
    ball_handle: round(clamp(40 + playmakingBlend * 3.9 + current.PTS * 0.55, 25, 99)),
    speed_with_ball: round(clamp(45 + playmakingBlend * 2.8 + current.PTS * 0.6, 25, 99)),
    hands: round(clamp(48 + playmakingBlend * 2.6 + current.AST * 1.8, 25, 99)),
    pass_accuracy: round(clamp(40 + playmakingBlend * 4.8, 25, 99)),
    pass_iq: round(clamp(40 + playmakingBlend * 4.2 + current.AST * 1.7, 25, 99)),
    pass_vision: round(clamp(40 + playmakingBlend * 4.6 + current.STL * 2.5, 25, 99)),
    offensive_consistency: round(clamp(48 + scoringBlend * 2.1, 35, 99)),
    interior_defense: round(clamp(34 + current.BLK * 13 * interiorBias + current.REB * 2.6, 25, 99)),
    perimeter_defense: round(clamp(36 + current.STL * 14 * perimeterBias + current.BLK * 4.5, 25, 99)),
    steal: round(clamp(35 + current.STL * 22 + previous.STL * 5, 25, 99)),
    block: round(clamp(30 + current.BLK * 23 + previous.BLK * 6, 25, 99)),
    offensive_rebound: round(clamp(34 + current.REB * 6.2 * interiorBias, 25, 99)),
    defensive_rebound: round(clamp(38 + current.REB * 6.8 * interiorBias, 25, 99)),
    help_defense_iq: round(clamp(44 + current.BLK * 9 + current.STL * 6.5, 30, 99)),
    pass_perception: round(clamp(38 + current.STL * 16 + playmakingBlend * 1.2, 25, 99)),
    defensive_consistency: round(clamp(46 + stocksBlend * 4.1, 35, 99)),
    speed: round(clamp(55 + current.STL * 7 + current.PTS * 0.5, 35, 99)),
    agility: round(clamp(55 + current.STL * 8 + current.AST * 1.5, 35, 99)),
    strength: round(clamp(40 + current.REB * 4.2 + current.BLK * 7, 30, 99)),
    vertical: round(clamp(44 + (current.REB + current.BLK) * 3.2 + current.PTS * 0.35, 25, 99)),
    stamina: round(clamp(64 + current.GP * 0.35 + current.PTS * 0.3, 45, 99)),
    intangibles: round(clamp(58 + scoringBlend * 0.7 + stocksBlend * 2.2, 25, 99)),
    hustle: round(clamp(52 + current.REB * 2.4 + current.STL * 4.5, 25, 99)),
    overall_durability: round(clamp(55 + current.GP * 0.45, 25, 99)),
    potential: round(clamp(70 + current.PTS * 0.2 + current.AST * 0.3 + current.REB * 0.2, 25, 99)),
  };

  const tendencyRegistry = loadTendencyRegistry();
  const tendencies: Record<string, number> = {};
  for (const row of tendencyRegistry) {
    const canonical = row.canonical_name;
    const raw = tendencyScore(row, stats, attributes);
    const cap = Number.isFinite(Number(row.hard_cap)) ? Number(row.hard_cap) : 100;
    tendencies[canonical] = round(clamp(raw, 0, cap));
  }

  // Sub-zones should track their parent tendency while respecting each sub-zone hard cap.
  for (const row of tendencyRegistry) {
    if (!row.is_sub_zone || !row.parent_tendency || !(row.parent_tendency in tendencies)) continue;
    const cap = Number.isFinite(Number(row.hard_cap)) ? Number(row.hard_cap) : 100;
    const parentValue = tendencies[row.parent_tendency];
    tendencies[row.canonical_name] = round(clamp(parentValue, 0, cap));
  }

  const archetypeScores = {
    Sharpshooter: attributes.three_point_shot * 0.45 + attributes.mid_range_shot * 0.2 + attributes.shot_iq * 0.35,
    Slasher: attributes.driving_layup * 0.4 + attributes.driving_dunk * 0.35 + attributes.speed * 0.25,
    Playmaker: attributes.pass_vision * 0.4 + attributes.ball_handle * 0.35 + attributes.pass_accuracy * 0.25,
    Defender: attributes.perimeter_defense * 0.35 + attributes.interior_defense * 0.25 + attributes.steal * 0.2 + attributes.block * 0.2,
  };

  const archetype = Object.entries(archetypeScores).sort((a, b) => b[1] - a[1])[0][0];

  const flat = Object.entries(attributes).map(([canonical, value]) => ({
    category: ATTRIBUTE_CATEGORIES[canonical] ?? "other",
    label: ATTRIBUTE_LABELS[canonical] ?? canonical,
    value,
  }));

  const strengths = flat
    .slice()
    .sort((a, b) => b.value - a.value)
    .slice(0, 4)
    .map((item) => item.label);

  const weaknesses = flat
    .slice()
    .sort((a, b) => a.value - b.value)
    .slice(0, 4)
    .map((item) => item.label);

  const offenseKeys = [
    "close_shot",
    "mid_range_shot",
    "three_point_shot",
    "free_throw",
    "driving_layup",
    "driving_dunk",
    "standing_dunk",
    "post_hook",
    "post_fade",
    "post_control",
    "ball_handle",
    "speed_with_ball",
    "pass_accuracy",
    "pass_vision",
  ];
  const defenseKeys = ["interior_defense", "perimeter_defense", "steal", "block", "help_defense_iq", "pass_perception", "defensive_consistency"];
  const physicalKeys = ["speed", "agility", "strength", "vertical", "stamina"];
  const mentalKeys = ["shot_iq", "offensive_consistency", "intangibles", "hustle"];

  const avgFor = (keys: string[]) => {
    const values = keys.map((key) => attributes[key] ?? 0);
    return values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  };

  const offenseAvg = avgFor(offenseKeys);
  const defenseAvg = avgFor(defenseKeys);
  const physicalAvg = avgFor(physicalKeys);
  const mentalAvg = avgFor(mentalKeys);

  const attributeGroups = groupAttributes(attributes);
  const tendencyGroups = groupTendencies(tendencies, tendencyRegistry);

  const ovr = round(clamp(offenseAvg * 0.45 + defenseAvg * 0.22 + physicalAvg * 0.2 + mentalAvg * 0.13, 40, 99));

  return {
    attributes,
    tendencies,
    attributeGroups,
    tendencyGroups,
    archetype,
    strengths,
    weaknesses,
    role: roleFromArchetype(archetype),
    playStylePriorities: [],
    usage: usageFromArchetype(archetype, stats),
    ovr,
  };
}
