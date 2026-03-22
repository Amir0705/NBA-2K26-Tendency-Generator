import { apiGet } from "@/lib/api/client";
import {
  parseCategoryValueRows,
  parsePlayerCore,
  parsePlayerExtras,
  parsePlayerSearchRows,
  parsePlayerStatsRows,
  parseTeamSearchRows,
} from "@/lib/supabase/adapters";
import { hasSupabaseConfig, supabase } from "@/lib/supabase/client";
import { resolveColumn, selectClause, tableName } from "@/lib/supabase/schema";
import type { PlayerProfile, SearchPlayer, SearchTeam } from "@/types/player";

type GeneratedPlayerResponse = {
  player_id: number;
  player_name: string;
  team: string;
  position: string;
  season: string;
  attributes: Record<string, { value: number; label: string; category: string }>;
  tendencies: Record<string, { value: number; label: string }>;
  play_style_priorities?: string[];
};

function defaultStats() {
  return {
    GP: "N/A",
    PTS: "N/A",
    REB: "N/A",
    AST: "N/A",
    STL: "N/A",
    BLK: "N/A",
    "FG%": "N/A",
    "3PT%": "N/A",
  } as Record<string, number | string>;
}

function foldRows(rows: Array<{ category: string; label: string; value: number }>) {
  return rows.reduce<Record<string, Record<string, number>>>((acc, row) => {
    const category = row.category || "core";
    if (!acc[category]) acc[category] = {};
    acc[category][row.label] = Number(row.value || 0);
    return acc;
  }, {});
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item) => item && typeof item === "object" && !Array.isArray(item)) as Record<string, unknown>[];
}

function seasonLabelFromYear(season: number): string {
  return `${season}-${String(season + 1).slice(2)}`;
}

function formatHeightFromInches(value: unknown): string {
  const inches = Number(value);
  if (!Number.isFinite(inches) || inches <= 0) return "N/A";
  const rounded = Math.round(inches);
  const feet = Math.floor(rounded / 12);
  const rem = rounded % 12;
  return `${feet}'${rem}"`;
}

function formatWeightFromLbs(value: unknown): string {
  const lbs = Number(value);
  if (!Number.isFinite(lbs) || lbs <= 0) return "N/A";
  return `${Math.round(lbs)} lbs`;
}

function ageFromBirthdate(value: unknown): number | null {
  const raw = String(value ?? "").trim();
  if (!raw) return null;
  const birth = new Date(raw);
  if (Number.isNaN(birth.getTime())) return null;

  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  const dayDiff = today.getDate() - birth.getDate();
  if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) age -= 1;
  return age >= 0 ? age : null;
}

function isUnauthorizedError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  return msg.includes("unauthorized") || msg.includes("401");
}

function normalizeGeneratedToProfile(data: GeneratedPlayerResponse, season: number): PlayerProfile {
  const groupedAttributes: Record<string, Record<string, number>> = {};
  Object.entries(data.attributes ?? {}).forEach(([key, entry]) => {
    const cat = entry.category || "core";
    if (!groupedAttributes[cat]) groupedAttributes[cat] = {};
    groupedAttributes[cat][entry.label || key] = Number(entry.value || 0);
  });

  const groupedTendencies: Record<string, Record<string, number>> = {
    shooting: {},
    driving: {},
    passing: {},
    defense: {},
  };

  Object.entries(data.tendencies ?? {}).forEach(([key, entry]) => {
    const label = entry.label || key;
    const lower = `${key} ${label}`.toLowerCase();
    const category = lower.includes("pass")
      ? "passing"
      : lower.includes("def") || lower.includes("block") || lower.includes("steal")
        ? "defense"
        : lower.includes("drive") || lower.includes("layup") || lower.includes("dunk")
          ? "driving"
          : "shooting";
    groupedTendencies[category][label] = Number(entry.value || 0);
  });

  const allAttributes = Object.values(groupedAttributes).flatMap((cat) => Object.values(cat));
  const ovr = allAttributes.length
    ? Math.round(allAttributes.reduce((sum, x) => sum + x, 0) / allAttributes.length)
    : 75;

  return {
    id: String(data.player_id),
    name: data.player_name,
    team: data.team || "N/A",
    season,
    ovr,
    images: {
      headshot: data.player_id ? `https://cdn.nba.com/headshots/nba/latest/1040x760/${data.player_id}.png` : "",
      action: "",
    },
    info: {
      height: "N/A",
      weight: "N/A",
      age: 0,
      yearsPro: 0,
      draft: "N/A",
      school: "N/A",
      archetype: data.play_style_priorities?.[0] || "Balanced",
      position: data.position || "N/A",
    },
    attributes: groupedAttributes,
    tendencies: groupedTendencies,
    stats: {
      current: defaultStats(),
      previous: defaultStats(),
      career: defaultStats(),
    },
    strengths: Object.entries(groupedAttributes)
      .flatMap(([, v]) => Object.entries(v))
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([name]) => name),
    weaknesses: Object.entries(groupedAttributes)
      .flatMap(([, v]) => Object.entries(v))
      .sort((a, b) => a[1] - b[1])
      .slice(0, 4)
      .map(([name]) => name),
    role: data.play_style_priorities?.[0] || "Balanced Creator",
    usage: (data.play_style_priorities ?? []).slice(0, 4).map((x) => `Lean into ${x.toLowerCase()} actions.`),
  };
}

export async function fetchPlayerProfile(playerId: string, season: number): Promise<PlayerProfile> {
  if (!hasSupabaseConfig || !supabase) {
    const generated = await apiGet<GeneratedPlayerResponse>(
      `/generate/id/${playerId}?season=${season}-${String(season + 1).slice(2)}`,
    );
    return normalizeGeneratedToProfile(generated, season);
  }

  // Query the optimized player_profiles_v1 view (single call instead of 5)
  const result = await supabase
    .from("player_profiles_v1")
    .select("*")
    .eq("player_id", playerId)
    .eq("season", season)
    .maybeSingle();

  if (result.error || !result.data) {
    const generated = await apiGet<GeneratedPlayerResponse>(
      `/generate/id/${playerId}?season=${season}-${String(season + 1).slice(2)}`,
    );
    return normalizeGeneratedToProfile(generated, season);
  }

  const data = result.data as Record<string, unknown>;

  // Parse JSON aggregates from the view
  const attributesRaw = asRecordArray(JSON.parse(String(data.attributes || "[]")));
  const tendenciesRaw = asRecordArray(JSON.parse(String(data.tendencies || "[]")));
  const statsRaw = asRecordArray(JSON.parse(String(data.stats || "[]")));
  const extrasRaw = asRecord(data.extras) ?? {};

  const attributes = foldRows(
    attributesRaw.map((row) => ({
      category: String(row.category ?? "core"),
      label: String(row.label ?? ""),
      value: Number(row.value ?? 0),
    })),
  );

  const tendencies = foldRows(
    tendenciesRaw.map((row) => ({
      category: String(row.category ?? "core"),
      label: String(row.label ?? ""),
      value: Number(row.value ?? 0),
    })),
  );

  const statsByScope: Record<string, Record<string, number | string>> = {
    current: defaultStats(),
    previous: defaultStats(),
    career: defaultStats(),
  };

  statsRaw.forEach((row) => {
    const scope = String(row.scope ?? "current");
    statsByScope[scope] = {
      GP: (row.gp as number | null) ?? "N/A",
      PTS: (row.pts_pg as number | null) ?? "N/A",
      REB: (row.reb_pg as number | null) ?? "N/A",
      AST: (row.ast_pg as number | null) ?? "N/A",
      STL: (row.stl_pg as number | null) ?? "N/A",
      BLK: (row.blk_pg as number | null) ?? "N/A",
      "FG%": (row.fg_pct as number | null) ?? "N/A",
      "3PT%": (row.fg3_pct as number | null) ?? "N/A",
    };
  });

  const player = {
    id: String(data.player_id ?? ""),
    name: String(data.full_name ?? ""),
    team: String(data.team_abbr ?? ""),
    headshotUrl: String(data.headshot_url ?? ""),
    height: formatHeightFromInches(data.height_in ?? data.height),
    weight: formatWeightFromLbs(data.weight_lbs ?? data.weight),
    age: ageFromBirthdate(data.birthdate) ?? Number(data.age ?? 0),
    yearsPro: Number(data.years_pro ?? 0),
    draft: String(data.draft ?? "N/A"),
    school: String(data.school ?? "N/A"),
    archetype: String(data.archetype ?? "Balanced"),
    position: String(data.position ?? "SF"),
  };

  const extras = {
    strengths: asRecordArray(extrasRaw.strengths as unknown[] ?? []).map(String),
    weaknesses: asRecordArray(extrasRaw.weaknesses as unknown[] ?? []).map(String),
    role: String(extrasRaw.role ?? ""),
    usage: asRecordArray(extrasRaw.usage as unknown[] ?? []).map(String),
  };

  const all = Object.values(attributes).flatMap((cat) => Object.values(cat));
  const ovr = all.length ? Math.round(all.reduce((sum, val) => sum + val, 0) / all.length) : 75;

  const derivedStrengths = Object.entries(attributes)
    .flatMap(([, values]) => Object.entries(values))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([name]) => name);

  const derivedWeaknesses = Object.entries(attributes)
    .flatMap(([, values]) => Object.entries(values))
    .sort((a, b) => a[1] - b[1])
    .slice(0, 4)
    .map(([name]) => name);

  return {
    id: player.id,
    name: player.name,
    team: player.team,
    season,
    ovr,
    images: {
      headshot: player.headshotUrl,
      action: "",
    },
    info: {
      height: player.height,
      weight: player.weight,
      age: player.age,
      yearsPro: player.yearsPro,
      draft: player.draft,
      school: player.school,
      archetype: player.archetype,
      position: player.position,
    },
    attributes,
    tendencies,
    stats: {
      current: statsByScope.current,
      previous: statsByScope.previous,
      career: statsByScope.career,
    },
    strengths: extras.strengths.length ? extras.strengths : derivedStrengths,
    weaknesses: extras.weaknesses.length ? extras.weaknesses : derivedWeaknesses,
    role: extras.role || player.archetype || "Balanced Creator",
    usage: extras.usage.length ? extras.usage : ["Primary ball-handler reps", "Half-court scoring actions"],
  };
}

export async function searchPlayers(term: string): Promise<SearchPlayer[]> {
  if (!term.trim()) return [];

  if (!hasSupabaseConfig || !supabase) {
    const payload = await apiGet<{ query: string; results: Array<Record<string, unknown>> }>(
      `/search/${encodeURIComponent(term)}`,
    );
    const rows = Array.isArray(payload?.results) ? payload.results : [];
    return rows
      .map((row) => ({
        id: String(row.player_id ?? ""),
        name: String(row.full_name ?? ""),
        team: String(row.team ?? ""),
        position: String(row.position ?? ""),
      }))
      .filter((row) => row.id && row.name)
      .slice(0, 8);
  }

  const playersTable = tableName("players");
  const playerNameColumn = resolveColumn("players", "name");

  const { data } = await supabase
    .from(playersTable)
    .select(selectClause("players", ["id", "name", "team", "position"]))
    .ilike(playerNameColumn, `%${term}%`)
    .limit(8);

  return parsePlayerSearchRows((data as Record<string, unknown>[] | null) ?? []).map((row) => ({
    id: row.id,
    name: row.name,
    team: row.team,
    position: row.position,
  }));
}

export async function searchTeams(term: string): Promise<SearchTeam[]> {
  if (!term.trim() || !hasSupabaseConfig || !supabase) return [];

  const teamsTable = tableName("teams");
  const teamNameColumn = resolveColumn("teams", "name");
  const teamAbbreviationColumn = resolveColumn("teams", "abbreviation");

  const { data } = await supabase
    .from(teamsTable)
    .select(selectClause("teams", ["id", "name", "abbreviation"]))
    .or(`${teamNameColumn}.ilike.%${term}%,${teamAbbreviationColumn}.ilike.%${term}%`)
    .limit(8);

  return parseTeamSearchRows((data as Record<string, unknown>[] | null) ?? []).map((row) => ({
    id: row.id,
    name: row.name,
    abbreviation: row.abbreviation,
  }));
}

export async function findTeamPrimaryPlayer(teamAbbr: string, season: number): Promise<string | null> {
  if (!hasSupabaseConfig || !supabase) {
    const seasonCandidates = [season, season - 1].filter((value, idx, all) => value >= 1997 && all.indexOf(value) === idx);
    let sawUnauthorized = false;

    for (const candidateSeason of seasonCandidates) {
      try {
        const seasonLabel = seasonLabelFromYear(candidateSeason);
        const payload = await apiGet<{ players?: Array<Record<string, unknown>> }>(
          `/team/${encodeURIComponent(teamAbbr)}?season=${encodeURIComponent(seasonLabel)}`,
        );
        const first = Array.isArray(payload?.players) ? payload.players[0] : null;
        if (!first) continue;
        const playerId = first.player_id;
        return playerId == null ? null : String(playerId);
      } catch (error) {
        if (isUnauthorizedError(error)) {
          sawUnauthorized = true;
        }
        continue;
      }
    }

    if (sawUnauthorized) {
      throw new Error("Session expired. Please log out and log back in.");
    }

    return null;
  }

  const playersTable = tableName("players");
  const idColumn = resolveColumn("players", "id");
  const teamColumn = resolveColumn("players", "team");
  const seasonColumn = resolveColumn("players", "season");
  const ovrColumn = resolveColumn("players", "ovr");

  const { data } = await supabase
    .from(playersTable)
    .select(idColumn)
    .eq(teamColumn, teamAbbr)
    .eq(seasonColumn, season)
    .order(ovrColumn, { ascending: false })
    .limit(1)
    .maybeSingle();

  if (!data || typeof data !== "object") return null;
  const value = (data as Record<string, unknown>)[idColumn];
  return value == null ? null : String(value);
}
