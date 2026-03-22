import { hasSupabaseConfig, supabase } from "@/lib/supabase/client";
import { resolveColumn, selectClause, tableName } from "@/lib/supabase/schema";
import type { GeneratorSourceStats, PlayerInfo, SearchPlayerOption, StatLine } from "@/lib/generator/types";

type Row = Record<string, unknown>;

function emptyStatLine(): StatLine {
  return {
    GP: 0,
    PTS: 0,
    REB: 0,
    AST: 0,
    STL: 0,
    BLK: 0,
    FG_PCT: 0,
    FG3_PCT: 0,
  };
}

function toStatLine(row: { gp: number | null; pts: number | null; reb: number | null; ast: number | null; stl: number | null; blk: number | null; fgPct: number | null; fg3Pct: number | null } | null): StatLine {
  if (!row) return emptyStatLine();
  return {
    GP: Number(row.gp ?? 0),
    PTS: Number(row.pts ?? 0),
    REB: Number(row.reb ?? 0),
    AST: Number(row.ast ?? 0),
    STL: Number(row.stl ?? 0),
    BLK: Number(row.blk ?? 0),
    FG_PCT: Number(row.fgPct ?? 0),
    FG3_PCT: Number(row.fg3Pct ?? 0),
  };
}

function n(value: unknown): number {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function ageFromBirthdate(value: unknown): number {
  const raw = String(value ?? "").trim();
  if (!raw) return 0;
  const birth = new Date(raw);
  if (Number.isNaN(birth.getTime())) return 0;

  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  const dayDiff = today.getDate() - birth.getDate();
  if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) age -= 1;
  return age >= 0 ? age : 0;
}

function createSupabaseServerClient() {
  if (!hasSupabaseConfig || !supabase) {
    throw new Error("Supabase env vars are missing. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.");
  }
  return supabase;
}

function seasonToLabel(season: number): string {
  return `${season}-${String(season + 1).slice(2)}`;
}

function applySeasonFilter<T extends { eq: (column: string, value: number | string) => T; or: (filters: string) => T }>(
  query: T,
  seasonColumn: string,
  season: number,
): T {
  const seasonLabel = seasonToLabel(season);
  return query.or(`${seasonColumn}.eq.${season},${seasonColumn}.eq.${seasonLabel}`);
}

export async function fetchPlayerStatsFromSupabase(playerId: string, season: number): Promise<{ info: PlayerInfo; stats: GeneratorSourceStats }> {
  const supabase = createSupabaseServerClient();
  const viewRowResult = await supabase
    .from("player_generation_source_v1")
    .select("*")
    .eq("player_id", playerId)
    .eq("season_start", season)
    .limit(1)
    .maybeSingle();

  if (viewRowResult.error || !viewRowResult.data) {
    throw new Error(`Failed to fetch player stats from player_generation_source_v1 for ${playerId} (${seasonToLabel(season)}).`);
  }

  const row = viewRowResult.data as Row;

  const info: PlayerInfo = {
    id: String(row.player_id ?? playerId),
    name: String(row.full_name ?? "N/A"),
    team: String(row.team_abbr ?? "N/A"),
    position: String(row.position ?? "N/A"),
    season,
    height: (() => {
      const inches = n(row.height_in);
      if (!inches) return "N/A";
      const rounded = Math.round(inches);
      const feet = Math.floor(rounded / 12);
      const rem = rounded % 12;
      return `${feet}'${rem}"`;
    })(),
    weight: (() => {
      const lbs = n(row.weight_lbs);
      return lbs ? `${Math.round(lbs)} lbs` : "N/A";
    })(),
    age: (() => {
      const direct = Math.round(n(row.age));
      return direct > 0 ? direct : ageFromBirthdate(row.birthdate);
    })(),
    yearsPro: Math.max(0, Math.round(n(row.years_pro))),
    draft: String(row.draft ?? "").trim() || "N/A",
    school: String(row.school ?? "").trim() || "N/A",
    headshotUrl: String(row.player_id ? `https://cdn.nba.com/headshots/nba/latest/1040x760/${row.player_id}.png` : ""),
  };

  const currentLine = toStatLine({
    gp: n(row.gp),
    pts: n(row.pts_pg),
    reb: n(row.reb_pg),
    ast: n(row.ast_pg),
    stl: n(row.stl_pg),
    blk: n(row.blk_pg),
    fgPct: n(row.fg_pct),
    fg3Pct: n(row.fg3_pct),
  });

  const previousLine = toStatLine({
    gp: n(row.prev_gp),
    pts: n(row.prev_pts_pg),
    reb: n(row.prev_reb_pg),
    ast: n(row.prev_ast_pg),
    stl: n(row.prev_stl_pg),
    blk: n(row.prev_blk_pg),
    fgPct: n(row.prev_fg_pct),
    fg3Pct: n(row.prev_fg3_pct),
  });

  const careerLine = toStatLine({
    gp: n(row.career_gp),
    pts: n(row.career_pts_pg),
    reb: n(row.career_reb_pg),
    ast: n(row.career_ast_pg),
    stl: n(row.career_stl_pg),
    blk: n(row.career_blk_pg),
    fgPct: n(row.career_fg_pct),
    fg3Pct: n(row.career_fg3_pct),
  });

  const stats: GeneratorSourceStats = {
    current: currentLine,
    previous: previousLine,
    career: careerLine,
  };

  if (!stats.current.GP && !stats.current.PTS && !stats.current.REB && !stats.current.AST) {
    throw new Error(`No current-season stat line found in Supabase for player ${playerId} (${seasonToLabel(season)}).`);
  }

  return { info, stats };
}

export async function searchPlayersFromSupabase(term: string, season: number): Promise<SearchPlayerOption[]> {
  const cleaned = term.trim();
  if (cleaned.length < 2) return [];

  const supabase = createSupabaseServerClient();
  const playersTable = tableName("players");
  const nameColumn = resolveColumn("players", "name");
  const idColumn = resolveColumn("players", "id");
  const seasonColumn = resolveColumn("players", "season");

  const { data, error } = await supabase
    .from(playersTable)
    .select(selectClause("players", ["id", "name", "team", "position"]))
    .ilike(nameColumn, `%${cleaned}%`)
    .or(`${seasonColumn}.eq.${season},${seasonColumn}.eq.${seasonToLabel(season)}`)
    .order(nameColumn, { ascending: true })
    .limit(10);

  if (error) {
    throw new Error(error.message || "Failed to search players from Supabase.");
  }

  const rows = (data as unknown as Row[] | null) ?? [];
  return rows
    .map((row) => ({
      id: String(row[idColumn] ?? row.id ?? ""),
      name: String(row[nameColumn] ?? row.name ?? ""),
      team: String(row.team ?? ""),
      position: String(row.position ?? ""),
    }))
    .filter((row) => row.id && row.name);
}
