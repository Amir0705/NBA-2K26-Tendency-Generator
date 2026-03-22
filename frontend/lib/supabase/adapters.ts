import { resolveValue } from "@/lib/supabase/schema";

type AnyRow = Record<string, unknown>;

export type PlayerCore = {
  id: string;
  name: string;
  team: string;
  position: string;
  archetype: string;
  headshotUrl: string;
  height: string;
  weight: string;
  age: number;
  yearsPro: number;
  draft: string;
  school: string;
};

export type CategoryValueRow = {
  category: string;
  label: string;
  value: number;
};

export type StatsScope = "current" | "previous" | "career";

export type PlayerStats = {
  scope: StatsScope;
  gp: number | null;
  pts: number | null;
  reb: number | null;
  ast: number | null;
  stl: number | null;
  blk: number | null;
  fgPct: number | null;
  fg3Pct: number | null;
};

export type PlayerExtras = {
  strengths: string[];
  weaknesses: string[];
  role: string;
  usage: string[];
};

export type PlayerSearchRow = {
  id: string;
  name: string;
  team: string;
  position: string;
};

export type TeamSearchRow = {
  id: string;
  name: string;
  abbreviation: string;
};

function toStringValue(input: unknown, fallback = ""): string {
  return typeof input === "string" ? input : input == null ? fallback : String(input);
}

function toNumberValue(input: unknown, fallback = 0): number {
  const n = Number(input);
  return Number.isFinite(n) ? n : fallback;
}

function toStringArray(input: unknown): string[] {
  if (Array.isArray(input)) return input.map((x) => toStringValue(x)).filter(Boolean);
  if (typeof input === "string" && input.trim()) return [input.trim()];
  return [];
}

export function parsePlayerCore(row: AnyRow): PlayerCore {
  return {
    id: toStringValue(resolveValue(row, "players", "id", true), ""),
    name: toStringValue(resolveValue(row, "players", "name", true), "Unknown Player"),
    team: toStringValue(resolveValue(row, "players", "team"), "N/A"),
    position: toStringValue(resolveValue(row, "players", "position"), "N/A"),
    archetype: toStringValue(resolveValue(row, "players", "archetype"), "Balanced"),
    headshotUrl: toStringValue(resolveValue(row, "players", "headshotUrl"), ""),
    height: toStringValue(resolveValue(row, "players", "height"), "N/A"),
    weight: toStringValue(resolveValue(row, "players", "weight"), "N/A"),
    age: toNumberValue(resolveValue(row, "players", "age"), 0),
    yearsPro: toNumberValue(resolveValue(row, "players", "yearsPro"), 0),
    draft: toStringValue(resolveValue(row, "players", "draft"), "N/A"),
    school: toStringValue(resolveValue(row, "players", "school"), "N/A"),
  };
}

export function parseCategoryValueRows(
  rows: AnyRow[] | null,
  table: "playerAttributes" | "playerTendencies",
): CategoryValueRow[] {
  return (rows ?? []).map((row) => ({
    category: toStringValue(resolveValue(row, table, "category"), "core"),
    label: toStringValue(resolveValue(row, table, "label"), "Unknown"),
    value: toNumberValue(resolveValue(row, table, "value"), 0),
  }));
}

export function parsePlayerStatsRows(rows: AnyRow[] | null): PlayerStats[] {
  return (rows ?? []).map((row) => {
    const rawScope = toStringValue(resolveValue(row, "playerStats", "scope"), "current").toLowerCase();
    const scope: StatsScope = rawScope === "previous" || rawScope === "career" ? rawScope : "current";

    return {
      scope,
      gp: Number.isFinite(Number(resolveValue(row, "playerStats", "gp"))) ? Number(resolveValue(row, "playerStats", "gp")) : null,
      pts: Number.isFinite(Number(resolveValue(row, "playerStats", "pts"))) ? Number(resolveValue(row, "playerStats", "pts")) : null,
      reb: Number.isFinite(Number(resolveValue(row, "playerStats", "reb"))) ? Number(resolveValue(row, "playerStats", "reb")) : null,
      ast: Number.isFinite(Number(resolveValue(row, "playerStats", "ast"))) ? Number(resolveValue(row, "playerStats", "ast")) : null,
      stl: Number.isFinite(Number(resolveValue(row, "playerStats", "stl"))) ? Number(resolveValue(row, "playerStats", "stl")) : null,
      blk: Number.isFinite(Number(resolveValue(row, "playerStats", "blk"))) ? Number(resolveValue(row, "playerStats", "blk")) : null,
      fgPct: Number.isFinite(Number(resolveValue(row, "playerStats", "fgPct"))) ? Number(resolveValue(row, "playerStats", "fgPct")) : null,
      fg3Pct: Number.isFinite(Number(resolveValue(row, "playerStats", "fg3Pct"))) ? Number(resolveValue(row, "playerStats", "fg3Pct")) : null,
    };
  });
}

export function parsePlayerExtras(row: AnyRow | null): PlayerExtras {
  return {
    strengths: toStringArray(resolveValue(row ?? {}, "playerExtras", "strengths")),
    weaknesses: toStringArray(resolveValue(row ?? {}, "playerExtras", "weaknesses")),
    role: toStringValue(resolveValue(row ?? {}, "playerExtras", "role"), ""),
    usage: toStringArray(resolveValue(row ?? {}, "playerExtras", "usage")),
  };
}

export function parsePlayerSearchRows(rows: AnyRow[] | null): PlayerSearchRow[] {
  return (rows ?? []).map((row) => ({
    id: toStringValue(resolveValue(row, "players", "id", true), ""),
    name: toStringValue(resolveValue(row, "players", "name", true), "Unknown Player"),
    team: toStringValue(resolveValue(row, "players", "team"), ""),
    position: toStringValue(resolveValue(row, "players", "position"), ""),
  }));
}

export function parseTeamSearchRows(rows: AnyRow[] | null): TeamSearchRow[] {
  return (rows ?? []).map((row) => ({
    id: toStringValue(resolveValue(row, "teams", "id", true), ""),
    name: toStringValue(resolveValue(row, "teams", "name", true), "Unknown Team"),
    abbreviation: toStringValue(resolveValue(row, "teams", "abbreviation"), ""),
  }));
}
