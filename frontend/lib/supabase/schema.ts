type TableKey = "players" | "playerAttributes" | "playerTendencies" | "playerStats" | "playerExtras" | "teams";

type ColumnKey =
  | "id"
  | "name"
  | "team"
  | "position"
  | "season"
  | "ovr"
  | "archetype"
  | "headshotUrl"
  | "height"
  | "weight"
  | "age"
  | "yearsPro"
  | "draft"
  | "school"
  | "playerId"
  | "category"
  | "label"
  | "value"
  | "scope"
  | "gp"
  | "pts"
  | "reb"
  | "ast"
  | "stl"
  | "blk"
  | "fgPct"
  | "fg3Pct"
  | "strengths"
  | "weaknesses"
  | "role"
  | "usage"
  | "abbreviation";

type TableConfig = {
  name: string;
  columns: Record<ColumnKey, string[]>;
};

const env = (key: string, fallback: string) => process.env[key] || fallback;

const tables: Record<TableKey, TableConfig> = {
  players: {
    name: env("NEXT_PUBLIC_SB_TABLE_PLAYERS", "players"),
    columns: {
      id: ["player_id", "id"],
      name: ["full_name", "name", "player_name"],
      team: ["team_abbr", "team", "team_abbreviation"],
      position: ["position", "pos"],
      season: ["season"],
      ovr: ["ovr", "overall"],
      archetype: ["archetype", "role"],
      headshotUrl: ["headshot_url", "headshot", "image_headshot"],
      height: ["height"],
      weight: ["weight"],
      age: ["age"],
      yearsPro: ["years_pro", "yearspro"],
      draft: ["draft", "draft_info"],
      school: ["school", "college"],
      playerId: ["player_id", "id"],
      category: ["category"],
      label: ["label"],
      value: ["value"],
      scope: ["scope"],
      gp: ["gp"],
      pts: ["pts"],
      reb: ["reb"],
      ast: ["ast"],
      stl: ["stl"],
      blk: ["blk"],
      fgPct: ["fg_pct", "fg%", "fgPercent"],
      fg3Pct: ["fg3_pct", "3pt_pct", "fg3Percent"],
      strengths: ["strengths"],
      weaknesses: ["weaknesses"],
      role: ["role"],
      usage: ["usage"],
      abbreviation: ["abbreviation", "team_abbreviation"],
    },
  },
  playerAttributes: {
    name: env("NEXT_PUBLIC_SB_TABLE_PLAYER_ATTRIBUTES", "player_attributes"),
    columns: {
      id: ["id"],
      name: ["name"],
      team: ["team"],
      position: ["position"],
      season: ["season"],
      ovr: ["ovr"],
      archetype: ["archetype"],
      headshotUrl: ["headshot_url"],
      height: ["height"],
      weight: ["weight"],
      age: ["age"],
      yearsPro: ["years_pro"],
      draft: ["draft"],
      school: ["school"],
      playerId: ["player_id", "id"],
      category: ["category", "attr_category"],
      label: ["label", "attr_label"],
      value: ["value", "rating"],
      scope: ["scope"],
      gp: ["gp"],
      pts: ["pts"],
      reb: ["reb"],
      ast: ["ast"],
      stl: ["stl"],
      blk: ["blk"],
      fgPct: ["fg_pct"],
      fg3Pct: ["fg3_pct"],
      strengths: ["strengths"],
      weaknesses: ["weaknesses"],
      role: ["role"],
      usage: ["usage"],
      abbreviation: ["abbreviation"],
    },
  },
  playerTendencies: {
    name: env("NEXT_PUBLIC_SB_TABLE_PLAYER_TENDENCIES", "player_tendencies"),
    columns: {
      id: ["id"],
      name: ["name"],
      team: ["team"],
      position: ["position"],
      season: ["season"],
      ovr: ["ovr"],
      archetype: ["archetype"],
      headshotUrl: ["headshot_url"],
      height: ["height"],
      weight: ["weight"],
      age: ["age"],
      yearsPro: ["years_pro"],
      draft: ["draft"],
      school: ["school"],
      playerId: ["player_id", "id"],
      category: ["category", "tendency_category"],
      label: ["label", "tendency_label"],
      value: ["value", "rating"],
      scope: ["scope"],
      gp: ["gp"],
      pts: ["pts"],
      reb: ["reb"],
      ast: ["ast"],
      stl: ["stl"],
      blk: ["blk"],
      fgPct: ["fg_pct"],
      fg3Pct: ["fg3_pct"],
      strengths: ["strengths"],
      weaknesses: ["weaknesses"],
      role: ["role"],
      usage: ["usage"],
      abbreviation: ["abbreviation"],
    },
  },
  playerStats: {
    name: env("NEXT_PUBLIC_SB_TABLE_PLAYER_STATS", "player_stats"),
    columns: {
      id: ["id"],
      name: ["name"],
      team: ["team"],
      position: ["position"],
      season: ["season"],
      ovr: ["ovr"],
      archetype: ["archetype"],
      headshotUrl: ["headshot_url"],
      height: ["height"],
      weight: ["weight"],
      age: ["age"],
      yearsPro: ["years_pro"],
      draft: ["draft"],
      school: ["school"],
      playerId: ["player_id", "id"],
      category: ["category"],
      label: ["label"],
      value: ["value"],
      scope: ["season", "scope", "split"],
      gp: ["gp", "games_played"],
      pts: ["pts_pg", "pts", "points"],
      reb: ["reb_pg", "reb", "rebounds"],
      ast: ["ast_pg", "ast", "assists"],
      stl: ["stl_pg", "stl", "steals"],
      blk: ["blk_pg", "blk", "blocks"],
      fgPct: ["fg_pct", "fg%", "fg_percent"],
      fg3Pct: ["fg3_pct", "3pt_pct", "fg3_percent"],
      strengths: ["strengths"],
      weaknesses: ["weaknesses"],
      role: ["role"],
      usage: ["usage"],
      abbreviation: ["abbreviation"],
    },
  },
  playerExtras: {
    name: env("NEXT_PUBLIC_SB_TABLE_PLAYER_EXTRAS", "player_extras"),
    columns: {
      id: ["id"],
      name: ["name"],
      team: ["team"],
      position: ["position"],
      season: ["season"],
      ovr: ["ovr"],
      archetype: ["archetype"],
      headshotUrl: ["headshot_url"],
      height: ["height"],
      weight: ["weight"],
      age: ["age"],
      yearsPro: ["years_pro"],
      draft: ["draft"],
      school: ["school"],
      playerId: ["player_id", "id"],
      category: ["category"],
      label: ["label"],
      value: ["value"],
      scope: ["scope"],
      gp: ["gp"],
      pts: ["pts"],
      reb: ["reb"],
      ast: ["ast"],
      stl: ["stl"],
      blk: ["blk"],
      fgPct: ["fg_pct"],
      fg3Pct: ["fg3_pct"],
      strengths: ["strengths", "strength_list"],
      weaknesses: ["weaknesses", "weakness_list"],
      role: ["role"],
      usage: ["usage", "usage_notes"],
      abbreviation: ["abbreviation"],
    },
  },
  teams: {
    name: env("NEXT_PUBLIC_SB_TABLE_TEAMS", "teams"),
    columns: {
      id: ["id", "team_id"],
      name: ["name", "team_name"],
      team: ["team"],
      position: ["position"],
      season: ["season"],
      ovr: ["ovr"],
      archetype: ["archetype"],
      headshotUrl: ["headshot_url"],
      height: ["height"],
      weight: ["weight"],
      age: ["age"],
      yearsPro: ["years_pro"],
      draft: ["draft"],
      school: ["school"],
      playerId: ["player_id"],
      category: ["category"],
      label: ["label"],
      value: ["value"],
      scope: ["scope"],
      gp: ["gp"],
      pts: ["pts"],
      reb: ["reb"],
      ast: ["ast"],
      stl: ["stl"],
      blk: ["blk"],
      fgPct: ["fg_pct"],
      fg3Pct: ["fg3_pct"],
      strengths: ["strengths"],
      weaknesses: ["weaknesses"],
      role: ["role"],
      usage: ["usage"],
      abbreviation: ["abbreviation", "team_abbreviation", "abbr"],
    },
  },
};

export function tableName(key: TableKey): string {
  return tables[key].name;
}

export function resolveColumn(key: TableKey, column: ColumnKey): string {
  return tables[key].columns[column][0];
}

export function resolveValue<T = unknown>(
  row: Record<string, unknown>,
  table: TableKey,
  column: ColumnKey,
  required = false,
): T | null {
  const aliases = tables[table].columns[column];
  for (const alias of aliases) {
    if (alias in row && row[alias] !== undefined && row[alias] !== null) {
      return row[alias] as T;
    }
  }

  if (required) {
    throw new Error(`Missing required column '${column}' in table '${tables[table].name}'. Tried aliases: ${aliases.join(", ")}`);
  }

  return null;
}

export function selectClause(table: TableKey, columns: ColumnKey[]): string {
  const selected = new Set<string>();
  columns.forEach((column) => {
    const first = tables[table].columns[column][0];
    selected.add(first);
  });
  return Array.from(selected).join(",");
}
