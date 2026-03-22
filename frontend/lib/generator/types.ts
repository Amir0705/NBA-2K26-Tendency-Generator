export type StatLine = {
  GP: number;
  PTS: number;
  REB: number;
  AST: number;
  STL: number;
  BLK: number;
  FG_PCT: number;
  FG3_PCT: number;
};

export type GeneratorSourceStats = {
  current: StatLine;
  previous: StatLine;
  career: StatLine;
};

export type PlayerInfo = {
  id: string;
  name: string;
  team: string;
  position: string;
  season: number;
  height: string;
  weight: string;
  age: number;
  yearsPro: number;
  draft: string;
  school: string;
  headshotUrl: string;
};

export type GeneratedOutput = {
  attributes: Record<string, number>;
  tendencies: Record<string, number>;
  attributeGroups: Record<string, Record<string, number>>;
  tendencyGroups: Record<string, Record<string, number>>;
  archetype: string;
  archetypes?: string[];
  strengths: string[];
  weaknesses: string[];
  role: string;
  playStylePriorities: string[];
  usage: string[];
  ovr: number;
};

export type GeneratedPlayerPayload = {
  info: PlayerInfo;
  stats: GeneratorSourceStats;
  attributes: GeneratedOutput["attributes"];
  tendencies: GeneratedOutput["tendencies"];
  attributeGroups: GeneratedOutput["attributeGroups"];
  tendencyGroups: GeneratedOutput["tendencyGroups"];
  archetype: string;
  archetypes?: string[];
  strengths: string[];
  weaknesses: string[];
  role: string;
  playStylePriorities: string[];
  usage: string[];
  ovr: number;
};

export type GeneratorRequest = {
  playerId: string;
  season: number;
};

export type SearchPlayerOption = {
  id: string;
  name: string;
  team: string;
  position: string;
};
