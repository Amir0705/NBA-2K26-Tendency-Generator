export type PlayerProfile = {
  id: string;
  name: string;
  team: string;
  season: number;
  ovr: number;
  images: {
    headshot: string;
    action: string;
  };
  info: {
    height: string;
    weight: string;
    age: number;
    yearsPro: number;
    draft: string;
    school: string;
    archetype: string;
    position?: string;
  };
  attributes: Record<string, Record<string, number>>;
  tendencies: Record<string, Record<string, number>>;
  stats: {
    current: Record<string, number | string>;
    previous: Record<string, number | string>;
    career: Record<string, number | string>;
  };
  strengths: string[];
  weaknesses: string[];
  role: string;
  usage: string[];
};

export type SearchPlayer = {
  id: string;
  name: string;
  team: string;
  position?: string;
};

export type SearchTeam = {
  id: string;
  name: string;
  abbreviation: string;
};
