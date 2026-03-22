"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ShieldCheck, Sparkles, Target } from "lucide-react";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useAuth } from "@/components/auth/auth-provider";
import { getApiToken } from "@/lib/api/client";
import type { GeneratedPlayerPayload } from "@/lib/generator/types";

type PanelTab = "attributes" | "tendencies";
const DISABLE_CLIENT_CACHE = true;

function gradeFromAverage(value: number) {
  if (value >= 92) return "A+";
  if (value >= 86) return "A";
  if (value >= 80) return "A-";
  if (value >= 74) return "B+";
  if (value >= 68) return "B";
  return "C";
}

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, x) => sum + x, 0) / values.length;
}

function avgByKeys(map: Record<string, number>, keys: string[]): number {
  return avg(keys.map((key) => Number(map[key] ?? 0)));
}

function seasonToLabel(season: number): string {
  return `${season}-${String(season + 1).slice(2)}`;
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
    offense,
    defense,
    physical,
  };
}

export default function PlayerPage({ params }: { params: { id: string } }) {
  const { profile } = useAuth();
  const searchParams = useSearchParams();
  const season = Number(searchParams.get("season") ?? "2025");

  const [payload, setPayload] = useState<GeneratedPlayerPayload | null>(null);
  const [activeTab, setActiveTab] = useState<PanelTab>("attributes");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load(force = false, showLoadingState = true) {
    if (showLoadingState) {
      setLoading(true);
    }
    setError(null);
    try {
      const token = getApiToken();
      const response = await fetch("/api/generate-player", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ playerId: params.id, season, forceRefresh: force }),
        cache: "no-store",
      });

      const data = (await response.json()) as GeneratedPlayerPayload & { error?: string };
      if (!response.ok) {
        throw new Error(data.error || "Failed to generate player profile.");
      }

      setPayload(data);
      if (!DISABLE_CLIENT_CACHE && !force) {
        sessionStorage.setItem(`generated:${params.id}:${season}`, JSON.stringify(data));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate player profile.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!DISABLE_CLIENT_CACHE) {
      const cacheKey = `generated:${params.id}:${season}`;
      const cached = sessionStorage.getItem(cacheKey);
      if (cached) {
        try {
          const parsed = JSON.parse(cached) as GeneratedPlayerPayload;
          setPayload(parsed);
          setLoading(false);
          return;
        } catch {
          // Ignore invalid cache and regenerate.
        }
      }
    }

    void load(false, true);
  }, [params.id, season]);

  if (loading && !payload) {
    return (
      <section className="rounded-3xl border border-white/10 bg-white/5 p-10 text-center backdrop-blur-xl">
        <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-sky/30 border-t-sky" />
        <p className="mt-4 text-sm text-white/70">Fetching Supabase stats and generating profile...</p>
      </section>
    );
  }

  if (error || !payload) {
    return <section className="rounded-2xl border border-red-400/30 bg-red-500/10 p-4 text-sm text-red-100">{error || "Player not found."}</section>;
  }

  const categoryScores = computeCategoryScores(payload.attributes, payload.info.position);
  const statsSeason = Number(payload.info.season || season);
  const offenseAverage = categoryScores.offense;
  const defenseAverage = categoryScores.defense;
  const physicalAverage = categoryScores.physical;
  const archetypeChips = (payload.archetypes && payload.archetypes.length > 0 ? payload.archetypes : [payload.archetype]).slice(0, 4);

  const gradeCards = [
    {
      icon: <Target className="h-4 w-4" />,
      title: "Offense",
      grade: gradeFromAverage(offenseAverage),
      score: Math.round(offenseAverage),
      tone: "border-prime/50 bg-prime/15",
    },
    {
      icon: <ShieldCheck className="h-4 w-4" />,
      title: "Defense",
      grade: gradeFromAverage(defenseAverage),
      score: Math.round(defenseAverage),
      tone: "border-sky/50 bg-sky/15",
    },
    {
      icon: <Sparkles className="h-4 w-4" />,
      title: "Physical",
      grade: gradeFromAverage(physicalAverage),
      score: Math.round(physicalAverage),
      tone: "border-emerald-400/45 bg-emerald-400/15",
    },
  ];

  return (
    <div className="space-y-4">
      <section className="relative overflow-hidden rounded-3xl border border-white/15 bg-white/5 shadow-glass backdrop-blur-xl">
        <div className="absolute inset-0">
          <div className="h-full w-full bg-gradient-to-br from-sky/30 via-transparent to-prime/30" />
          <div className="absolute inset-0 bg-gradient-to-r from-ink via-ink/75 to-ink/45" />
        </div>

        <div className="relative grid gap-6 p-6 lg:grid-cols-[1fr_auto] lg:items-end">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-sky/80">Generated Player Profile</p>
            <h1 className="mt-2 text-4xl font-black tracking-tight text-white sm:text-5xl">{payload.info.name}</h1>
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1">{payload.info.team}</span>
              <span className="rounded-full border border-white/20 bg-white/10 px-3 py-1">{payload.info.position || "N/A"}</span>
              {archetypeChips.map((tag) => (
                <span key={tag} className="rounded-full border border-sky/35 bg-sky/20 px-3 py-1 font-semibold text-sky-100">
                  {tag}
                </span>
              ))}
            </div>

            <div className="mt-5 flex flex-wrap gap-2" />
          </div>

          <div className="flex items-center gap-4 lg:flex-col">
            <div className="relative h-28 w-28 overflow-hidden rounded-2xl border border-white/25 bg-white/10">
              {payload.info.headshotUrl ? (
                <Image src={payload.info.headshotUrl} alt={`${payload.info.name} headshot`} fill className="object-cover" />
              ) : (
                <div className="grid h-full place-items-center text-xs text-white/60">No headshot</div>
              )}
            </div>
            <div className="grid h-24 w-24 place-items-center rounded-2xl border border-white/20 bg-white/10">
              <span className="text-xs uppercase tracking-widest text-white/65">OVR</span>
              <span className="text-3xl font-black text-white">{payload.ovr}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-4 xl:grid-cols-[320px_1fr]">
        <aside className="space-y-4">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-md">
            <h2 className="mb-3 text-xs uppercase tracking-[0.2em] text-sky/80">Player Info</h2>
            <div className="space-y-2 text-sm">
              <InfoRow label="Height" value={payload.info.height || "N/A"} />
              <InfoRow label="Weight" value={payload.info.weight || "N/A"} />
              <InfoRow label="Age" value={payload.info.age || "N/A"} />
              <InfoRow label="Years Pro" value={payload.info.yearsPro || "N/A"} />
              <InfoRow label="Draft" value={payload.info.draft || "N/A"} />
              <InfoRow label="School" value={payload.info.school || "N/A"} />
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-md">
            <h2 className="mb-3 text-xs uppercase tracking-[0.2em] text-sky/80">Play Style Priorities</h2>
            {payload.playStylePriorities && payload.playStylePriorities.length > 0 ? (
              <ol className="space-y-1 text-sm text-white/90">
                {payload.playStylePriorities.map((priority, index) => (
                  <li key={`${priority}-${index}`} className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                    <span className="text-xs uppercase tracking-[0.16em] text-white/55">#{index + 1}</span>
                    <span className="flex-1 text-right font-medium text-white/95">{priority}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-sm text-white/70">No play style priorities available.</p>
            )}
          </div>

          <div className="rounded-2xl border border-emerald-300/25 bg-emerald-500/10 p-4 backdrop-blur-md">
            <h3 className="mb-2 text-xs uppercase tracking-[0.2em] text-emerald-200">Strengths</h3>
            <ul className="list-disc space-y-1 pl-5 text-sm text-emerald-100">
              {payload.strengths.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="rounded-2xl border border-rose-300/25 bg-rose-500/10 p-4 backdrop-blur-md">
            <h3 className="mb-2 text-xs uppercase tracking-[0.2em] text-rose-200">Weaknesses</h3>
            <ul className="list-disc space-y-1 pl-5 text-sm text-rose-100">
              {payload.weaknesses.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </aside>

        <div className="space-y-4">
          <section className="grid gap-3 md:grid-cols-3">
            {gradeCards.map((card) => (
              <article key={card.title} className={`rounded-2xl border p-4 backdrop-blur-md ${card.tone}`}>
                <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-white/75">
                  <span className="inline-flex items-center gap-2">{card.icon}{card.title}</span>
                  <span>{card.grade}</span>
                </div>
                <p className="mt-3 text-3xl font-black text-white">{card.score}</p>
              </article>
            ))}
          </section>

          <section className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-md">
            <h2 className="mb-3 text-xs uppercase tracking-[0.2em] text-sky/80">Stats (From Supabase)</h2>
            <div className="grid gap-3 lg:grid-cols-3">
              <StatCard title={`Current (${seasonToLabel(statsSeason)})`} stats={payload.stats.current} />
              <StatCard title={`Previous (${seasonToLabel(statsSeason - 1)})`} stats={payload.stats.previous} />
              <StatCard title="Career" stats={payload.stats.career} />
            </div>
          </section>

          <section className="flex flex-wrap gap-2">
            <button
              onClick={() => setActiveTab("attributes")}
              className={`rounded-full border px-4 py-2 text-sm font-semibold transition ${
                activeTab === "attributes" ? "border-sky/70 bg-sky/25" : "border-white/20 bg-white/10 hover:border-sky/60"
              }`}
            >
              View Attributes
            </button>
            <button
              onClick={() => setActiveTab("tendencies")}
              className={`rounded-full border px-4 py-2 text-sm font-semibold transition ${
                activeTab === "tendencies" ? "border-prime/70 bg-prime/25" : "border-white/20 bg-white/10 hover:border-prime/60"
              }`}
            >
              View Tendencies
            </button>
          </section>

          <section className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-md">
            <h2 className="mb-3 text-xs uppercase tracking-[0.2em] text-sky/80">
              {activeTab === "attributes" ? "Attributes (Generated)" : "Tendencies (Generated)"}
            </h2>
            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                className="space-y-3"
              >
                {Object.entries(activeTab === "attributes" ? payload.attributeGroups : payload.tendencyGroups).map(([category, values]) => (
                  <div key={category} className="rounded-xl border border-white/10 bg-black/20 p-3">
                    <h3 className="mb-2 text-xs uppercase tracking-[0.18em] text-white/65">{category}</h3>
                    <div className="grid gap-2 md:grid-cols-2">
                      {Object.entries(values).map(([label, value]) => (
                        <div key={label} className="flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-2 py-1.5 text-sm">
                          <span className="text-white/80">{label}</span>
                          <span className="font-semibold text-white">{value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </motion.div>
            </AnimatePresence>
          </section>

          <section>
            <article className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-md">
              <h3 className="mb-2 text-xs uppercase tracking-[0.2em] text-sky/80">Usage</h3>
              <ul className="list-disc space-y-1 pl-5 text-sm text-white/90">
                {payload.usage.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          </section>
        </div>
      </section>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-white/10 bg-black/20 px-3 py-2">
      <span className="text-white/65">{label}</span>
      <span className="font-medium text-white">{value}</span>
    </div>
  );
}

function StatCard({ title, stats }: { title: string; stats: { GP: number; PTS: number; REB: number; AST: number; STL: number; BLK: number; FG_PCT: number; FG3_PCT: number } }) {
  return (
    <article className="rounded-xl border border-white/10 bg-black/20 p-3">
      <h3 className="mb-2 text-xs uppercase tracking-[0.18em] text-white/70">{title}</h3>
      <dl className="grid grid-cols-2 gap-2 text-sm">
        <StatCell label="PTS" value={stats.PTS} />
        <StatCell label="REB" value={stats.REB} />
        <StatCell label="AST" value={stats.AST} />
        <StatCell label="STL" value={stats.STL} />
        <StatCell label="BLK" value={stats.BLK} />
        <StatCell label="FG%" value={stats.FG_PCT} />
        <StatCell label="3PT%" value={stats.FG3_PCT} />
        <StatCell label="GP" value={stats.GP} />
      </dl>
    </article>
  );
}

function formatStatValue(label: string, value: number): string {
  if (!Number.isFinite(value)) return "0.0";
  if (label === "GP") return String(Math.round(value));
  return value.toFixed(1);
}

function StatCell({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-white/10 bg-white/5 px-2 py-1">
      <dt className="text-white/70">{label}</dt>
      <dd className="font-semibold text-white">{formatStatValue(label, value)}</dd>
    </div>
  );
}
