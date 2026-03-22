"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Search, Sparkles } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { useSeason } from "@/contexts/season-context";
import { getApiToken } from "@/lib/api/client";
import type { GeneratedPlayerPayload, SearchPlayerOption } from "@/lib/generator/types";

export default function DashboardPage() {
  const { season, setSeason, seasons } = useSeason();
  const router = useRouter();

  const [term, setTerm] = useState("");
  const [options, setOptions] = useState<SearchPlayerOption[]>([]);
  const [selected, setSelected] = useState<SearchPlayerOption | null>(null);
  const [searchBusy, setSearchBusy] = useState(false);
  const [generateBusy, setGenerateBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const cleaned = term.trim();
    if (cleaned.length < 2) {
      setOptions([]);
      return;
    }

    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      setSearchBusy(true);
      setError(null);
      try {
        const response = await fetch(`/api/search-players?term=${encodeURIComponent(cleaned)}&season=${season}`, {
          method: "GET",
          cache: "no-store",
          signal: ctrl.signal,
        });

        const payload = (await response.json()) as { results?: SearchPlayerOption[]; error?: string };
        if (!response.ok) {
          throw new Error(payload.error || "Failed to search players.");
        }

        const rows = Array.isArray(payload.results) ? payload.results : [];
        setOptions(rows);
      } catch (err) {
        if (!ctrl.signal.aborted) {
          setOptions([]);
          setError(err instanceof Error ? err.message : "Failed to search players.");
        }
      } finally {
        if (!ctrl.signal.aborted) setSearchBusy(false);
      }
    }, 220);

    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [term, season]);

  const selectedLabel = useMemo(() => {
    if (!selected) return "No player selected";
    return `${selected.name} · ${selected.team}${selected.position ? ` · ${selected.position}` : ""}`;
  }, [selected]);

  async function onGenerate() {
    if (!selected) {
      setError("Select a player from autocomplete first.");
      return;
    }

    setGenerateBusy(true);
    setError(null);
    try {
      const token = getApiToken();
      const response = await fetch("/api/generate-player", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ playerId: selected.id, season }),
      });

      const payload = (await response.json()) as GeneratedPlayerPayload & { error?: string };
      if (!response.ok) {
        throw new Error(payload.error || "Generation failed.");
      }

      sessionStorage.setItem(`generated:${selected.id}:${season}`, JSON.stringify(payload));
      router.push(`/player/${selected.id}?season=${season}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setGenerateBusy(false);
    }
  }

  return (
    <section className="relative overflow-hidden rounded-3xl border border-white/15 bg-white/5 p-6 shadow-glass backdrop-blur-xl sm:p-8">
      <div className="pointer-events-none absolute -top-24 right-[-80px] h-64 w-64 rounded-full bg-sky/20 blur-3xl" />
      <div className="pointer-events-none absolute -bottom-24 left-[-90px] h-64 w-64 rounded-full bg-prime/20 blur-3xl" />

      <div className="relative space-y-6">
        <header>
          <p className="text-xs uppercase tracking-[0.28em] text-sky/80">Generator Panel</p>
          <h1 className="mt-2 text-3xl font-black tracking-tight text-white sm:text-4xl">Build NBA 2K Player Profiles</h1>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-white/70">
            Supabase stats are fetched first, then the generator engine computes attributes, tendencies, archetype, strengths,
            weaknesses, role, usage, and OVR.
          </p>
        </header>

        <div className="grid gap-3 md:grid-cols-[1fr_150px_auto]">
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-3.5 h-4 w-4 text-white/50" />
            <input
              value={term}
              onChange={(e) => {
                setTerm(e.target.value);
                setSelected(null);
              }}
              placeholder="Search player from Supabase"
              className="w-full rounded-xl border border-white/20 bg-black/30 px-10 py-3 text-sm outline-none transition focus:border-sky/70"
            />
          </div>

          <select
            value={season}
            onChange={(e) => setSeason(Number(e.target.value))}
            className="rounded-xl border border-white/20 bg-black/30 px-3 py-3 text-sm outline-none transition focus:border-sky/70"
          >
            {seasons.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>

          <button
            onClick={onGenerate}
            disabled={generateBusy || !selected}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-prime/50 bg-prime/20 px-5 py-3 text-sm font-semibold transition hover:bg-prime/30 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Sparkles className="h-4 w-4" />
            {generateBusy ? "Generating..." : "Generate Player"}
          </button>
        </div>

        <AnimatePresence initial={false}>
          {term.trim().length >= 2 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="rounded-2xl border border-white/10 bg-black/30 p-2"
            >
              {searchBusy && <p className="px-2 py-2 text-xs text-white/60">Searching Supabase players...</p>}

              {!searchBusy && options.length === 0 && (
                <p className="px-2 py-2 text-xs text-white/60">No players found for this season.</p>
              )}

              {!searchBusy && options.length > 0 && (
                <ul className="space-y-1">
                  {options.map((player) => {
                    const active = selected?.id === player.id;
                    return (
                      <li key={player.id}>
                        <button
                          onClick={() => {
                            setSelected(player);
                            setTerm(player.name);
                            setOptions([]);
                          }}
                          className={`flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-sm transition ${
                            active ? "border border-sky/60 bg-sky/20" : "border border-transparent bg-white/0 hover:bg-white/10"
                          }`}
                        >
                          <span className="font-medium text-white">{player.name}</span>
                          <span className="text-xs text-white/65">{player.team}{player.position ? ` · ${player.position}` : ""}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <div className="rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-white/75">
          <span className="text-white/50">Selected:</span> {selectedLabel}
        </div>

        {error && <p className="rounded-xl border border-red-300/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</p>}
      </div>
    </section>
  );
}
