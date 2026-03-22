"use client";

import { createContext, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";

type SeasonContextValue = {
  season: number;
  setSeason: (value: number) => void;
  seasons: number[];
};

const SeasonContext = createContext<SeasonContextValue | null>(null);

export function SeasonProvider({ children }: { children: ReactNode }) {
  const years = useMemo(() => {
    const list: number[] = [];
    for (let y = 2024; y >= 2000; y -= 1) {
      list.push(y);
    }
    return list;
  }, []);

  const [season, setSeason] = useState<number>(2024);

  return (
    <SeasonContext.Provider value={{ season, setSeason, seasons: years }}>
      {children}
    </SeasonContext.Provider>
  );
}

export function useSeason() {
  const ctx = useContext(SeasonContext);
  if (!ctx) throw new Error("useSeason must be used inside SeasonProvider.");
  return ctx;
}
