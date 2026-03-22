import { NextRequest, NextResponse } from "next/server";
import { searchPlayersFromSupabase } from "@/lib/generator/supabase-source";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const term = String(searchParams.get("term") ?? "").trim();
    const season = Number(searchParams.get("season") ?? "2025");

    if (!term) {
      return NextResponse.json({ results: [] }, { status: 200 });
    }

    const results = await searchPlayersFromSupabase(term, season);
    return NextResponse.json({ results }, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to search players.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
