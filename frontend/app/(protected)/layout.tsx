"use client";

import { LogOut } from "lucide-react";
import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { AuthGuard } from "@/components/auth/auth-guard";
import { useAuth } from "@/components/auth/auth-provider";
import { logoutCurrentUser } from "@/lib/firebase/auth";

export default function ProtectedLayout({ children }: { children: ReactNode }) {
  const { profile } = useAuth();
  const router = useRouter();

  async function onLogout() {
    await logoutCurrentUser();
    router.replace("/login");
  }

  return (
    <AuthGuard>
      <main className="mx-auto min-h-screen w-full max-w-[1380px] px-4 py-5 sm:px-6">
        <header className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 backdrop-blur-md">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-sky/75">NBA 2K26</p>
            <p className="text-sm text-white/80">Generator-first player workflow</p>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-right">
              <p className="text-sm font-semibold text-white">{profile?.username || profile?.email}</p>
              <p className="text-xs uppercase tracking-widest text-sky/70">{profile?.role || "user"}</p>
            </div>

            <button
              onClick={onLogout}
              className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm transition hover:border-sky/60 hover:bg-sky/15"
            >
              <LogOut className="h-4 w-4" />
              Logout
            </button>
          </div>
        </header>

        <section>{children}</section>
      </main>
    </AuthGuard>
  );
}
