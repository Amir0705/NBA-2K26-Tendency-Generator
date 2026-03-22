"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { useAuth } from "@/components/auth/auth-provider";
import { apiLogin, setApiToken } from "@/lib/api/client";
import { loginWithUsername } from "@/lib/firebase/auth";

export default function LoginPage() {
  const { firebaseUser, profile } = useAuth();
  const router = useRouter();

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!firebaseUser) return;
    if (profile?.mustChangePassword) {
      router.replace("/change-password");
      return;
    }
    router.replace("/dashboard");
  }, [firebaseUser, profile, router]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await loginWithUsername(username, password);
      try {
        await apiLogin(username, password);
      } catch {
        // Firebase-backed API mode accepts ID tokens, so keep backend endpoints usable.
        try {
          const fallbackToken = await user.getIdToken();
          if (fallbackToken) setApiToken(fallbackToken);
        } catch {
          // Keep Firebase-authenticated flow usable even if backend token exchange is unavailable.
        }
      }
      if (user) {
        // Route immediately after successful auth; auth-provider will hydrate profile state in parallel.
        router.replace("/dashboard");
        router.refresh();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to login.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid min-h-screen place-items-center px-4">
      <section className="w-full max-w-md rounded-3xl border border-white/20 bg-white/10 p-8 shadow-glass backdrop-blur-xl">
        <h1 className="text-3xl font-bold">Welcome Back</h1>
        <p className="mt-2 text-sm text-white/70">Private access only. Contact an admin to create your account.</p>

        <form onSubmit={onSubmit} className="mt-6 space-y-4">
          <div>
            <label className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/60">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full rounded-xl border border-white/20 bg-ink/70 px-4 py-3 outline-none focus:border-sky/70"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/60">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full rounded-xl border border-white/20 bg-ink/70 px-4 py-3 outline-none focus:border-sky/70"
            />
          </div>

          {error && <p className="rounded-xl border border-red-300/40 bg-red-500/10 p-3 text-sm text-red-200">{error}</p>}

          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-full border border-prime/60 bg-prime/20 px-4 py-3 font-semibold transition hover:bg-prime/30 disabled:opacity-60"
          >
            {busy ? "Signing in..." : "Login"}
          </button>
        </form>
      </section>
    </main>
  );
}
