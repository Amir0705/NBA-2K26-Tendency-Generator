"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { useAuth } from "@/components/auth/auth-provider";
import { completeFirstPasswordChange } from "@/lib/firebase/auth";

export default function ChangePasswordPage() {
  const { firebaseUser, profile, refreshProfile } = useAuth();
  const router = useRouter();

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!firebaseUser) {
      router.replace("/login");
      return;
    }
    if (!profile?.mustChangePassword) {
      router.replace("/dashboard");
    }
  }, [firebaseUser, profile, router]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);

    if (newPassword !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    if (!firebaseUser) {
      setError("No authenticated user.");
      return;
    }

    setBusy(true);
    try {
      await completeFirstPasswordChange(firebaseUser, currentPassword, newPassword);
      await refreshProfile();
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid min-h-screen place-items-center px-4">
      <section className="w-full max-w-md rounded-3xl border border-white/20 bg-white/10 p-8 shadow-glass backdrop-blur-xl">
        <h1 className="text-3xl font-bold">Set New Password</h1>
        <p className="mt-2 text-sm text-white/70">First login detected. You must change your password before continuing.</p>

        <form onSubmit={onSubmit} className="mt-6 space-y-4">
          <div>
            <label className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/60">Current Password</label>
            <input
              type="password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              required
              className="w-full rounded-xl border border-white/20 bg-ink/70 px-4 py-3 outline-none focus:border-sky/70"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/60">New Password</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              className="w-full rounded-xl border border-white/20 bg-ink/70 px-4 py-3 outline-none focus:border-sky/70"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs uppercase tracking-[0.16em] text-white/60">Confirm Password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
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
            {busy ? "Updating..." : "Save New Password"}
          </button>
        </form>
      </section>
    </main>
  );
}
