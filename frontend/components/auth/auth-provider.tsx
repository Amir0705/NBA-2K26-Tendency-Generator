"use client";

import { onAuthStateChanged, User } from "firebase/auth";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { firebaseAuth, hasFirebaseConfig } from "@/lib/firebase/client";
import { clearApiToken, hasApiToken, setApiToken } from "@/lib/api/client";
import { getUserProfileByEmail } from "@/lib/firebase/auth";
import type { AppUserProfile } from "@/types/auth";

type AuthContextValue = {
  firebaseUser: User | null;
  profile: AppUserProfile | null;
  loading: boolean;
  refreshProfile: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [firebaseUser, setFirebaseUser] = useState<User | null>(null);
  const [profile, setProfile] = useState<AppUserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  async function refreshProfile() {
    if (!firebaseUser?.email) {
      setProfile(null);
      return;
    }
    try {
      const userProfile = await getUserProfileByEmail(firebaseUser.email);
      setProfile(userProfile);
    } catch {
      setProfile(null);
    }
  }

  useEffect(() => {
    if (!hasFirebaseConfig || !firebaseAuth) {
      setFirebaseUser(null);
      setProfile(null);
      setLoading(false);
      return () => undefined;
    }

    const unsub = onAuthStateChanged(firebaseAuth, async (u) => {
      setFirebaseUser(u);
      try {
        if (u) {
          if (!hasApiToken()) {
            try {
              const fallbackToken = await u.getIdToken();
              if (fallbackToken) setApiToken(fallbackToken);
            } catch {
              // Keep auth state usable even when token refresh fails transiently.
            }
          }
        } else {
          clearApiToken();
        }

        if (u?.email) {
          const userProfile = await getUserProfileByEmail(u.email);
          setProfile(userProfile);
        } else {
          setProfile(null);
        }
      } catch {
        setProfile(null);
      } finally {
        setLoading(false);
      }
    });

    return () => unsub();
  }, []);

  const value = useMemo(
    () => ({ firebaseUser, profile, loading, refreshProfile }),
    [firebaseUser, profile, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider.");
  return ctx;
}
