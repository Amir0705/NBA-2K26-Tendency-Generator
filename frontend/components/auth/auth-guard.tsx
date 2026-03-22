"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import type { ReactNode } from "react";
import { useAuth } from "@/components/auth/auth-provider";

export function AuthGuard({ children }: { children: ReactNode }) {
  const { loading, firebaseUser, profile } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (loading) return;

    if (!firebaseUser) {
      router.replace("/login");
      return;
    }

    if (profile?.mustChangePassword && pathname !== "/change-password") {
      router.replace("/change-password");
      return;
    }

    if (!profile?.mustChangePassword && pathname === "/change-password") {
      router.replace("/dashboard");
    }
  }, [loading, firebaseUser, profile, pathname, router]);

  if (loading) {
    return (
      <div className="grid min-h-screen place-items-center bg-ink">
        <div className="h-10 w-10 animate-spin rounded-full border-2 border-sky/30 border-t-sky" />
      </div>
    );
  }

  if (!firebaseUser) return null;
  if (profile?.mustChangePassword && pathname !== "/change-password") return null;
  if (!profile?.mustChangePassword && pathname === "/change-password") return null;

  return <>{children}</>;
}
