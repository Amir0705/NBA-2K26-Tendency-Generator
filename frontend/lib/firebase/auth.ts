import {
  AuthError,
  EmailAuthProvider,
  User,
  reauthenticateWithCredential,
  signInWithEmailAndPassword,
  signOut,
  updatePassword,
} from "firebase/auth";
import {
  collection,
  getDocs,
  limit,
  query,
  updateDoc,
  where,
} from "firebase/firestore";
import { clearApiToken } from "@/lib/api/client";
import { firebaseAuth, firestore, hasFirebaseConfig } from "@/lib/firebase/client";
import type { AppUserProfile } from "@/types/auth";

function isPermissionDeniedError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const code = "code" in error ? String((error as { code?: unknown }).code ?? "") : "";
  const msg = "message" in error ? String((error as { message?: unknown }).message ?? "") : "";
  return code.includes("permission-denied") || msg.toLowerCase().includes("insufficient permissions");
}

function mapAuthError(error: unknown): Error {
  const authError = error as Partial<AuthError> | undefined;
  const code = String(authError?.code ?? "").toLowerCase();

  if (
    code.includes("auth/invalid-credential") ||
    code.includes("auth/wrong-password") ||
    code.includes("auth/user-not-found") ||
    code.includes("auth/invalid-email")
  ) {
    return new Error("Invalid username or password.");
  }

  if (code.includes("auth/too-many-requests")) {
    return new Error("Too many login attempts. Please wait a moment and try again.");
  }

  if (error instanceof Error) return error;
  return new Error("Failed to login.");
}

function getFirebaseContext() {
  if (!hasFirebaseConfig || !firebaseAuth || !firestore) {
    throw new Error("Firebase is not configured. Set NEXT_PUBLIC_FIREBASE_* environment variables.");
  }
  return { auth: firebaseAuth, db: firestore };
}

export async function loginWithEmail(email: string, password: string): Promise<User> {
  const { auth } = getFirebaseContext();
  const credential = await signInWithEmailAndPassword(auth, email, password);
  return credential.user;
}

function usernameToEmail(username: string): string {
  const uname = (username || "").trim().toLowerCase();
  if (!uname) return "";
  if (uname.includes("@")) return uname;
  return `${uname}@atd.local`;
}

async function findProfileByUsernameOrEmail(login: string) {
  const { db } = getFirebaseContext();
  const normalized = (login || "").trim().toLowerCase();

  const collectionsToTry = ["app_users", "users"];
  for (const name of collectionsToTry) {
    const col = collection(db, name);

    const byUsername = await getDocs(query(col, where("username", "==", normalized), limit(1)));
    if (!byUsername.empty) {
      const snap = byUsername.docs[0];
      return { collectionName: name, snap, data: snap.data() as Record<string, unknown> };
    }

    const byEmail = await getDocs(query(col, where("email", "==", normalized), limit(1)));
    if (!byEmail.empty) {
      const snap = byEmail.docs[0];
      return { collectionName: name, snap, data: snap.data() as Record<string, unknown> };
    }
  }

  return null;
}

export async function loginWithUsername(username: string, password: string): Promise<User> {
  const { auth } = getFirebaseContext();
  const normalized = (username || "").trim().toLowerCase();
  if (!normalized) throw new Error("Username is required.");

  let profile: Awaited<ReturnType<typeof findProfileByUsernameOrEmail>> = null;
  try {
    profile = await findProfileByUsernameOrEmail(normalized);
  } catch (error) {
    // Most Firestore rules block unauthenticated profile reads; fall back to deterministic username email.
    if (!isPermissionDeniedError(error)) throw error;
  }
  const emailFromProfile = profile?.data?.email ? String(profile.data.email) : "";
  const loginEmail = emailFromProfile || usernameToEmail(normalized);

  if (!loginEmail) {
    throw new Error("Could not resolve login email from username.");
  }

  try {
    const credential = await signInWithEmailAndPassword(auth, loginEmail, password);
    return credential.user;
  } catch (error) {
    throw mapAuthError(error);
  }
}

export async function logoutCurrentUser(): Promise<void> {
  const { auth } = getFirebaseContext();
  await signOut(auth);
  clearApiToken();
}

export async function getUserProfileByEmail(email: string): Promise<AppUserProfile | null> {
  const normalized = (email || "").toLowerCase();
  let profile: Awaited<ReturnType<typeof findProfileByUsernameOrEmail>> = null;
  try {
    profile = await findProfileByUsernameOrEmail(normalized);
  } catch (error) {
    // If rules deny profile reads, keep the signed-in session usable with a minimal profile.
    if (isPermissionDeniedError(error)) {
      return {
        email: normalized,
        role: "viewer",
        mustChangePassword: false,
      };
    }
    throw error;
  }
  if (!profile) return null;

  const data = profile.data;
  const rawRole = String(data.role ?? "viewer").toLowerCase();
  const role = rawRole === "admin" || rawRole === "editor" || rawRole === "viewer" ? rawRole : "user";
  const mustChangePassword = Boolean(data.mustChangePassword ?? data.must_change_password ?? false);

  return {
    username: data.username ? String(data.username) : undefined,
    email: String(data.email ?? normalized),
    mustChangePassword,
    role,
  };
}

export async function completeFirstPasswordChange(
  user: User,
  currentPassword: string,
  newPassword: string,
): Promise<void> {
  const { db } = getFirebaseContext();
  if (!user.email) throw new Error("Missing user email.");

  const credential = EmailAuthProvider.credential(user.email, currentPassword);
  await reauthenticateWithCredential(user, credential);
  await updatePassword(user, newPassword);

  for (const name of ["app_users", "users"]) {
    const usersRef = collection(db, name);
    const q = query(usersRef, where("email", "==", user.email.toLowerCase()), limit(1));
    let docs;
    try {
      docs = await getDocs(q);
    } catch (error) {
      if (isPermissionDeniedError(error)) continue;
      throw error;
    }
    if (!docs.empty) {
      try {
        await updateDoc(docs.docs[0].ref, {
          mustChangePassword: false,
          must_change_password: false,
        });
      } catch (error) {
        if (!isPermissionDeniedError(error)) throw error;
      }
    }
  }
}
