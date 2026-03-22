export type UserRole = "admin" | "editor" | "viewer" | "user";

export type AppUserProfile = {
  username?: string;
  email: string;
  mustChangePassword: boolean;
  role: UserRole;
};
