import type { Metadata } from "next";
import { Sora } from "next/font/google";
import type { ReactNode } from "react";
import { AuthProvider } from "@/components/auth/auth-provider";
import { SeasonProvider } from "@/contexts/season-context";
import "./globals.css";

const sora = Sora({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NBA 2K26 Premium Generator",
  description: "Modern premium NBA player generator dashboard.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={sora.className}>
        <AuthProvider>
          <SeasonProvider>{children}</SeasonProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
