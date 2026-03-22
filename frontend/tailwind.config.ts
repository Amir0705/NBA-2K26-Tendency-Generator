import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./contexts/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#070b14",
        panel: "#11192a",
        line: "#2c3a56",
        prime: "#ff6a3d",
        sky: "#53c2ff",
      },
      boxShadow: {
        glass: "0 12px 40px rgba(0,0,0,0.35)",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
      },
      backdropBlur: {
        xs: "2px",
      },
      keyframes: {
        glow: {
          "0%, 100%": { boxShadow: "0 0 0 rgba(83, 194, 255, 0)" },
          "50%": { boxShadow: "0 0 30px rgba(83, 194, 255, 0.22)" },
        },
      },
      animation: {
        glow: "glow 2.5s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
