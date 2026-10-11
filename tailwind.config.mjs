// tailwind.config.mjs
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        // System monospace stack — no external font load needed.
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      colors: {
        // Extend neutral palette with a slightly warmer near-black.
        neutral: {
          950: "#0a0a0a",
        },
        // Centralized semantic design tokens (variables defined in global.css)
        theme: {
          page: "var(--bg-page)",
          surface: "var(--bg-surface)",
          elevated: "var(--bg-surface-elevated)",
          hover: "var(--bg-hover)",
          selected: "var(--bg-selected)",
          overlay: "var(--bg-overlay)",
          input: "var(--bg-input)",
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
          dim: "var(--text-dim)",
          border: "var(--border-primary)",
          "border-subtle": "var(--border-subtle)",
          "border-accent": "var(--border-accent)",
          accent: "var(--accent)",
          "accent-hover": "var(--accent-hover)",
          "accent-subtle": "var(--accent-subtle)",
          "accent-text": "var(--accent-text)",
          link: "var(--text-link)",
          placeholder: "var(--text-placeholder)",
        },
      },
    },
  },
  plugins: [],
};
