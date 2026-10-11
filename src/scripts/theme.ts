// src/scripts/theme.ts — Foolproof Client Theme Manager
export type ThemePreference = "light" | "dark" | "system";

const THEME_KEY = "ai-pulse-theme";

export function getThemePreference(): ThemePreference {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark" || saved === "system") {
      return saved;
    }
  } catch {
    // Graceful fallback if localStorage is disabled or throws
  }
  return "system";
}

export function applyTheme(preference: ThemePreference): void {
  const isDark =
    preference === "dark"
      ? true
      : preference === "light"
      ? false
      : typeof window !== "undefined" &&
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: dark)").matches;

  const root = document.documentElement;
  if (isDark) {
    root.classList.add("dark");
  } else {
    root.classList.remove("dark");
  }

  root.setAttribute("data-theme-preference", preference);
  root.setAttribute("data-theme-resolved", isDark ? "dark" : "light");
  root.style.colorScheme = isDark ? "dark" : "light";

  syncThemeButtons(preference);
}

let themeTransitionTimer: number | null = null;

export function triggerThemeTransition(): void {
  if (typeof window === "undefined") return;
  // Respect prefers-reduced-motion
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return;
  }
  const root = document.documentElement;
  root.classList.add("theme-transitioning");
  if (themeTransitionTimer !== null) {
    window.clearTimeout(themeTransitionTimer);
  }
  themeTransitionTimer = window.setTimeout(() => {
    root.classList.remove("theme-transitioning");
    themeTransitionTimer = null;
  }, 220);
}

export function setThemePreference(pref: ThemePreference): void {
  try {
    localStorage.setItem(THEME_KEY, pref);
  } catch {
    // Graceful fallback
  }
  triggerThemeTransition();
  applyTheme(pref);
}

export function syncThemeButtons(pref?: ThemePreference): void {
  const current = pref || getThemePreference();
  const buttons = document.querySelectorAll<HTMLButtonElement>("[data-theme-value]");
  buttons.forEach((btn) => {
    const val = btn.dataset.themeValue;
    const isSelected = val === current;
    btn.setAttribute("aria-pressed", isSelected ? "true" : "false");
    if (isSelected) {
      btn.classList.add("theme-active");
    } else {
      btn.classList.remove("theme-active");
    }
  });
}

let initialized = false;

export function initThemeManager(): void {
  // Always synchronize button states to reflect the active preference
  syncThemeButtons();

  if (initialized) {
    return;
  }
  initialized = true;

  // Global delegated click handler for any theme button in the DOM
  document.addEventListener("click", (e) => {
    const target = e.target as HTMLElement | null;
    const btn = target?.closest<HTMLButtonElement>("[data-theme-value]");
    if (btn) {
      e.preventDefault();
      const val = btn.dataset.themeValue as ThemePreference;
      if (val === "light" || val === "dark" || val === "system") {
        setThemePreference(val);
      }
    }
  });

  // System appearance change listener
  if (typeof window !== "undefined" && window.matchMedia) {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleMediaChange = () => {
      if (getThemePreference() === "system") {
        triggerThemeTransition();
        applyTheme("system");
      }
    };
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", handleMediaChange);
    } else if (typeof (mediaQuery as any)["addListener"] === "function") {
      try {
        (mediaQuery as any)["addListener"](handleMediaChange);
      } catch {}
    }
  }

  // Cross-tab synchronization via storage event
  window.addEventListener("storage", (e) => {
    if (e.key === THEME_KEY) {
      triggerThemeTransition();
      applyTheme(getThemePreference());
    }
  });

  // Astro client-side navigation lifecycles
  document.addEventListener("astro:after-swap", () => {
    applyTheme(getThemePreference());
  });

  document.addEventListener("astro:page-load", () => {
    syncThemeButtons();
  });
}

// Auto-run if executed in browser
if (typeof window !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => initThemeManager(), { once: true });
  } else {
    initThemeManager();
  }
}
