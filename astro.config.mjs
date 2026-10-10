// astro.config.mjs
import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";

export default defineConfig({
  integrations: [tailwind()],
  site: "https://swaroophegde07.github.io",
  base: "/ai-pulse",
  output: "static",
});
