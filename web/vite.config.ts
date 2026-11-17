/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// In development, /api is proxied to the service with the API key added, exactly as
// nginx does in production, so the browser never holds the key.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, "..", "FRAUDLENS_");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": {
          target: env.FRAUDLENS_SERVICE_URL ?? "http://localhost:8000",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
          headers: { "X-API-Key": env.FRAUDLENS_API_KEY ?? "" },
        },
      },
    },
    test: {
      environment: "jsdom",
      globals: true,
      setupFiles: ["src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
    },
  };
});
