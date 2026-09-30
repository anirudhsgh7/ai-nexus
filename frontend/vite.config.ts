import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Port 5173 with strictPort so the backend's default CORS allow-list
// (http://localhost:5173, http://127.0.0.1:5173) always matches.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true },
  preview: { port: 5173, strictPort: true },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    restoreMocks: true,
  },
});
