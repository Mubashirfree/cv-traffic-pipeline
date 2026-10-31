import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the UI runs on :5173 and forwards API calls to the FastAPI server on :8000.
const api = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/search": api, "/analyze": api, "/cameras": api, "/media": api, "/metrics": api, "/health": api },
  },
});
