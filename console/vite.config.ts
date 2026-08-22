import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Bound to 127.0.0.1, like every other port in this project. CLAUDE.md's rule
// is about the compose ports and it is about the same fact: this is a public
// VPS, and a dev server on 0.0.0.0 is the console served to the internet
// without authentication in front of it.
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 5173 },
});
