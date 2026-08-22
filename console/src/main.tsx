import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { initialiseI18n } from "./i18n";

import "./theme/tokens.css";
import "./shell.css";

/* i18n first, then render. `initialiseI18n` sets `lang` and `dir` on the root
 * element before anything paints, so an Arabic console never flashes through
 * an LTR layout on load — which is the whole page jumping, not a subtlety. */
void initialiseI18n().then(() => {
  const root = document.getElementById("root");
  if (!root) {
    throw new Error("no #root element");
  }
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
