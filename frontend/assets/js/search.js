// Type-ahead stock search (WAI-ARIA combobox pattern).
//
// Keyboard: Down/Up move through suggestions, Enter opens the highlighted (or
// first) stock, Escape closes the list. Requests are debounced, and answers
// to an older query are ignored so the list never shows stale results.

import { endpoints } from "./api.js";
import { debounce, escapeHtml } from "./format.js";

export function initSearch({ input, list, onPick }) {
  let results = [];
  let active = -1;
  let seq = 0;

  const close = () => {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    active = -1;
  };

  const open = () => {
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
  };

  const highlight = (index) => {
    active = index;
    [...list.querySelectorAll('[role="option"]')].forEach((li, i) => {
      li.setAttribute("aria-selected", String(i === index));
      if (i === index) {
        input.setAttribute("aria-activedescendant", li.id);
        li.scrollIntoView({ block: "nearest" });
      }
    });
  };

  const render = (message) => {
    if (message) {
      list.innerHTML = `<li class="empty" role="option" aria-disabled="true">${escapeHtml(message)}</li>`;
    } else {
      list.innerHTML = results
        .map(
          (r, i) => `<li id="search-opt-${i}" role="option" aria-selected="false" data-symbol="${escapeHtml(r.symbol)}">
            <span class="sym">${escapeHtml(r.symbol)}</span>
            <span class="name">${escapeHtml(r.name)}</span>
            <span class="sector">${escapeHtml(r.sector)}</span></li>`,
        )
        .join("");
    }
    open();
  };

  const pick = (symbol) => {
    close();
    input.value = "";
    input.blur();
    onPick(symbol);
  };

  const query = debounce(async (q) => {
    const mine = ++seq;
    try {
      const data = await endpoints.search(q);
      if (mine !== seq) return;
      results = data.results;
      active = -1;
      render(results.length ? null : `No stock matches “${q}”. FinSight covers 10 large-cap US stocks.`);
    } catch (error) {
      if (mine !== seq) return;
      results = [];
      render(error.unreachable ? "Search is unavailable: the API cannot be reached." : `Search failed: ${error.message}`);
    }
  }, 150);

  input.addEventListener("input", () => {
    const q = input.value.trim();
    if (!q) {
      seq += 1;
      results = [];
      close();
      return;
    }
    query(q);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (!results.length) return;
      event.preventDefault();
      if (list.hidden) open();
      const step = event.key === "ArrowDown" ? 1 : -1;
      highlight((active + step + results.length) % results.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const choice = results[active >= 0 ? active : 0];
      if (choice) pick(choice.symbol);
    } else if (event.key === "Escape") {
      if (!list.hidden) {
        event.preventDefault();
        close();
      } else {
        input.value = "";
      }
    }
  });

  // mousedown (not click) so the choice lands before the input loses focus.
  list.addEventListener("mousedown", (event) => {
    const li = event.target.closest("[data-symbol]");
    if (!li) return;
    event.preventDefault();
    pick(li.dataset.symbol);
  });

  input.addEventListener("blur", () => setTimeout(close, 100));
  input.addEventListener("focus", () => {
    if (input.value.trim() && results.length) render();
  });

  // "/" focuses the search box from anywhere (unless typing in a field).
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && !/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName)) {
      event.preventDefault();
      input.focus();
    }
  });
}
