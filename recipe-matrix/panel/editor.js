// editor.js — in-place editing: contenteditable commits, row drag-reorder,
// step merge/split via the toolbar, deletions. All mutations flow through
// app.mutate(fn) so undo snapshots stay consistent.

import { deleteIngredient, moveIngredient } from "./matrix.js";

export function initEditor(container, app) {
  // --- contenteditable commits -------------------------------------------
  container.addEventListener("focusout", (e) => {
    const el = e.target.closest?.("[contenteditable]");
    if (!el) return;
    const kind = el.dataset.edit;
    const text = el.textContent.replace(/\s+/g, " ").trim();
    const { recipe } = app.state;
    const ing = el.dataset.id ? recipe.ingredients.find((i) => i.id === el.dataset.id) : null;

    const commit = (fn) => app.mutate(fn, { rerender: true });
    switch (kind) {
      case "title":
        if (text !== recipe.title) commit((r) => (r.recipe.title = text));
        break;
      case "op": {
        const id = el.dataset.node;
        commit((r) => {
          const hit = (function find(n) {
            if (n.id === id) return n;
            for (const c of n.children || []) {
              const h = find(c);
              if (h) return h;
            }
            return null;
          })(r.recipe.tree);
          if (hit) hit.label = text || "…";
        });
        break;
      }
      case "qty": {
        if (!ing) break;
        const m = text.match(/^(.*?)\s*\(([^)]*)\)\s*$/);
        const imperial = m ? m[1].trim() : text;
        const metric = m ? m[2].trim() : "";
        if (imperial !== ing.imperial || metric !== ing.metric)
          commit((r) => {
            const target = r.recipe.ingredients.find((i) => i.id === ing.id);
            target.imperial = imperial;
            target.metric = metric;
          });
        break;
      }
      case "name":
      case "note":
      case "section":
        if (ing && text !== ing[kind === "name" ? "name" : kind]) {
          commit((r) => {
            const target = r.recipe.ingredients.find((i) => i.id === ing.id);
            target[kind] = text;
          });
        }
        break;
      case "prep": {
        const idx = +el.dataset.index;
        if (text !== recipe.prepHeaders[idx])
          commit((r) => {
            if (text) r.recipe.prepHeaders[idx] = text;
            else r.recipe.prepHeaders.splice(idx, 1);
          });
        break;
      }
      case "footnote": {
        const idx = +el.dataset.index;
        if (text !== recipe.footnotes[idx])
          commit((r) => {
            if (text) r.recipe.footnotes[idx] = text;
            else r.recipe.footnotes.splice(idx, 1);
          });
        break;
      }
      case "notes":
        if (text !== recipe.notes.trim())
          app.mutate((r) => (r.recipe.notes = el.textContent.trim()), { rerender: false });
        break;
    }
  });

  // Keep Enter from inserting rows inside single-line cells.
  container.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.dataset?.edit && e.target.dataset.edit !== "notes") {
      e.preventDefault();
      e.target.blur();
    }
  });

  // --- clicks: delete buttons, op-cell selection --------------------------
  container.addEventListener("click", (e) => {
    const del = e.target.closest("[data-act]");
    if (del) {
      if (del.dataset.act === "del-ing") {
        app.mutate((r) => deleteIngredient(r.recipe, del.dataset.id), { rerender: true });
      } else if (del.dataset.act === "del-prep") {
        app.mutate((r) => r.recipe.prepHeaders.splice(+del.dataset.index, 1), { rerender: true });
      }
      return;
    }
    const op = e.target.closest("td.op");
    app.selectNode(op ? op.dataset.node : null);
  });

  // --- drag to reorder ingredient rows -----------------------------------
  let dragFrom = null;
  container.addEventListener("dragstart", (e) => {
    const grip = e.target.closest?.(".row-grip");
    if (!grip) return;
    dragFrom = +grip.closest("td.ingredient").dataset.row;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", String(dragFrom));
  });
  container.addEventListener("dragover", (e) => {
    if (dragFrom === null) return;
    const td = e.target.closest?.("td.ingredient");
    if (!td) return;
    e.preventDefault();
    container.querySelectorAll("tr.drag-over").forEach((tr) => tr.classList.remove("drag-over"));
    td.closest("tr").classList.add("drag-over");
  });
  container.addEventListener("dragleave", (e) => {
    e.target.closest?.("tr")?.classList.remove("drag-over");
  });
  container.addEventListener("drop", (e) => {
    if (dragFrom === null) return;
    const td = e.target.closest?.("td.ingredient");
    container.querySelectorAll("tr.drag-over").forEach((tr) => tr.classList.remove("drag-over"));
    if (!td) return;
    e.preventDefault();
    const to = +td.dataset.row;
    const from = dragFrom;
    dragFrom = null;
    if (to !== from) app.mutate((r) => moveIngredient(r.recipe, from, to), { rerender: true });
  });
  container.addEventListener("dragend", () => {
    dragFrom = null;
    container.querySelectorAll("tr.drag-over").forEach((tr) => tr.classList.remove("drag-over"));
  });
}
