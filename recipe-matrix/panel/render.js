// render.js — turn a normalized recipe into the themed <table> chart.

import { computeLayout } from "./matrix.js";

export const MISE_PALETTE = {
  appetizer: "#4C7A34",
  salad: "#4C7A34",
  soup: "#B26A19",
  stew: "#B26A19",
  main: "#8C2F2F",
  seafood: "#1F6F7A",
  side: "#6B8E23",
  vegetable: "#6B8E23",
  baking: "#A9752F",
  bread: "#A9752F",
  dessert: "#8E3B62",
  breakfast: "#C4762B",
  drink: "#3B4B8C",
  cocktail: "#3B4B8C",
  sauce: "#7A5C2E",
  condiment: "#7A5C2E",
  other: "#1B5E20",
};

export function miseAccent(course) {
  return MISE_PALETTE[String(course || "other").toLowerCase()] || MISE_PALETTE.other;
}

const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function qtyText(ing) {
  if (ing.imperial && ing.metric) return `${ing.imperial} (${ing.metric})`;
  return ing.imperial || ing.metric || "";
}

/**
 * Render the chart into `container`. Interactive hooks (grips, delete
 * buttons, contenteditable, data-* ids) are included; export strips them
 * via the `.exporting` class.
 */
export function renderChart(container, state) {
  const { recipe, meta, theme } = state;
  const layout = computeLayout(recipe.tree);
  const totalCols = layout.columnCount;

  container.dataset.theme = theme === "mise" ? "mise" : theme;
  container.style.setProperty(
    "--accent",
    theme === "mise" ? miseAccent(recipe.course) : ""
  );

  const metaBits = [];
  if (meta.domain) metaBits.push(`<span class="m-domain">${esc(meta.domain)}</span>`);
  if (recipe.servings) metaBits.push(`<span>${esc(recipe.servings)}</span>`);
  if (recipe.totalTime) metaBits.push(`<span>${esc(recipe.totalTime)}</span>`);

  const prepRows = recipe.prepHeaders
    .map(
      (h, i) => `<tr><td class="prep-header" colspan="${totalCols}">
        <span contenteditable="plaintext-only" data-edit="prep" data-index="${i}">${esc(h)}</span>
        <button class="hdr-del" data-act="del-prep" data-index="${i}" title="Remove header">✕</button>
      </td></tr>`
    )
    .join("");

  const bodyRows = layout.rows
    .map((cells, rowIdx) => {
      const tds = cells
        .map((cell) => {
          const span =
            (cell.rowspan > 1 ? ` rowspan="${cell.rowspan}"` : "") +
            (cell.colspan > 1 ? ` colspan="${cell.colspan}"` : "");
          if (cell.type === "ingredient") {
            const ing = recipe.ingredients[rowIdx];
            const prev = recipe.ingredients[rowIdx - 1];
            const sectionLabel =
              ing.section && ing.section !== (prev?.section || "")
                ? `<span class="ing-section" contenteditable="plaintext-only" data-edit="section" data-id="${esc(ing.id)}">${esc(ing.section)}</span>`
                : "";
            const qty = qtyText(ing);
            return `<td class="ingredient"${span} data-row="${rowIdx}">
              <span class="row-grip" draggable="true" title="Drag to reorder">⠿</span>
              ${sectionLabel}
              ${qty ? `<span class="ing-qty" contenteditable="plaintext-only" data-edit="qty" data-id="${esc(ing.id)}">${esc(qty)}</span>` : ""}<span class="ing-name" contenteditable="plaintext-only" data-edit="name" data-id="${esc(ing.id)}">${esc(ing.name)}</span>${
                ing.note ? `<span class="ing-note" contenteditable="plaintext-only" data-edit="note" data-id="${esc(ing.id)}">${esc(ing.note)}</span>` : ""
              }
              <button class="row-del" data-act="del-ing" data-id="${esc(ing.id)}" title="Delete ingredient">✕</button>
            </td>`;
          }
          const isRoot = cell.node.id === recipe.tree.id;
          return `<td class="op${isRoot ? " root" : ""}"${span} data-node="${esc(cell.node.id)}"><span contenteditable="plaintext-only" data-edit="op" data-node="${esc(cell.node.id)}">${esc(cell.label)}</span></td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");

  const footnotes = recipe.footnotes.length
    ? `<ul class="chart-footnotes">${recipe.footnotes
        .map((f, i) => `<li contenteditable="plaintext-only" data-edit="footnote" data-index="${i}">${esc(f)}</li>`)
        .join("")}</ul>`
    : "";

  const notes =
    state.showNotes || recipe.notes
      ? `<div class="chart-notes" contenteditable="plaintext-only" data-edit="notes">${esc(recipe.notes)}</div>`
      : "";

  container.innerHTML = `
    <div class="chart-header">
      <div class="chart-title" contenteditable="plaintext-only" data-edit="title">${esc(recipe.title || "Untitled recipe")}</div>
      ${metaBits.length ? `<div class="chart-meta">${metaBits.join("")}</div>` : ""}
    </div>
    <table class="matrix">
      <tbody>
        ${prepRows}
        ${bodyRows}
      </tbody>
    </table>
    ${footnotes}
    ${notes}
    ${meta.url ? `<div class="chart-footer">${esc(meta.url)}</div>` : ""}
  `;
}
