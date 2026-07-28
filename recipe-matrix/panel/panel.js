// panel.js — orchestration: extract → structure → render → edit → export.

import { normalizeRecipe, naiveTree, mergeIntoParent, groupWithNextSibling, wrapInStep, freshId } from "./matrix.js";
import { renderChart } from "./render.js";
import { initEditor } from "./editor.js";
import { exportPng, exportPdf, copyPngToClipboard } from "./export.js";
import { getProvider, loadSettings } from "../lib/llmProvider.js";

const $ = (id) => document.getElementById(id);
const chartEl = $("chart");

const app = {
  state: {
    recipe: null,
    meta: { url: "", domain: "" },
    theme: "mise",
    showNotes: false,
    selectedNodeId: null,
  },
  undoStack: [],

  mutate(fn, { rerender = true } = {}) {
    this.undoStack.push(JSON.stringify({ recipe: this.state.recipe, showNotes: this.state.showNotes }));
    if (this.undoStack.length > 60) this.undoStack.shift();
    $("btn-undo").disabled = false;
    fn(this.state);
    if (rerender) this.render();
  },

  undo() {
    const snap = this.undoStack.pop();
    if (!snap) return;
    const { recipe, showNotes } = JSON.parse(snap);
    this.state.recipe = recipe;
    this.state.showNotes = showNotes;
    $("btn-undo").disabled = this.undoStack.length === 0;
    this.render();
  },

  selectNode(id) {
    this.state.selectedNodeId = id;
    chartEl.querySelectorAll("td.op.selected").forEach((td) => td.classList.remove("selected"));
    if (id) chartEl.querySelector(`td.op[data-node="${CSS.escape(id)}"]`)?.classList.add("selected");
    $("cell-tools").hidden = !id;
  },

  render() {
    renderChart(chartEl, this.state);
    chartEl.hidden = false;
    $("edit-bar").hidden = false;
    this.selectNode(
      this.state.selectedNodeId &&
        chartEl.querySelector(`td.op[data-node="${CSS.escape(this.state.selectedNodeId)}"]`)
        ? this.state.selectedNodeId
        : null
    );
  },
};

// --- status helpers --------------------------------------------------------

function status(text, isError = false) {
  const el = $("status");
  el.hidden = !text;
  el.textContent = text || "";
  el.classList.toggle("error", isError);
}
function banner(html) {
  const el = $("banner");
  el.hidden = !html;
  el.innerHTML = html || "";
}
function spinner(text) {
  $("spinner").hidden = !text;
  if (text) $("spinner-text").textContent = text;
}

// --- naive (no-LLM) structuring -------------------------------------------

const UNIT = /(?:cups?|tsp|tbsp|teaspoons?|tablespoons?|fl\s?oz|oz|ounces?|lbs?|pounds?|kg|g|grams?|m[lL]|liters?|litres?|l\b|pinch(?:es)?|dash(?:es)?|cloves?|cans?|sticks?|slices?|sprigs?|bunch(?:es)?|packages?|pkgs?|shots?|large|medium|small)\.?/;
const QTY = new RegExp(
  `^((?:\\d+\\s+\\d/\\d|\\d/\\d|\\d+(?:[.,]\\d+)?|[¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])(?:\\s*[-–]\\s*\\d+(?:[.,]\\d+)?)?\\s*(?:${UNIT.source})?)\\s+(.+)$`,
  "i"
);

function parseIngredientString(s, section) {
  let imperial = "";
  let metric = "";
  let rest = s.trim();
  const m = rest.match(QTY);
  if (m) {
    imperial = m[1].trim();
    rest = m[2].trim();
  }
  const paren = rest.match(/^\(([^)]{1,20})\)\s*(.+)$/);
  if (paren && /\d/.test(paren[1])) {
    metric = paren[1].trim();
    rest = paren[2].trim();
  }
  let note = "";
  const comma = rest.match(/^([^,]+),\s*(.+)$/);
  if (comma && comma[2].length < 40) {
    rest = comma[1].trim();
    note = comma[2].trim();
  }
  return { id: freshId("i"), imperial, metric, name: rest, note, section: section || "" };
}

function guessCookLabel(instructions) {
  const VERB = /\b(bake|roast|simmer|boil|fry|sauté|saute|grill|broil|steam|braise|chill|freeze)/;
  // The final cooking step is usually the last instruction that names a
  // cooking verb; pull temp/time from that same line.
  const lines = (instructions || []).map((l) => l.toLowerCase());
  const line = [...lines].reverse().find((l) => VERB.test(l));
  if (!line) return "cook";
  let label = line.match(VERB)[1];
  const temp = line.match(/(\d{2,3})\s*°?\s*([cf])\b/i);
  const time = line.match(/(\d+)(?:\s*[-–]\s*(\d+))?\s*(?:minutes|min|hours|hrs?)\b/);
  if (temp) label += ` ${temp[1]}°${temp[2].toUpperCase()}`;
  if (time) label += ` ${time[1]}${time[2] ? "-" + time[2] : ""} min`;
  return label;
}

function naiveRecipe(extracted, meta) {
  let section = "";
  const ingredients = [];
  for (const raw of extracted.ingredients) {
    const s = raw.trim();
    if (/^(for the|for)\b.*:$/i.test(s) || (/:$/.test(s) && s.length < 48 && !/\d/.test(s))) {
      section = s.replace(/:$/, "");
      continue;
    }
    if (s) ingredients.push(parseIngredientString(s, section));
  }
  if (!ingredients.length) return null;
  const recipe = {
    title: extracted.title || meta.domain || "Recipe",
    servings: extracted.servings || "",
    totalTime: extracted.totalTime || "",
    course: guessCourse(extracted),
    prepHeaders: [],
    ingredients,
    tree: null,
    footnotes: [],
    notes: "",
  };
  recipe.tree = naiveTree(ingredients);
  recipe.tree.label = guessCookLabel(extracted.instructions);
  return normalizeRecipe(recipe);
}

function guessCourse(extracted) {
  const hay = `${extracted.category || ""} ${extracted.title || ""}`.toLowerCase();
  for (const [key, course] of [
    ["dessert", "dessert"], ["cake", "dessert"], ["cookie", "dessert"], ["brownie", "dessert"],
    ["bread", "baking"], ["appetizer", "appetizer"], ["starter", "appetizer"], ["salad", "salad"],
    ["soup", "soup"], ["stew", "soup"], ["breakfast", "breakfast"], ["brunch", "breakfast"],
    ["drink", "drink"], ["cocktail", "drink"], ["sauce", "sauce"], ["dressing", "sauce"],
    ["side", "side"], ["fish", "seafood"], ["shrimp", "seafood"], ["salmon", "seafood"],
    ["main", "main"], ["dinner", "main"], ["entree", "main"], ["entrée", "main"],
  ]) {
    if (hay.includes(key)) return course;
  }
  return "other";
}

// --- pasted-text fallback --------------------------------------------------

function extractedFromPaste(text) {
  const lines = text.split(/\n/).map((l) => l.replace(/^[•*\-–]\s*/, "").trim());
  const marker = lines.findIndex((l) =>
    /^(instructions?|directions?|method|steps?|preparation)\b[:.]?$/i.test(l)
  );
  let ingredientLines;
  let instructionLines;
  if (marker >= 0) {
    ingredientLines = lines.slice(0, marker);
    instructionLines = lines.slice(marker + 1);
  } else {
    ingredientLines = lines.filter((l) => l && (QTY.test(l) || /:$/.test(l)) && l.length < 120);
    instructionLines = lines.filter((l) => l && !ingredientLines.includes(l));
  }
  const ingHeader = ingredientLines.findIndex((l) => /^ingredients\b[:.]?$/i.test(l));
  if (ingHeader >= 0) ingredientLines.splice(ingHeader, 1);
  return {
    title: "",
    ingredients: ingredientLines.filter(Boolean),
    instructions: instructionLines.filter(Boolean),
    servings: "",
    totalTime: "",
    category: "",
  };
}

// --- main flow -------------------------------------------------------------

async function getActiveTab() {
  const fromQuery = new URLSearchParams(location.search).get("tabId");
  if (fromQuery) return { id: +fromQuery };
  const [tab] =
    (await chrome.tabs.query({ active: true, currentWindow: true })).length
      ? await chrome.tabs.query({ active: true, currentWindow: true })
      : await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab;
}

async function runExtraction() {
  status("");
  banner("");
  $("no-recipe").hidden = true;
  chartEl.hidden = true;
  $("edit-bar").hidden = true;
  spinner("Reading page…");

  let extraction = null;
  try {
    const tab = await getActiveTab();
    const [result] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content/extract.js"],
    });
    extraction = result?.result;
  } catch (e) {
    spinner(null);
    status(`Couldn't read this page (${e.message}). Paste the recipe below instead.`, true);
    $("no-recipe").hidden = false;
    return;
  }

  app.state.meta = {
    url: extraction?.url || "",
    domain: extraction?.domain || "",
  };

  const settings = await loadSettings();
  const provider = getProvider(settings);

  if (extraction?.ok) {
    await structureExtracted(extraction.recipe, provider);
  } else if (extraction?.pageText && provider) {
    await structureRawText(extraction.pageText, provider);
  } else {
    spinner(null);
    $("no-recipe").hidden = false;
    if (!provider && extraction?.pageText) {
      banner(
        `No structured recipe found. With a free Gemini key (<a href="#" id="banner-options">Options</a>) I could read the page text itself.`
      );
      bindBannerOptions();
    }
  }
}

async function structureExtracted(extracted, provider) {
  if (provider) {
    spinner(`Structuring with ${provider.name}…`);
    try {
      const raw = await provider.structureRecipe(extracted, app.state.meta.domain, {
        onStatus: (t) => spinner(t),
      });
      const recipe = normalizeRecipe(raw);
      if (recipe) {
        finishRender(recipe);
        return;
      }
      status("The model's structure was unusable — using naive grouping instead.", true);
    } catch (e) {
      status(`${e.message} Using naive grouping instead.`, e.kind !== "rate-limit");
    }
  }
  const recipe = naiveRecipe(extracted, app.state.meta);
  if (!recipe) {
    spinner(null);
    $("no-recipe").hidden = false;
    return;
  }
  if (!provider) {
    banner(
      `No API key — grouped naively (combine → cook). Edit the cells by hand, or add a free Gemini key in <a href="#" id="banner-options">Options</a> for smart structuring.`
    );
    bindBannerOptions();
  }
  finishRender(recipe);
}

async function structureRawText(text, provider) {
  spinner(`No structured recipe markup — asking ${provider.name} to read the page…`);
  try {
    const raw = await provider.extractAndStructure(text, app.state.meta.domain, {
      onStatus: (t) => spinner(t),
    });
    if (raw?.error) throw new Error(raw.error);
    const recipe = normalizeRecipe(raw);
    if (!recipe) throw new Error("no usable recipe in response");
    finishRender(recipe);
  } catch (e) {
    spinner(null);
    status(`Couldn't extract a recipe: ${e.message}`, true);
    $("no-recipe").hidden = false;
  }
}

function finishRender(recipe) {
  spinner(null);
  app.state.recipe = recipe;
  app.undoStack.length = 0;
  $("btn-undo").disabled = true;
  app.render();
}

function bindBannerOptions() {
  document.getElementById("banner-options")?.addEventListener("click", (e) => {
    e.preventDefault();
    chrome.runtime.openOptionsPage();
  });
}

// --- toolbar wiring --------------------------------------------------------

function selectedNode() {
  return app.state.selectedNodeId;
}

$("btn-undo").addEventListener("click", () => app.undo());
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.target.closest("[contenteditable]")) {
    e.preventDefault();
    app.undo();
  }
});

$("btn-rerun").addEventListener("click", runExtraction);
$("link-options").addEventListener("click", (e) => {
  e.preventDefault();
  chrome.runtime.openOptionsPage();
});

$("theme-select").addEventListener("change", (e) => {
  app.state.theme = e.target.value;
  if (app.state.recipe) app.render();
});

$("btn-add-header").addEventListener("click", () =>
  app.mutate((s) => s.recipe.prepHeaders.push("Preheat oven to 350°F (175°C)"))
);
$("btn-add-notes").addEventListener("click", () =>
  app.mutate((s) => {
    s.showNotes = true;
  })
);

$("btn-merge").addEventListener("click", () => {
  const id = selectedNode();
  if (id) app.mutate((s) => mergeIntoParent(s.recipe.tree, id));
});
$("btn-remove-step").addEventListener("click", () => {
  const id = selectedNode();
  if (!id) return;
  app.mutate((s) => {
    const root = s.recipe.tree;
    if (root.id === id) {
      // Removing the root: promote its only child if that keeps a valid tree.
      if (root.children?.length === 1 && root.children[0].children) s.recipe.tree = root.children[0];
    } else {
      mergeIntoParent(root, id);
    }
  });
});
$("btn-group").addEventListener("click", () => {
  const id = selectedNode();
  if (id) app.mutate((s) => groupWithNextSibling(s.recipe.tree, id, "mix"));
});
$("btn-wrap").addEventListener("click", () => {
  const id = selectedNode();
  if (!id) return;
  app.mutate((s) => {
    if (s.recipe.tree.id === id) {
      s.recipe.tree = { id: freshId(), label: "step", children: [s.recipe.tree] };
    } else {
      wrapInStep(s.recipe.tree, id, "step");
    }
  });
});

// --- export ----------------------------------------------------------------

function exportBase() {
  return {
    filename: (app.state.recipe?.title || "recipe").replace(/[^\w\d-]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase() || "recipe",
  };
}
$("btn-png").addEventListener("click", async () => {
  try {
    await exportPng(chartEl, { ...exportBase(), transparent: $("png-transparent").checked });
  } catch (e) {
    status(`PNG export failed: ${e.message}`, true);
  }
});
$("btn-pdf").addEventListener("click", async () => {
  try {
    await exportPdf(chartEl, { ...exportBase(), page: $("pdf-size").value });
  } catch (e) {
    status(`PDF export failed: ${e.message}`, true);
  }
});
$("btn-copy").addEventListener("click", async () => {
  try {
    await copyPngToClipboard(chartEl);
    status("Copied chart image to clipboard.");
    setTimeout(() => status(""), 2500);
  } catch (e) {
    status(`Copy failed: ${e.message}`, true);
  }
});

// --- paste fallback --------------------------------------------------------

$("btn-paste-go").addEventListener("click", async () => {
  const text = $("paste-text").value.trim();
  if (!text) return;
  $("no-recipe").hidden = true;
  const settings = await loadSettings();
  const provider = getProvider(settings);
  app.state.meta = app.state.meta.url ? app.state.meta : { url: "", domain: "pasted text" };
  if (provider) {
    await structureRawText(text, provider);
    if (!app.state.recipe) $("no-recipe").hidden = false;
  } else {
    const recipe = naiveRecipe(extractedFromPaste(text), app.state.meta);
    if (recipe) {
      banner(
        `No API key — grouped naively. Add a free Gemini key in <a href="#" id="banner-options">Options</a> for smart structuring.`
      );
      bindBannerOptions();
      finishRender(recipe);
    } else {
      status("Couldn't find ingredient lines in that text.", true);
      $("no-recipe").hidden = false;
    }
  }
});

// --- boot ------------------------------------------------------------------

initEditor(chartEl, app);
runExtraction();
