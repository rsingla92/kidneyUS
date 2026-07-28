// panel.js — orchestration: extract → structure → render → edit → export.

import { normalizeRecipe, mergeIntoParent, groupWithNextSibling, wrapInStep, freshId } from "./matrix.js";
import { buildRecipeFromExtraction, QTY } from "./heuristic.js";
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

  if (extraction?.ok) {
    // Deterministic by default — the LLM only runs when the user asks
    // (the "AI restructure" button) because the mechanical chart looks wrong.
    app.lastExtracted = extraction.recipe;
    const recipe = buildRecipeFromExtraction(extraction.recipe, app.state.meta);
    if (recipe) {
      finishRender(recipe);
      return;
    }
  }
  spinner(null);
  $("no-recipe").hidden = false;
  if (extraction?.pageText) {
    app.lastPageText = extraction.pageText;
    const provider = getProvider(await loadSettings());
    $("btn-ai-page").hidden = !provider;
    if (!provider) {
      banner(
        `No structured recipe found. With a free Gemini key (<a href="#" id="banner-options">Options</a>) I could read the page text itself.`
      );
      bindBannerOptions();
    }
  }
}

/** The opt-in LLM pass: re-group the last extraction with Gemini. */
async function aiRestructure() {
  const provider = getProvider(await loadSettings());
  if (!provider) {
    status("AI restructure needs a free Gemini API key — opening Options.", true);
    chrome.runtime.openOptionsPage();
    return;
  }
  if (!app.lastExtracted) {
    status("Nothing to restructure — re-extract first.", true);
    return;
  }
  $("btn-ai").disabled = true;
  spinner(`Restructuring with ${provider.name}…`);
  try {
    const raw = await provider.structureRecipe(app.lastExtracted, app.state.meta.domain, {
      onStatus: (t) => spinner(t),
    });
    const recipe = normalizeRecipe(raw);
    if (!recipe) throw new Error("The model's structure was unusable.");
    spinner(null);
    app.mutate((s) => {
      s.recipe = recipe;
    });
    status("AI restructure applied — Undo reverts to the mechanical grouping.");
    setTimeout(() => status(""), 4000);
  } catch (e) {
    spinner(null);
    status(`AI restructure failed: ${e.message}`, e.kind !== "rate-limit");
  } finally {
    $("btn-ai").disabled = false;
  }
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
  app.state.meta = app.state.meta.url ? app.state.meta : { url: "", domain: "pasted text" };
  const extracted = extractedFromPaste(text);
  app.lastExtracted = extracted;
  const recipe = buildRecipeFromExtraction(extracted, app.state.meta);
  if (recipe) {
    finishRender(recipe);
    return;
  }
  // Deterministic parse found no ingredient lines — the LLM is the only
  // remaining reader for this text.
  const provider = getProvider(await loadSettings());
  if (provider) {
    await structureRawText(text, provider);
    if (!app.state.recipe) $("no-recipe").hidden = false;
  } else {
    status("Couldn't find ingredient lines in that text.", true);
    $("no-recipe").hidden = false;
  }
});

$("btn-ai").addEventListener("click", aiRestructure);
$("btn-ai-page").addEventListener("click", async () => {
  const provider = getProvider(await loadSettings());
  if (!provider || !app.lastPageText) return;
  $("no-recipe").hidden = true;
  await structureRawText(app.lastPageText, provider);
  if (!app.state.recipe) $("no-recipe").hidden = false;
});

// --- boot ------------------------------------------------------------------

initEditor(chartEl, app);
runExtraction();
