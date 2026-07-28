import { DEFAULT_SETTINGS, loadSettings } from "../lib/llmProvider.js";
import { geminiProvider } from "../lib/gemini.js";

const $ = (id) => document.getElementById(id);

async function init() {
  const settings = await loadSettings();
  $("apiKey").value = settings.apiKey;
  $("model").value = settings.model;
}

function currentInput() {
  return {
    provider: "gemini",
    apiKey: $("apiKey").value.trim(),
    model: $("model").value.trim() || DEFAULT_SETTINGS.model,
  };
}

function show(ok, message) {
  const el = $("result");
  el.className = ok ? "ok" : "err";
  el.textContent = message;
}

$("save").addEventListener("click", async () => {
  await chrome.storage.local.set(currentInput());
  show(true, "Saved.");
});

$("test").addEventListener("click", async () => {
  const input = currentInput();
  if (!input.apiKey) {
    show(false, "Enter an API key first.");
    return;
  }
  show(true, "Testing…");
  const { ok, message } = await geminiProvider(input).testKey();
  show(ok, message);
});

init();
