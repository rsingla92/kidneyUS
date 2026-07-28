# Recipe Matrix

A Chrome extension (Manifest V3) that converts any recipe web page into a
*tabular recipe chart* — the format popularized by Michael Chu's
[Cooking for Engineers](https://www.cookingforengineers.com/): ingredients as
rows on the left, nested operation cells (`melt`, `mix`, `fold in`,
`bake 350°F 30-40 min`) merging rightward via `rowspan` until the final
cooking step.

Everything runs client-side in your browser. The page you already loaded is
read in place (no server fetch, so paywalls/bot-walls that let *you* in let
the extension in too). The only network call is the optional Gemini
structuring request.

## Install (Load unpacked)

1. Open `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and select this `recipe-matrix/` directory.
4. Pin the icon. On a recipe page, click it — the chart opens in the side
   panel (or a popup window on browsers without `chrome.sidePanel`).

## Getting a free Gemini key (optional but recommended)

Without a key the extension still extracts recipes and renders a chart with a
naive grouping (all ingredients → *combine* → *cook*) that you can reshape by
hand. With a key, Gemini infers the real operation tree.

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and
   sign in with a Google account. The free tier needs **no credit card**.
2. Click **Create API key** and copy it.
3. Right-click the extension icon → **Options** (or the ⚙ in the panel),
   paste the key, click **Save**, then **Test key**.

Notes on the free tier:

- **Privacy:** free-tier prompts may be used by Google for model improvement.
  The recipe text you convert is sent to Google. The options page warns about
  this too.
- **Rate limits:** roughly 10–15 requests/minute. On a 429 the panel shows a
  countdown and retries with backoff; daily-quota exhaustion (resets midnight
  Pacific) is reported separately since retrying won't help.
- The key is stored in `chrome.storage.local` only — never synced, never sent
  anywhere but `generativelanguage.googleapis.com`.

## Changing the model id

Free-tier model ids rotate over time. The default is `gemini-2.5-flash`. If
structuring fails with *model not found*, look up the current free Flash
model at [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models),
paste its id into **Options → Model id**, save, and hit **Test key** — the
button reports the API's actual error text on failure.

Other providers: the LLM call sits behind `lib/llmProvider.js`. Implementing
`{ structureRecipe, extractAndStructure, testKey }` in a new file and
registering it there is the whole job.

## Using it

1. Open a recipe page and click the icon.
2. Extraction is deterministic and instant for most sites (JSON-LD →
   microdata → heuristic scrape). Only structuring uses the LLM.
3. Edit before exporting:
   - Click any text (title, quantities, names, operation labels) to edit in
     place.
   - Drag the `⠿` grip to reorder ingredient rows; hover `✕` deletes a row.
   - Click an operation cell to select it, then use the toolbar:
     **Merge ⤴** (fold the step into its parent), **Group ⌒** (group with the
     next cell under a new step), **Wrap +** (add a step around it),
     **✕ Step** (remove the step).
   - **+ Prep header** adds a full-width row ("Preheat oven to …");
     **+ Notes** adds a free-text block under the grid.
   - **↩ Undo** or `Ctrl+Z`.
4. Pick a theme — default is **Auto by course (Mise)**, which colors the
   chart by dish type; or Test Kitchen, Blueprint, Newsprint, Chalkboard.
5. Export: **PNG** (2×, optional transparency), **PDF** (fits one Letter/A4
   page), or **Copy** to clipboard as an image.

If no recipe is detected you get a paste box — drop any recipe text in and
chart it the same way.

## Development

```
cd recipe-matrix
npm test        # node --test: layout/rowspan invariants + PDF writer
```

No build step, no dependencies. Layout logic lives in `panel/matrix.js`
(`computeLayout` — height/leaf-count annotation → rowspan/colspan grid),
rendering in `panel/render.js`, export in `panel/export.js` (SVG
`foreignObject` rasterization + a hand-rolled single-page PDF writer; MV3's
CSP forbids remote scripts, so nothing external is loaded).
