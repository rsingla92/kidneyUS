// heuristic.js — deterministic, no-LLM recipe structuring.
//
// Recipe instructions are already an ordered pipeline: most steps name the
// ingredients they consume. So instead of dumping everything into
// combine→cook, we walk the instructions in order, match each step's text
// against the ingredient names, and left-fold a real operation tree:
// step N's node contains the running result plus whatever ingredients the
// step introduces. That reproduces the lateral structure of the chart for
// free on any site with decent instructions.

import { freshId, naiveTree, normalizeRecipe } from "./matrix.js";

// ---------------------------------------------------------------------------
// Ingredient string parsing
// ---------------------------------------------------------------------------

const UNIT =
  /(?:cups?|tsp|tbsp|teaspoons?|tablespoons?|fl\s?oz|oz|ounces?|lbs?|pounds?|kg|g|grams?|m[lL]|liters?|litres?|l\b|pinch(?:es)?|dash(?:es)?|cloves?|cans?|sticks?|slices?|sprigs?|bunch(?:es)?|packages?|pkgs?|shots?|stalks?|heads?|large|medium|small)\.?/;

export const QTY = new RegExp(
  `^((?:\\d+\\s+\\d/\\d|\\d/\\d|\\d+(?:[.,]\\d+)?|[¼½¾⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])(?:\\s*[-–]\\s*\\d+(?:[.,]\\d+)?)?\\s*(?:${UNIT.source})?)\\s+(.+)$`,
  "i"
);

export function parseIngredientString(s, section) {
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

// ---------------------------------------------------------------------------
// Matching ingredients to instruction steps
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  "the", "a", "an", "of", "and", "or", "to", "in", "into", "with", "for",
  "fresh", "freshly", "large", "medium", "small", "finely", "coarsely",
  "chopped", "diced", "minced", "sliced", "grated", "ground", "whole",
  "unsalted", "salted", "all-purpose", "plain", "extra", "virgin", "light",
  "dark", "granulated", "powdered", "boneless", "skinless", "low", "sodium",
  "room", "temperature", "plus", "more", "divided", "optional", "taste",
  "cut", "pieces", "piece", "about", "packed", "heaping", "level",
]);

function stem(word) {
  // crude singular/plural + -ed folding: tomatoes→tomato, onions→onion,
  // melted→melt. Enough for containment checks.
  let w = word;
  if (w.length > 4 && w.endsWith("oes")) w = w.slice(0, -2);
  else if (w.length > 3 && w.endsWith("ies")) w = w.slice(0, -3) + "y";
  else if (w.length > 3 && w.endsWith("es") && !w.endsWith("ses")) w = w.slice(0, -1);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1);
  return w;
}

function tokens(text) {
  return String(text)
    .toLowerCase()
    .replace(/[()]/g, " ")
    .split(/[^a-zà-ü'-]+/i)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w))
    .map(stem);
}

/** Does this instruction step mention this ingredient? */
export function stepMentions(stepTokens, ingredient) {
  const ingTokens = tokens(ingredient.name);
  if (!ingTokens.length) return false;
  // The head noun (last token) matching is the strong signal — "flour" in
  // "all-purpose flour". Any other distinctive token (≥5 chars) also counts,
  // so "espresso" matches "1 shot espresso, cooled".
  const head = ingTokens[ingTokens.length - 1];
  if (stepTokens.has(head)) return true;
  return ingTokens.some((t) => t.length >= 5 && stepTokens.has(t));
}

// ---------------------------------------------------------------------------
// Terse operation labels from step prose
// ---------------------------------------------------------------------------

const VERBS = [
  "preheat", "marinate", "macerate", "brine", "melt", "cream", "beat",
  "whisk", "whip", "fold", "stir", "mix", "combine", "toss", "knead",
  "sift", "dissolve", "bloom", "sauté", "saute", "sweat", "sear", "brown",
  "fry", "bake", "roast", "grill", "broil", "boil", "simmer", "poach",
  "steam", "braise", "stew", "reduce", "deglaze", "caramelize", "toast",
  "chill", "freeze", "rest", "cool", "proof", "rise", "drain", "strain",
  "purée", "puree", "blend", "mash", "season", "garnish", "top", "serve",
  "add", "pour", "spread", "layer", "roll", "shape", "cut", "slice", "heat",
  "cook",
];
const VERB_RE = new RegExp(`\\b(${VERBS.join("|")})`, "gi");
// Cooking-method verbs beat generic handling verbs ("add", "pour", "mix")
// when a step contains both: "add the lentils and simmer" → "simmer".
const PRIMARY = new Set([
  "marinate", "macerate", "brine", "melt", "sauté", "saute", "sweat", "sear",
  "brown", "fry", "bake", "roast", "grill", "broil", "boil", "simmer",
  "poach", "steam", "braise", "stew", "reduce", "caramelize", "toast",
  "chill", "freeze", "rest", "proof", "rise", "knead", "whip", "cream",
]);

/** Steps that are plating/serving fluff — never worth a column. */
const SKIP_RE = /^(serve|enjoy|garnish and serve|plate)\b/i;

// Words that, when preceding a verb candidate, mark it as a noun instead:
// "the cream", "heavy cream", "light brown sugar".
const NOUN_CONTEXT = new Set([
  "the", "a", "an", "of", "heavy", "sour", "whipped", "whipping", "double",
  "light", "dark", "ice", "golden",
]);

export function terseLabel(step) {
  const text = String(step).replace(/^##\s*/, "").trim();
  const lower = text.toLowerCase();
  const verbs = [...lower.matchAll(VERB_RE)]
    .filter((m) => {
      // Past participles are results, not actions: "until cooked through",
      // "the sautéed onions".
      if (lower.slice(m.index + m[1].length).startsWith("ed")) return false;
      const before = lower.slice(0, m.index).trimEnd().split(/\s+/).pop() || "";
      return !NOUN_CONTEXT.has(before);
    })
    .map((m) => m[1]);
  const verb = verbs.filter((v) => PRIMARY.has(v)).pop() || verbs[0] || "";
  const temp = lower.match(/(\d{2,3})\s*°\s*([cf])\b/i) || lower.match(/(\d{2,3})\s*degrees\s*([cf])?/i);
  const time = lower.match(/(\d+)(?:\s*(?:to|[-–])\s*(\d+))?\s*(minutes?|mins?|hours?|hrs?|seconds?|secs?)\b/);

  let label = verb || lower.split(/\s+/).slice(0, 2).join(" ");
  // "fold" reads better as "fold in", "stir in", "mix in" when adding things.
  if (/^(fold|stir|mix|whisk)$/.test(label) && /\b(in|into)\b/.test(lower)) label += " in";
  if (temp) label += ` ${temp[1]}°${(temp[2] || "F").toUpperCase()}`;
  if (time) {
    const unit = /hour|hr/.test(time[3]) ? "hr" : /sec/.test(time[3]) ? "sec" : "min";
    label += ` ${time[1]}${time[2] ? "-" + time[2] : ""} ${unit}`;
  }
  return label.length > 26 ? label.slice(0, 26).trimEnd() : label;
}

/** Prep-header steps: oven preheats, pan prep — they precede the grid. */
function isPrepHeader(step) {
  return /^(preheat|butter and flour|grease|line (a|the)|lightly (grease|oil|butter))/i.test(
    String(step).trim()
  );
}

// ---------------------------------------------------------------------------
// The structurer
// ---------------------------------------------------------------------------

/** Steps that start a new parallel component ("Meanwhile, make the sauce"). */
const NEW_BRANCH_RE =
  /\b(meanwhile|while (the|it|that)|in a (separate|clean|second|another|large|medium|small) (bowl|pan|pot|saucepan|skillet)|in another (bowl|pan|pot|saucepan|skillet))\b/i;

/**
 * Build an operation tree by walking instructions in order and folding in
 * the ingredients each step mentions. Branch-aware: "Meanwhile…" /
 * "In a separate bowl…" / HowToSection markers park the running result and
 * start a parallel component; a later step that re-mentions a parked
 * component's ingredients (or its section name — "add the chicken to the
 * sauce") merges that branch back in. Returns { tree, prepHeaders, usedAll }.
 */
export function buildTree(ingredients, instructions) {
  const prepHeaders = [];
  const steps = [];
  for (const raw of instructions || []) {
    let s = String(raw).trim();
    if (!s) continue;
    const section = s.match(/^##\s*(.+)$/);
    if (section) {
      steps.push({ text: "", sectionName: section[1] });
      continue;
    }
    if (isPrepHeader(s)) prepHeaders.push(s.replace(/\.$/, ""));
    else steps.push({ text: s });
  }

  const unused = [...ingredients];
  const take = (stepTokens) => {
    const found = [];
    for (let i = unused.length - 1; i >= 0; i--) {
      if (stepMentions(stepTokens, unused[i])) found.unshift(unused.splice(i, 1)[0]);
    }
    return found;
  };

  // A branch is a parallel component under construction: its running node,
  // the ingredients folded into it so far, and any name tokens ("sauce").
  const newBranch = () => ({ node: null, used: [], nameTokens: new Set() });
  let current = newBranch();
  const parked = [];

  const branchMentioned = (branch, stepTokens) => {
    if ([...branch.nameTokens].some((t) => stepTokens.has(t))) return true;
    return branch.used.some((ing) => stepMentions(stepTokens, ing));
  };

  for (const step of steps) {
    if (step.sectionName) {
      if (current.node) parked.push(current);
      current = newBranch();
      for (const t of tokens(step.sectionName)) current.nameTokens.add(t);
      continue;
    }
    const text = step.text;
    const stepTokens = new Set(tokens(text));

    if (NEW_BRANCH_RE.test(text) && current.node) {
      parked.push(current);
      current = newBranch();
    }

    const matched = take(stepTokens);
    const label = terseLabel(text);
    const leaves = matched.map((ing) => ({ id: freshId(), ingredientId: ing.id }));

    // Parked branches this step calls back into ("add the chicken to the
    // sauce" while the sauce simmers in `current`).
    const merging = [];
    for (let i = parked.length - 1; i >= 0; i--) {
      if (branchMentioned(parked[i], stepTokens)) merging.unshift(...parked.splice(i, 1));
    }

    const children = [
      ...merging.map((b) => b.node),
      ...(current.node ? [current.node] : []),
      ...leaves,
    ];
    if (children.length === 0) continue;

    if (children.length === 1 && current.node && !leaves.length) {
      // Ingredient-less step ("let rest 10 min") wraps the running result —
      // but don't stack wrappers; extend the previous label instead.
      if (SKIP_RE.test(text)) continue;
      const prev = current.node;
      if (prev.children?.length === 1 && !prev.children[0].ingredientId) {
        prev.label = `${prev.label}, ${label}`.slice(0, 30);
      } else {
        current.node = { id: freshId(), label, children: [prev] };
      }
    } else {
      current.node = { id: freshId(), label, children };
    }
    current.used.push(...merging.flatMap((b) => b.used), ...matched);
  }

  // Branches never merged by a step join the final result; earlier branches
  // read from the top, so they go above the running chain.
  if (parked.length) {
    const nodes = parked.map((b) => b.node);
    if (!current.node) {
      current.node = { id: freshId(), label: "combine", children: nodes };
    } else if (current.node.children) {
      current.node.children.unshift(...nodes);
    }
  }

  // Anything never mentioned joins at the last step.
  if (unused.length) {
    const strays = unused.map((ing) => ({ id: freshId(), ingredientId: ing.id }));
    if (!current.node) {
      // No step named any ingredient at all ("whisk everything…") — fall
      // back to combine → <final cooking step>.
      current.node = { id: freshId(), label: "combine", children: strays };
      const lastCook = [...steps].reverse().find((s) => s.text && !SKIP_RE.test(s.text));
      if (lastCook) {
        const label = terseLabel(lastCook.text);
        if (label && label !== "combine") {
          current.node = { id: freshId(), label, children: [current.node] };
        }
      }
    } else if (current.node.children) {
      current.node.children.push(...strays);
    } else {
      current.node = { id: freshId(), label: "combine", children: [current.node, ...strays] };
    }
  }

  return { tree: current.node, prepHeaders, usedAll: unused.length === 0 };
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

/**
 * Full no-LLM pipeline: extraction result → normalized recipe with a real
 * tree when instructions allow, else the naive combine→cook fallback.
 */
export function buildRecipeFromExtraction(extracted, meta = {}) {
  let section = "";
  const ingredients = [];
  for (const raw of extracted.ingredients || []) {
    const s = String(raw).trim();
    if (/^##\s/.test(s) || /^(for the|for)\b.*:$/i.test(s) || (/:$/.test(s) && s.length < 48 && !/\d/.test(s))) {
      section = s.replace(/^##\s*/, "").replace(/:$/, "");
      continue;
    }
    if (s) ingredients.push(parseIngredientString(s, section));
  }
  if (!ingredients.length) return null;

  const { tree, prepHeaders } = buildTree(ingredients, extracted.instructions);
  const recipe = {
    title: extracted.title || meta.domain || "Recipe",
    servings: extracted.servings || "",
    totalTime: extracted.totalTime || "",
    course: guessCourse(extracted),
    prepHeaders,
    ingredients,
    tree: tree || naiveTree(ingredients),
    footnotes: [],
    notes: "",
  };
  return normalizeRecipe(recipe);
}
