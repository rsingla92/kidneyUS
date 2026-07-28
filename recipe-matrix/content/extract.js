// Deterministic recipe extraction, injected into the page on demand.
// Runs as a classic script; its final expression is returned to the panel
// via chrome.scripting.executeScript.
//
// Priority: JSON-LD → microdata → heuristic DOM scrape. If all fail, we
// return readable page text so the panel can fall back to LLM extraction.

(() => {
  const clean = (s) =>
    String(s ?? "")
      // Some sites ship raw HTML inside JSON-LD strings (shop links in
      // ingredients, <br> in steps) — strip markup before anything else.
      .replace(/<[^>]*>/g, " ")
      .replace(/&nbsp;/gi, " ")
      .replace(/&amp;/gi, "&")
      .replace(/&#39;|&apos;/gi, "'")
      .replace(/&quot;/gi, '"')
      .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(+n))
      .replace(/\s+/g, " ")
      .trim();

  // --- 1. JSON-LD ---------------------------------------------------------

  function* jsonLdObjects(value) {
    if (!value || typeof value !== "object") return;
    if (Array.isArray(value)) {
      for (const v of value) yield* jsonLdObjects(v);
      return;
    }
    yield value;
    if (value["@graph"]) yield* jsonLdObjects(value["@graph"]);
    if (value.mainEntity) yield* jsonLdObjects(value.mainEntity);
  }

  function isRecipeType(t) {
    if (!t) return false;
    const types = Array.isArray(t) ? t : [t];
    return types.some((x) => String(x).toLowerCase().includes("recipe"));
  }

  function instructionsFrom(value, out = []) {
    if (!value) return out;
    if (typeof value === "string") {
      // Some sites stuff all steps into one HTML string.
      const parts = value.split(/<\/?(?:li|p|br)[^>]*>/i).map(clean).filter(Boolean);
      out.push(...(parts.length ? parts : [clean(value)]));
      return out;
    }
    if (Array.isArray(value)) {
      for (const v of value) instructionsFrom(v, out);
      return out;
    }
    if (typeof value === "object") {
      const type = String(value["@type"] || "");
      if (/HowToSection/i.test(type)) {
        if (value.name) out.push(`## ${clean(value.name)}`);
        instructionsFrom(value.itemListElement, out);
      } else {
        const text = clean(value.text || value.name || "");
        if (text) out.push(text);
        else instructionsFrom(value.itemListElement, out);
      }
    }
    return out;
  }

  function durationToText(iso) {
    if (!iso) return "";
    if (typeof iso !== "string" || !/^P/i.test(iso)) return clean(String(iso));
    const m = iso.match(/P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?/i);
    if (!m) return "";
    const [, d, h, min] = m.map((x) => parseInt(x || "0", 10));
    const parts = [];
    if (d) parts.push(`${d} d`);
    if (h) parts.push(`${h} hr`);
    if (min) parts.push(`${min} min`);
    return parts.join(" ");
  }

  function fromJsonLd() {
    for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
      let data;
      try {
        data = JSON.parse(script.textContent);
      } catch {
        continue;
      }
      for (const obj of jsonLdObjects(data)) {
        if (!isRecipeType(obj["@type"])) continue;
        const ingredients = (obj.recipeIngredient || obj.ingredients || [])
          .map(clean)
          .filter(Boolean);
        const instructions = instructionsFrom(obj.recipeInstructions);
        if (!ingredients.length) continue;
        return {
          method: "json-ld",
          title: clean(obj.name),
          ingredients,
          instructions,
          servings: clean(Array.isArray(obj.recipeYield) ? obj.recipeYield[0] : obj.recipeYield),
          totalTime: durationToText(obj.totalTime) ||
            [durationToText(obj.prepTime), durationToText(obj.cookTime)].filter(Boolean).join(" + "),
          category: clean(Array.isArray(obj.recipeCategory) ? obj.recipeCategory[0] : obj.recipeCategory),
          cuisine: clean(Array.isArray(obj.recipeCuisine) ? obj.recipeCuisine[0] : obj.recipeCuisine),
        };
      }
    }
    return null;
  }

  // --- 2. Microdata -------------------------------------------------------

  function fromMicrodata() {
    const scope = document.querySelector('[itemtype*="schema.org/Recipe" i]');
    if (!scope) return null;
    const props = (name) =>
      [...scope.querySelectorAll(`[itemprop="${name}" i]`)].map((el) =>
        clean(el.content || el.textContent)
      );
    const ingredients = [...props("recipeIngredient"), ...props("ingredients")].filter(Boolean);
    if (!ingredients.length) return null;
    return {
      method: "microdata",
      title: clean(props("name")[0] || ""),
      ingredients,
      instructions: props("recipeInstructions").filter(Boolean),
      servings: props("recipeYield")[0] || "",
      totalTime: props("totalTime")[0] || "",
      category: props("recipeCategory")[0] || "",
      cuisine: props("recipeCuisine")[0] || "",
    };
  }

  // --- 3. Heuristic DOM scrape -------------------------------------------

  function textItems(container) {
    const items = [...container.querySelectorAll("li")];
    const src = items.length ? items : [...container.querySelectorAll("p")];
    return src.map((el) => clean(el.textContent)).filter((t) => t && t.length < 300);
  }

  function bestMatch(regex) {
    const candidates = [...document.querySelectorAll("ul, ol, div, section")].filter((el) => {
      const key = `${el.className} ${el.id}`.toLowerCase();
      return regex.test(key);
    });
    // Prefer the candidate with the most list items and the least nesting noise.
    let best = null;
    let bestCount = 0;
    for (const el of candidates) {
      const n = textItems(el).length;
      if (n > bestCount) {
        best = el;
        bestCount = n;
      }
    }
    return best;
  }

  function fromHeuristics() {
    const ingEl = bestMatch(/ingredient/);
    const insEl = bestMatch(/instruction|direction|method|preparation|steps?\b/);
    if (!ingEl) return null;
    const ingredients = textItems(ingEl);
    if (ingredients.length < 2) return null;
    return {
      method: "heuristic",
      title: clean(document.querySelector("h1")?.textContent || document.title),
      ingredients,
      instructions: insEl ? textItems(insEl) : [],
      servings: "",
      totalTime: "",
      category: "",
      cuisine: "",
    };
  }

  // --- 4. Readable text for the LLM fallback ------------------------------

  function readableText() {
    const clone = document.body.cloneNode(true);
    clone
      .querySelectorAll("script, style, nav, header, footer, iframe, noscript, svg, form, aside")
      .forEach((el) => el.remove());
    return clean(clone.textContent).slice(0, 20000);
  }

  const recipe = fromJsonLd() || fromMicrodata() || fromHeuristics();
  return {
    ok: !!recipe,
    recipe,
    pageText: recipe ? "" : readableText(),
    url: location.href,
    domain: location.hostname,
    pageTitle: document.title,
  };
})();
