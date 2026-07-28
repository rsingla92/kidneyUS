// node --test test/heuristic.test.js — deterministic no-LLM structuring.
import test from "node:test";
import assert from "node:assert/strict";
import {
  buildRecipeFromExtraction,
  parseIngredientString,
  terseLabel,
} from "../panel/heuristic.js";
import { computeLayout, flattenLeaves } from "../panel/matrix.js";

test("parseIngredientString splits qty / name / note / metric", () => {
  assert.deepEqual(
    (({ imperial, metric, name, note }) => ({ imperial, metric, name, note }))(
      parseIngredientString("1 cup (200 g) granulated sugar, sifted")
    ),
    { imperial: "1 cup", metric: "200 g", name: "granulated sugar", note: "sifted" }
  );
  const eggs = parseIngredientString("2 large eggs");
  assert.equal(eggs.imperial, "2 large");
  assert.equal(eggs.name, "eggs");
});

test("terseLabel compresses steps to chart-sized labels", () => {
  assert.equal(terseLabel("Bake at 350°F for 30 to 40 minutes until set."), "bake 350°F 30-40 min");
  assert.equal(terseLabel("Fold in the flour mixture gently."), "fold in");
  assert.equal(terseLabel("Simmer, stirring occasionally, for 35 minutes."), "simmer 35 min");
  assert.equal(terseLabel("Let the dough rest for 1 hour."), "rest 1 hr");
});

const BROWNIES = {
  title: "Espresso Brownies",
  category: "Dessert",
  ingredients: [
    "4 oz unsalted butter",
    "1 cup granulated sugar",
    "1/4 tsp vanilla extract",
    "1 shot espresso",
    "2 large eggs",
    "1/2 cup all-purpose flour",
    "1/3 cup cocoa powder",
    "1/4 tsp baking soda",
    "1/4 tsp salt",
  ],
  instructions: [
    "Butter and flour an 8x8-in pan.",
    "Preheat oven to 350°F.",
    "Melt the butter in a saucepan.",
    "Stir in the sugar.",
    "Mix in the vanilla, espresso, and eggs.",
    "Fold in the flour, cocoa powder, baking soda, and salt.",
    "Bake at 350°F for 30 to 40 minutes.",
  ],
};

test("brownies: instruction-order fold reproduces the CfE structure", () => {
  const recipe = buildRecipeFromExtraction(BROWNIES, { domain: "example.com" });
  assert.ok(recipe);

  // Pan prep and preheat became full-width headers, not tree steps.
  assert.equal(recipe.prepHeaders.length, 2);

  const { columnCount, rows } = computeLayout(recipe.tree);
  // melt | stir in | mix | fold in | bake → 5 op columns + ingredients
  assert.equal(columnCount, 6);
  assert.equal(rows.length, 9);

  // Ingredient display order = usage order.
  assert.deepEqual(
    recipe.ingredients.map((i) => i.name),
    ["unsalted butter", "granulated sugar", "vanilla extract", "espresso", "eggs",
     "all-purpose flour", "cocoa powder", "baking soda", "salt"]
  );

  // Row 0 reads laterally: melt → stir in → mix → fold in → bake.
  assert.deepEqual(
    rows[0].filter((c) => c.type === "op").map((c) => c.label),
    ["melt", "stir in", "mix in", "fold in", "bake 350°F 30-40 min"]
  );
  // The bake cell spans every ingredient.
  const bake = rows[0].find((c) => c.label?.startsWith("bake"));
  assert.equal(bake.rowspan, 9);
  // Espresso (row 3) joins at "mix in": its cell spans across melt+stir cols.
  const espresso = rows[3][0];
  assert.equal(espresso.type, "ingredient");
  assert.equal(espresso.colspan, 3);

  const area = rows.flat().reduce((s, c) => s + c.colspan * c.rowspan, 0);
  assert.equal(area, rows.length * columnCount);
});

test("unmentioned ingredients join at the last step", () => {
  const recipe = buildRecipeFromExtraction(
    {
      title: "Mystery",
      ingredients: ["1 cup flour", "1 tsp saffron threads"],
      instructions: ["Whisk the flour with water.", "Bake for 20 minutes."],
    },
    {}
  );
  const leaves = flattenLeaves(recipe.tree).map(
    (l) => recipe.ingredients.find((i) => i.id === l.ingredientId).name
  );
  assert.ok(leaves.includes("saffron threads"));
  const { rows, columnCount } = computeLayout(recipe.tree);
  const area = rows.flat().reduce((s, c) => s + c.colspan * c.rowspan, 0);
  assert.equal(area, rows.length * columnCount);
});

test("no instructions → combine fallback still renders", () => {
  const recipe = buildRecipeFromExtraction(
    { title: "X", ingredients: ["1 cup a", "2 cups b"], instructions: [] },
    {}
  );
  const { rows } = computeLayout(recipe.tree);
  assert.equal(rows.length, 2);
});

test("ingredient sections survive and members stay adjacent", () => {
  const recipe = buildRecipeFromExtraction(
    {
      title: "Tacos",
      ingredients: [
        "For the salsa:",
        "2 tomatoes",
        "1 lime",
        "For the filling:",
        "1 lb ground beef",
        "1 tsp cumin",
      ],
      // No instructions mentioning them — combine fallback keeps list order.
      instructions: [],
    },
    {}
  );
  assert.deepEqual(
    recipe.ingredients.map((i) => i.section),
    ["For the salsa", "For the salsa", "For the filling", "For the filling"]
  );
});

test("plural/derived word forms still match steps", () => {
  const recipe = buildRecipeFromExtraction(
    {
      title: "Soup",
      ingredients: ["3 tomatoes", "2 onions", "1 cup lentils"],
      instructions: [
        "Sauté the onion until soft.",
        "Add the tomato and lentils and simmer for 30 minutes.",
      ],
    },
    {}
  );
  const { rows, columnCount } = computeLayout(recipe.tree);
  assert.equal(columnCount, 3); // sauté | simmer
  // Onion is used first → displayed first.
  assert.equal(recipe.ingredients[0].name, "onions");
  const simmer = rows.flat().find((c) => c.label?.startsWith("simmer"));
  assert.equal(simmer.rowspan, 3);
});

const TIKKA = {
  title: "Chicken Tikka Masala",
  category: "Main Course",
  ingredients: [
    "For the marinade:",
    "1.5 lbs boneless chicken thighs",
    "1 cup plain yogurt",
    "2 tbsp lemon juice",
    "2 tsp garam masala",
    "For the sauce:",
    "3 tbsp butter",
    "1 large onion, finely diced",
    "4 cloves garlic, minced",
    "1 tbsp grated ginger",
    "1 can (400 g) tomato purée",
    "1 cup heavy cream",
    "1 tsp paprika",
    "Fresh cilantro, chopped",
  ],
  instructions: [
    "## Marinade",
    "Combine the chicken, yogurt, lemon juice, and garam masala; marinate for 1 hour.",
    "Grill the chicken pieces until charred, about 8 minutes.",
    "## Sauce",
    "Melt the butter and sauté the onion for 5 minutes.",
    "Add the garlic and ginger and cook for 1 minute.",
    "Stir in the tomato purée and paprika and simmer for 15 minutes.",
    "Pour in the heavy cream.",
    "Add the grilled chicken to the sauce and simmer for 10 minutes.",
    "Garnish with cilantro.",
  ],
};

test("tikka masala: parallel branches merge on ingredient re-mention", () => {
  const recipe = buildRecipeFromExtraction(TIKKA, { domain: "example.com" });
  const { columnCount, rows } = computeLayout(recipe.tree);

  // Two parallel chains → the merge step's cell spans both branches
  // (everything except the garnish, which joins at the root).
  const merge = rows.flat().find((c) => c.label?.startsWith("simmer 10 min"));
  assert.ok(merge, "merge step exists");
  assert.equal(merge.rowspan, recipe.ingredients.length - 1);
  const root = rows.flat().find((c) => c.label?.startsWith("garnish"));
  assert.equal(root.rowspan, recipe.ingredients.length);

  // The marinade branch: chicken/yogurt/lemon/garam under one "marinate" cell
  // that does NOT span sauce ingredients.
  const marinate = rows.flat().find((c) => c.label?.startsWith("marinate"));
  assert.ok(marinate);
  assert.equal(marinate.rowspan, 4);

  // The sauce branch exists in parallel: its sauté cell spans butter+onion.
  const saute = rows.flat().find((c) => c.label?.includes("sauté"));
  assert.ok(saute);
  assert.equal(saute.rowspan, 2);

  // Layout invariant.
  const area = rows.flat().reduce((s, c) => s + c.colspan * c.rowspan, 0);
  assert.equal(area, rows.length * columnCount);

  // Marinade ingredients appear before sauce ingredients (top-down reading).
  const names = recipe.ingredients.map((i) => i.name);
  assert.ok(names.indexOf("boneless chicken thighs") < names.indexOf("butter"));
});
