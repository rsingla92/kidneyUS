// node --test test/matrix.test.js
import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeRecipe,
  computeLayout,
  flattenLeaves,
  mergeIntoParent,
  groupWithNextSibling,
  wrapInStep,
  deleteIngredient,
  moveIngredient,
  cloneTree,
} from "../panel/matrix.js";

// The Cooking for Engineers brownie chart from the spec:
//   butter → melt ┐
//   sugar ────────┴ mix ┐
//   vanilla ──────┐     │
//   espresso ─────┼ mix ┴──────┐
//   eggs ─────────┘            ├ fold in → bake
//   flour ┐                    │
//   cocoa ├────────────────────┘
//   soda  │
//   salt  ┘
function brownies() {
  const ing = (id, imperial, metric, name) => ({ id, imperial, metric, name });
  return {
    title: "Espresso Brownies",
    course: "dessert",
    prepHeaders: ["Butter and flour an 8x8-in pan", "Preheat oven to 350°F (170°C)"],
    ingredients: [
      ing("i1", "4 oz", "115 g", "unsalted butter"),
      ing("i2", "1 cup", "200 g", "sugar"),
      ing("i3", "1/4 tsp", "2.5 mL", "vanilla"),
      ing("i4", "1 shot", "60 mL", "espresso"),
      ing("i5", "2 large", "100 g", "eggs"),
      ing("i6", "1/2 cup", "80 g", "flour"),
      ing("i7", "1/3 cup", "80 g", "cocoa powder"),
      ing("i8", "1/4 tsp", "1.3 g", "baking soda"),
      ing("i9", "1/4 tsp", "1.5 g", "salt"),
    ],
    tree: {
      label: "bake 350°F 30-40 min",
      children: [
        {
          label: "fold in",
          children: [
            {
              label: "mix",
              children: [
                {
                  label: "mix",
                  children: [
                    { label: "melt", children: [{ ingredientId: "i1" }] },
                    { ingredientId: "i2" },
                  ],
                },
                { ingredientId: "i3" },
                { ingredientId: "i4" },
                { ingredientId: "i5" },
              ],
            },
            { ingredientId: "i6" },
            { ingredientId: "i7" },
            { ingredientId: "i8" },
            { ingredientId: "i9" },
          ],
        },
      ],
    },
  };
}

test("brownie layout: columns, rowspans, colspans", () => {
  const recipe = normalizeRecipe(brownies());
  const { columnCount, rows } = computeLayout(recipe.tree);

  // ingredient col + melt(1) mix(2) mix(3) fold-in(4) bake(5)
  assert.equal(columnCount, 6);
  assert.equal(rows.length, 9);

  const cell = (r, type, label) =>
    rows[r].find((c) => c.type === type && (label === undefined || c.label === label));

  // Row 0 carries butter, melt, mix, mix, fold in, bake.
  assert.deepEqual(
    rows[0].map((c) => [c.type === "op" ? c.label : "ING", c.col, c.colspan, c.rowspan]),
    [
      ["ING", 0, 1, 1],
      ["melt", 1, 1, 1],
      ["mix", 2, 1, 2], // butter+sugar
      ["mix", 3, 1, 5], // through eggs
      ["fold in", 4, 1, 9],
      ["bake 350°F 30-40 min", 5, 1, 9],
    ]
  );

  // Sugar joins "mix" directly: its ingredient cell spans cols 0-1.
  assert.deepEqual(rows[1].map((c) => [c.type, c.col, c.colspan]), [["ingredient", 0, 2]]);

  // Espresso goes straight to the inner mix at col 3: spans cols 0-2.
  assert.equal(cell(3, "ingredient").colspan, 3);

  // Flour joins "fold in" at col 4: spans cols 0-3.
  assert.equal(cell(5, "ingredient").colspan, 4);

  // Row-emission accounting: total cell area == rows × columns.
  const area = rows.flat().reduce((s, c) => s + c.colspan * c.rowspan, 0);
  assert.equal(area, rows.length * columnCount);
});

test("normalize repairs leaf/ingredient order mismatches by reordering ingredients", () => {
  const r = brownies();
  // Shuffle the ingredient list; leaf order in the tree must win.
  r.ingredients.reverse();
  const recipe = normalizeRecipe(r);
  assert.deepEqual(
    recipe.ingredients.map((ing) => ing.id),
    ["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"]
  );
});

test("normalize appends ingredients missing from the tree", () => {
  const r = brownies();
  r.ingredients.push({ id: "i10", imperial: "1 pinch", metric: "", name: "nutmeg" });
  const recipe = normalizeRecipe(r);
  const leaves = flattenLeaves(recipe.tree).map((l) => l.ingredientId);
  assert.ok(leaves.includes("i10"));
  assert.equal(recipe.ingredients.length, 10);
  const { rows, columnCount } = computeLayout(recipe.tree);
  const area = rows.flat().reduce((s, c) => s + c.colspan * c.rowspan, 0);
  assert.equal(area, rows.length * columnCount);
});

test("normalize drops leaves referencing unknown ingredients", () => {
  const r = brownies();
  r.tree.children[0].children.push({ ingredientId: "bogus" });
  const recipe = normalizeRecipe(r);
  assert.equal(flattenLeaves(recipe.tree).length, 9);
});

test("normalize builds a naive tree when tree is missing", () => {
  const r = brownies();
  delete r.tree;
  const recipe = normalizeRecipe(r);
  const { columnCount, rows } = computeLayout(recipe.tree);
  assert.equal(columnCount, 3); // ingredients | combine | cook
  assert.equal(rows[0].map((c) => c.label).join(","), ",combine,cook");
  assert.equal(rows.length, 9);
});

test("edit ops keep the layout consistent", () => {
  const recipe = normalizeRecipe(brownies());
  const check = () => {
    const { rows, columnCount } = computeLayout(cloneTree(recipe.tree));
    const area = rows.flat().reduce((s, c) => s + c.colspan * c.rowspan, 0);
    assert.equal(area, rows.length * columnCount);
  };

  // Merge the "melt" node into its parent mix.
  const melt = (function find(n) {
    if (n.label === "melt") return n;
    for (const c of n.children || []) {
      const hit = find(c);
      if (hit) return hit;
    }
    return null;
  })(recipe.tree);
  assert.ok(mergeIntoParent(recipe.tree, melt.id));
  check();

  // Group flour with cocoa under "sift".
  const flourLeaf = flattenLeaves(recipe.tree)[5];
  assert.ok(groupWithNextSibling(recipe.tree, flourLeaf.id, "sift"));
  check();

  // Wrap the fold-in node in a rest step.
  const foldIn = recipe.tree.children[0];
  assert.ok(wrapInStep(recipe.tree, foldIn.id, "rest 10 min"));
  check();

  // Delete an ingredient.
  assert.ok(deleteIngredient(recipe, "i3"));
  assert.equal(recipe.ingredients.length, 8);
  check();

  // Move salt (last) up to position 0.
  assert.ok(moveIngredient(recipe, recipe.ingredients.length - 1, 0));
  assert.equal(recipe.ingredients[0].name, "salt");
  assert.deepEqual(
    flattenLeaves(recipe.tree).map((l) => l.ingredientId),
    recipe.ingredients.map((ing) => ing.id)
  );
  check();
});

test("single-ingredient recipe still lays out", () => {
  const recipe = normalizeRecipe({
    ingredients: [{ id: "i1", imperial: "1", metric: "", name: "egg" }],
    tree: { ingredientId: "i1" },
  });
  const { columnCount, rows } = computeLayout(recipe.tree);
  assert.equal(rows.length, 1);
  assert.ok(columnCount >= 2);
});
