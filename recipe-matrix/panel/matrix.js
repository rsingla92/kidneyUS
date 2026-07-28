// matrix.js — the tabular-recipe data model and rowspan/colspan layout.
//
// A recipe chart is a tree whose leaves are ingredient references and whose
// internal nodes are operations ("melt", "mix", "bake 350°F 30-40 min").
// The chart is an HTML <table>: ingredients occupy column 0, one per row;
// each operation cell spans (rowspan) every ingredient row in its subtree,
// and stretches right (colspan) until it meets its parent's column.

let idCounter = 0;
export function freshId(prefix = "n") {
  return `${prefix}${++idCounter}_${Math.random().toString(36).slice(2, 7)}`;
}

// ---------------------------------------------------------------------------
// Tree helpers
// ---------------------------------------------------------------------------

export function isLeaf(node) {
  return node && node.ingredientId != null;
}

/** Flatten leaf nodes left-to-right (display order of ingredient rows). */
export function flattenLeaves(node, out = []) {
  if (!node) return out;
  if (isLeaf(node)) {
    out.push(node);
  } else {
    for (const child of node.children || []) flattenLeaves(child, out);
  }
  return out;
}

export function walk(node, fn, parent = null) {
  if (!node) return;
  fn(node, parent);
  if (!isLeaf(node)) for (const c of node.children || []) walk(c, fn, node);
}

export function findNode(root, id) {
  let hit = null;
  walk(root, (n) => {
    if (n.id === id) hit = n;
  });
  return hit;
}

export function findParent(root, id) {
  let hit = null;
  walk(root, (n, p) => {
    if (n.id === id) hit = p;
  });
  return hit;
}

/** Deep clone (plain JSON data only). */
export function cloneTree(x) {
  return JSON.parse(JSON.stringify(x));
}

// ---------------------------------------------------------------------------
// Validation & repair
// ---------------------------------------------------------------------------

/**
 * Normalize a recipe object (from the LLM or the naive fallback) in place:
 *  - give every tree node and ingredient an id
 *  - drop leaves that reference unknown ingredients
 *  - append ingredients missing from the tree as direct children of the root
 *  - collapse internal nodes left with zero children
 *  - reorder `ingredients` to match the tree's leaf order (leaf order wins;
 *    rowspans cannot be computed otherwise)
 * Returns the same object, or null if unsalvageable.
 */
export function normalizeRecipe(recipe) {
  if (!recipe || !Array.isArray(recipe.ingredients) || recipe.ingredients.length === 0) {
    return null;
  }
  recipe.ingredients.forEach((ing, i) => {
    if (!ing.id) ing.id = `i${i + 1}`;
    ing.name = String(ing.name ?? "").trim();
    ing.imperial = String(ing.imperial ?? "").trim();
    ing.metric = String(ing.metric ?? "").trim();
    ing.note = String(ing.note ?? "").trim();
    ing.section = String(ing.section ?? "").trim();
  });
  const byId = new Map(recipe.ingredients.map((ing) => [ing.id, ing]));

  if (!recipe.tree || typeof recipe.tree !== "object") {
    recipe.tree = naiveTree(recipe.ingredients);
  }

  // Assign node ids, prune bad leaves / empty branches.
  const seenIngredients = new Set();
  const prune = (node) => {
    if (!node || typeof node !== "object") return null;
    if (node.ingredientId != null) {
      if (!byId.has(node.ingredientId) || seenIngredients.has(node.ingredientId)) return null;
      seenIngredients.add(node.ingredientId);
      return { id: node.id || freshId(), ingredientId: node.ingredientId };
    }
    const children = (Array.isArray(node.children) ? node.children : [])
      .map(prune)
      .filter(Boolean);
    if (children.length === 0) return null;
    return { id: node.id || freshId(), label: String(node.label ?? "").trim() || "…", children };
  };
  let root = prune(recipe.tree);

  // Ingredients the tree forgot become direct children of the root.
  const missing = recipe.ingredients.filter((ing) => !seenIngredients.has(ing.id));
  if (!root) {
    root = naiveTree(recipe.ingredients);
  } else if (missing.length) {
    if (isLeaf(root)) root = { id: freshId(), label: "combine", children: [root] };
    for (const ing of missing) root.children.push({ id: freshId(), ingredientId: ing.id });
  }

  // Leaf order is authoritative: reorder the ingredient list to match it.
  const leafOrder = flattenLeaves(root).map((l) => l.ingredientId);
  recipe.ingredients = leafOrder.map((id) => byId.get(id));

  recipe.tree = root;
  recipe.title = String(recipe.title ?? "").trim();
  recipe.servings = String(recipe.servings ?? "").trim();
  recipe.totalTime = String(recipe.totalTime ?? "").trim();
  recipe.course = String(recipe.course ?? "other").trim().toLowerCase();
  recipe.prepHeaders = (recipe.prepHeaders || []).map((h) => String(h).trim()).filter(Boolean);
  recipe.footnotes = (recipe.footnotes || []).map((f) => String(f).trim()).filter(Boolean);
  recipe.notes = String(recipe.notes ?? "");
  return recipe;
}

/** No-LLM fallback: all ingredients → "combine" → "cook". */
export function naiveTree(ingredients) {
  return {
    id: freshId(),
    label: "cook",
    children: [
      {
        id: freshId(),
        label: "combine",
        children: ingredients.map((ing) => ({ id: freshId(), ingredientId: ing.id })),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Layout — the heart of the chart
// ---------------------------------------------------------------------------

/**
 * Compute the table layout for a normalized recipe tree.
 *
 * Every node is annotated with:
 *   height         leaves = 0; internal = max(child.height) + 1
 *   leafCount      number of ingredient rows the cell spans (its rowspan)
 *   firstLeafIndex row on which the cell is emitted
 *
 * Column model: ingredient cells live in column 0; an internal node's cell
 * starts at column = its height. A cell stretches right until its parent's
 * column begins: colspan = parent.col − own.col (the root runs to the last
 * column). This is what lets an ingredient that joins late — espresso going
 * straight into "fold in" — span the empty columns in between.
 *
 * Returns { columnCount, rows } where rows[i] is the ordered list of cells
 * to emit for <tr> i: { type: 'ingredient'|'op', node, ingredientId?,
 * label?, rowspan, colspan }.
 */
export function computeLayout(tree) {
  const root = tree;
  let leafIndex = 0;

  const annotate = (node) => {
    if (isLeaf(node)) {
      node._height = 0;
      node._leafCount = 1;
      node._firstLeaf = leafIndex++;
      return;
    }
    let h = 0;
    let leaves = 0;
    for (const c of node.children) {
      annotate(c);
      h = Math.max(h, c._height);
      leaves += c._leafCount;
    }
    node._height = h + 1;
    node._leafCount = leaves;
    node._firstLeaf = node.children[0]._firstLeaf;
  };
  annotate(root);

  // A bare-leaf root still gets one op column so the table has shape.
  const columnCount = Math.max(root._height, 1) + 1; // +1 for the ingredient column

  const rows = Array.from({ length: root._leafCount }, () => []);

  const place = (node, parentCol) => {
    const col = isLeaf(node) ? 0 : node._height;
    const colspan = Math.max(parentCol - col, 1);
    rows[node._firstLeaf].push({
      type: isLeaf(node) ? "ingredient" : "op",
      node,
      ingredientId: node.ingredientId,
      label: node.label,
      col,
      colspan,
      rowspan: node._leafCount,
    });
    if (!isLeaf(node)) for (const c of node.children) place(c, col);
  };
  place(root, columnCount);

  // Emit cells left-to-right within each row.
  for (const row of rows) row.sort((a, b) => a.col - b.col);

  return { columnCount, rows };
}

// ---------------------------------------------------------------------------
// Tree edits (used by the editor; all return true on success)
// ---------------------------------------------------------------------------

/** Splice a node out, hoisting its children into its parent at its position. */
export function mergeIntoParent(root, nodeId) {
  const parent = findParent(root, nodeId);
  if (!parent) return false; // can't merge the root
  const idx = parent.children.findIndex((c) => c.id === nodeId);
  const node = parent.children[idx];
  if (isLeaf(node)) return false;
  parent.children.splice(idx, 1, ...node.children);
  return true;
}

/** Wrap a node in a new single-child parent operation. */
export function wrapInStep(root, nodeId, label = "step") {
  if (root.id === nodeId) return false; // root wrap handled by caller (new root)
  const parent = findParent(root, nodeId);
  if (!parent) return false;
  const idx = parent.children.findIndex((c) => c.id === nodeId);
  parent.children[idx] = { id: freshId(), label, children: [parent.children[idx]] };
  return true;
}

/** Group a node with its next sibling under a new operation. */
export function groupWithNextSibling(root, nodeId, label = "mix") {
  const parent = findParent(root, nodeId);
  if (!parent) return false;
  const idx = parent.children.findIndex((c) => c.id === nodeId);
  if (idx < 0 || idx >= parent.children.length - 1) return false;
  const pair = parent.children.splice(idx, 2);
  parent.children.splice(idx, 0, { id: freshId(), label, children: pair });
  return true;
}

/** Remove an ingredient row everywhere (list + tree), pruning empty branches. */
export function deleteIngredient(recipe, ingredientId) {
  const before = recipe.ingredients.length;
  if (before <= 1) return false; // never delete the last row
  recipe.ingredients = recipe.ingredients.filter((ing) => ing.id !== ingredientId);
  const prune = (node) => {
    if (isLeaf(node)) return node.ingredientId === ingredientId ? null : node;
    node.children = node.children.map(prune).filter(Boolean);
    return node.children.length ? node : null;
  };
  recipe.tree = prune(recipe.tree) || naiveTree(recipe.ingredients);
  // If the root collapsed to a lone leaf chain, keep it; layout handles it.
  return recipe.ingredients.length < before;
}

/**
 * Move the ingredient at leaf position `from` to position `to` (display
 * order). The leaf is re-attached next to the leaf currently at `to`, inside
 * that leaf's parent, so the tree stays valid and flattens in the new order.
 */
export function moveIngredient(recipe, from, to) {
  const leaves = flattenLeaves(recipe.tree);
  if (from === to || from < 0 || to < 0 || from >= leaves.length || to >= leaves.length) {
    return false;
  }
  const moving = leaves[from];
  const target = leaves[to];
  const oldParent = findParent(recipe.tree, moving.id);
  if (!oldParent) return false;
  oldParent.children.splice(oldParent.children.indexOf(moving), 1);

  const newParent = findParent(recipe.tree, target.id) || recipe.tree;
  const tIdx = newParent.children.indexOf(target);
  newParent.children.splice(from < to ? tIdx + 1 : tIdx, 0, moving);

  // The old parent may now be empty — prune upward.
  const prune = (node) => {
    if (isLeaf(node)) return node;
    node.children = node.children.map(prune).filter(Boolean);
    return node.children.length ? node : null;
  };
  recipe.tree = prune(recipe.tree);

  recipe.ingredients = flattenLeaves(recipe.tree).map((l) =>
    recipe.ingredients.find((ing) => ing.id === l.ingredientId)
  );
  return true;
}
