// export.js — PNG / PDF / clipboard export with no external libraries.
//
// PNG: the chart subtree is serialized into an SVG <foreignObject> along
// with our own stylesheet text, loaded into an <img>, and drawn to a canvas
// at 2x. Everything is same-document and inline, so the canvas stays
// untainted and MV3's CSP is never involved.
//
// PDF: a minimal one-page PDF is assembled by hand; the rendered chart is
// embedded as a lossless FlateDecode RGB image scaled to fit Letter/A4.

const SCALE = 2; // 2x device pixel ratio

function ownStylesText() {
  let css = "";
  for (const sheet of document.styleSheets) {
    try {
      for (const rule of sheet.cssRules) {
        // @font-face rules reference chrome-extension:// URLs, which the
        // isolated SVG rasterization context refuses to load — they're
        // re-added as data: URIs by embeddedFontCss() instead.
        if (rule instanceof CSSFontFaceRule) continue;
        css += rule.cssText + "\n";
      }
    } catch {
      /* cross-origin sheet — none in this extension */
    }
  }
  return css;
}

// Bundled Source Serif 4 (SIL OFL 1.1), inlined for export.
const FONT_FILES = [
  { file: "SourceSerif4-Regular.ttf.woff2", weight: 400, style: "normal" },
  { file: "SourceSerif4-It.ttf.woff2", weight: 400, style: "italic" },
  { file: "SourceSerif4-Semibold.ttf.woff2", weight: 600, style: "normal" },
  { file: "SourceSerif4-Bold.ttf.woff2", weight: 700, style: "normal" },
];
let fontCssCache = null;

async function embeddedFontCss() {
  if (fontCssCache !== null) return fontCssCache;
  const faces = await Promise.all(
    FONT_FILES.map(async ({ file, weight, style }) => {
      try {
        const res = await fetch(chrome.runtime.getURL(`fonts/${file}`));
        const buf = new Uint8Array(await res.arrayBuffer());
        let bin = "";
        for (let i = 0; i < buf.length; i += 0x8000) {
          bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
        }
        return `@font-face{font-family:"Source Serif 4";font-weight:${weight};font-style:${style};src:url(data:font/woff2;base64,${btoa(bin)}) format("woff2");}`;
      } catch {
        return ""; // fall back to the CSS font stack
      }
    })
  );
  fontCssCache = faces.join("\n");
  return fontCssCache;
}

export async function renderToCanvas(chartEl, { transparent = false } = {}) {
  chartEl.classList.add("exporting");
  const rect = chartEl.getBoundingClientRect();
  const width = Math.ceil(rect.width);
  const height = Math.ceil(rect.height);

  const clone = chartEl.cloneNode(true);
  clone.removeAttribute("hidden");
  // Freeze layout: the clone must not depend on the panel's flex context.
  clone.style.width = width + "px";
  clone.style.margin = "0";
  if (transparent) clone.style.background = "transparent";
  clone.querySelectorAll("[contenteditable]").forEach((el) => el.removeAttribute("contenteditable"));

  chartEl.classList.remove("exporting");

  const wrapper = document.createElement("div");
  wrapper.appendChild(clone);
  const html = new XMLSerializer().serializeToString(wrapper);

  await document.fonts.ready;
  const css = (await embeddedFontCss()) + "\n" + ownStylesText();
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
    `<foreignObject width="100%" height="100%">` +
    `<div xmlns="http://www.w3.org/1999/xhtml"><style>${css
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")}</style>${html}</div>` +
    `</foreignObject></svg>`;

  // A data: URL, not a blob: URL — Chromium taints the canvas when a
  // foreignObject SVG is drawn from a blob URL, but not from a data URL.
  const img = new Image();
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("could not rasterize chart"));
  });
  await img.decode().catch(() => {}); // ensure embedded fonts are rasterized
  const canvas = document.createElement("canvas");
  canvas.width = width * SCALE;
  canvas.height = height * SCALE;
  const ctx = canvas.getContext("2d");
  ctx.scale(SCALE, SCALE);
  ctx.drawImage(img, 0, 0);
  return canvas;
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  if (chrome?.downloads?.download) {
    chrome.downloads.download({ url, filename, saveAs: false }, () =>
      setTimeout(() => URL.revokeObjectURL(url), 30000)
    );
  } else {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }
}

function canvasToBlob(canvas, type = "image/png") {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("canvas.toBlob failed"))), type)
  );
}

export async function exportPng(chartEl, { filename = "recipe", transparent = false } = {}) {
  const canvas = await renderToCanvas(chartEl, { transparent });
  download(await canvasToBlob(canvas), `${filename}.png`);
}

export async function copyPngToClipboard(chartEl) {
  const canvas = await renderToCanvas(chartEl);
  const blob = await canvasToBlob(canvas);
  await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
}

// ---------------------------------------------------------------------------
// Minimal PDF writer
// ---------------------------------------------------------------------------

const PAGE_SIZES = {
  letter: [612, 792],
  a4: [595.28, 841.89],
};

async function deflate(bytes) {
  // CompressionStream "deflate" = zlib (RFC 1950), which is exactly what
  // PDF's FlateDecode expects.
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function exportPdf(chartEl, { filename = "recipe", page = "letter" } = {}) {
  const canvas = await renderToCanvas(chartEl, { transparent: false });
  const { width: pxW, height: pxH } = canvas;

  // Strip alpha → tightly packed RGB.
  const ctx = canvas.getContext("2d");
  const rgba = ctx.getImageData(0, 0, pxW, pxH).data;
  const rgb = new Uint8Array(pxW * pxH * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  const compressed = await deflate(rgb);
  const bytes = buildPdf({ pxW, pxH, compressed, page, scale: SCALE });
  download(new Blob([bytes], { type: "application/pdf" }), `${filename}.pdf`);
}

/** Assemble a one-page PDF embedding a FlateDecode RGB image. Pure bytes. */
export function buildPdf({ pxW, pxH, compressed, page = "letter", scale = SCALE }) {
  const [pageW, pageH] = PAGE_SIZES[page] || PAGE_SIZES.letter;
  const margin = 36;
  const maxW = pageW - margin * 2;
  const maxH = pageH - margin * 2;
  // CSS pixel → point at 96dpi, then shrink if the chart overflows the page.
  let drawW = (pxW / scale) * 0.75;
  let drawH = (pxH / scale) * 0.75;
  const fit = Math.min(1, maxW / drawW, maxH / drawH);
  drawW *= fit;
  drawH *= fit;
  const x = margin + (maxW - drawW) / 2;
  const y = pageH - margin - drawH; // PDF origin is bottom-left; pin to top.

  const content = `q\n${drawW.toFixed(2)} 0 0 ${drawH.toFixed(2)} ${x.toFixed(2)} ${y.toFixed(2)} cm\n/Im0 Do\nQ\n`;

  const enc = new TextEncoder();
  const parts = [];
  let offset = 0;
  const offsets = [];
  const push = (data) => {
    const bytes = typeof data === "string" ? enc.encode(data) : data;
    parts.push(bytes);
    offset += bytes.length;
  };
  const beginObj = (n) => {
    offsets[n] = offset;
    push(`${n} 0 obj\n`);
  };

  // Binary-marker comment: raw latin1 bytes, not UTF-8.
  push(Uint8Array.from("%PDF-1.4\n%\xE2\xE3\xCF\xD3\n", (c) => c.charCodeAt(0)));
  beginObj(1);
  push("<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
  beginObj(2);
  push("<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
  beginObj(3);
  push(
    `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pageW} ${pageH}] ` +
      `/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>\nendobj\n`
  );
  beginObj(4);
  push(`<< /Length ${content.length} >>\nstream\n${content}endstream\nendobj\n`);
  beginObj(5);
  push(
    `<< /Type /XObject /Subtype /Image /Width ${pxW} /Height ${pxH} ` +
      `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode ` +
      `/Length ${compressed.length} >>\nstream\n`
  );
  push(compressed);
  push("\nendstream\nendobj\n");

  const xrefStart = offset;
  push(`xref\n0 6\n0000000000 65535 f \n`);
  for (let n = 1; n <= 5; n++) push(`${String(offsets[n]).padStart(10, "0")} 00000 n \n`);
  push(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF\n`);

  const out = new Uint8Array(offset);
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}
