// node --test test/pdf.test.js — sanity-check the hand-rolled PDF writer.
import test from "node:test";
import assert from "node:assert/strict";
import zlib from "node:zlib";
import { buildPdf } from "../panel/export.js";

test("buildPdf produces a structurally valid one-page PDF", () => {
  const pxW = 40;
  const pxH = 20;
  const rgb = new Uint8Array(pxW * pxH * 3).fill(200);
  const compressed = new Uint8Array(zlib.deflateSync(rgb)); // zlib == FlateDecode
  const bytes = buildPdf({ pxW, pxH, compressed, page: "letter", scale: 2 });

  const latin1 = Buffer.from(bytes).toString("latin1");
  assert.ok(latin1.startsWith("%PDF-1.4\n"));
  assert.equal(bytes[9], 0x25); // '%' of the binary marker line
  assert.equal(bytes[10], 0xe2); // raw high bytes, not UTF-8 expanded
  assert.ok(latin1.trimEnd().endsWith("%%EOF"));

  // Verify every xref offset actually points at "N 0 obj".
  const xrefStart = parseInt(latin1.match(/startxref\n(\d+)\n/)[1], 10);
  assert.equal(latin1.slice(xrefStart, xrefStart + 4), "xref");
  const entries = latin1
    .slice(xrefStart)
    .match(/^\d{10} \d{5} n $/gm);
  assert.equal(entries.length, 5);
  entries.forEach((entry, i) => {
    const off = parseInt(entry.slice(0, 10), 10);
    assert.equal(latin1.slice(off, off + `${i + 1} 0 obj`.length), `${i + 1} 0 obj`);
  });

  // The image stream must round-trip through inflate at the right length.
  const streamStart = latin1.indexOf("stream\n", latin1.indexOf("/Subtype /Image")) + 7;
  const imageBytes = bytes.slice(streamStart, streamStart + compressed.length);
  const inflated = zlib.inflateSync(Buffer.from(imageBytes));
  assert.equal(inflated.length, pxW * pxH * 3);

  // Page size and fit: image must be placed inside the media box.
  const [, w, h, x, y] = latin1.match(/([\d.]+) 0 0 ([\d.]+) ([\d.]+) ([\d.]+) cm/).map(Number);
  assert.ok(x >= 36 && y >= 36 && x + w <= 612 - 35 && y + h <= 792 - 35);
});

test("buildPdf A4 fit for an oversized chart", () => {
  const pxW = 4000; // wider than a page at 96dpi
  const pxH = 1000;
  const compressed = new Uint8Array(zlib.deflateSync(new Uint8Array(12)));
  const bytes = buildPdf({ pxW, pxH, compressed, page: "a4", scale: 2 });
  const latin1 = Buffer.from(bytes).toString("latin1");
  const [, w, h] = latin1.match(/([\d.]+) 0 0 ([\d.]+) [\d.]+ [\d.]+ cm/).map(Number);
  assert.ok(w <= 595.28 - 72 + 0.01, "fits width");
  assert.ok(h <= 841.89 - 72 + 0.01, "fits height");
  // Aspect ratio preserved.
  assert.ok(Math.abs(w / h - pxW / pxH) < 0.01);
});
