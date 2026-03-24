import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import workerSrc from 'pdfjs-dist/build/pdf.worker.min?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

const PAGE_SCALE = 1.4;

function normalizeText(value) {
  return String(value || '').toLowerCase().replace(/\s+/g, ' ').trim();
}

function buildCandidatePhrases(chunkText) {
  const normalized = normalizeText(chunkText);
  if (!normalized) return [];

  const words = normalized.split(' ');
  const candidates = [normalized];

  if (words.length > 14) candidates.push(words.slice(0, 14).join(' '));
  if (words.length > 8) candidates.push(words.slice(0, 8).join(' '));

  return [...new Set(candidates)].filter((phrase) => phrase.length >= 8);
}

function findHighlightsOnPage(textContent, viewport, chunks) {
  const normalizedItems = textContent.items.map((item) => normalizeText(item.str));
  const spans = [];

  let cursor = 0;
  normalizedItems.forEach((itemText, index) => {
    const start = cursor;
    const end = start + itemText.length;
    spans.push({ index, start, end });
    cursor = end + 1;
  });

  const pageText = normalizedItems.join(' ');
  const highlights = [];

  chunks.forEach((chunk) => {
    const phrases = buildCandidatePhrases(chunk.text || chunk.sentence || '');
    if (phrases.length === 0) return;

    let matchStart = -1;
    let matchLen = 0;

    for (const phrase of phrases) {
      const found = pageText.indexOf(phrase);
      if (found !== -1) {
        matchStart = found;
        matchLen = phrase.length;
        break;
      }
    }

    if (matchStart === -1) return;

    const matchEnd = matchStart + matchLen;
    const matchedSpans = spans.filter((span) => span.end > matchStart && span.start < matchEnd);

    matchedSpans.forEach((span) => {
      const item = textContent.items[span.index];
      if (!item?.transform || typeof item.width !== 'number' || typeof item.height !== 'number') return;

      const x = item.transform[4];
      const y = item.transform[5];
      const pdfRect = [x, y, x + item.width, y + item.height];
      const [vx0, vy0, vx1, vy1] = viewport.convertToViewportRectangle(pdfRect);

      const left = Math.min(vx0, vx1);
      const top = Math.min(vy0, vy1);
      const width = Math.abs(vx1 - vx0);
      const height = Math.abs(vy1 - vy0);

      if (width > 0 && height > 0) {
        highlights.push({
          clauseId: chunk.clauseId,
          label: chunk.label,
          x: left,
          y: top,
          width,
          height,
        });
      }
    });
  });

  return highlights;
}

export default function PDFViewer({ fileURL, chunks = [], activeClauseId }) {
  const containerRef = useRef(null);
  const [pages, setPages] = useState([]);

  const validChunks = useMemo(
    () =>
      (Array.isArray(chunks) ? chunks : []).filter((chunk) => (chunk.text || chunk.sentence || '').trim()),
    [chunks]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadAndRender() {
      if (!fileURL) {
        setPages([]);
        return;
      }

      try {
        const loadingTask = pdfjsLib.getDocument(fileURL);
        const pdf = await loadingTask.promise;
        const nextPages = [];

        for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
          const page = await pdf.getPage(pageNumber);
          const viewport = page.getViewport({ scale: PAGE_SCALE });

          const canvas = document.createElement('canvas');
          canvas.width = viewport.width;
          canvas.height = viewport.height;

          const context = canvas.getContext('2d');
          await page.render({ canvasContext: context, viewport }).promise;

          const textContent = await page.getTextContent();
          const highlights = findHighlightsOnPage(textContent, viewport, validChunks);

          nextPages.push({
            pageNumber,
            imageDataUrl: canvas.toDataURL(),
            width: viewport.width,
            height: viewport.height,
            highlights,
          });
        }

        if (!cancelled) {
          setPages(nextPages);
        }

        if (pdf.destroy) {
          pdf.destroy();
        }
      } catch (error) {
        console.error('Failed to render PDF document:', error);
        if (!cancelled) {
          setPages([]);
        }
      }
    }

    loadAndRender();

    return () => {
      cancelled = true;
    };
  }, [fileURL, validChunks]);

  useEffect(() => {
    if (!Number.isInteger(activeClauseId)) return;
    if (!containerRef.current) return;

    const firstActivePage = pages.find((page) =>
      page.highlights.some((highlight) => highlight.clauseId === activeClauseId)
    );

    if (!firstActivePage) return;

    const target = containerRef.current.querySelector(`[data-page-num="${firstActivePage.pageNumber}"]`);
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeClauseId, pages]);

  if (!fileURL) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-slate-500">No document uploaded.</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full overflow-auto bg-slate-100 p-4">
      <div className="mx-auto max-w-5xl space-y-4">
        {pages.length === 0 && (
          <div className="flex items-center justify-center h-40 rounded-lg border border-slate-200 bg-white">
            <p className="text-slate-500">Rendering document...</p>
          </div>
        )}

        {pages.map((page) => (
          <div
            key={page.pageNumber}
            data-page-num={page.pageNumber}
            className="relative bg-white shadow-md"
            style={{ width: page.width, height: page.height }}
          >
            <img
              src={page.imageDataUrl}
              alt={`Page ${page.pageNumber}`}
              className="w-full h-full block"
            />

            <svg
              className="absolute top-0 left-0 pointer-events-none"
              width={page.width}
              height={page.height}
              viewBox={`0 0 ${page.width} ${page.height}`}
            >
              {page.highlights.map((highlight, idx) => {
                const isActive = highlight.clauseId === activeClauseId;
                return (
                  <rect
                    key={`${page.pageNumber}-${idx}`}
                    x={highlight.x}
                    y={highlight.y}
                    width={highlight.width}
                    height={highlight.height}
                    fill={isActive ? 'rgba(59, 130, 246, 0.45)' : 'rgba(250, 204, 21, 0.35)'}
                    stroke={isActive ? 'rgb(37, 99, 235)' : 'rgb(217, 119, 6)'}
                    strokeWidth={isActive ? 2 : 1}
                    rx={2}
                    ry={2}
                  />
                );
              })}
            </svg>
          </div>
        ))}
      </div>
    </div>
  );
}
