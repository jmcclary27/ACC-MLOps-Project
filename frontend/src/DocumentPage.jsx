import React, { useMemo, useRef } from 'react';

function formatConfidence(value) {
  if (typeof value !== 'number') return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
}

function normalizeLabel(label) {
  return label || 'Unknown';
}

function buildHighlightedSegments(text, clauses) {
  if (!text) return [{ text: '', highlighted: false, clause: null }];

  const validClauses = [...clauses]
    .filter(
      (clause) =>
        typeof clause.start_char === 'number' &&
        typeof clause.end_char === 'number' &&
        clause.start_char >= 0 &&
        clause.end_char > clause.start_char &&
        clause.end_char <= text.length
    )
    .sort((a, b) => a.start_char - b.start_char);

  const segments = [];
  let cursor = 0;

  validClauses.forEach((clause, index) => {
    if (clause.start_char > cursor) {
      segments.push({
        key: `plain-${cursor}-${clause.start_char}`,
        text: text.slice(cursor, clause.start_char),
        highlighted: false,
        clause: null,
      });
    }

    segments.push({
      key: `highlight-${index}-${clause.start_char}-${clause.end_char}`,
      text: text.slice(clause.start_char, clause.end_char),
      highlighted: true,
      clause,
    });

    cursor = clause.end_char;
  });

  if (cursor < text.length) {
    segments.push({
      key: `plain-${cursor}-${text.length}`,
      text: text.slice(cursor),
      highlighted: false,
      clause: null,
    });
  }

  return segments;
}

function groupClausesByLabel(clauses) {
  const grouped = clauses.reduce((acc, clause, index) => {
    const label = normalizeLabel(clause.label);
    if (!acc[label]) {
      acc[label] = [];
    }
    acc[label].push({ ...clause, originalIndex: index });
    return acc;
  }, {});

  return Object.entries(grouped).sort((a, b) => {
    if (b[1].length !== a[1].length) {
      return b[1].length - a[1].length;
    }
    return a[0].localeCompare(b[0]);
  });
}

export default function DocumentPage({
  fileName,
  apiResults,
  onBack,
  confidenceThreshold,
  onConfidenceThresholdChange,
}) {
  const documentText = apiResults?.extracted_text || '';
  const results = apiResults?.results || [];

  const filteredResults = useMemo(() => {
    return results.filter((item) => {
      const confidence = typeof item.confidence === 'number' ? item.confidence : 0;
      return confidence >= confidenceThreshold;
    });
  }, [results, confidenceThreshold]);

  const groupedClauses = useMemo(() => {
    return groupClausesByLabel(filteredResults);
  }, [filteredResults]);

  const highlightRefs = useRef({});

  const segments = useMemo(() => {
    return buildHighlightedSegments(documentText, filteredResults);
  }, [documentText, filteredResults]);

  const scrollToClause = (clauseKey) => {
    const element = highlightRefs.current[clauseKey];
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  return (
    <div className="h-screen bg-white flex flex-col">
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
        <div>
          {fileName && <p className="text-sm text-slate-600">{fileName}</p>}
          {apiResults?.num_predictions != null && (
            <p className="text-sm text-slate-500">
              {apiResults.num_predictions} prediction
              {apiResults.num_predictions === 1 ? '' : 's'}
            </p>
          )}
        </div>

        <button
          onClick={onBack}
          className="px-4 py-2 text-medium font-medium text-white bg-blue-500 hover:bg-blue-600 rounded-md transition-colors"
        >
          ← Home
        </button>
      </header>

      <div className="flex flex-1 min-h-0">
        <main className="flex-1 bg-white overflow-auto border-r border-slate-200">
          <div className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur px-6 py-4">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-xl font-semibold text-slate-900">Document Viewer</h2>
                <p className="text-sm text-slate-600">
                  Highlighted clauses are shown based on the selected confidence threshold.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <label
                  htmlFor="confidence-threshold"
                  className="text-sm font-medium text-slate-700"
                >
                  Show clauses above:
                </label>
                <select
                  id="confidence-threshold"
                  value={confidenceThreshold}
                  onChange={(e) => onConfidenceThresholdChange(Number(e.target.value))}
                  className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-800 shadow-sm"
                >
                  <option value={0.7}>70%</option>
                  <option value={0.8}>80%</option>
                  <option value={0.9}>90%</option>
                </select>
              </div>
            </div>
          </div>

          {!apiResults && (
            <div className="flex items-center justify-center h-full px-8">
              <p className="text-slate-500 text-lg">No document data available.</p>
            </div>
          )}

          {apiResults && !documentText && (
            <div className="flex items-center justify-center h-full px-8">
              <p className="text-slate-500 text-lg">
                The API did not return extracted document text, so highlights cannot be rendered.
              </p>
            </div>
          )}

          {apiResults && documentText && (
            <div className="p-6">
              {filteredResults.length === 0 && (
                <div className="mb-6 rounded-lg border border-yellow-200 bg-yellow-50 px-4 py-3 text-yellow-800">
                  No clauses meet the selected confidence threshold.
                </div>
              )}

              <div className="rounded-xl border border-slate-200 bg-slate-50 p-6 shadow-sm">
                <div className="whitespace-pre-wrap break-words text-[15px] leading-7 text-slate-800">
                  {segments.map((segment) => {
                    if (!segment.highlighted) {
                      return <span key={segment.key}>{segment.text}</span>;
                    }

                    const clauseKey = `${segment.clause.start_char}-${segment.clause.end_char}-${segment.clause.label}`;

                    return (
                      <mark
                        key={segment.key}
                        ref={(node) => {
                          if (node) {
                            highlightRefs.current[clauseKey] = node;
                          }
                        }}
                        className="rounded px-1 py-0.5 bg-yellow-200 text-slate-900"
                        title={`${segment.clause.label} • ${formatConfidence(segment.clause.confidence)}`}
                      >
                        {segment.text}
                      </mark>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </main>

        <aside className="w-[420px] bg-slate-50 overflow-auto">
          <div className="p-5 border-b border-slate-200 bg-white sticky top-0 z-10">
            <h2 className="text-xl font-semibold text-slate-900">Extracted Clauses</h2>
            <p className="text-sm text-slate-600 mt-1">
              Grouped by label. Click a clause to jump to its highlighted text.
            </p>
          </div>

          <div className="p-4 space-y-4">
            {!apiResults && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 text-slate-600">
                No model output available.
              </div>
            )}

            {apiResults && results.length === 0 && (
              <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4 text-yellow-800">
                The API returned no clause predictions.
              </div>
            )}

            {apiResults && results.length > 0 && filteredResults.length === 0 && (
              <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4 text-yellow-800">
                No clauses are visible at the current confidence threshold.
              </div>
            )}

            {groupedClauses.map(([label, clauses]) => (
              <div
                key={label}
                className="rounded-xl border border-slate-200 bg-white shadow-sm"
              >
                <div className="border-b border-slate-100 px-4 py-3">
                  <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-700">
                    {label} ({clauses.length})
                  </h3>
                </div>

                <div className="divide-y divide-slate-100">
                  {clauses.map((item) => {
                    const clauseText = item.sentence || item.text || '';
                    const clauseKey = `${item.start_char}-${item.end_char}-${item.label}`;

                    return (
                      <button
                        key={clauseKey}
                        onClick={() => scrollToClause(clauseKey)}
                        className="w-full text-left px-4 py-3 hover:bg-blue-50 transition-colors"
                      >
                        <div className="flex items-start justify-between gap-3 mb-2">
                          <span className="text-xs font-medium text-blue-700">
                            {formatConfidence(item.confidence)}
                          </span>
                          <span className="text-xs text-slate-500">
                            {item.start_char}–{item.end_char}
                          </span>
                        </div>

                        <p className="text-sm leading-6 text-slate-800">
                          {clauseText}
                        </p>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}