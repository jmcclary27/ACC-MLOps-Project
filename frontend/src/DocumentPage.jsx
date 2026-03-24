import React, { useMemo, useState } from 'react';
import PDFViewer from './PDFViewer';

function formatConfidence(value) {
  if (typeof value !== 'number') return 'N/A';
  return `${(value * 100).toFixed(1)}%`;
}

export default function DocumentPage({
  fileURL,
  fileName,
  apiResults,
  onBack,
}) {
  const [activeClauseId, setActiveClauseId] = useState(null);
  const results = apiResults?.results || [];
  const extractedText = apiResults?.extracted_text || '';
  const CONFIDENCE_THRESHOLD = 0.80;

  const highConfidenceResults = useMemo(
    () =>
      results
        .map((result, index) => ({
          ...result,
          text: result.text || result.sentence || '',
          clauseId: index,
          sourceIndex: index,
        }))
        .filter((r) => r.confidence >= CONFIDENCE_THRESHOLD),
    [results]
  );

  const groupedByLabel = useMemo(() => {
    const groups = new Map();

    highConfidenceResults.forEach((item) => {
      const existing = groups.get(item.label) || [];
      existing.push(item);
      groups.set(item.label, existing);
    });

    return [...groups.entries()]
      .map(([label, items]) => ({ label, items }))
      .sort((a, b) => b.items.length - a.items.length || a.label.localeCompare(b.label));
  }, [highConfidenceResults]);

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
          {fileURL ? (
            <PDFViewer
              fileURL={fileURL}
              extractedText={extractedText}
              chunks={highConfidenceResults}
              activeClauseId={activeClauseId}
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-slate-500 text-lg">No document uploaded.</p>
            </div>
          )}
        </main>

        <aside className="w-[420px] bg-slate-50 overflow-auto">
          <div className="p-5 border-b border-slate-200 bg-white sticky top-0">
            <h2 className="text-xl font-semibold text-slate-900">Model Results</h2>
            <p className="text-sm text-slate-600 mt-1">
              Grouped by label and sorted by count
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

            {apiResults && groupedByLabel.map((group) => (
              <div key={group.label} className="rounded-xl border border-slate-200 bg-white shadow-sm overflow-hidden">
                <div className="px-4 py-3 border-b border-slate-100 bg-slate-50">
                  <p className="text-sm font-semibold text-slate-900">
                    {group.label} ({group.items.length})
                  </p>
                </div>

                <div className="divide-y divide-slate-100">
                  {group.items.map((item) => (
                    <button
                      key={`nav-${item.clauseId}`}
                      type="button"
                      onClick={() => setActiveClauseId(item.clauseId)}
                      className={`w-full text-left px-4 py-3 transition-colors text-slate-800 ${
                        activeClauseId === item.clauseId ? 'bg-blue-50' : 'hover:bg-slate-50'
                      }`}
                    >
                      <p className="text-xs text-slate-500 mb-1">
                        Clause {item.sourceIndex + 1} • {formatConfidence(item.confidence)}
                      </p>
                      <p className="text-sm leading-5 line-clamp-2">{item.text}</p>
                    </button>
                  ))}
                </div>
              </div>
            ))}


          </div>
        </aside>
      </div>
    </div>
  );
}