import React from 'react';

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
  const results = apiResults?.results || [];

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
            <iframe
              src={fileURL}
              className="w-full h-full"
              title="Uploaded Document"
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
              Clause predictions returned by the API
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

            {results.map((item, index) => (
              <div
                key={`${index}-${item.sentence?.slice(0, 20) || 'clause'}`}
                className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
              >
                <div className="flex items-start justify-between gap-3 mb-3">
                  <span className="inline-flex rounded-full bg-blue-100 px-3 py-1 text-sm font-medium text-blue-700">
                    {item.label}
                  </span>
                  <span className="text-sm font-medium text-slate-500">
                    {formatConfidence(item.confidence)}
                  </span>
                </div>

                <p className="text-sm leading-6 text-slate-800">
                  {item.sentence}
                </p>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}