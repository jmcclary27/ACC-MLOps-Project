import React from "react";

export default function DocumentPage({
  fileURL,
  fileName,
  onBack,
}) {

  return (
    <div className="h-screen bg-white flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
        {fileName && <p className="text-sm text-slate-600">{fileName}</p>}
        <button
          onClick={onBack}
          className="px-4 py-2 text-medium font-medium text-white bg-blue-500 hover:bg-blue-600 rounded-md transition-colors"
        >
          ← Home
        </button>
      </header>

      {/* PDF Viewer */}
      <main className="flex-1 bg-white overflow-auto">
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
    </div>
  );
}
