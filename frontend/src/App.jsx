import React, { useEffect, useRef, useState } from 'react';
import DocumentPage from './DocumentPage';
import uploadDocPng from './uploadDoc.png';

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileURL, setFileURL] = useState(null);
  const [view, setView] = useState('home');

  const [apiResults, setApiResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (fileURL) {
        URL.revokeObjectURL(fileURL);
      }
    };
  }, [fileURL]);

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFile(file.name);

    const url = URL.createObjectURL(file);
    setFileURL(url);

    setLoading(true);
    setError(null);
    setApiResults(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://127.0.0.1:8000/upload-contract', {
        method: 'POST',
        body: formData,
      });

      let data = null;
      try {
        data = await response.json();
      } catch {
        data = null;
      }

      if (!response.ok) {
        throw new Error(data?.detail || 'Upload failed');
      }

      setApiResults(data);
      setView('viewer');
    } catch (err) {
      setError(err.message || 'Something went wrong');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    setView('home');
    setUploadedFile(null);
    setApiResults(null);
    setError(null);

    if (fileURL) {
      URL.revokeObjectURL(fileURL);
    }
    setFileURL(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  if (view === 'viewer') {
    return (
      <DocumentPage
        fileName={uploadedFile}
        fileURL={fileURL}
        apiResults={apiResults}
        onBack={handleBack}
      />
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.pdf,.docx"
        onChange={handleFileChange}
        className="hidden"
      />

      <section
        className="relative min-h-[70vh] flex items-center overflow-hidden
        bg-gradient-to-br from-blue-200 via-blue-300 to-indigo-200"
      >
        <div
          className="pointer-events-none absolute inset-0
          bg-[radial-gradient(circle_at_1px_1px,rgba(0,0,0,0.15)_1px,transparent_0)]
          bg-[size:16px_16px] opacity-10"
        />

        <div
          className="pointer-events-none absolute -top-40 -left-40
          w-[520px] h-[520px] rounded-full
          bg-white/30 blur-3xl"
        />

        <div className="relative max-w-5xl mx-auto flex items-center justify-between px-8 w-full">
          <h1 className="text-[5.5rem] md:text-[9rem] font-extrabold leading-none text-slate-900">
            Contract AI
          </h1>

          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-0 bg-transparent border-none shadow-none hover:scale-[1.02] transition-transform"
            style={{ width: 'auto', height: 'auto' }}
            disabled={loading}
          >
            <img
              src={uploadDocPng}
              alt="Upload Document"
              style={{ width: 300, height: 300, objectFit: 'contain', display: 'block' }}
            />
          </button>
        </div>
      </section>

      <section className="flex flex-col items-center pt-12 pb-20 px-0 gap-10">
        <p className="max-w-3xl text-center text-xl text-slate-800 leading-relaxed">
          Have a lease agreement, employment contract, or any legal document you want to
          understand better? Upload it and let our AI assistant break down the key points,
          obligations, and risks in plain language. No more legalese confusion, just clear
          insights at your fingertips.
        </p>

        {loading && (
          <p className="text-lg font-medium text-blue-700">
            Uploading contract and running model...
          </p>
        )}

        {error && (
          <div className="max-w-2xl rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-red-700">
            {error}
          </div>
        )}

        <div className="w-40 h-1 rounded-full bg-gradient-to-r from-blue-400 via-indigo-400 to-blue-300" />
      </section>
    </div>
  );
}