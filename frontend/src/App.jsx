import React, { useState, useRef } from 'react';
import DocumentPage from './DocumentPage';
import uploadDocPng from './uploadDoc.png';

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [fileURL, setFileURL] = useState(null);
  const [view, setView] = useState('home'); // 'home' | 'viewer' 

  const fileInputRef = useRef(null);

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
      formData.append("file", file);

      const response = await fetch("http://127.0.0.1:8000/upload-contract", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await response.json();
      setApiResults(data);
      setView("viewer");
    } catch (err) {
      setError(err.message || "Something went wrong");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    setView('home');
    setUploadedFile(null);
    if (fileURL) URL.revokeObjectURL(fileURL);
    setFileURL(null);
  };

  if (view === 'viewer') {
    return (
      <DocumentPage
        fileName={uploadedFile}
        fileURL={fileURL}
        onBack={handleBack}
      />
    );
  }


  return (
    <div className="min-h-screen bg-white">
      {/* hidden file input used by hero button */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.pdf,.docx"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Hero Section */}
      <section className="relative min-h-[70vh] flex items-center overflow-hidden
        bg-gradient-to-br from-blue-200 via-blue-300 to-indigo-200">

        {/* Subtle grain / noise */}
        <div className="pointer-events-none absolute inset-0
          bg-[radial-gradient(circle_at_1px_1px,rgba(0,0,0,0.15)_1px,transparent_0)]
          bg-[size:16px_16px] opacity-10" />

        {/* Soft abstract shape */}
        <div className="pointer-events-none absolute -top-40 -left-40
          w-[520px] h-[520px] rounded-full
          bg-white/30 blur-3xl" />

        <div className="relative max-w-5xl mx-auto flex items-center justify-between px-8 w-full">
          <h1 className="text-[5.5rem] md:text-[9rem] font-extrabold leading-none text-slate-900">
            Contract AI
          </h1>

          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-0 bg-transparent border-none shadow-none hover:scale-[1.02] transition-transform"
            style={{ width: 'auto', height: 'auto' }}
          >
            <img
              src={uploadDocPng}
              alt="Upload Document"
              style={{ width: 300, height: 300, objectFit: 'contain', display: 'block' }}
            />
          </button>

        </div>
      </section>


      {/* Description Section */}
      <section className="flex flex-col items-center pt-12 pb-20 px-0 gap-10">
        <p className="max-w-3xl text-center text-xl text-slate-800 leading-relaxed">
          Have a lease agreement, employment contract, or any legal document you want to understand better? Upload it and let our AI assistant break down the key points, obligations, and risks in plain language. No more legalese confusion, just clear insights at your fingertips.
        </p>
        <div className="w-40 h-1 rounded-full bg-gradient-to-r from-blue-400 via-indigo-400 to-blue-300" />
      </section>
      
    </div>
  );
}