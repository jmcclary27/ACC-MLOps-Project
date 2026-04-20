import React, { useEffect, useMemo, useRef, useState } from 'react';

function buildHighlightedSegments(text, citations, activeCitationIndex) {
  if (!text) return [{ key: 'empty', text: '', highlighted: false, active: false }];

  const validCitations = [...citations]
    .map((citation, index) => ({ ...citation, __index: index }))
    .filter(
      (citation) =>
        typeof citation.start_char === 'number' &&
        typeof citation.end_char === 'number' &&
        citation.start_char >= 0 &&
        citation.end_char > citation.start_char &&
        citation.end_char <= text.length
    )
    .sort((a, b) => a.start_char - b.start_char);

  if (validCitations.length === 0) {
    return [{ key: 'plain-full', text, highlighted: false, active: false }];
  }

  const segments = [];
  let cursor = 0;

  validCitations.forEach((citation, index) => {
    if (citation.start_char > cursor) {
      segments.push({
        key: `plain-${cursor}-${citation.start_char}`,
        text: text.slice(cursor, citation.start_char),
        highlighted: false,
        active: false,
        citation: null,
      });
    }

    segments.push({
      key: `highlight-${index}-${citation.start_char}-${citation.end_char}`,
      text: text.slice(citation.start_char, citation.end_char),
      highlighted: true,
      active: citation.__index === activeCitationIndex,
      citation,
    });

    cursor = citation.end_char;
  });

  if (cursor < text.length) {
    segments.push({
      key: `plain-${cursor}-${text.length}`,
      text: text.slice(cursor),
      highlighted: false,
      active: false,
      citation: null,
    });
  }

  return segments;
}

function ChatBubble({ message }) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={
          isUser
            ? 'max-w-[85%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-3 text-white shadow-sm'
            : 'max-w-[85%] rounded-2xl rounded-bl-md bg-white px-4 py-3 text-slate-800 shadow-sm border border-slate-200'
        }
      >
        <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
      </div>
    </div>
  );
}

function CitationNavigator({
  citations,
  activeCitationIndex,
  onPrevious,
  onNext,
  onJumpToCitation,
}) {
  if (!citations.length) return null;

  const activeCitation = citations[activeCitationIndex] || citations[0];

  return (
    <div className="sticky top-[88px] z-20 mx-6 mt-4 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 shadow-sm">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-amber-900">
            Reference {Math.min(activeCitationIndex + 1, citations.length)} of {citations.length}
          </p>
          <p className="mt-1 text-sm text-amber-800 truncate">
            {activeCitation?.text || ''}
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={onPrevious}
            className="rounded-md border border-amber-300 bg-white px-3 py-2 text-sm font-medium text-amber-900 hover:bg-amber-100 transition-colors"
          >
            Previous
          </button>
          <button
            onClick={() => onJumpToCitation(activeCitationIndex)}
            className="rounded-md border border-amber-300 bg-white px-3 py-2 text-sm font-medium text-amber-900 hover:bg-amber-100 transition-colors"
          >
            Jump to active
          </button>
          <button
            onClick={onNext}
            className="rounded-md border border-amber-300 bg-white px-3 py-2 text-sm font-medium text-amber-900 hover:bg-amber-100 transition-colors"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

export default function DocumentPage({ fileName, documentData, onBack }) {
  const [messages, setMessages] = useState(() => {
    if (!documentData?.summary) return [];

    return [
      {
        id: 'summary',
        role: 'assistant',
        content: documentData.summary,
      },
      {
        id: 'prompt',
        role: 'assistant',
        content: 'Is there anything else you would like to know?',
      },
    ];
  });

  const [question, setQuestion] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);

  const [activeCitations, setActiveCitations] = useState([]);
  const [activeCitationIndex, setActiveCitationIndex] = useState(0);

  const chatContainerRef = useRef(null);
  const highlightRefs = useRef({});

  const documentText = documentData?.extracted_text || '';
  const documentId = documentData?.document_id || '';

  useEffect(() => {
    setMessages(
      documentData?.summary
        ? [
            {
              id: 'summary',
              role: 'assistant',
              content: documentData.summary,
            },
            {
              id: 'prompt',
              role: 'assistant',
              content: 'Is there anything else you would like to know?',
            },
          ]
        : []
    );
    setActiveCitations([]);
    setActiveCitationIndex(0);
    setQuestion('');
    setChatError(null);
  }, [documentData]);

  useEffect(() => {
    if (!chatContainerRef.current) return;
    chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [messages, chatLoading]);

  useEffect(() => {
    if (!activeCitations.length) return;

    const activeCitation = activeCitations[activeCitationIndex];
    if (!activeCitation) return;

    const key = `${activeCitation.start_char}-${activeCitation.end_char}-${activeCitation.chunk_id}`;
    const element = highlightRefs.current[key];

    if (element) {
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [activeCitationIndex, activeCitations]);

  const highlightedSegments = useMemo(() => {
    return buildHighlightedSegments(documentText, activeCitations, activeCitationIndex);
  }, [documentText, activeCitations, activeCitationIndex]);

  const handleAskQuestion = async (e) => {
    e.preventDefault();

    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !documentId || chatLoading) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmedQuestion,
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setChatLoading(true);
    setChatError(null);

    try {
      const response = await fetch('/qa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          document_id: documentId,
          question: trimmedQuestion,
          top_k: 5,
        }),
      });

      let data = null;
      try {
        data = await response.json();
      } catch {
        data = null;
      }

      if (!response.ok) {
        throw new Error(data?.error || data?.detail || 'Question failed');
      }

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: data?.answer || 'No answer returned.',
      };

      setMessages((prev) => [...prev, assistantMessage]);

      const citations = Array.isArray(data?.citations) ? data.citations : [];
      setActiveCitations(citations);
      setActiveCitationIndex(0);
    } catch (err) {
      setChatError(err.message || 'Something went wrong');
      console.error(err);
    } finally {
      setChatLoading(false);
    }
  };

  const jumpToCitation = (index) => {
    if (!activeCitations.length) return;

    const safeIndex = ((index % activeCitations.length) + activeCitations.length) % activeCitations.length;
    setActiveCitationIndex(safeIndex);

    const citation = activeCitations[safeIndex];
    if (!citation) return;

    const key = `${citation.start_char}-${citation.end_char}-${citation.chunk_id}`;
    const element = highlightRefs.current[key];

    if (element) {
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  };

  const goToPreviousCitation = () => {
    if (!activeCitations.length) return;
    jumpToCitation(activeCitationIndex - 1);
  };

  const goToNextCitation = () => {
    if (!activeCitations.length) return;
    jumpToCitation(activeCitationIndex + 1);
  };

  return (
    <div className="h-screen bg-white flex flex-col">
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-200">
        <div>
          {fileName && <p className="text-sm text-slate-600">{fileName}</p>}
          {documentData?.chunk_count != null && (
            <p className="text-sm text-slate-500">
              {documentData.chunk_count} chunk{documentData.chunk_count === 1 ? '' : 's'}
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
        <main className="flex-1 bg-white overflow-auto border-r border-slate-200 relative">
          <div className="sticky top-0 z-10 border-b border-slate-200 bg-white/95 backdrop-blur px-6 py-4">
            <div>
              <h2 className="text-xl font-semibold text-slate-900">Document Viewer</h2>
              <p className="text-sm text-slate-600">
                Ask questions in the sidebar and jump through highlighted references.
              </p>
            </div>
          </div>

          <CitationNavigator
            citations={activeCitations}
            activeCitationIndex={activeCitationIndex}
            onPrevious={goToPreviousCitation}
            onNext={goToNextCitation}
            onJumpToCitation={jumpToCitation}
          />

          {!documentData && (
            <div className="flex items-center justify-center h-full px-8">
              <p className="text-slate-500 text-lg">No document data available.</p>
            </div>
          )}

          {documentData && !documentText && (
            <div className="flex items-center justify-center h-full px-8">
              <p className="text-slate-500 text-lg">
                The API did not return extracted document text.
              </p>
            </div>
          )}

          {documentData && documentText && (
            <div className="p-6">
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-6 shadow-sm">
                <div className="whitespace-pre-wrap break-words text-[15px] leading-7 text-slate-800">
                  {highlightedSegments.map((segment) => {
                    if (!segment.highlighted) {
                      return <span key={segment.key}>{segment.text}</span>;
                    }

                    const citationKey = `${segment.citation.start_char}-${segment.citation.end_char}-${segment.citation.chunk_id}`;

                    return (
                      <mark
                        key={segment.key}
                        ref={(node) => {
                          if (node) {
                            highlightRefs.current[citationKey] = node;
                          }
                        }}
                        className={
                          segment.active
                            ? 'rounded px-1 py-0.5 bg-yellow-300 text-slate-900 citation-active'
                            : 'rounded px-1 py-0.5 bg-orange-200 text-slate-900 citation-match'
                        }
                        title={`${segment.citation.start_char}–${segment.citation.end_char}`}
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

        <aside className="w-[430px] bg-slate-50 flex flex-col min-h-0">
          <div className="p-5 border-b border-slate-200 bg-white sticky top-0 z-10">
            <h2 className="text-xl font-semibold text-slate-900">Contract Assistant</h2>
            <p className="text-sm text-slate-600 mt-1">
              Ask questions about this document and review highlighted references.
            </p>
          </div>

          <div
            ref={chatContainerRef}
            className="flex-1 min-h-0 overflow-auto p-4 space-y-4 soft-scroll"
          >
            {!messages.length && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 text-slate-600">
                No conversation available yet.
              </div>
            )}

            {messages.map((message) => (
              <ChatBubble key={message.id} message={message} />
            ))}

            {chatLoading && (
              <div className="flex justify-start">
                <div className="max-w-[85%] rounded-2xl rounded-bl-md bg-white px-4 py-3 text-slate-800 shadow-sm border border-slate-200">
                  <p className="text-sm leading-6">Thinking...</p>
                </div>
              </div>
            )}

            {chatError && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
                <p className="font-semibold text-sm">Request failed</p>
                <p className="mt-1 text-sm">{chatError}</p>
              </div>
            )}
          </div>

          <div className="border-t border-slate-200 bg-white p-4">
            <form onSubmit={handleAskQuestion} className="space-y-3">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question about this contract..."
                className="w-full min-h-[96px] resize-none rounded-xl border border-slate-300 px-4 py-3 text-sm text-slate-800 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs text-slate-500">
                  Answers will highlight the cited sections in the document.
                </p>
                <button
                  type="submit"
                  disabled={chatLoading || !question.trim()}
                  className="rounded-xl bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Send
                </button>
              </div>
            </form>
          </div>
        </aside>
      </div>
    </div>
  );
}