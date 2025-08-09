import { useState, useRef } from 'react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  const API_BASE = "http://196.115.125.145:8000";

  const askAgent = async () => {
    setLoading(true);
    setStreaming(false);
    setError('');
    setResponse('');
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Unknown error");
      }
      const data = await res.json();
      setResponse(data.response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const askAgentStream = async () => {
    setLoading(false);
    setStreaming(true);
    setError('');
    setResponse('');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API_BASE}/ask_stream`, {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
        signal: controller.signal
      });

      if (!res.ok || !res.body) {
        const errTxt = await res.text().catch(() => "");
        throw new Error(errTxt || "Stream failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          setResponse(prev => prev + chunk);
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setError('Stream aborted by user.');
      } else {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  };

  const stopStream = () => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  };

  const busy = loading || streaming;

  return (
    <div style={{ padding: 20, maxWidth: 800, margin: "0 auto", fontFamily: "ui-sans-serif, system-ui" }}>
      <h2>🧠 Local AI Agent</h2>

      <textarea
        rows={5}
        style={{ width: "100%", padding: 8 }}
        value={prompt}
        placeholder="Ask me anything…"
        onChange={e => setPrompt(e.target.value)}
      />

      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button onClick={askAgent} disabled={busy || !prompt.trim()}>
          {loading ? "Loading…" : "Ask (agent)"}
        </button>

        {!streaming ? (
          <button onClick={askAgentStream} disabled={busy || !prompt.trim()}>
            Ask (stream)
          </button>
        ) : (
          <button onClick={stopStream} style={{ background: '#eee' }}>
            ⏹ Stop
          </button>
        )}
      </div>

      {(loading || streaming) && <div style={{ marginTop: 8 }}>⏳ {streaming ? 'Streaming…' : 'Waiting for response…'}</div>}
      {error && <div style={{ color: 'red', marginTop: 8 }}>Error: {error}</div>}

      <div style={{
        marginTop: 16, whiteSpace: 'pre-wrap', padding: 12,
        border: '1px solid #ddd', borderRadius: 8, minHeight: 120
      }}>
        <strong>Response:</strong>
        <div style={{ marginTop: 8 }}>{response}</div>
      </div>
    </div>
  );
}

export default App;
