import { useState } from 'react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const askAgent = async () => {
    setLoading(true);
    setError('');
    setResponse('');
    try {
      const res = await fetch("http://196.115.125.145:8000/ask", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Unknown error");
      }
      const data = await res.json();
      setResponse(data.response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const askAgentStream = async () => {
    setLoading(true);
    setError('');
    setResponse('');
    try {
      const res = await fetch("http://196.115.125.145:8000/ask_stream", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      if (!res.ok || !res.body) {
        const errTxt = await res.text();
        throw new Error(errTxt || "Stream failed");
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          setResponse(prev => prev + chunk);
        }
      }
    } catch (err){
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>🧠 Local AI Agent</h2>
      <textarea
        rows="4"
        cols="50"
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
      />
      <br />
      <button onClick={askAgent} disabled={loading || !prompt.trim()}>
        {loading ? "Loading..." : "Ask"}
      </button>
      <button onClick={askAgentStream} disabled={loading || !prompt.trim()} style={{ marginLeft: 8 }}>
        {loading ? "Streaming..." : "Ask (stream)"}
      </button>
      {loading && <div>⏳ Waiting for response...</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      <p><strong>Response:</strong> {typeof response === "string" ? response : JSON.stringify(response)}</p>
    </div>
  );
}

export default App;
