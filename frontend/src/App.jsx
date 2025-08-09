import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, FileText, Loader, AlertCircle, Trash2, Copy, Check } from 'lucide-react';

function App() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const copyToClipboard = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(messageId);
      setTimeout(() => setCopied(null), 2000);
    } catch (err) {
      console.error('Failed to copy: ', err);
    }
  };

  const askAgent = async (useStreaming = false) => {
    if (!prompt.trim()) return;

    const userMessage = { 
      id: Date.now(), 
      type: 'user', 
      content: prompt, 
      timestamp: new Date().toLocaleTimeString() 
    };
    
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setError('');
    
    const currentPrompt = prompt;
    setPrompt('');

    if (useStreaming) {
      // Streaming response handling
      try {
        const botMessageId = Date.now() + 1;
        const initialBotMessage = {
          id: botMessageId,
          type: 'bot',
          content: '',
          timestamp: new Date().toLocaleTimeString(),
          streaming: true
        };
        
        setMessages(prev => [...prev, initialBotMessage]);

        const response = await fetch("http://196.115.125.145:8000/ask_stream", {
          method: "POST",
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: currentPrompt })
        });

        if (!response.ok) {
          throw new Error("Streaming request failed");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let streamedContent = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          streamedContent += chunk;

          // Update the message content in real-time
          setMessages(prev => prev.map(msg => 
            msg.id === botMessageId 
              ? { ...msg, content: streamedContent }
              : msg
          ));
        }

        // Mark streaming as complete
        setMessages(prev => prev.map(msg => 
          msg.id === botMessageId 
            ? { ...msg, streaming: false }
            : msg
        ));

      } catch (err) {
        setError(err.message);
        const errorMessage = {
          id: Date.now() + 1,
          type: 'error',
          content: err.message,
          timestamp: new Date().toLocaleTimeString()
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setLoading(false);
      }
    } else {
      // Regular response handling
      try {
        const res = await fetch("http://196.115.125.145:8000/ask", {
          method: "POST",
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: currentPrompt })
        });
        
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Unknown error");
        }
        
        const data = await res.json();
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: data.response,
          timestamp: new Date().toLocaleTimeString()
        };
        
        setMessages(prev => [...prev, botMessage]);
      } catch (err) {
        setError(err.message);
        const errorMessage = {
          id: Date.now() + 1,
          type: 'error',
          content: err.message,
          timestamp: new Date().toLocaleTimeString()
        };
        setMessages(prev => [...prev, errorMessage]);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      askAgent();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError('');
  };

  const examplePrompts = [
    "List all files in the directory",
    "Summarize the content of a text file",
    "Write a Python script to calculate fibonacci numbers",
    "Show current directory with shell command"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-md border-b border-white/10 sticky top-0 z-10">
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                AI Agent Assistant
              </h1>
              <p className="text-sm text-gray-400">Powered by Mistral & LangChain</p>
            </div>
          </div>
          <button
            onClick={clearChat}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            title="Clear chat"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-6 flex flex-col h-[calc(100vh-80px)]">
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 mb-6 scrollbar-thin scrollbar-thumb-white/20">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mx-auto mb-4 flex items-center justify-center">
                <Bot className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-semibold mb-2">Welcome to AI Agent Assistant</h3>
              <p className="text-gray-400 mb-6">Ask me to help with files, code, or shell commands</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
                {examplePrompts.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setPrompt(example)}
                    className="p-3 bg-white/5 hover:bg-white/10 rounded-lg border border-white/10 text-left transition-colors"
                  >
                    <FileText className="w-4 h-4 inline mr-2" />
                    {example}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.type === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.type !== 'user' && (
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.type === 'error' ? 'bg-red-500/20' : 'bg-gradient-to-br from-blue-500 to-purple-600'
                }`}>
                  {message.type === 'error' ? <AlertCircle className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                </div>
              )}
              
              <div
                className={`max-w-3xl rounded-2xl px-4 py-3 ${
                  message.type === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white ml-12'
                    : message.type === 'error'
                    ? 'bg-red-500/10 border border-red-500/20 text-red-300'
                    : 'bg-white/5 backdrop-blur-sm border border-white/10'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed">
                      {message.content}
                    </pre>
                  </div>
                  <div className="flex items-center gap-2 ml-2">
                    <span className="text-xs opacity-50">{message.timestamp}</span>
                    <button
                      onClick={() => copyToClipboard(message.content, message.id)}
                      className="p-1 hover:bg-white/10 rounded opacity-50 hover:opacity-100 transition-opacity"
                      title="Copy message"
                    >
                      {copied === message.id ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                    </button>
                  </div>
                </div>
              </div>

              {message.type === 'user' && (
                <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-2xl px-4 py-3">
                <div className="flex items-center gap-2">
                  <Loader className="w-4 h-4 animate-spin" />
                  <span className="text-sm text-gray-300">AI is thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 p-4">
          <div className="flex gap-3">
            <div className="flex-1">
              <textarea
                ref={textareaRef}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask me to help with files, code, or shell commands... (Ctrl/Cmd + Enter to send)"
                className="w-full bg-transparent resize-none outline-none text-white placeholder-gray-400 min-h-[60px] max-h-40"
                rows="1"
                style={{
                  height: 'auto',
                  minHeight: '60px',
                  height: Math.min(Math.max(60, prompt.split('\n').length * 24), 160) + 'px'
                }}
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => askAgent(false)}
                disabled={loading || !prompt.trim()}
                className="flex-1 p-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed rounded-xl transition-colors flex items-center justify-center gap-2"
              >
                {loading ? <Loader className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
                <span className="hidden sm:inline">Send</span>
              </button>
              <button
                onClick={() => askAgent(true)}
                disabled={loading || !prompt.trim()}
                className="p-3 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed rounded-xl transition-colors flex items-center justify-center"
                title="Stream response"
              >
                <Bot className="w-5 h-5" />
              </button>
            </div>
          </div>
          
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-white/10">
            <div className="text-xs text-gray-400">
              <kbd className="bg-white/10 px-1 rounded">Ctrl</kbd> + <kbd className="bg-white/10 px-1 rounded">Enter</kbd> to send • 
              <span className="ml-2">Blue = Normal • Green = Stream</span>
            </div>
            <div className="text-xs text-gray-400">
              {prompt.length > 0 && `${prompt.length} characters`}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;