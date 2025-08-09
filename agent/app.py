# --------------------- IMPORTS ---------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain_experimental.tools import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import subprocess, os, re
from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator

# ===================================================
# ================ APP & LLM CONFIG =================
# ===================================================

app = FastAPI()

# CORS (open to all for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = Ollama(model="mistral", base_url=OLLAMA_BASE_URL)

# Secure files directory (mount as volume: ./files:/app/files)
SAFE_DIR = "./files"
os.makedirs(SAFE_DIR, exist_ok=True)

# ===================================================
# ================== GENERIC HELPERS ================
# ===================================================

def _strip_quotes(s: str) -> str:
    """Removes quotes/backticks and extra spaces around an argument."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or \
       (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("`") and s.endswith("`")):
        s = s[1:-1].strip()
    return s.strip(" '\"`")

def _safe_join(filename: str) -> str:
    """Builds a safe path strictly under SAFE_DIR."""
    filename = _strip_quotes(filename)
    path = os.path.abspath(os.path.join(SAFE_DIR, filename))
    safe_root = os.path.abspath(SAFE_DIR)
    if os.path.commonpath([path, safe_root]) != safe_root:
        raise PermissionError("Access denied: Unsafe file path.")
    return path

# ===================================================
# ================== FILE OPS MODULE ================
# ===================================================

def list_files() -> str:
    items = sorted(os.listdir(SAFE_DIR))
    return "\n".join(items) if items else "(empty)"

def read_file(filename: str) -> str:
    path = _safe_join(filename)
    if not os.path.exists(path):
        return f"File not found: {filename}"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(filename: str, content: str = "") -> str:
    path = _safe_join(filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")
    return f"Written to {filename}"

# Low-level API for agent (list/read/write via a single command)
def file_tool(action: str, filename: str = "", content: str = "") -> str:
    try:
        action = (action or "").strip().lower()
        if action == "list":
            return list_files()
        elif action == "read":
            if not filename:
                return "Missing filename."
            return read_file(filename)
        elif action == "write":
            if not filename:
                return "Missing filename."
            return write_file(filename, content)
        else:
            return "Invalid action. Use 'list', 'read <file>', or 'write <file> <content>'."
    except PermissionError as pe:
        return str(pe)
    except Exception as e:
        return str(e)

def file_exploit_tool(user_input: str) -> str:
    """
    Robustly parses a one-line command:
      - list
      - read filename.txt
      - write filename.txt content...
    Tolerates quotes/backticks and multiple spaces.
    """
    if not user_input or not user_input.strip():
        return "No input provided."

    cleaned = _strip_quotes(user_input.strip())
    parts = re.split(r"\s+", cleaned, maxsplit=2)
    if not parts:
        return "No input provided."

    action = (parts[0] or "").lower()
    filename = _strip_quotes(parts[1]) if len(parts) > 1 else ""
    content  = parts[2] if len(parts) > 2 else ""

    return file_tool(action, filename, content)

# ===================================================
# =================== AI TOOLS ======================
# ===================================================

def shell_tool(cmd: str) -> str:
    """Restricted shell for allowed commands only."""
    allowed = {"ls", "pwd", "whoami"}
    token = (cmd or "").strip().split()[0] if cmd else ""
    if token in allowed:
        try:
            return subprocess.getoutput(cmd)
        except Exception as e:
            return f"Shell error: {str(e)}"
    return "Command not allowed"

def summarize_file_tool(filename: str) -> str:
    """Reads a file and asks the LLM for a concise summary (bullets + TL;DR), with chunking."""
    try:
        filename = _strip_quotes(filename)
        path = _safe_join(filename)
        if not os.path.exists(path):
            return f"File not found: {filename}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return f"File is empty: {filename}"

        # If content is short enough, summarize directly without chunking
        if len(content) < 3000:
            direct_prompt = f"""Please provide a concise summary of this text in bullet points (3-5 points) followed by a one-sentence TL;DR.

Text to summarize:
{content}

Summary:
-"""
            try:
                result = llm.invoke(direct_prompt)
                return result.strip()
            except Exception as e:
                return f"Direct summary error: {str(e)}"

        # For longer content, use map-reduce approach
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = splitter.split_text(content)
        docs = [Document(page_content=c) for c in chunks]

        # Improved prompts with clearer instructions
        map_prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant. Please summarize the following text passage in 3-5 bullet points. Do not copy-paste, but provide a concise summary of the key points.

Text passage:
{text}

Summary:
-"""
        )
        
        combine_prompt = PromptTemplate.from_template(
            """You are a helpful AI assistant. Please combine these individual summaries into a final comprehensive summary with 5-7 clear bullet points (avoid repetition), followed by a one-sentence TL;DR.

Individual summaries:
{text}

Final Summary:
-"""
        )

        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            return_intermediate_steps=False,
            verbose=False,
        )
        result = chain.run(docs)
        return result.strip()

    except PermissionError as pe:
        return str(pe)
    except Exception as e:
        return f"Summary error: {str(e)}"

def question_on_file_tool(input_str: str) -> str:
    """
    Ask a question about a file.
    Expected format: 'filename.txt | my question'
    (tolerates spaces and quotes around)
    """
    try:
        if not input_str or "|" not in input_str:
            return "Invalid format. Use: 'filename.txt | question'"
        left, right = input_str.split("|", 1)
        filename = _strip_quotes(left.strip())
        question = (right or "").strip()

        content = read_file(filename)
        if content.startswith("File not found:"):
            return content
        if not content.strip():
            return f"File is empty: {filename}"

        prompt = (
            f"Here is the content of a file:\n{content}\n\n"
            f"Based only on this text, answer clearly the question: {question}"
        )
        return llm.invoke(prompt)
    except Exception as e:
        return f"Question error: {str(e)}"

# ===================================================
# ================ TOOLS DECLARATION ================
# ===================================================

tools = [
    Tool(name="Python", func=PythonREPLTool().run,
         description="Executes Python code."),
    Tool(name="Shell", func=shell_tool,
         description="Executes secure shell commands (ls/pwd/whoami)."),
    Tool(name="FileExploitation", func=file_exploit_tool,
         description="List/read/write in the secure directory. Syntax: 'list' | 'read <file>' | 'write <file> <content>'"),
    Tool(name="SummarizeFile", func=summarize_file_tool,
         description="Simple file summarizer (give only the file name)."),
    Tool(name="QuestionOnFile", func=question_on_file_tool,
         description="Ask a question about a file. Format: 'filename.txt | my question'"),
]

# ===================================================
# ================== AGENT & ENDPOINT ===============
# ===================================================

agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, handle_parsing_errors=True,
    max_iterations=10, max_execution_time=30,
)

class Prompt(BaseModel):
    prompt: str

def _normalize_agent_result(res) -> str:
    """Always returns a readable string from AgentExecutor."""
    if isinstance(res, dict):
        # most common case: {'input': ..., 'output': ...}
        if "output" in res and isinstance(res["output"], str):
            return res["output"]
        if "text" in res and isinstance(res["text"], str):
            return res["text"]
        return str(res)
    return str(res)


@app.post("/ask")
def ask_user(prompt: Prompt):
    try:
        res = agent.invoke(prompt.prompt)
        return {"response": _normalize_agent_result(res)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

# ===================================================
# ================== STREAMING RESPONSE =============
# ===================================================

@app.post("/ask_stream")
async def ask_stream(prompt: Prompt):
    """
    Stream the response from the AI agent using Server-Sent Events (SSE).
    """
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # For streaming with Ollama, we need to create a streaming LLM instance
            streaming_llm = Ollama(
                model="mistral", 
                base_url=OLLAMA_BASE_URL,
                callbacks=[]  # We'll handle streaming manually
            )
            
            # Create a streaming agent
            streaming_agent = initialize_agent(
                tools,
                streaming_llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=30,
            )
            
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'content': 'Processing your request...'})}\n\n"
            
            # Execute the agent
            result = await streaming_agent.ainvoke(prompt.prompt)
            
            # Normalize and send the final result
            final_response = _normalize_agent_result(result)
            
            # Split response into chunks for streaming effect
            words = final_response.split()
            current_chunk = ""
            
            for i, word in enumerate(words):
                current_chunk += word + " "
                
                # Send chunk every 3-5 words or at the end
                if (i + 1) % 4 == 0 or i == len(words) - 1:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': current_chunk.strip()})}\n\n"
                    current_chunk = ""
                    await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'content': final_response})}\n\n"
            
        except Exception as e:
            error_msg = f"Streaming error: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        
        finally:
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )
    """
    Stream the model's tokens as the are generated.
    For now, we stream the *LLM text* (including agent thought tokens if it decides to think out loud).
    """

    # Create a streaming callback
    cb = AsyncIteratorCallbackHandler()


    # Build a fresh agent using the same tools but the streaming LLM
    streaming_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=30,
    )

    async def run_agent():
        try:
            # Use ainvoke to avoid blocking the event loop
            await streaming_agent.ainvoke(prompt.prompt, callbacks=[cb])
        except Exception as e:
            # Surface the error to the stream
            try:
                await cb.on_llm_error(e)
            except TypeError:
                cb.on_llm_error(e)
        finally:
            await cb.done() # Signal end of stream
    
    async def token_generator():

        # Start the agent in background
        task = asyncio.create_task(run_agent())
        # Yield tokens as they arrive
        async for token in cb.aiter():
                # You can wrap as SSE if you prefer; here we just stream plain text
                yield token
        await task

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")