from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain_experimental.tools import PythonREPLTool  # Correct import
from langchain.agents.agent_types import AgentType
import subprocess, os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = Ollama(model="mistral", base_url=ollama_base_url)

# --------------------- File-safe tool ---------------------
SAFE_DIR = "./files"
os.makedirs(SAFE_DIR, exist_ok=True)

def file_tool(action: str, filename: str = "", content: str = "") -> str:
    path = os.path.abspath(os.path.join(SAFE_DIR, filename))
    if not path.startswith(os.path.abspath(SAFE_DIR)):
        return "Access denied: Unsafe file path."
    try:
        if action == "list":
            return "\n".join(os.listdir(SAFE_DIR))
        elif action == "read":
            with open(path, "r") as f:
                return f.read()
        elif action == "write":
            with open(path, "w") as f:
                f.write(content)
            return f"Written to {filename}"
        else:
            return "Invalid action. Use 'list', 'read', or 'write'."
    except Exception as e:
        return f"File error: {str(e)}"

def file_exploit_tool(input: str) -> str:
    """
    Usage:
    - To list files: 'list'
    - To read a file: 'read filename.txt'
    - To write a file: 'write filename.txt content'
    """
    parts = input.strip().split(" ", 2)
    if not parts:
        return "No input provided."
    action = parts[0]
    filename = parts[1] if len(parts) > 1 else ""
    content = parts[2] if len(parts) > 2 else ""
    return file_tool(action, filename, content)

def shell_tool(cmd: str) -> str:
    allowed = ["ls", "pwd", "whoami"]
    # Only allow exact matches for safety
    if cmd.strip().split()[0] in allowed:
        try:
            return subprocess.getoutput(cmd)
        except Exception as e:
            return f"Shell error: {str(e)}"
    return "Command not allowed"

tools = [
    Tool(name="Python", func=PythonREPLTool().run, description="Execute Python code."),
    Tool(name="Shell", func=shell_tool, description="Run safe shell commands."),
    Tool(name="FileExploitation", func=file_exploit_tool, description="List, read, or write files in a safe directory. Usage: 'list', 'read filename', 'write filename content'")

]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True)

class Prompt(BaseModel):
    prompt: str

@app.post("/ask")
def ask_user(prompt: Prompt):
    try:
        response = agent.invoke(prompt.prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
