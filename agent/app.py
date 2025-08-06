from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain_experimental.tools import PythonREPLTool  # Correct import
from langchain.agents.agent_types import AgentType
import subprocess
import os

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
    Tool(name="Shell", func=shell_tool, description="Run safe shell commands.")
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
