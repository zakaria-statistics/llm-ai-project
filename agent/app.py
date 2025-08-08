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

# ===================================================
# ================ APP & LLM CONFIG =================
# ===================================================

app = FastAPI()

# CORS (ouvre à tout pour dev; restreindre en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = Ollama(model="mistral", base_url=OLLAMA_BASE_URL)

# Dossier fichiers sécurisé (montre-le en volume: ./files:/app/files)
SAFE_DIR = "./files"
os.makedirs(SAFE_DIR, exist_ok=True)

# ===================================================
# ================== HELPERS GÉNÉRIQUES =============
# ===================================================

def _strip_quotes(s: str) -> str:
    """Nettoie quotes/backticks et espaces superflus autour d'un arg."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or \
       (s.startswith('"') and s.endswith('"')) or \
       (s.startswith("`") and s.endswith("`")):
        s = s[1:-1].strip()
    return s.strip(" '\"`")

def _safe_join(filename: str) -> str:
    """Construit un chemin sûr, strictement sous SAFE_DIR."""
    filename = _strip_quotes(filename)  # <- nettoyage de base
    path = os.path.abspath(os.path.join(SAFE_DIR, filename))
    safe_root = os.path.abspath(SAFE_DIR)
    if os.path.commonpath([path, safe_root]) != safe_root:
        raise PermissionError("Access denied: Unsafe file path.")
    return path

# ===================================================
# ================== MODULE FILE OPS =================
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

# API “bas niveau” pour l’agent (list/read/write via une seule commande)
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
    Parse robuste d'une commande en une ligne :
      - list
      - read filename.txt
      - write filename.txt contenu...
    Tolère quotes/backticks et espaces multiples.
    """
    if not user_input or not user_input.strip():
        return "No input provided."

    cleaned = _strip_quotes(user_input.strip())
    parts = re.split(r"\s+", cleaned, maxsplit=2)  # split sur espaces multiples
    if not parts:
        return "No input provided."

    action = (parts[0] or "").lower()
    filename = _strip_quotes(parts[1]) if len(parts) > 1 else ""
    content  = parts[2] if len(parts) > 2 else ""

    return file_tool(action, filename, content)

# ===================================================
# =================== OUTILS “IA” ====================
# ===================================================

def shell_tool(cmd: str) -> str:
    """Shell restreint aux commandes autorisées."""
    allowed = {"ls", "pwd", "whoami"}
    token = (cmd or "").strip().split()[0] if cmd else ""
    if token in allowed:
        try:
            return subprocess.getoutput(cmd)
        except Exception as e:
            return f"Shell error: {str(e)}"
    return "Command not allowed"

def summarize_file_tool(filename: str) -> str:
    """Lis un fichier et demande au LLM un résumé concis (puces + TL;DR), avec découpage."""
    try:
        filename = _strip_quotes(filename)
        path = _safe_join(filename)
        if not os.path.exists(path):
            return f"File not found: {filename}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return f"File is empty: {filename}"

        # Découpage pour éviter l’écho du texte complet
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = splitter.split_text(content)
        docs = [Document(page_content=c) for c in chunks]

        map_prompt = PromptTemplate.from_template(
            "Tu es concis. Résume ce passage en 3–5 puces sans copier-coller :\n\n{text}\n\n-"
        )
        combine_prompt = PromptTemplate.from_template(
            "Combine ces résumés en 5–7 puces claires (sans redite), puis ajoute un TL;DR d'une phrase.\n\n{text}\n\nRésumé final :"
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
    Poser une question sur un fichier.
    Format attendu : 'fichier.txt | ma question'
    (tolère espaces et quotes/autour)
    """
    try:
        if not input_str or "|" not in input_str:
            return "Format invalide. Utilise: 'fichier.txt | question'"
        left, right = input_str.split("|", 1)
        filename = _strip_quotes(left.strip())
        question = (right or "").strip()

        content = read_file(filename)
        if content.startswith("File not found:"):
            return content
        if not content.strip():
            return f"File is empty: {filename}"

        prompt = (
            f"Voici le contenu d'un fichier :\n{content}\n\n"
            f"En te basant uniquement sur ce texte, réponds clairement à la question : {question}"
        )
        return llm.invoke(prompt)
    except Exception as e:
        return f"Question error: {str(e)}"

# ===================================================
# ================ DÉCLARATION DES TOOLS ============
# ===================================================

tools = [
    Tool(name="Python", func=PythonREPLTool().run,
         description="Exécute du code Python."),
    Tool(name="Shell", func=shell_tool,
         description="Exécute des commandes shell sécurisées (ls/pwd/whoami)."),
    Tool(name="FileExploitation", func=file_exploit_tool,
         description="Lister/lire/écrire dans le répertoire sécurisé. Syntaxe: 'list' | 'read <fichier>' | 'write <fichier> <contenu>'"),
    Tool(name="SummarizeFile", func=summarize_file_tool,
         description="Résume le contenu d’un fichier (donne uniquement le nom du fichier)."),
    Tool(name="QuestionOnFile", func=question_on_file_tool,
         description="Question sur un fichier. Format: 'fichier.txt | ma question'"),
]

# ===================================================
# ================== AGENT & ENDPOINT ===============
# ===================================================

agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, handle_parsing_errors=True,
    max_iterations=10, max_execution_time=30
)

class Prompt(BaseModel):
    prompt: str

def _normalize_agent_result(res) -> str:
    """Retourne toujours une string lisible depuis AgentExecutor."""
    if isinstance(res, dict):
        # cas le plus courant : {'input': ..., 'output': ...}
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
