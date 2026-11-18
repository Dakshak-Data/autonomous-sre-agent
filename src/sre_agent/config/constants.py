"""
Centralized Constants: All URLs, model names, default budgets, and prompts live here.
Change a model or token limit in ONE place without hunting across files!
"""

from __future__ import annotations

# --- Embedding Model (Local, Free ONNX) ---
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- LLM Provider Endpoints (OpenAI-Compatible) ---
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"

# --- Default Model Pool: (provider, model_id, base_url, api_key_env, tier) ---
DEFAULT_MODEL_CONFIGS: list[tuple[str, str, str, str, str]] = [
    # Smart Tier (Complex debugging, root-cause, multi-hop fixes): Gemini Primary -> Groq Fallback
    ("google", "gemini-2.5-flash", GOOGLE_OPENAI_BASE_URL, "GEMINI_API_KEY", "smart"),
    ("groq", "qwen/qwen3.8-27b", GROQ_OPENAI_BASE_URL, "GROQ_API_KEY", "smart"),
    ("google", "gemini-2.5-pro", GOOGLE_OPENAI_BASE_URL, "GEMINI_API_KEY", "smart"),
    ("groq", "openai/gpt-oss-120b", GROQ_OPENAI_BASE_URL, "GROQ_API_KEY", "smart"),
    # Fast Tier (Simple status checks, commits, summaries, compaction): Groq Primary -> Gemini Fallback
    ("groq", "qwen/qwen3.8-27b", GROQ_OPENAI_BASE_URL, "GROQ_API_KEY", "fast"),
    ("google", "gemini-2.5-flash-lite", GOOGLE_OPENAI_BASE_URL, "GEMINI_API_KEY", "fast"),
    ("google", "gemini-2.5-flash", GOOGLE_OPENAI_BASE_URL, "GEMINI_API_KEY", "fast"),
]

# --- Router & Circuit Breaker Defaults ---
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_AUTH_COOLDOWN_SECONDS = 300.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0

# --- Context & Token Budgeting Defaults ---
DEFAULT_MAX_CONTEXT_TOKENS = 4000
DEFAULT_MAX_TOOL_OUTPUT_TOKENS = 600
DEFAULT_COMPACTION_THRESHOLD = 0.75
DEFAULT_KEEP_RECENT_MESSAGES = 4
DEFAULT_MAX_AGENT_STEPS = 8

# --- Dynamic Tool Loading Defaults ---
DEFAULT_TOP_K_TOOLS = 3
DEFAULT_MIN_TOOL_SCORE = 0.40

# --- Semantic Cache Defaults (0-Token Short-Circuit for Repeated Queries) ---
DEFAULT_SEMANTIC_CACHE_THRESHOLD = 0.92
DEFAULT_SEMANTIC_CACHE_TTL_SECONDS = 300  # 5 minutes

# --- Destructive Tools Requiring Human-in-the-Loop Approval ---
DESTRUCTIVE_TOOLS: set[str] = {
    "rollback_vercel_deployment",
    "create_github_pull_request",
}

# --- External Cloud API URLs ---
VERCEL_API_BASE_URL = "https://api.vercel.com"
GITHUB_API_BASE_URL = "https://api.github.com"

# --- Default SQLite Filenames ---
DEFAULT_ENGRAM_DB = "sre_engram_memory.db"
DEFAULT_WORKFLOW_DB = "sre_workflow_state.db"

# --- Prompts ---
DEFAULT_SRE_SYSTEM_PROMPT = (
    "You are an Autonomous SRE & Production Incident Investigator. "
    "Investigate incidents methodically using the provided tools (Vercel logs, "
    "Supabase PostgreSQL metrics, GitHub commits). "
    "Always verify root causes with real tool evidence before proposing fixes."
)

COMPACTION_SYSTEM_PROMPT = (
    "Summarize the key SRE investigation facts, errors found, and "
    "tool results below into 3-5 concise bullet points."
)
