import os
from pathlib import Path

# Configuration settings for the RAG chatbot

# ============================================================
# 模型後端切換
#   "openai" = OpenAI API（GPT-4o + text-embedding-3-small），需要 API key
#   "ollama" = 本地 GPU（需先安裝 Ollama 並 pull 模型），不需要 API key
# 可用環境變數覆寫：  $env:LLM_PROVIDER="ollama"
# ============================================================
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

# --- OpenAI Configuration ---
OPENAI_CHAT_MODEL_NAME = "gpt-4o"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# --- Ollama Configuration (本地 GPU) ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# 聊天模型。GTX 1050 Ti (4GB，實際可用約 3GB) 建議 qwen2.5:3b。
# 其他可選：gemma3:1b（更小更快）、gemma3:4b（會 offload 到 CPU，較慢）
OLLAMA_CHAT_MODEL_NAME = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
# 嵌入模型。bge-m3 為多語（含繁中）表現最好的選擇，約 1.2GB。
OLLAMA_EMBEDDING_MODEL_NAME = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
# 上下文窗口。愈大 KV cache 愈吃 VRAM；3GB 可用時 8192 是安全上限。
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = 1500   # 對應 OpenAI 路徑的 max_tokens
OLLAMA_TEMPERATURE = 0.1    # 與 OpenAI 路徑一致

# ============================================================
# 路徑：全部以專案根目錄為錨點（src/ 的上一層），
# 這樣不論從哪個工作目錄啟動都指向同一份資料。
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"       # 執行時產生的資料
CACHE_DIR = PROJECT_ROOT / ".cache"    # 第三方套件的快取

# File handling
UPLOAD_DIR = str(DATA_DIR / "uploaded_docs")
QDRANT_PATH = str(DATA_DIR / "qdrant_data")
# collection 名稱帶上 provider：不同後端的向量維度不同（OpenAI 1536 / bge-m3 1024），
# 分開命名可避免切換後端時誤用到維度不符的舊集合。
QDRANT_COLLECTION_NAME = f"rag_collection_contextual_{LLM_PROVIDER}_v1"
TIKTOKEN_CACHE_DIR = str(CACHE_DIR / "tiktoken")
MPLCONFIGDIR = str(CACHE_DIR / "matplotlib")

# Contextual Retriever Configuration
CONTEXTUAL_WINDOW_SIZE = 1  # Retrieve 1 chunk before and 1 chunk after the seed chunk
BASE_RETRIEVER_K = 3  # Number of initial seed documents to retrieve
FINAL_RETRIEVER_K = 5  # Max number of documents to pass to LLM after contextual expansion

# Gradio Configuration
GRADIO_SHARE = False  # Set to True for public sharing (only set True if needed)
GRADIO_DEBUG = False  # Set to True for debugging

# Environment setup
def setup_directories():
    """Create necessary directories if they don't exist"""
    # parents=True：data/ 與 .cache/ 若不存在也一併建立
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    Path(TIKTOKEN_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(MPLCONFIGDIR).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(TIKTOKEN_CACHE_DIR).resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(MPLCONFIGDIR).resolve()))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

def get_openai_api_key():
    """Get OpenAI API key from environment or prompt user"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in environment variables.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    return api_key

def check_ollama_ready():
    """
    預檢 Ollama：服務是否可連線、所需模型是否已 pull。
    回傳 (ok: bool, message: str)。不拋例外，讓呼叫端決定怎麼處理。
    """
    import json
    import urllib.request

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return False, (
            f"無法連線到 Ollama 服務 ({OLLAMA_BASE_URL})：{e}\n"
            "請確認：\n"
            "  1. 已安裝 Ollama（https://ollama.com/download）\n"
            "  2. Ollama 服務正在執行（終端機執行 `ollama list` 應能列出模型）"
        )

    def _norm(tag):
        """Ollama 省略 tag 時等同 :latest，比對前先正規化。"""
        return tag if ":" in tag else tag + ":latest"

    installed = sorted(m.get("name", "") for m in data.get("models", []))
    installed_norm = {_norm(n) for n in installed}
    needed = [OLLAMA_CHAT_MODEL_NAME, OLLAMA_EMBEDDING_MODEL_NAME]
    missing = [tag for tag in needed if _norm(tag) not in installed_norm]

    if missing:
        lines = [f"Ollama 已連線，但缺少模型：{', '.join(missing)}", "", "請執行："]
        lines += [f"  ollama pull {tag}" for tag in missing]
        for tag in missing:
            family = tag.split(":")[0]
            near = [n for n in installed if n.split(":")[0] == family]
            if near:
                lines.append(f"（偵測到同系列但標籤不同的模型：{', '.join(near)}"
                             f" — 若要改用，請設定 OLLAMA_CHAT_MODEL / OLLAMA_EMBED_MODEL）")
        lines += ["", f"目前已安裝：{', '.join(installed) if installed else '（無）'}"]
        return False, "\n".join(lines)

    return True, (
        f"Ollama 就緒 — chat: {OLLAMA_CHAT_MODEL_NAME} / embed: {OLLAMA_EMBEDDING_MODEL_NAME}\n"
        f"已安裝模型：{', '.join(installed)}"
    )


def startup_check():
    """
    依 LLM_PROVIDER 做啟動前檢查。回傳 (ok: bool, message: str)。
    openai → 取得 API key（必要時提示輸入）；ollama → 檢查服務與模型。
    """
    if LLM_PROVIDER == "ollama":
        return check_ollama_ready()

    if LLM_PROVIDER != "openai":
        return False, (f"未知的 LLM_PROVIDER: '{LLM_PROVIDER}'。"
                       "可用值為 'openai' 或 'ollama'。")

    api_key = get_openai_api_key()
    if not api_key:
        return False, "無法獲取 OpenAI API 金鑰。"
    return True, f"OpenAI API 金鑰已設置 (開頭: {api_key[:4]}...)"
