# 本地 Ollama 後端啟動腳本（不需要 API key，全部跑在本機 GPU）
# 用法（從專案根目錄執行）：  .\scripts\run_ollama.ps1
# 需先安裝 Ollama 並 pull 模型：
#   ollama pull qwen2.5:3b
#   ollama pull bge-m3
$env:PYTHONUTF8 = "1"
$env:LLM_PROVIDER = "ollama"

# 想換模型就改這兩行（未設定時用 config.py 的預設值）
$env:OLLAMA_CHAT_MODEL = "qwen2.5:3b"     # 或 gemma3:1b / gemma3:4b
$env:OLLAMA_EMBED_MODEL = "bge-m3"

# GTX 1050 Ti 只有約 3GB 可用 VRAM，聊天模型與嵌入模型無法同時常駐。
# 限制同時載入 1 個模型，讓 Ollama 自動換出換入，避免 VRAM 不足而整個掉到 CPU。
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# ── Python 解譯器解析（依序嘗試）──
#   1. $env:RAG_PYTHON        明確指定，優先於一切
#   2. conda 環境 RAG_damo    本專案的開發環境
#   3. PATH 上的 python       其他機器的預設
$python = $env:RAG_PYTHON
if (-not $python) {
    $conda = Join-Path $env:USERPROFILE "anaconda3/envs/RAG_damo/python.exe"
    if (Test-Path $conda) { $python = $conda } else { $python = 'python' }
}
# 腳本在 scripts/ 底下，工作目錄要切到專案根目錄（上一層）
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
& $python app.py
