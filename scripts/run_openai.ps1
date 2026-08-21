# OpenAI 後端啟動腳本（GPT-4o + text-embedding-3-small）
# 用法（從專案根目錄執行）：  .\scripts\run_openai.ps1
# 需先設好 OPENAI_API_KEY，否則啟動時會提示輸入。
$env:PYTHONUTF8 = "1"
$env:LLM_PROVIDER = "openai"
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
