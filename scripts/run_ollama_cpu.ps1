# 本地 Ollama 後端 —— 純 CPU 版（驅動修好前的備用方案）
#
# 用法（從專案根目錄執行）：  .\scripts\run_ollama_cpu.ps1
#
# 為什麼需要這個：
#   GTX 1050 Ti 是 Pascal (compute capability 6.1)，而 NVIDIA 驅動 560.94 只支援
#   CUDA 12.6。Ollama 0.32.15 的 cuda_v12 runner 是用更新的 toolkit 編譯的，
#   驅動無法 JIT 那份 PTX，llama-server 會以 0xc0000409 崩潰：
#     "CUDA error: the provided PTX was compiled with an unsupported toolchain"
#   正解是把驅動升到 580.x 分支（支援 CUDA 13），之後改用 run_ollama.ps1 走 GPU。
#
# 這個腳本另開一個停用 CUDA 的 Ollama 實例在 11435 埠，不影響你原本
# 跑在 11434 的 Ollama 服務。模型檔案兩邊共用，不會重複下載。
#
# 實測效能（i7-8700 純 CPU，qwen2.5:3b Q4_K_M）：
#   建索引 約 1.7 秒/chunk，回答 約 40-50 秒/題

$ErrorActionPreference = "Stop"
# ollama.exe 位置：先找 PATH，再找 Windows 的預設安裝位置
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    $ollama = Join-Path $env:LOCALAPPDATA "Programs/Ollama/ollama.exe"
}
if (-not (Test-Path $ollama)) {
    Write-Error "找不到 ollama.exe。請先安裝 Ollama：winget install Ollama.Ollama"
    exit 1
}
$cpuUrl = "http://127.0.0.1:11435"

# 若 11435 還沒有實例，就開一個（停用 CUDA）
$running = $false
try {
    Invoke-WebRequest -Uri "$cpuUrl/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
    $running = $true
    Write-Host "CPU 版 Ollama 已在 11435 執行中。"
} catch { }

if (-not $running) {
    Write-Host "啟動 CPU 版 Ollama (11435)..."
    # 注意：Start-Process 的 -Environment 參數只有 PowerShell 7+ 才有，
    # Windows PowerShell 5.1 要先設在當前 session，子行程會繼承。
    $env:CUDA_VISIBLE_DEVICES = "-1"
    $env:OLLAMA_HOST          = "127.0.0.1:11435"
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    # 設完就清掉，避免影響後面啟動的 python 行程
    Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    Remove-Item Env:\OLLAMA_HOST -ErrorAction SilentlyContinue
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 800
        try { Invoke-WebRequest -Uri "$cpuUrl/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null; break } catch { }
    }
}

$env:PYTHONUTF8      = "1"
$env:LLM_PROVIDER    = "ollama"
$env:OLLAMA_BASE_URL = $cpuUrl
$env:OLLAMA_CHAT_MODEL  = "qwen2.5:3b"
$env:OLLAMA_EMBED_MODEL = "bge-m3"

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
