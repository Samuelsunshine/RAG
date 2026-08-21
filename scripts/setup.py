"""
Setup script for Windows environment

注意：本腳本位於 scripts/ 之下，所有路徑均以專案根目錄（上一層）為基準，
      因此可以從任何工作目錄執行。
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("錯誤: 需要 Python 3.8 或更高版本")
        return False
    print(f"Python 版本: {sys.version}")
    return True


def install_requirements():
    """Install required packages"""
    print("正在安裝所需套件...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "-r", str(PROJECT_ROOT / "requirements.txt")])
        print("套件安裝完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"安裝套件時發生錯誤: {e}")
        return False


def setup_environment():
    """Setup environment variables and directories"""
    print("正在設置環境...")
    
    # Create necessary directories（與 src/config.py 的路徑定義一致）
    directories = [
        PROJECT_ROOT / "data" / "uploaded_docs",
        PROJECT_ROOT / "data" / "qdrant_data",
        PROJECT_ROOT / ".cache" / "tiktoken",
        PROJECT_ROOT / ".cache" / "matplotlib",
    ]
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
        print(f"已創建目錄: {path.relative_to(PROJECT_ROOT)}")
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n警告: 未檢測到 OPENAI_API_KEY 環境變數")
        print("您可以:")
        print("1. 設置環境變數: set OPENAI_API_KEY=your_key_here")
        print("2. 或在啟動應用時手動輸入")
    else:
        print("已檢測到 OPENAI_API_KEY 環境變數")
    
    return True


def main():
    """Main setup function"""
    print("開始設置 RAG Chatbot 環境...")
    
    if not check_python_version():
        return False
    
    if not install_requirements():
        return False
    
    if not setup_environment():
        return False
    
    print("")
    print("設置完成! 現在可以運行以下命令啟動應用:")
    print("  .\\scripts\\run_openai.ps1      (OpenAI 後端)")
    print("  .\\scripts\\run_ollama.ps1      (本地 Ollama，GPU)")
    print("  .\\scripts\\run_ollama_cpu.ps1  (本地 Ollama，純 CPU)")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)