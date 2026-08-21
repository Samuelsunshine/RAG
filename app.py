import os
import sys
import shutil
import gradio as gr
import gc
try:
    import torch
except ImportError:  # torch 僅用於選擇性的 CUDA 快取清理
    torch = None
from typing import List, Tuple
from pathlib import Path

# 核心模組放在 src/，先把它加進 import 路徑。
# 必須在任何專案模組的 import 之前執行。
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import (
    UPLOAD_DIR, 
    QDRANT_PATH, 
    GRADIO_SHARE, 
    GRADIO_DEBUG,
    LLM_PROVIDER,
    OPENAI_CHAT_MODEL_NAME,
    OLLAMA_CHAT_MODEL_NAME,
    OLLAMA_EMBEDDING_MODEL_NAME,
    setup_directories,
    get_openai_api_key,
    startup_check
)
from document_loader import load_documents, split_documents_with_order_metadata
from rag_system import RAGSystem


class ChatbotApp:
    def __init__(self):
        self.rag_system = RAGSystem()
        self.processed_files = []
        
    def process_uploaded_files(self, files_list, progress=gr.Progress(track_tqdm=True)):
        """Process uploaded files and create RAG system"""
        print("\n--- 開始處理上傳文件 ---")
        
        if not files_list:
            message = "請先上傳文件。"
            print(message)
            return message, [], None, None

        # Clear existing system
        self.rag_system.clear_system()
        
        # Clean up directories
        if os.path.exists(QDRANT_PATH):
            print(f"正在清理舊的 Qdrant 數據目錄: {QDRANT_PATH}")
            try:
                shutil.rmtree(QDRANT_PATH)
            except Exception as e:
                print(f"警告: 刪除 Qdrant 數據目錄失敗: {e}")
        
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        
        setup_directories()

        # Copy uploaded files
        uploaded_file_paths = []
        current_filenames = []
        
        for file_obj in files_list:
            filename = os.path.basename(file_obj.name)
            dest_path = os.path.join(UPLOAD_DIR, filename)
            try:
                shutil.copy(file_obj.name, dest_path)
                uploaded_file_paths.append(dest_path)
                current_filenames.append(filename)
            except Exception as e:
                message = f"複製文件 {filename} 失敗: {e}"
                print(message)
                return message, [], None, None

        self.processed_files = sorted(list(set(current_filenames)))

        # Process documents
        progress(0.1, desc="正在載入文件...")
        documents = load_documents(uploaded_file_paths)
        if not documents:
            message = "文件載入失敗。"
            print(message)
            return message, [], None, None

        progress(0.3, desc="正在分割文件並添加元數據...")
        chunks = split_documents_with_order_metadata(documents)
        if not chunks:
            message = "文件分割失敗。"
            print(message)
            return message, [], None, None

        progress(0.6, desc="正在創建向量數據庫...")
        vector_store = self.rag_system.create_vector_store(chunks)
        if not vector_store:
            message = "向量數據庫創建失敗。"
            print(message)
            return message, [], None, None

        progress(0.8, desc="正在創建 RAG 對話鏈...")
        rag_chain = self.rag_system.create_rag_chain(vector_store)
        if not rag_chain:
            message = "RAG 對話鏈創建失敗。請檢查 API 金鑰和網絡連接。"
            print(message)
            self.rag_system.clear_system()
            return message, [], None, None

        processed_files_str = ", ".join(self.processed_files)
        status_message = f"文件 '{processed_files_str}' 已成功處理！您可以開始提問了。"
        print(status_message)
        print("--- 文件處理完成 ---")
        
        return status_message, [], rag_chain, vector_store

    def chat_with_bot(self, message: str, chat_history: List[Tuple[str, str]], 
                     current_rag_chain, current_vector_store):
        """Handle chat with the bot"""
        
        if not message or message.strip() == "":
            return "", chat_history, current_rag_chain, current_vector_store

        if not current_rag_chain:
            error_msg = "錯誤：RAG 系統尚未初始化。請先上傳並處理文件。"
            print(error_msg)
            chat_history.append((message, error_msg))
            return "", chat_history, current_rag_chain, current_vector_store

        print(f"\n用戶提問: {message}")
        
        try:
            result = current_rag_chain.invoke({"question": message})
            answer = result["answer"].strip()
            source_documents = result.get("source_documents", [])

            print(f"LLM 回答: {answer[:300]}...")

            if not answer:
                answer = "模型返回了一個空回答。請嘗試不同的問題或檢查上下文。"

            # Add source information
            if source_documents:
                answer += "\n\n---\n**參考來源:**"
                unique_sources_info = {}

                source_documents.sort(key=lambda d: (d.metadata.get("doc_id", ""), d.metadata.get("chunk_id", -1)))

                for doc in source_documents:
                    source_name = os.path.basename(doc.metadata.get('source', '未知來源'))
                    page_content_preview = doc.page_content[:100].replace('\n', ' ') + "..."
                    doc_id = doc.metadata.get('doc_id', 'N/A')
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')

                    if source_name not in unique_sources_info:
                        unique_sources_info[source_name] = []

                    if len(unique_sources_info[source_name]) < 3:
                        unique_sources_info[source_name].append(f"  - Chunk {chunk_id}: \"{page_content_preview}\"")

                idx = 1
                for src_name, contents in unique_sources_info.items():
                    answer += f"\n{idx}. **{src_name}**:"
                    for content_preview_line in contents:
                        answer += f"\n  {content_preview_line}"
                    idx += 1

        except Exception as e:
            error_detail = f"回答時發生錯誤: {e}"
            print(error_detail)
            answer = f"抱歉，處理您的請求時發生錯誤。詳情: {str(e)[:150]}..."
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                answer += "\n請檢查您的 OpenAI API 金鑰餘額或使用限制。"

        chat_history.append((message, answer))
        return "", chat_history, current_rag_chain, current_vector_store

    def clear_chat_and_data(self):
        """Clear all chat data and RAG system"""
        print("\n--- 清除所有數據和 RAG 狀態 ---")
        
        self.rag_system.clear_system()
        self.processed_files = []

        # Clean up directories
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        if os.path.exists(QDRANT_PATH):
            try:
                shutil.rmtree(QDRANT_PATH)
            except Exception as e:
                print(f"警告: 刪除 Qdrant 目錄失敗: {e}")

        setup_directories()

        # Clean up memory
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        status_message = "所有聊天記錄、數據和 RAG 狀態已清除。"
        print(status_message)
        print("--- 數據清除完成 ---")
        
        return status_message, [], None, None, None

    def create_interface(self):
        """Create Gradio interface"""
        if LLM_PROVIDER == "ollama":
            backend_title = f"本地 Ollama ({OLLAMA_CHAT_MODEL_NAME})"
            backend_desc = (f"使用本地 GPU 運行 {OLLAMA_CHAT_MODEL_NAME}，"
                            f"嵌入模型為 {OLLAMA_EMBEDDING_MODEL_NAME}，資料不外流。")
            backend_step1 = "確認 Ollama 服務正在執行，且已 pull 所需模型。"
        else:
            backend_title = f"OpenAI {OPENAI_CHAT_MODEL_NAME}"
            backend_desc = f"使用 OpenAI {OPENAI_CHAT_MODEL_NAME} 提供高質量回答。"
            backend_step1 = "確保已設置 OPENAI_API_KEY 環境變數或在啟動時輸入。"

        with gr.Blocks(
            title=f"{backend_title} Contextual RAG Chatbot", 
            theme=gr.themes.Soft(primary_hue=gr.themes.colors.teal, secondary_hue=gr.themes.colors.cyan)
        ) as demo:
            
            gr.Markdown(f"""
            # 🤖 {backend_title} Contextual RAG Chatbot 🤖
            
            上傳您的 TXT, PDF, DOCX, CSV, 或 Excel 文件，然後向它們提問！
            {backend_desc}並以上下文感知檢索器增強答案的相關性。

            **使用說明:**
            1. {backend_step1}
            2. 點擊 "上傳文件" 並選擇您的文檔。
            3. 點擊 "📁 處理文件並準備 RAG" 按鈕。
            4. 處理完成後，在下方聊天框輸入問題並點擊 "💬 發送"。
            5. 點擊 "🗑️ 清除所有數據" 以重置系統。
            """)

            rag_chain_state = gr.State(None)
            vector_store_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    file_uploader = gr.File(
                        label="上傳文件 (TXT, PDF, DOCX, CSV, Excel)", 
                        file_count="multiple", 
                        file_types=[".txt", ".pdf", ".docx", ".csv", ".xlsx", ".xls"]
                    )
                    process_button = gr.Button("📁 處理文件並準備 RAG", variant="primary", scale=2)
                    clear_button = gr.Button("🗑️ 清除所有數據", variant="stop")
                    status_display = gr.Textbox(
                        label="系統狀態", 
                        interactive=False, 
                        lines=4, 
                        max_lines=4, 
                        placeholder="系統將在此處顯示處理狀態和消息..."
                    )

                with gr.Column(scale=2):
                    chatbot_display = gr.Chatbot(
                        label="RAG 聊天機器人",
                        height=600,
                        type="tuples"
                    )
                    with gr.Row():
                        message_input = gr.Textbox(
                            label="輸入您的問題:", 
                            placeholder="例如：總結一下這個文件的主要觀點...", 
                            show_label=False, 
                            lines=3, 
                            scale=5
                        )
                        submit_button = gr.Button("💬 發送", variant="primary", scale=1)

            # Event handlers
            process_button.click(
                fn=self.process_uploaded_files,
                inputs=[file_uploader],
                outputs=[status_display, chatbot_display, rag_chain_state, vector_store_state],
            )

            chat_inputs = [message_input, chatbot_display, rag_chain_state, vector_store_state]
            chat_outputs = [message_input, chatbot_display, rag_chain_state, vector_store_state]

            submit_button.click(fn=self.chat_with_bot, inputs=chat_inputs, outputs=chat_outputs)
            message_input.submit(fn=self.chat_with_bot, inputs=chat_inputs, outputs=chat_outputs)

            clear_button.click(
                fn=self.clear_chat_and_data, 
                inputs=[], 
                outputs=[status_display, chatbot_display, rag_chain_state, vector_store_state, file_uploader]
            )

        return demo


def main():
    """Main function to run the application"""
    print("正在啟動 RAG Chatbot...")
    
    # Setup environment
    setup_directories()

    # 依 LLM_PROVIDER 做啟動前檢查
    #   openai → 取得 API key；ollama → 檢查服務可連線且模型已 pull
    print(f"模型後端 (LLM_PROVIDER): {LLM_PROVIDER}")
    ok, message = startup_check()
    print(message)
    if not ok:
        print("")
        print("啟動中止。")
        return

    # Create and launch app
    app = ChatbotApp()
    demo = app.create_interface()
    
    print("\n正在啟動 Gradio 界面...")
    demo.launch(
        debug=GRADIO_DEBUG, 
        share=GRADIO_SHARE,
        server_name="127.0.0.1",  # Bind to localhost for Windows
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()