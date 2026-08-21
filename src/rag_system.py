import os
import traceback
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from qdrant_client import QdrantClient, models
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from config import (
    LLM_PROVIDER,
    OPENAI_CHAT_MODEL_NAME, 
    OPENAI_EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL_NAME,
    OLLAMA_EMBEDDING_MODEL_NAME,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    QDRANT_PATH, 
    QDRANT_COLLECTION_NAME,
    CONTEXTUAL_WINDOW_SIZE,
    BASE_RETRIEVER_K
)
from contextual_retriever import ContextualRetriever
from qdrant_vector_store import LocalQdrantVectorStore


class RAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.rag_chain = None
        
    def get_embedding_model(self) -> Optional[Embeddings]:
        """依 LLM_PROVIDER 初始化嵌入模型（openai 或 ollama）"""
        if self.embedding_model is not None:
            return self.embedding_model

        if LLM_PROVIDER == "ollama":
            print(f"初始化 Ollama 嵌入模型: {OLLAMA_EMBEDDING_MODEL_NAME} @ {OLLAMA_BASE_URL}...")
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError:
                print("錯誤: 未安裝 langchain-ollama。請執行 pip install langchain-ollama")
                return None
            try:
                self.embedding_model = OllamaEmbeddings(
                    model=OLLAMA_EMBEDDING_MODEL_NAME,
                    base_url=OLLAMA_BASE_URL,
                )
                print("Ollama 嵌入模型初始化完成。")
            except Exception as e:
                print(f"初始化 Ollama 嵌入模型時發生錯誤: {e}")
                traceback.print_exc()
                return None
            return self.embedding_model

        # 預設 openai 路徑（原本邏輯不變）
        print(f"初始化 OpenAI 嵌入模型: {OPENAI_EMBEDDING_MODEL_NAME}...")
        try:
            if not os.environ.get("OPENAI_API_KEY"):
                print("錯誤: OPENAI_API_KEY 未設置。")
                return None

            self.embedding_model = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL_NAME,
            )
            print(f"OpenAI 嵌入模型初始化完成。")
        except Exception as e:
            print(f"初始化 OpenAI 嵌入模型時發生錯誤: {e}")
            traceback.print_exc()
            return None
        return self.embedding_model

    # 向後相容：舊名稱仍可呼叫
    def get_openai_embedding_model(self) -> Optional[Embeddings]:
        return self.get_embedding_model()

    def create_vector_store(self, chunks: List[Document]) -> Optional[LocalQdrantVectorStore]:
        """Create Qdrant vector store"""
        print(f"開始創建向量數據庫...")
        if not chunks:
            print("錯誤: 沒有文本區塊可供創建向量數據庫。")
            return None

        embedding_model = self.get_embedding_model()
        if not embedding_model:
            print(f"錯誤: 嵌入模型初始化失敗 (provider={LLM_PROVIDER})。")
            return None

        try:
            print(f"使用 Qdrant 創建集合...")
            client = QdrantClient(path=QDRANT_PATH)

            try:
                client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)
            except Exception:
                pass

            embedding_size = len(embedding_model.embed_query("dimension probe"))
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                ),
            )

            qdrant_store = LocalQdrantVectorStore(
                client=client,
                collection_name=QDRANT_COLLECTION_NAME,
                embeddings=embedding_model,
            )
            qdrant_store.add_documents(chunks)
            print(f"向量數據庫創建成功。")
            self.vector_store = qdrant_store
            return qdrant_store
        except Exception as e:
            print(f"創建 Qdrant 向量數據庫時發生錯誤: {e}")
            traceback.print_exc()
            return None

    def get_llm(self) -> Optional[BaseChatModel]:
        """依 LLM_PROVIDER 初始化聊天模型（openai 或 ollama）"""
        if LLM_PROVIDER == "ollama":
            print(f"初始化 Ollama LLM: {OLLAMA_CHAT_MODEL_NAME} @ {OLLAMA_BASE_URL} "
                  f"(num_ctx={OLLAMA_NUM_CTX})...")
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                print("錯誤: 未安裝 langchain-ollama。請執行 pip install langchain-ollama")
                return None
            try:
                llm = ChatOllama(
                    model=OLLAMA_CHAT_MODEL_NAME,
                    base_url=OLLAMA_BASE_URL,
                    temperature=OLLAMA_TEMPERATURE,
                    num_ctx=OLLAMA_NUM_CTX,
                    num_predict=OLLAMA_NUM_PREDICT,
                )
                print("Ollama LLM 初始化完成。")
                return llm
            except Exception as e:
                print(f"初始化 Ollama LLM 時發生錯誤: {e}")
                traceback.print_exc()
                return None

        # 預設 openai 路徑（原本邏輯不變）
        print(f"初始化 OpenAI LLM: {OPENAI_CHAT_MODEL_NAME}...")
        try:
            if not os.environ.get("OPENAI_API_KEY"):
                print("錯誤: OPENAI_API_KEY 未設置。")
                return None
                
            llm = ChatOpenAI(
                model_name=OPENAI_CHAT_MODEL_NAME,
                temperature=0.1,
                max_tokens=1500,
            )
            print(f"OpenAI LLM 初始化完成。")
            return llm
        except Exception as e:
            print(f"初始化 OpenAI LLM 時發生錯誤: {e}")
            traceback.print_exc()
            return None

    # 向後相容：舊名稱仍可呼叫
    def get_openai_llm(self) -> Optional[BaseChatModel]:
        return self.get_llm()

    def create_rag_chain(self, vector_store: LocalQdrantVectorStore) -> Optional[ConversationalRetrievalChain]:
        """Create RAG conversation chain with contextual retriever"""
        if vector_store is None:
            print("錯誤: 向量數據庫未初始化。")
            return None

        print("準備創建 RAG 對話鏈...")
        llm = self.get_llm()
        if llm is None:
            print(f"錯誤: LLM 初始化失敗 (provider={LLM_PROVIDER})。")
            return None

        # Define prompts
        qa_prompt_template = """### 你是一位 AI 資訊檢索與問答助理 ###
**核心任務:** 你的 **唯一且最重要的任務** 是根據下面提供的「上下文資訊」來回答用戶提出的「問題」。你必須 **嚴格且僅僅** 依賴「上下文資訊」中的內容來生成你的回答。

**行為準則:**
1. **絕對基於上下文:** 你的回答必須 **完全且僅僅** 來源於「上下文資訊」。**嚴禁** 使用任何外部知識、個人觀點、預設信息，或任何「上下文資訊」之外的內容。
2. **處理資訊缺失:** 如果「上下文資訊」中 **沒有** 包含回答「問題」所需的內容，你 **必須** 明確指出。標準回答應為：「根據提供的文件資料，我無法找到關於『[此處簡述問題核心]』的具體資訊。」
3. **回答風格:** 請始終使用 **繁體中文** 進行回答。保持專業、客觀、中立且樂於助人的語氣。

**上下文資訊:**
{context}

**問題:**
{question}
"""

        condense_question_template = """### 你的任務：問題重述與獨立化 ###
**目標:** 根據提供的「對話記錄」和一個「後續問題」，你的任務是將「後續問題」改寫成一個 **完整、獨立且無歧義的繁體中文問題**。

**執行指令:**
1. **分析上下文:** 仔細閱讀「對話記錄」，理解它如何影響「後續問題」的含義。
2. **識別依賴:** 判斷「後續問題」是否依賴「對話記錄」中的信息。
3. **改寫或保留:** 如果「後續問題」已清晰獨立，直接原樣輸出。如果有依賴性，改寫成自包含的獨立問題。
4. **語言:** 輸出必須是 **繁體中文**。
5. **輸出格式:** **僅輸出改寫後的獨立問題本身。**

**對話記錄:**
{chat_history}

**後續問題:**
{question}
"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_prompt_template)
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

        # Setup contextual retriever
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': BASE_RETRIEVER_K}
        )

        contextual_retriever = ContextualRetriever(
            vectorstore=vector_store,
            base_retriever=base_retriever,
            window_size=CONTEXTUAL_WINDOW_SIZE
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        try:
            rag_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=contextual_retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                verbose=False
            )
            print("RAG 對話鏈創建完成。")
            self.rag_chain = rag_chain
            return rag_chain
        except Exception as e:
            print(f"創建 ConversationalRetrievalChain 時發生錯誤: {e}")
            traceback.print_exc()
            return None

    def clear_system(self):
        """Clear all system components"""
        print("正在清理 RAG 系統...")
        
        if self.rag_chain:
            if hasattr(self.rag_chain, 'llm'):
                del self.rag_chain.llm
            if hasattr(self.rag_chain, 'memory'):
                del self.rag_chain.memory
            if hasattr(self.rag_chain, 'retriever'):
                del self.rag_chain.retriever
            del self.rag_chain
            self.rag_chain = None

        if self.embedding_model:
            del self.embedding_model
            self.embedding_model = None

        if self.vector_store and hasattr(self.vector_store, 'client') and hasattr(self.vector_store.client, 'close'):
            try:
                self.vector_store.client.close()
            except Exception as e:
                print(f"關閉 Qdrant client 時出錯: {e}")
        self.vector_store = None
        
        print("RAG 系統清理完成。")