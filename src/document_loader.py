import os
import hashlib
import uuid
from typing import List
import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _dataframe_to_text(df: pd.DataFrame) -> str:
    """Convert tabular data into plain text that keeps row/column context."""
    cleaned_df = df.dropna(how="all").fillna("")
    cleaned_df = cleaned_df.loc[:, ~(cleaned_df == "").all()]
    if cleaned_df.empty:
        return ""

    return cleaned_df.to_csv(index=False, sep="\t")


def _load_csv_document(file_path: str) -> List[Document]:
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5"]
    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            text = _dataframe_to_text(df)
            if not text:
                return []
            return [Document(page_content=text, metadata={"file_type": "csv"})]
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise ValueError(f"無法使用常見編碼讀取 CSV: {last_error}")


def _load_excel_document(file_path: str) -> List[Document]:
    sheets = pd.read_excel(file_path, sheet_name=None)
    documents = []

    for sheet_name, df in sheets.items():
        text = _dataframe_to_text(df)
        if not text:
            continue

        page_content = f"工作表: {sheet_name}\n\n{text}"
        documents.append(
            Document(
                page_content=page_content,
                metadata={"file_type": "excel", "sheet_name": sheet_name},
            )
        )

    return documents


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load documents from various file types"""
    print("開始載入文件...")
    documents = []
    
    for file_path in file_paths:
        print(f"正在載入: {file_path}")
        try:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
                docs_from_file = loader.load()
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                docs_from_file = loader.load()
            elif file_extension == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                docs_from_file = loader.load()
            elif file_extension == ".csv":
                docs_from_file = _load_csv_document(file_path)
            elif file_extension in [".xlsx", ".xls"]:
                docs_from_file = _load_excel_document(file_path)
            else:
                print(f"警告: 不支持的文件類型: {file_path}")
                continue

            # Add unique document ID to each loaded document's metadata
            file_doc_id = hashlib.md5(file_path.encode()).hexdigest()
            for doc in docs_from_file:
                doc.metadata["original_file_doc_id"] = file_doc_id
                doc.metadata["source"] = os.path.basename(file_path)
            documents.extend(docs_from_file)

        except Exception as e:
            print(f"錯誤: 載入文件 {file_path} 失敗: {e}")
            
    print(f"文件載入完成，共載入 {len(documents)} 個文檔片段。")
    return documents


def split_documents_with_order_metadata(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks with order metadata"""
    print("開始分割文件並添加順序元數據...")
    if not documents:
        print("警告: 沒有文件可供分割。")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks_with_metadata = []

    # Group documents by their original file source
    docs_by_source = {}
    for doc in documents:
        source_file = doc.metadata.get("source", "unknown_source")
        if source_file not in docs_by_source:
            docs_by_source[source_file] = []
        docs_by_source[source_file].append(doc)

    for source_file, file_docs in docs_by_source.items():
        # Combine all pages/documents from the same file into one text
        combined_text = "\n\n".join([doc.page_content for doc in file_docs])
        doc_id = file_docs[0].metadata.get("original_file_doc_id", f"doc_{uuid.uuid4().hex}")
        
        print(f"處理文件 {source_file}: 合併 {len(file_docs)} 個頁面/片段")
        
        # Split the combined text
        chunks_from_file = text_splitter.split_text(combined_text)
        total_chunks_in_file = len(chunks_from_file)
        
        print(f"文件 {source_file} 分割成 {total_chunks_in_file} 個區塊")

        for chunk_idx, chunk_text in enumerate(chunks_from_file):
            chunk_metadata = file_docs[0].metadata.copy()  # Use metadata from first page
            chunk_metadata["doc_id"] = doc_id
            chunk_metadata["chunk_id"] = chunk_idx
            chunk_metadata["total_chunks_in_doc"] = total_chunks_in_file
            chunk_metadata["source"] = source_file

            new_chunk_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
            all_chunks_with_metadata.append(new_chunk_doc)

    print(f"文件分割完成，共分割成 {len(all_chunks_with_metadata)} 個區塊。")
    return all_chunks_with_metadata