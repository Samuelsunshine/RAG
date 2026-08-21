import uuid
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_client import QdrantClient, models


class LocalQdrantVectorStore:
    """Small Qdrant wrapper compatible with qdrant-client >= 1.16."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings: Embeddings,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def add_documents(self, documents: List[Document], batch_size: int = 64) -> None:
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start + batch_size]
            texts = [doc.page_content for doc in batch_docs]
            vectors = self.embeddings.embed_documents(texts)

            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                )
                for doc, vector in zip(batch_docs, vectors)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[models.Filter] = None,
        **_: Any,
    ) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=filter,
            limit=k,
            with_payload=True,
        )

        documents = []
        for point in response.points:
            payload = point.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )

        return documents

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        search_kwargs = kwargs.get("search_kwargs") or {}
        return LocalQdrantRetriever(
            vectorstore=self,
            search_kwargs=search_kwargs,
        )


class LocalQdrantRetriever(BaseRetriever):
    vectorstore: LocalQdrantVectorStore
    search_kwargs: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query, **self.search_kwargs)
