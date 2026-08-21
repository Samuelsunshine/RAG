from typing import Any, List, Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_client import models
import traceback


class ContextualRetriever(BaseRetriever):
    """Custom retriever that fetches contextual chunks around seed documents"""
    
    vectorstore: Any
    base_retriever: BaseRetriever
    window_size: int = 1

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        print(f"\nContextualRetriever: Original query: '{query[:50]}...'")

        # Get seed documents using the base retriever
        seed_documents = self.base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
        print(f"ContextualRetriever: Got {len(seed_documents)} seed documents.")

        all_retrieved_docs_map: Dict[str, Document] = {}

        for seed_doc in seed_documents:
            seed_doc_key = f"{seed_doc.metadata.get('doc_id', 'unknown_doc')}_{seed_doc.metadata.get('chunk_id', -1)}"
            all_retrieved_docs_map[seed_doc_key] = seed_doc

            doc_id = seed_doc.metadata.get("doc_id")
            chunk_id = seed_doc.metadata.get("chunk_id")
            total_chunks = seed_doc.metadata.get("total_chunks_in_doc")

            if doc_id is None or chunk_id is None or total_chunks is None:
                print(f"ContextualRetriever: Seed doc missing metadata, skipping contextual expansion.")
                continue

            print(f"ContextualRetriever: Processing seed: doc_id={doc_id}, chunk_id={chunk_id}, total_chunks={total_chunks}")

            contextual_chunks_to_fetch = []
            
            # Fetch "before" chunks
            for i in range(1, self.window_size + 1):
                target_chunk_id = chunk_id - i
                if target_chunk_id >= 0:
                    contextual_chunks_to_fetch.append(target_chunk_id)

            # Fetch "after" chunks
            for i in range(1, self.window_size + 1):
                target_chunk_id = chunk_id + i
                if target_chunk_id < total_chunks:
                    contextual_chunks_to_fetch.append(target_chunk_id)

            if not contextual_chunks_to_fetch:
                continue

            # Fetch contextual chunks
            for target_cid in contextual_chunks_to_fetch:
                context_chunk_key = f"{doc_id}_{target_cid}"
                if context_chunk_key in all_retrieved_docs_map:
                    continue

                qdrant_filter = models.Filter(
                    must=[
                        models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id)),
                        models.FieldCondition(key="metadata.chunk_id", match=models.MatchValue(value=target_cid)),
                    ]
                )
                
                try:
                    context_docs = self.vectorstore.similarity_search(
                        query="context",
                        k=1,
                        filter=qdrant_filter
                    )

                    if context_docs:
                        retrieved_context_doc = context_docs[0]
                        print(f"ContextualRetriever: Fetched context: doc_id={retrieved_context_doc.metadata.get('doc_id')}, chunk_id={retrieved_context_doc.metadata.get('chunk_id')}")
                        all_retrieved_docs_map[context_chunk_key] = retrieved_context_doc
                    else:
                        print(f"ContextualRetriever: Context chunk not found for doc_id={doc_id}, chunk_id={target_cid}")
                        
                except Exception as e:
                    print(f"ContextualRetriever: Error fetching context chunk ({doc_id}, {target_cid}): {e}")
                    traceback.print_exc()

        # Sort documents by doc_id and chunk_id
        final_docs = list(all_retrieved_docs_map.values())
        final_docs.sort(key=lambda d: (d.metadata.get("doc_id", ""), d.metadata.get("chunk_id", -1)))

        print(f"ContextualRetriever: Total documents after contextual expansion: {len(final_docs)}")
        return final_docs