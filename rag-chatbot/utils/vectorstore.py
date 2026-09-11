"""
Vector store implementation backed by NumPy + JSON.

Why not ChromaDB / SKLearn?
---------------------------
- chromadb requires the native `hnswlib` index, which the Windows Application
  Control policy on this machine blocks from loading (DLL load failed ...
  "An Application Control policy has blocked this file").
- SKLearnVectorStore requires scikit-learn (heavy) and still sits on numpy.
NumPy + OnnxRuntime (via FastEmbed) are verified working here, so we use a
small persistent vector store built directly on NumPy (cosine similarity).

Public API mirrors the old module so callers (ingest.py, rag_chain.py,
app.py) don't change:
    get_embeddings(model_name)
    create_vector_store(chunks, persist_directory, embedding_model_name)
    load_vector_store(persist_directory, embedding_model_name)
"""
import json
import os
from typing import Iterable, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

VECTORS_FILE = "vectors.npy"
DOCS_FILE = "docs.json"


def get_embeddings(model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Initializes the FastEmbed embeddings model.
    """
    return FastEmbedEmbeddings(model_name=model_name)


class NumPyVectorStore(VectorStore):
    """
    A minimal persistent vector store using NumPy (cosine similarity)
    with JSON metadata persistence. Drop-in replacement for ChromaDB here.
    """

    def __init__(
        self,
        embedding,
        persist_directory: Optional[str] = None,
        vectors: Optional[list] = None,
        docs: Optional[list] = None,
    ):
        self._embedding = embedding
        self._persist_directory = persist_directory
        self._vectors: list = vectors if vectors is not None else []
        self._docs: list = docs if docs is not None else []

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _paths(self):
        vpath = os.path.join(self._persist_directory, VECTORS_FILE)
        dpath = os.path.join(self._persist_directory, DOCS_FILE)
        return vpath, dpath

    def persist(self):
        """Writes vectors (npy) + docs (json) to the persist directory."""
        if not self._persist_directory:
            return
        os.makedirs(self._persist_directory, exist_ok=True)
        vpath, dpath = self._paths()
        if self._vectors:
            np.save(vpath, np.asarray(self._vectors, dtype=np.float32))
        elif os.path.exists(vpath):
            os.remove(vpath)
        payload = [
            {"page_content": d.page_content, "metadata": d.metadata} for d in self._docs
        ]
        with open(dpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, embedding, persist_directory: str) -> "NumPyVectorStore":
        """Loads a previously persisted store from disk."""
        vpath, dpath = os.path.join(persist_directory, VECTORS_FILE), os.path.join(
            persist_directory, DOCS_FILE
        )
        vectors: list = []
        docs: list = []
        if os.path.exists(vpath):
            vectors = np.load(vpath, allow_pickle=False).astype(np.float32).tolist()
        if os.path.exists(dpath):
            with open(dpath, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    docs.append(
                        Document(
                            page_content=item["page_content"],
                            metadata=item.get("metadata", {}),
                        )
                    )
        return cls(
            embedding=embedding,
            persist_directory=persist_directory,
            vectors=vectors,
            docs=docs,
        )

    # ------------------------------------------------------------------ #
    # LangChain VectorStore interface
    # ------------------------------------------------------------------ #
    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs
    ) -> List[str]:
        texts = list(texts)
        embeddings = self._embedding.embed_documents(texts)
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            meta = metadatas[i] if metadatas else {}
            self._docs.append(Document(page_content=text, metadata=meta))
            self._vectors.append(list(vector))
        self.persist()
        start = len(self._docs) - len(texts)
        return [str(idx) for idx in range(start, len(self._docs))]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        if not self._vectors:
            return []
        query_vec = np.asarray(self._embedding.embed_query(query), dtype=np.float32)
        matrix = np.asarray(self._vectors, dtype=np.float32)
        q_norm = float(np.linalg.norm(query_vec))
        m_norms = np.linalg.norm(matrix, axis=1)
        scores = (matrix @ query_vec) / (m_norms * q_norm + 1e-9)
        top_idx = np.argsort(-scores)[:k]
        return [self._docs[int(i)] for i in top_idx]

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding,
        persist_directory: Optional[str] = None,
        **kwargs,
    ) -> "NumPyVectorStore":
        store = cls(embedding=embedding, persist_directory=persist_directory)
        store.add_texts(
            [d.page_content for d in documents], [d.metadata for d in documents]
        )
        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding,
        metadatas: Optional[List[dict]] = None,
        persist_directory: Optional[str] = None,
        **kwargs,
    ) -> "NumPyVectorStore":
        store = cls(embedding=embedding, persist_directory=persist_directory)
        store.add_texts(texts, metadatas)
        return store


def create_vector_store(chunks, persist_directory: str, embedding_model_name: str):
    """
    Creates a new vector store from document chunks and persists it to disk.
    """
    print(f"Creating vector store at {persist_directory}...")
    embeddings = get_embeddings(embedding_model_name)
    return NumPyVectorStore.from_documents(
        chunks, embeddings, persist_directory=persist_directory
    )


def load_vector_store(persist_directory: str, embedding_model_name: str):
    """
    Loads an existing vector store from disk.
    """
    embeddings = get_embeddings(embedding_model_name)
    return NumPyVectorStore.load(embeddings, persist_directory)
