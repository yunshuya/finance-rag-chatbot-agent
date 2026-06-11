import json
import math
import os
from hashlib import md5
from pathlib import Path

from .chunker import Document, build_chunks


class DeterministicFakeEmbeddings:
    """Small deterministic embeddings for local Chroma tests without API keys."""

    def __init__(self, size=2048):
        self.size = size

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        vector = [0.0] * self.size
        text = str(text)
        tokens = _char_tokens(text)
        for token in tokens:
            index = int(md5(token.encode("utf-8")).hexdigest(), 16) % self.size
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class BgeM3Embeddings:
    """LangChain/Chroma-compatible wrapper around BAAI/bge-m3."""

    def __init__(
        self,
        model_name="BAAI/bge-m3",
        cache_dir=None,
        batch_size=12,
        use_fp16=False,
        model_factory=None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.model_factory = model_factory
        self._model = None

    def embed_documents(self, texts):
        texts = [str(text) for text in texts]
        if not texts:
            return []
        return self._dense_vectors(texts)

    def embed_query(self, text):
        return self._dense_vectors([str(text)])[0]

    def _dense_vectors(self, texts):
        encoded = self._model_instance().encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense_vectors = encoded["dense_vecs"]
        return [_as_float_list(vector) for vector in dense_vectors]

    def _model_instance(self):
        if self._model is not None:
            return self._model

        model_factory = self.model_factory
        if model_factory is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise ImportError(
                    "BGE-M3 embedding requires FlagEmbedding. "
                    "Install it with: pip install FlagEmbedding"
                ) from exc
            model_factory = BGEM3FlagModel

        kwargs = {"use_fp16": self.use_fp16}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_dir))
            os.environ.setdefault("HF_HUB_CACHE", str(self.cache_dir / "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir / "transformers"))
            kwargs["cache_dir"] = str(self.cache_dir)

        model_path = self._local_snapshot_path() or self.model_name

        try:
            self._model = model_factory(str(model_path), **kwargs)
        except TypeError:
            kwargs.pop("cache_dir", None)
            self._model = model_factory(str(model_path), **kwargs)
        return self._model

    def _local_snapshot_path(self):
        if not self.cache_dir or "/" not in self.model_name:
            return None

        cache_name = f"models--{self.model_name.replace('/', '--')}"
        cached_model_dir = self.cache_dir / cache_name
        ref_path = cached_model_dir / "refs" / "main"
        if not ref_path.exists():
            return None

        revision = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = cached_model_dir / "snapshots" / revision
        if snapshot_path.exists():
            return snapshot_path
        return None


def load_chunks_from_parsed_json(parsed_json_path, chunk_size=1000, chunk_overlap=150):
    parsed_json_path = Path(parsed_json_path)
    parsed_doc = json.loads(parsed_json_path.read_text(encoding="utf-8"))
    return build_chunks(
        parsed_doc,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def build_chroma_vectorstore(
    documents,
    persist_dir,
    embeddings=None,
    collection_name="finance_rag",
    reset_collection=True,
):
    """Persist retrieval chunks into a Chroma vectorstore."""
    import chromadb

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = embeddings or DeterministicFakeEmbeddings()
    documents = [_sanitize_document(document) for document in documents]
    client = chromadb.PersistentClient(path=str(persist_dir))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception as error:
            if error.__class__.__name__ != "NotFoundError":
                raise
            pass

    collection = client.get_or_create_collection(collection_name)
    if documents:
        texts = [document.page_content for document in documents]
        collection.add(
            ids=[document.metadata["chunk_id"] for document in documents],
            documents=texts,
            metadatas=[document.metadata for document in documents],
            embeddings=embeddings.embed_documents(texts),
        )
    return ChromaVectorStore(collection, embeddings)


class ChromaVectorStore:
    """Small adapter with the similarity_search method used by tests and scripts."""

    def __init__(self, collection, embeddings):
        self.collection = collection
        self.embeddings = embeddings

    def similarity_search(self, query, k=4):
        return [document for document, _ in self.similarity_search_with_scores(query, k=k)]

    def similarity_search_with_scores(self, query, k=4):
        results = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [
            (Document(page_content=text, metadata=metadata or {}), float(distance))
            for text, metadata, distance in zip(documents, metadatas, distances)
        ]

    def similarity_search_with_score(self, query, k=4):
        return self.similarity_search_with_scores(query, k=k)

    def count(self):
        return self.collection.count()

    def as_retriever(self, search_kwargs=None):
        search_kwargs = search_kwargs or {}
        return ChromaRetriever(
            vectorstore=self,
            k=search_kwargs.get("k", 4),
        )


class ChromaRetriever:
    def __init__(self, vectorstore, k=4):
        self.vectorstore = vectorstore
        self.k = k

    def get_relevant_documents(self, query):
        return self.vectorstore.similarity_search(query, k=self.k)

    def invoke(self, query):
        return self.get_relevant_documents(query)


def load_chroma_vectorstore(
    persist_dir,
    embeddings=None,
    collection_name="finance_rag",
):
    import chromadb

    persist_dir = Path(persist_dir)
    embeddings = embeddings or DeterministicFakeEmbeddings()
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    return ChromaVectorStore(
        collection=collection,
        embeddings=embeddings,
    )


def _sanitize_document(document):
    metadata = {
        key: value
        for key, value in document.metadata.items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }
    document.metadata = metadata
    return document


def _char_tokens(text):
    compact = "".join(str(text).split())
    chars = list(compact)
    bigrams = [compact[index : index + 2] for index in range(max(len(compact) - 1, 0))]
    return chars + bigrams


def _as_float_list(vector):
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]
