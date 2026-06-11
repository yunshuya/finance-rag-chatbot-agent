import os
from pathlib import Path


def _normalize_scores(scores):
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


class BgeReranker:
    """Lazy-loaded wrapper around BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        model_name="BAAI/bge-reranker-v2-m3",
        cache_dir=None,
        use_fp16=False,
        model_factory=None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_fp16 = use_fp16
        self.model_factory = model_factory
        self._model = None

    def score_pairs(self, query, documents):
        if not documents:
            return []

        pairs = [[query, str(document.page_content)[:2000]] for document in documents]
        raw_scores = self._predict_scores(pairs)
        return _normalize_scores(raw_scores)

    def _predict_scores(self, pairs):
        model = self._model_instance()
        if hasattr(model, "predict"):
            scores = model.predict(pairs)
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            return [float(score) for score in scores]

        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(score) for score in scores]

    def _model_instance(self):
        if self._model is not None:
            return self._model

        if self.model_factory is not None:
            self._model = self.model_factory(self.model_name)
            return self._model

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(self.cache_dir))
            os.environ.setdefault("HF_HUB_CACHE", str(self.cache_dir / "hub"))

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception:
            pass

        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise ImportError(
                "BGE reranker requires sentence-transformers or FlagEmbedding."
            ) from exc

        kwargs = {"use_fp16": self.use_fp16}
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        try:
            self._model = FlagReranker(self.model_name, **kwargs)
        except TypeError:
            kwargs.pop("cache_dir", None)
            self._model = FlagReranker(self.model_name, **kwargs)
        return self._model
