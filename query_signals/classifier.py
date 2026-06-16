import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


if importlib.util.find_spec("sentence_transformers") is None:
    raise ImportError(
        "query_signals requires sentence-transformers. "
        "Install with: pip install sentence-transformers"
    )


class EmbeddingKNNClassifier:
    """Embedding k-NN classifier with a process-wide shared MiniLM model."""

    _model = None
    _model_name = "sentence-transformers/all-MiniLM-L6-v2"
    ambiguity_confidence_threshold = 0.5
    ambiguity_margin_threshold = 0.15

    @classmethod
    def _get_model(cls):
        if EmbeddingKNNClassifier._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError(
                    "query_signals requires sentence-transformers. "
                    "Install with: pip install sentence-transformers"
                ) from error
            EmbeddingKNNClassifier._model = SentenceTransformer(
                EmbeddingKNNClassifier._model_name
            )
        return EmbeddingKNNClassifier._model

    def __init__(
        self,
        examples_path: str | Path,
        cache_path: str | Path,
        k: int = 5,
        ambiguity_confidence_threshold: float = 0.5,
        ambiguity_margin_threshold: float = 0.15,
    ):
        self.examples_path = Path(examples_path)
        self.cache_path = Path(cache_path)
        self.k = k
        self.ambiguity_confidence_threshold = (
            ambiguity_confidence_threshold
        )
        self.ambiguity_margin_threshold = ambiguity_margin_threshold
        self.examples_by_class = self._load_examples()
        self.all_classes = sorted(self.examples_by_class)
        self.texts = []
        self.labels = []
        for label in self.all_classes:
            for text in self.examples_by_class[label]:
                self.texts.append(text)
                self.labels.append(label)
        self.embeddings = None
        self.cache_hit = False

    def _load_examples(self) -> dict[str, list[str]]:
        if not self.examples_path.exists():
            raise FileNotFoundError(
                f"Query examples file not found: {self.examples_path}"
            )
        try:
            data = json.loads(self.examples_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed query examples JSON at {self.examples_path}: {error}"
            ) from error
        except OSError as error:
            raise ValueError(
                f"Could not read query examples at {self.examples_path}: {error}"
            ) from error

        if not isinstance(data, dict) or not data:
            raise ValueError(
                f"Query examples must be a non-empty object: {self.examples_path}"
            )

        validated = {}
        for label, examples in data.items():
            if not isinstance(label, str) or not label:
                raise ValueError("Query example labels must be non-empty strings")
            if not isinstance(examples, list) or not examples:
                raise ValueError(
                    f"Examples for {label!r} must be a non-empty list"
                )
            if any(
                not isinstance(text, str) or not text.strip()
                for text in examples
            ):
                raise ValueError(
                    f"Examples for {label!r} must be non-empty strings"
                )
            validated[label] = examples
        return validated

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(self._model_name.encode("utf-8"))
        digest.update(b"\0")
        for label, text in zip(self.labels, self.texts):
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _load_cache(self):
        if not self.cache_path.exists():
            return None
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                embeddings = np.asarray(data["embeddings"])
                cache_key = str(data["cache_key"].item())
            if cache_key != self._cache_key():
                return None
            if embeddings.ndim != 2 or len(embeddings) != len(self.labels):
                return None
            self.cache_hit = True
            return embeddings
        except (OSError, ValueError, KeyError):
            return None

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                embeddings=np.asarray(self.embeddings),
                cache_key=np.asarray(self._cache_key()),
            )
            os.replace(temporary, self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _normalize_embeddings(embeddings):
        array = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-10)

    def _ensure_embeddings(self) -> None:
        if self.embeddings is not None:
            return
        self.embeddings = self._load_cache()
        if self.embeddings is None:
            model = self._get_model()
            self.embeddings = model.encode(
                self.texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.embeddings = self._normalize_embeddings(self.embeddings)
            self._save_cache()
        else:
            self.embeddings = self._normalize_embeddings(self.embeddings)

    def rebuild_cache(self) -> None:
        self.cache_hit = False
        model = self._get_model()
        self.embeddings = model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embeddings = self._normalize_embeddings(self.embeddings)
        self._save_cache()

    def _scores_from_similarities(
        self,
        similarities: np.ndarray,
        excluded_index: int | None = None,
    ) -> dict:
        values = np.asarray(similarities, dtype=np.float32).copy()
        if excluded_index is not None:
            values[excluded_index] = -np.inf

        candidate_count = len(values) - (1 if excluded_index is not None else 0)
        count = min(self.k, candidate_count)
        if count <= 0:
            return self._uniform_result()

        indices = np.argpartition(values, -count)[-count:]
        indices = indices[np.argsort(values[indices])[::-1]]
        scores = {label: 0.0 for label in self.all_classes}
        for index in indices:
            scores[self.labels[int(index)]] += max(float(values[index]), 0.0)

        total = sum(scores.values())
        if total:
            scores = {
                label: value / total for label, value in scores.items()
            }
        else:
            uniform = 1.0 / len(self.all_classes)
            scores = {label: uniform for label in self.all_classes}
        scores = self._rounded_distribution(scores)
        return self._result_for_scores(scores)

    @staticmethod
    def _rounded_distribution(scores: dict[str, float]) -> dict[str, float]:
        rounded = {label: round(value, 4) for label, value in scores.items()}
        residual = round(1.0 - sum(rounded.values()), 4)
        if residual:
            largest = max(rounded, key=rounded.get)
            rounded[largest] = round(rounded[largest] + residual, 4)
        return rounded

    @classmethod
    def _prediction_result(
        cls,
        scores: dict[str, float],
        confidence_threshold: float | None = None,
        margin_threshold: float | None = None,
    ) -> dict:
        if confidence_threshold is None:
            confidence_threshold = cls.ambiguity_confidence_threshold
        if margin_threshold is None:
            margin_threshold = cls.ambiguity_margin_threshold
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        predicted, confidence = ranked[0]
        secondary, secondary_confidence = ranked[1]
        margin = round(confidence - secondary_confidence, 4)
        ambiguity_reasons = []
        if confidence < confidence_threshold:
            ambiguity_reasons.append("low_confidence")
        if margin < margin_threshold:
            ambiguity_reasons.append("narrow_margin")
        return {
            "scores": scores,
            "predicted": predicted,
            "confidence": confidence,
            "secondary": secondary,
            "secondary_confidence": secondary_confidence,
            "margin": margin,
            "ambiguous": bool(ambiguity_reasons),
            "ambiguity_reasons": ambiguity_reasons,
        }

    def _result_for_scores(self, scores: dict[str, float]) -> dict:
        return self._prediction_result(
            scores,
            confidence_threshold=self.ambiguity_confidence_threshold,
            margin_threshold=self.ambiguity_margin_threshold,
        )

    def _uniform_result(self) -> dict:
        uniform = 1.0 / len(self.all_classes)
        scores = self._rounded_distribution(
            {label: uniform for label in self.all_classes}
        )
        return self._result_for_scores(scores)

    def predict(self, query: str) -> dict:
        model = self._get_model()
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return self.predict_from_embedding(query_embedding)

    def predict_from_embedding(self, query_embedding: np.ndarray) -> dict:
        self._ensure_embeddings()
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        similarities = self.embeddings @ query
        return self._scores_from_similarities(similarities)

    def predict_loo(self, index: int) -> dict:
        self._ensure_embeddings()
        if index < 0 or index >= len(self.labels):
            raise IndexError(index)
        similarities = self.embeddings @ self.embeddings[index]
        return self._scores_from_similarities(
            similarities,
            excluded_index=index,
        )


class EmbeddingLogisticClassifier(EmbeddingKNNClassifier):
    """Regularized logistic head over frozen, normalized embeddings."""

    def __init__(
        self,
        examples_path: str | Path,
        cache_path: str | Path,
        model_cache_path: str | Path,
        c_value: float,
        ambiguity_confidence_threshold: float = 0.5,
        ambiguity_margin_threshold: float = 0.15,
    ):
        super().__init__(
            examples_path,
            cache_path,
            ambiguity_confidence_threshold=ambiguity_confidence_threshold,
            ambiguity_margin_threshold=ambiguity_margin_threshold,
        )
        self.model_cache_path = Path(model_cache_path)
        self.c_value = c_value
        self.coef_ = None
        self.intercept_ = None
        self.model_classes_ = None
        self.model_cache_hit = False

    def _model_cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(self._cache_key().encode("ascii"))
        digest.update(f"\0logistic\0{self.c_value}".encode("ascii"))
        return digest.hexdigest()

    def _load_model_cache(self) -> bool:
        if not self.model_cache_path.exists():
            return False
        try:
            with np.load(self.model_cache_path, allow_pickle=False) as data:
                cache_key = str(data["cache_key"].item())
                coef = np.asarray(data["coef"], dtype=np.float32)
                intercept = np.asarray(data["intercept"], dtype=np.float32)
                classes = np.asarray(data["classes"]).astype(str)
            if cache_key != self._model_cache_key():
                return False
            expected_rows = 1 if len(classes) == 2 else len(classes)
            if coef.shape != (expected_rows, self.embeddings.shape[1]):
                return False
            if intercept.shape != (expected_rows,):
                return False
            if set(classes) != set(self.all_classes):
                return False
            self.coef_ = coef
            self.intercept_ = intercept
            self.model_classes_ = classes
            self.model_cache_hit = True
            return True
        except (OSError, ValueError, KeyError):
            return False

    def _save_model_cache(self) -> None:
        self.model_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.model_cache_path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                coef=self.coef_,
                intercept=self.intercept_,
                classes=self.model_classes_,
                cache_key=np.asarray(self._model_cache_key()),
            )
            os.replace(temporary, self.model_cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_prediction_model(self) -> None:
        self._ensure_embeddings()
        if self.coef_ is not None:
            return
        if self._load_model_cache():
            return
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=self.c_value,
            max_iter=5000,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(self.embeddings, np.asarray(self.labels))
        self.coef_ = np.asarray(model.coef_, dtype=np.float32)
        self.intercept_ = np.asarray(model.intercept_, dtype=np.float32)
        self.model_classes_ = np.asarray(model.classes_)
        self._save_model_cache()

    def rebuild_cache(self) -> None:
        super().rebuild_cache()
        self.coef_ = None
        self.intercept_ = None
        self.model_classes_ = None
        self.model_cache_hit = False
        self._ensure_prediction_model()

    def predict_from_embedding(self, query_embedding: np.ndarray) -> dict:
        self._ensure_prediction_model()
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        logits = self.coef_ @ query + self.intercept_
        if len(self.model_classes_) == 2 and len(logits) == 1:
            positive = 1.0 / (1.0 + np.exp(-logits[0]))
            probabilities = np.asarray([1.0 - positive, positive])
        else:
            logits = logits - logits.max()
            probabilities = np.exp(logits)
            probabilities = probabilities / probabilities.sum()
        raw_scores = {
            str(label): float(probability)
            for label, probability in zip(
                self.model_classes_,
                probabilities,
            )
        }
        scores = self._rounded_distribution(
            {label: raw_scores.get(label, 0.0) for label in self.all_classes}
        )
        return self._result_for_scores(scores)


class EmbeddingMultiLabelLogisticClassifier:
    """Independent sigmoid logistic heads over one shared text embedding."""

    def __init__(
        self,
        examples_path: str | Path,
        cache_path: str | Path,
        model_cache_path: str | Path,
        labels: list[str],
        c_value: float = 3.0,
        thresholds: float | dict[str, float] = 0.5,
    ):
        self.examples_path = Path(examples_path)
        self.cache_path = Path(cache_path)
        self.model_cache_path = Path(model_cache_path)
        self.all_classes = list(labels)
        self.c_value = c_value
        self.thresholds = (
            {label: float(thresholds) for label in self.all_classes}
            if isinstance(thresholds, (int, float))
            else {
                label: float(thresholds.get(label, 0.5))
                for label in self.all_classes
            }
        )
        self.records = self._load_examples()
        self.texts = [record["text"] for record in self.records]
        self.targets = np.asarray(
            [
                [int(label in record["labels"]) for label in self.all_classes]
                for record in self.records
            ],
            dtype=np.int8,
        )
        self.embeddings = None
        self.coef_ = None
        self.intercept_ = None
        self.cache_hit = False
        self.model_cache_hit = False

    def _load_examples(self) -> list[dict]:
        if not self.examples_path.exists():
            raise FileNotFoundError(
                f"Query concern examples file not found: {self.examples_path}"
            )
        try:
            data = json.loads(self.examples_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed query concern JSON at {self.examples_path}: {error}"
            ) from error
        if not isinstance(data, list) or not data:
            raise ValueError("Query concern examples must be a non-empty list")
        allowed = set(self.all_classes)
        records = []
        for record in data:
            if not isinstance(record, dict):
                raise ValueError("Each query concern example must be an object")
            text = record.get("text")
            labels = record.get("labels")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Query concern text must be non-empty")
            if (
                not isinstance(labels, list)
                or any(label not in allowed for label in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError(
                    f"Invalid concern labels for query {text!r}"
                )
            records.append({"text": text, "labels": labels})
        return records

    def _cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(EmbeddingKNNClassifier._model_name.encode("utf-8"))
        digest.update(b"\0multilabel-concern-v1\0")
        for label in self.all_classes:
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
        for record in self.records:
            digest.update(record["text"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(",".join(sorted(record["labels"])).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _load_embedding_cache(self):
        if not self.cache_path.exists():
            return None
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)
                cache_key = str(data["cache_key"].item())
            if cache_key != self._cache_key():
                return None
            if embeddings.ndim != 2 or len(embeddings) != len(self.texts):
                return None
            self.cache_hit = True
            return EmbeddingKNNClassifier._normalize_embeddings(embeddings)
        except (OSError, ValueError, KeyError):
            return None

    def _save_embedding_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                embeddings=np.asarray(self.embeddings),
                cache_key=np.asarray(self._cache_key()),
            )
            os.replace(temporary, self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_embeddings(self) -> None:
        if self.embeddings is not None:
            return
        self.embeddings = self._load_embedding_cache()
        if self.embeddings is None:
            model = EmbeddingKNNClassifier._get_model()
            self.embeddings = model.encode(
                self.texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.embeddings = EmbeddingKNNClassifier._normalize_embeddings(
                self.embeddings
            )
            self._save_embedding_cache()

    def _model_cache_key(self) -> str:
        return hashlib.sha256(
            f"{self._cache_key()}\0ovr-logistic\0{self.c_value}".encode("ascii")
        ).hexdigest()

    def _load_model_cache(self) -> bool:
        if not self.model_cache_path.exists():
            return False
        try:
            with np.load(self.model_cache_path, allow_pickle=False) as data:
                cache_key = str(data["cache_key"].item())
                coef = np.asarray(data["coef"], dtype=np.float32)
                intercept = np.asarray(data["intercept"], dtype=np.float32)
                classes = np.asarray(data["classes"]).astype(str).tolist()
            if cache_key != self._model_cache_key():
                return False
            if classes != self.all_classes:
                return False
            if coef.shape != (len(self.all_classes), self.embeddings.shape[1]):
                return False
            if intercept.shape != (len(self.all_classes),):
                return False
            self.coef_ = coef
            self.intercept_ = intercept
            self.model_cache_hit = True
            return True
        except (OSError, ValueError, KeyError):
            return False

    def _save_model_cache(self) -> None:
        self.model_cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.model_cache_path.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                coef=self.coef_,
                intercept=self.intercept_,
                classes=np.asarray(self.all_classes),
                cache_key=np.asarray(self._model_cache_key()),
            )
            os.replace(temporary, self.model_cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_prediction_model(self) -> None:
        self._ensure_embeddings()
        if self.coef_ is not None:
            return
        if self._load_model_cache():
            return
        from sklearn.linear_model import LogisticRegression

        coefficients = []
        intercepts = []
        for index in range(len(self.all_classes)):
            model = LogisticRegression(
                C=self.c_value,
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=42,
            )
            model.fit(self.embeddings, self.targets[:, index])
            coefficients.append(model.coef_[0])
            intercepts.append(model.intercept_[0])
        self.coef_ = np.asarray(coefficients, dtype=np.float32)
        self.intercept_ = np.asarray(intercepts, dtype=np.float32)
        self._save_model_cache()

    def rebuild_cache(self) -> None:
        self.embeddings = None
        self.coef_ = None
        self.intercept_ = None
        self.cache_hit = False
        self.model_cache_hit = False
        self._ensure_prediction_model()

    def predict_from_embedding(self, query_embedding: np.ndarray) -> dict:
        self._ensure_prediction_model()
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        logits = self.coef_ @ query + self.intercept_
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        scores = {
            label: round(float(probability), 4)
            for label, probability in zip(self.all_classes, probabilities)
        }
        detected = [
            label
            for label in sorted(
                self.all_classes,
                key=lambda item: scores[item],
                reverse=True,
            )
            if scores[label] >= self.thresholds[label]
        ]
        primary = (
            max(self.all_classes, key=scores.get)
            if self.all_classes
            else "none"
        )
        return {
            "scores": scores,
            "detected": detected,
            "primary": primary if detected else "none",
            "confidence": scores[primary] if detected else 0.0,
            "thresholds": dict(self.thresholds),
            "multi_label": True,
            "method": "embedding_logistic_one_vs_rest",
        }
