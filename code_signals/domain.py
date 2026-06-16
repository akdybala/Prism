import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .node_types import IMPORT_FROM_STATEMENT, IMPORT_STATEMENT
from .parser import get_node_text, parse, walk


ALL_DOMAINS = [
    "cryptography",
    "systems_programming",
    "machine_learning",
    "concurrency",
    "algorithms",
    "database",
    "backend_api",
    "frontend",
    "css_styling",
    "config_boilerplate",
    "general_python",
    "unknown",
]
DEFAULT_EMBEDDING_MODEL = "nomic-ai/CodeRankEmbed"
DEFAULT_EMBEDDING_REPRESENTATION = "tree_sitter_v7"
DEFAULT_EMBEDDING_SIMILARITY_POWER = 3.0
EMBEDDING_REPRESENTATION_VERSIONS = {
    "full_code": "1",
    "ranked_sketch": "6",
    "tree_sitter_v7": "4",
}


class _FallbackSketchTokenizer:
    def encode(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
    ):
        return re.findall(r"\w+|[^\w\s]", text)

IMPORT_FINGERPRINTS = {
    "cryptography": {
        "cryptography", "hashlib", "hmac", "ssl", "jwt", "bcrypt",
        "passlib", "secrets", "argon2", "nacl",
    },
    "systems_programming": {
        "ctypes", "cffi", "socket", "selectors", "signal", "mmap",
        "resource", "fcntl", "termios", "struct",
    },
    "machine_learning": {
        "torch", "tensorflow", "keras", "sklearn", "xgboost",
        "transformers", "datasets", "wandb", "mlflow", "numpy",
    },
    "concurrency": {
        "asyncio", "threading", "multiprocessing", "concurrent", "trio",
        "curio", "uvloop", "aiofiles",
    },
    "algorithms": {"heapq", "bisect", "collections"},
    "database": {
        "sqlalchemy", "psycopg2", "pymongo", "redis", "alembic",
        "peewee", "sqlite3", "mysql", "cassandra", "elasticsearch",
    },
    "backend_api": {
        "flask", "django", "fastapi", "starlette", "uvicorn", "sanic",
        "aiohttp", "rest_framework",
    },
    "frontend": {
        "streamlit", "gradio", "dash", "panel", "reflex", "pywebio",
        "nicegui", "tkinter",
    },
    "css_styling": {"cssutils", "sass", "scss", "lesscpy", "tinycss2"},
    "config_boilerplate": {
        "configparser", "dotenv", "pydantic_settings", "yaml", "tomllib",
        "logging",
    },
    "general_python": set(),
    "unknown": set(),
}

API_FINGERPRINTS = {
    "cryptography": {
        "hashlib.", "hmac.", "bcrypt.", "jwt.encode", "jwt.decode", "encrypt(",
        "decrypt(", "sign(", "verify(", "token_bytes(", "compare_digest(",
    },
    "systems_programming": {
        "ctypes.", "socket.socket", "struct.pack", "struct.unpack", "mmap.",
        "os.read(", "os.write(", "signal.signal", "memoryview(",
    },
    "machine_learning": {
        "model.fit", "model.predict", "model.eval", "model.train", ".backward()",
        "optimizer.step", "DataLoader(", "torch.no_grad", "pipeline(",
    },
    "concurrency": {
        "await ", "async for", "async with", "asyncio.gather",
        "asyncio.create_task", "lock.acquire", "queue.put", "pool.submit",
        "Semaphore(", "Thread(", "Process(",
    },
    "algorithms": {
        "stack.pop", "stack.append", "heapq.heappush", "heapq.heappop",
        "bisect.", "collections.deque", "defaultdict(", "visited",
    },
    "database": {
        "cursor.execute", "session.query", "session.commit", "session.add",
        "fetchone", "fetchall", ".filter(", ".values(", "bulk_create",
    },
    "backend_api": {
        "request.get", "request.post", "request.json", "jsonify",
        "HTTPException", "status_code", "@app.", "@router.", "Response(",
    },
    "frontend": {
        "st.", "gr.", "ui.", "render_template", "Button(", "Label(",
        "page.add", "app.layout", "launch(",
    },
    "css_styling": {
        "stylesheet", "cssText", "style.", "background-color", "font-family",
        "border-radius", "@media", "className",
    },
    "config_boilerplate": {
        "ConfigParser(", "load_dotenv(", "logging.basicConfig",
        "BaseSettings", "os.getenv(", "yaml.safe_load", "tomllib.load",
    },
    "general_python": set(),
    "unknown": set(),
}

VARIABLE_FINGERPRINTS = {
    "cryptography": {
        r"\bnonce\b", r"\bciphertext\b", r"\bplaintext\b", r"\bsalt\b",
        r"\bprivate_key\b", r"\bpublic_key\b",
    },
    "systems_programming": {
        r"\bbuffer\b", r"\bpointer\b", r"\baddress\b", r"\bdescriptor\b",
        r"\bpid\b", r"\bsignal\b",
    },
    "machine_learning": {
        r"\bX_train\b", r"\by_pred\b", r"\bepochs?\b", r"\bbatch_size\b",
        r"\blearning_rate\b", r"\blogits\b", r"\bembedding\b",
    },
    "concurrency": {
        r"\block\b", r"\bmutex\b", r"\bsemaphore\b", r"\bfuture\b",
        r"\bworker\b", r"\bpool\b",
    },
    "algorithms": {
        r"\bleft\b", r"\bright\b", r"\bmid\b", r"\bvisited\b", r"\bstack\b",
        r"\bheap\b", r"\bgraph\b", r"\badj\b", r"\bdp\b", r"\bmemo\b",
    },
    "database": {
        r"\bcursor\b", r"\bquery\b", r"\btransaction\b", r"\brow\b",
        r"\btable\b", r"\bschema\b",
    },
    "backend_api": {
        r"\brequest\b", r"\bresponse\b", r"\brouter\b", r"\bmiddleware\b",
        r"\bpayload\b", r"\bendpoint\b",
    },
    "frontend": {
        r"\bwidget\b", r"\bcomponent\b", r"\blayout\b", r"\bbutton\b",
        r"\bview\b", r"\bpage\b",
    },
    "css_styling": {
        r"\bcolor\b", r"\bmargin\b", r"\bpadding\b", r"\bfont\b",
        r"\bstylesheet\b",
    },
    "config_boilerplate": {
        r"\bconfig\b", r"\bsettings\b", r"\benv\b", r"\bdebug\b",
        r"\blog_level\b",
    },
    "general_python": set(),
    "unknown": set(),
}


def is_unknown_code(code: str) -> bool:
    """Return whether code contains only placeholders or no implementation."""
    meaningful_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("@"):
            continue
        if stripped.endswith(":") and stripped.startswith(
            ("def ", "async def ", "class ")
        ):
            continue
        if stripped in {
            "pass",
            "...",
            "return None",
            "raise NotImplementedError",
            "raise NotImplementedError()",
        }:
            continue
        if re.fullmatch(
            r"[A-Za-z_]\w*\s*=\s*(?:None|True|False|\[\]|\{\}|\(\))",
            stripped,
        ):
            continue
        if re.fullmatch(r"(?:None|True|False|\d+(?:\.\d+)?|['\"].*['\"])", stripped):
            continue
        meaningful_lines.append(stripped)
    return not meaningful_lines


def extract_imports(root_node) -> set[str]:
    modules = set()
    for node in walk(root_node):
        if node.type not in {IMPORT_STATEMENT, IMPORT_FROM_STATEMENT}:
            continue
        text = get_node_text(node)
        if node.type == IMPORT_STATEMENT:
            match = re.match(r"\s*import\s+(.+)", text)
            if match:
                for name in match.group(1).split(","):
                    modules.add(name.strip().split()[0].split(".")[0])
        else:
            match = re.match(r"\s*from\s+([.\w]+)\s+import\b", text)
            if match:
                module = match.group(1).lstrip(".").split(".")[0]
                if module:
                    modules.add(module)
    return modules


def compute_rule_scores(code: str, root_node=None) -> dict[str, float]:
    root_node = root_node or parse(code).root_node
    imports = extract_imports(root_node)
    raw = {}
    for domain in ALL_DOMAINS:
        import_fp = IMPORT_FINGERPRINTS[domain]
        api_fp = API_FINGERPRINTS[domain]
        variable_fp = VARIABLE_FINGERPRINTS[domain]
        import_score = len(imports & import_fp) / max(len(import_fp), 1)
        api_score = sum(item in code for item in api_fp) / max(len(api_fp), 1)
        variable_score = sum(
            bool(re.search(pattern, code)) for pattern in variable_fp
        ) / max(len(variable_fp), 1)
        raw[domain] = max(import_score, api_score * 0.7, variable_score * 0.5)
    total = sum(raw.values())
    if not total:
        fallback = "unknown" if is_unknown_code(code) else "general_python"
        return {
            domain: 1.0 if domain == fallback else 0.0
            for domain in ALL_DOMAINS
        }
    return {domain: score / total for domain, score in raw.items()}


class DomainClassifier:
    def __init__(
        self,
        examples_path: str | Path = "domain_examples.json",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_representation: str = DEFAULT_EMBEDDING_REPRESENTATION,
        embedding_batch_size: int = 1,
        embedding_cache_dir: str | Path | None = None,
        rebuild_embedding_cache: bool = False,
    ):
        if embedding_representation not in EMBEDDING_REPRESENTATION_VERSIONS:
            raise ValueError(
                "embedding_representation must be 'full_code', "
                "'ranked_sketch', or 'tree_sitter_v7'"
            )
        self.examples_path = Path(examples_path)
        self.model_name = model_name
        self.embedding_representation = embedding_representation
        self.embedding_batch_size = embedding_batch_size
        self.embedding_cache_dir = (
            Path(embedding_cache_dir)
            if embedding_cache_dir is not None
            else self.examples_path.resolve().parent / ".code_signals_cache"
        )
        self.rebuild_embedding_cache = rebuild_embedding_cache
        self.examples = self._load_examples()
        self.example_counts = self._count_examples()
        self._model = None
        self._embeddings = None
        self._model_failed = False
        self.embedding_error = None
        self.embedding_cache_hit = False

    def _load_examples(self) -> list[dict]:
        if not self.examples_path.exists():
            return []
        try:
            data = json.loads(self.examples_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(data, dict):
            return []
        examples = []
        for domain in ALL_DOMAINS:
            if domain == "unknown":
                continue
            records = data.get(domain, [])
            if not isinstance(records, list):
                continue
            valid = [
                record
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("code"), str)
                and record["code"].strip()
            ]
            for record in valid:
                examples.append(
                    {
                        "name": record.get("name", ""),
                        "description": record.get("description", ""),
                        "code": record["code"],
                        "domain": domain,
                    }
                )
        return examples

    def _count_examples(self) -> dict[str, int]:
        counts = {domain: 0 for domain in ALL_DOMAINS}
        for example in self.examples:
            domain = example["domain"]
            if domain in counts:
                counts[domain] += 1
        return counts

    def _embedding_cache_key(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.embedding_representation.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            EMBEDDING_REPRESENTATION_VERSIONS[
                self.embedding_representation
            ].encode("utf-8")
        )
        digest.update(b"\0")
        for example in self.examples:
            digest.update(example["domain"].encode("utf-8"))
            digest.update(b"\0")
            digest.update(example["code"].encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _embedding_text(self, code: str, root_node=None) -> str:
        if self.embedding_representation == "full_code":
            return code

        root_node = root_node or parse(code).root_node
        if self.embedding_representation == "tree_sitter_v7":
            from .tree_sitter_coherent_sketch import (
                build_tree_sitter_coherent_v7_sketch,
            )

            tokenizer = (
                self._model.tokenizer
                if self._model is not None
                and getattr(self._model, "tokenizer", None) is not None
                else _FallbackSketchTokenizer()
            )
            sketch, _ = build_tree_sitter_coherent_v7_sketch(
                code,
                tokenizer,
                root_node=root_node,
            )
            return sketch

        # Imported lazily because domain_sketch uses the API fingerprints
        # declared in this module.
        from .domain_sketch import build_domain_sketch

        return build_domain_sketch(
            code,
            root_node,
            call_format="comment",
            logic_selection="ranked",
        )

    @property
    def embedding_cache_path(self) -> Path:
        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", self.model_name)
        return self.embedding_cache_dir / (
            f"{model_slug}-{self._embedding_cache_key()[:16]}.npz"
        )

    def _load_embedding_cache(self):
        if self.rebuild_embedding_cache:
            return None
        path = self.embedding_cache_path
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as cached:
                embeddings = np.asarray(cached["embeddings"])
                cache_key = str(cached["cache_key"].item())
            if cache_key != self._embedding_cache_key():
                return None
            if embeddings.ndim != 2 or len(embeddings) != len(self.examples):
                return None
            self.embedding_cache_hit = True
            return embeddings
        except (OSError, ValueError, KeyError):
            return None

    def _save_embedding_cache(self) -> None:
        path = self.embedding_cache_path
        temporary = path.with_suffix(".tmp.npz")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                temporary,
                embeddings=np.asarray(self._embeddings),
                cache_key=np.asarray(self._embedding_cache_key()),
            )
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)

    def _ensure_embeddings(self) -> bool:
        if not self.examples or self._model_failed:
            return False
        if self._embeddings is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
            )
            self._embeddings = self._load_embedding_cache()
            if self._embeddings is None:
                self._embeddings = self._model.encode(
                    [
                        self._embedding_text(example["code"])
                        for example in self.examples
                    ],
                    batch_size=self.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self._save_embedding_cache()
            return True
        except Exception as error:
            self._model_failed = True
            self._model = None
            self._embeddings = None
            self.embedding_error = error
            return False

    def compute_rule_scores(self, code: str, root_node=None):
        return compute_rule_scores(code, root_node)

    def compute_embedding_scores(self, code: str, k: int = 5, root_node=None):
        if is_unknown_code(code):
            return {
                domain: 1.0 if domain == "unknown" else 0.0
                for domain in ALL_DOMAINS
            }
        if not self._ensure_embeddings():
            return None
        embedding_text = self._embedding_text(code, root_node)
        query = self._model.encode(
            [embedding_text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        similarities = np.asarray(self._embeddings) @ np.asarray(query)
        count = min(k, len(similarities))
        indices = np.argsort(similarities)[-count:][::-1]
        votes = defaultdict(float)
        for index in indices:
            domain = self.examples[int(index)]["domain"]
            class_count = self.example_counts.get(domain, 0)
            if class_count:
                similarity = max(float(similarities[index]), 0.0)
                votes[domain] += (
                    similarity ** DEFAULT_EMBEDDING_SIMILARITY_POWER
                ) / class_count
        total = sum(votes.values())
        if not total:
            return {domain: 0.0 for domain in ALL_DOMAINS}
        return {
            domain: votes.get(domain, 0.0) / total for domain in ALL_DOMAINS
        }

    def predict(self, code: str, root_node=None) -> dict:
        rule_scores = self.compute_rule_scores(code, root_node)
        embedding_scores = self.compute_embedding_scores(
            code,
            root_node=root_node,
        )
        embedding_sketch = self._embedding_text(code, root_node)
        if embedding_scores is None:
            prediction_scores = rule_scores
        else:
            prediction_scores = embedding_scores

        predicted = max(prediction_scores, key=prediction_scores.get)
        confidence = prediction_scores[predicted]
        return {
            "rule_scores": rule_scores,
            "embedding_scores": embedding_scores,
            "predicted_domain": predicted,
            "confidence": round(confidence, 3),
            "embedding_representation": self.embedding_representation,
            "embedding_sketch": embedding_sketch,
        }
