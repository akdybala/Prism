#!/usr/bin/env python
"""Local web UI for inspecting code and query signals."""

import argparse
import json
import threading
import warnings
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MODEL_LOCK = threading.Lock()
_ROUTER_MODEL_CACHE = None
_ROUTER_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "router_training_data"
    / "trusted_v1"
    / "lightgbm_router"
    / "lightgbm_router.joblib"
)

DEFAULT_CODE = """\
from collections import deque


def shortest_path(graph, start, target):
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return -1
"""

DEFAULT_QUERY = "Can this be made faster without using extra memory?"

PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Signal Inspector</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #090b10;
      --panel: #11151d;
      --panel-2: #171c26;
      --border: #293140;
      --text: #eef2f8;
      --muted: #909bad;
      --accent: #7c9cff;
      --accent-2: #66d9bd;
      --warning: #f6c177;
      --danger: #ef7d8d;
      --shadow: 0 18px 50px rgba(0, 0, 0, 0.28);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 15% 0%, rgba(124, 156, 255, 0.12), transparent 30rem),
        radial-gradient(circle at 90% 25%, rgba(102, 217, 189, 0.08), transparent 25rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      max-width: 1500px;
      margin: 0 auto;
      padding: 36px 28px 22px;
    }

    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.6rem);
      letter-spacing: -0.055em;
      line-height: 1;
    }

    header p {
      max-width: 720px;
      margin: 14px 0 0;
      color: var(--muted);
      line-height: 1.6;
    }

    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 0 28px 50px;
      display: grid;
      gap: 24px;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }

    .router-panel {
      grid-column: 1 / -1;
    }

    .panel {
      background: linear-gradient(160deg, rgba(23, 28, 38, 0.96), rgba(14, 18, 25, 0.96));
      border: 1px solid var(--border);
      border-radius: 20px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--border);
    }

    .panel-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
    }

    .tag {
      padding: 4px 8px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 650;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }

    textarea {
      display: block;
      width: 100%;
      min-height: 390px;
      resize: vertical;
      border: 0;
      outline: 0;
      padding: 20px;
      background: rgba(4, 7, 11, 0.68);
      color: #dce6f5;
      font: 13px/1.65 "Cascadia Code", "SFMono-Regular", Consolas, monospace;
      tab-size: 4;
    }

    #query-input {
      min-height: 150px;
      font-family: inherit;
      font-size: 15px;
      line-height: 1.6;
    }

    .actions {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 18px;
      border-top: 1px solid var(--border);
    }

    button {
      border: 0;
      border-radius: 11px;
      padding: 10px 16px;
      background: var(--accent);
      color: #07101f;
      font-weight: 800;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }

    button:hover { transform: translateY(-1px); }
    button:disabled { cursor: wait; opacity: 0.55; transform: none; }

    .status {
      color: var(--muted);
      font-size: 0.82rem;
    }

    .status.error { color: var(--danger); }

    .results {
      padding: 18px;
      display: grid;
      gap: 16px;
    }

    .empty {
      padding: 34px 20px;
      border: 1px dashed var(--border);
      border-radius: 14px;
      color: var(--muted);
      text-align: center;
      line-height: 1.6;
    }

    .section {
      padding: 16px;
      background: rgba(9, 12, 18, 0.55);
      border: 1px solid var(--border);
      border-radius: 15px;
    }

    .section h3 {
      margin: 0 0 14px;
      font-size: 0.9rem;
      letter-spacing: 0.025em;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
      gap: 9px;
    }

    .metric {
      padding: 11px;
      border-radius: 11px;
      background: var(--panel-2);
      min-width: 0;
    }

    .metric-name {
      color: var(--muted);
      font-size: 0.72rem;
      overflow-wrap: anywhere;
    }

    .metric-value {
      margin-top: 5px;
      font: 700 1rem/1.2 "Cascadia Code", Consolas, monospace;
      overflow-wrap: anywhere;
    }

    .prediction {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 12px;
      margin-bottom: 14px;
    }

    .prediction strong {
      font-size: 1.15rem;
    }

    .badge {
      padding: 5px 9px;
      border-radius: 999px;
      background: rgba(102, 217, 189, 0.12);
      color: var(--accent-2);
      font-size: 0.76rem;
      font-weight: 750;
    }

    .badge.warning {
      background: rgba(246, 193, 119, 0.12);
      color: var(--warning);
    }

    .vector {
      display: grid;
      gap: 8px;
    }

    .vector-row {
      display: grid;
      grid-template-columns: minmax(118px, 0.8fr) minmax(100px, 2fr) 52px;
      align-items: center;
      gap: 10px;
      font-size: 0.76rem;
    }

    .vector-label {
      color: #c6cfdd;
      overflow-wrap: anywhere;
    }

    .track {
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #262d39;
    }

    .fill {
      height: 100%;
      min-width: 1px;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), #a99cff);
    }

    .vector-row.secondary .fill {
      background: linear-gradient(90deg, var(--accent-2), #80c7e8);
    }

    .vector-value {
      color: var(--muted);
      font-family: "Cascadia Code", Consolas, monospace;
      text-align: right;
    }

    details {
      border: 1px solid var(--border);
      border-radius: 13px;
      overflow: hidden;
    }

    summary {
      padding: 12px 14px;
      cursor: pointer;
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
    }

    pre {
      max-height: 440px;
      margin: 0;
      overflow: auto;
      padding: 16px;
      border-top: 1px solid var(--border);
      background: #080b10;
      color: #c9d4e5;
      font: 12px/1.55 "Cascadia Code", Consolas, monospace;
    }

    .sketch-code {
      max-height: 520px;
      border: 1px solid var(--border);
      border-radius: 11px;
      white-space: pre;
    }

    .hint {
      color: var(--muted);
      font-size: 0.75rem;
      margin-left: auto;
    }

    @media (max-width: 980px) {
      .workspace { grid-template-columns: 1fr; }
      .router-panel { grid-column: auto; }
      textarea { min-height: 310px; }
    }

    @media (max-width: 560px) {
      header, main { padding-left: 14px; padding-right: 14px; }
      .vector-row { grid-template-columns: 100px minmax(70px, 1fr) 48px; }
      .hint { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Signal Inspector</h1>
    <p>Inspect Python code and natural-language queries as the routing system sees them. Code embeddings use coherent Tree-sitter V7 sketches.</p>
  </header>

  <main>
    <div class="workspace">
      <section class="panel">
        <div class="panel-header">
          <div class="panel-title">Code input <span class="tag">Python</span></div>
          <span class="tag">CodeRankEmbed + V7</span>
        </div>
        <textarea id="code-input" spellcheck="false">__DEFAULT_CODE__</textarea>
        <div class="actions">
          <button id="code-button" type="button">Extract code signals</button>
          <span id="code-status" class="status">First embedding run may take a while.</span>
        </div>
        <div id="code-results" class="results">
          <div class="empty">Run extraction to see structural, semantic, rule-domain, and embedding-domain signals.</div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-header">
          <div class="panel-title">Query input <span class="tag">Natural language</span></div>
          <span class="tag">MiniLM</span>
        </div>
        <textarea id="query-input">__DEFAULT_QUERY__</textarea>
        <div class="actions">
          <button id="query-button" type="button">Extract query signals</button>
          <span id="query-status" class="status">Type and domain share one embedding.</span>
        </div>
        <div id="query-results" class="results">
          <div class="empty">Run extraction to see operation, concern, and domain signals.</div>
        </div>
      </section>
    </div>

    <section class="panel router-panel">
      <div class="panel-header">
        <div class="panel-title">Minimum Capable Router <span class="tag">LightGBM PoC</span></div>
        <span class="tag">light / medium / heavy</span>
      </div>
      <div class="actions">
        <button id="router-button" type="button">Predict minimum model tier</button>
        <span id="router-status" class="status">Uses generated signal vector from the query and code above.</span>
      </div>
      <div id="router-results" class="results">
        <div class="empty">Run the router model to see light, medium, and heavy probabilities.</div>
      </div>
    </section>
  </main>

  <script>
    const byId = (id) => document.getElementById(id);

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function metricGrid(values) {
      return `<div class="metrics">${Object.entries(values).map(([name, value]) => `
        <div class="metric">
          <div class="metric-name">${escapeHtml(name)}</div>
          <div class="metric-value">${escapeHtml(value)}</div>
        </div>`).join("")}</div>`;
    }

    function vector(scores, style = "") {
      if (!scores) {
        return `<div class="empty">Embedding vector unavailable; rule fallback was used.</div>`;
      }
      const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
      return `<div class="vector">${entries.map(([name, value]) => `
        <div class="vector-row ${style}">
          <div class="vector-label">${escapeHtml(name)}</div>
          <div class="track"><div class="fill" style="width:${Math.max(0, Math.min(100, value * 100))}%"></div></div>
          <div class="vector-value">${Number(value).toFixed(4)}</div>
        </div>`).join("")}</div>`;
    }

    function rawJson(data) {
      return `<details><summary>Raw JSON</summary><pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre></details>`;
    }

    function probabilityBadges(probabilities) {
      return Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([name, value]) => `<span class="badge">${escapeHtml(name)}: ${Number(value).toFixed(4)}</span>`)
        .join("");
    }

    function renderCode(data) {
      const domain = data.domain;
      const embeddingEntries = domain.embedding_scores
        ? Object.entries(domain.embedding_scores).sort((a, b) => b[1] - a[1])
        : [];
      const embeddingPrediction = embeddingEntries[0];
      const domainHeadline = embeddingPrediction
        ? `<strong>${escapeHtml(embeddingPrediction[0])}</strong>
            <span class="badge">${Number(embeddingPrediction[1]).toFixed(4)} embedding confidence</span>`
        : `<strong>Embedding unavailable</strong>
            <span class="badge warning">No embedding prediction</span>`;
      const syntax = data.has_errors
        ? `<span class="badge warning">${data.error_count} syntax issue${data.error_count === 1 ? "" : "s"}</span>`
        : `<span class="badge">Syntax parsed</span>`;
      const representation = domain.embedding_representation === "tree_sitter_v7"
        ? `<span class="badge">Tree-sitter V7</span>`
        : `<span class="badge warning">${escapeHtml(domain.embedding_representation || "unknown representation")}</span>`;
      byId("code-results").innerHTML = `
        <div class="section">
          <div class="prediction">
            ${domainHeadline}
            ${syntax}
            ${representation}
          </div>
        </div>
        <div class="section"><h3>Structural signals</h3>${metricGrid(data.structural)}</div>
        <div class="section"><h3>Data-flow signals</h3>${metricGrid(data.data_flow)}</div>
        <div class="section"><h3>Semantic signals</h3>${metricGrid(data.semantic)}</div>
        <div class="section">
          <h3>V7 embedding sketch</h3>
          <pre class="sketch-code">${escapeHtml(domain.embedding_sketch || "")}</pre>
        </div>
        <div class="section"><h3>Rule-domain vector</h3>${vector(domain.rule_scores)}</div>
        <div class="section"><h3>Embedding-domain vector</h3>${vector(domain.embedding_scores, "secondary")}</div>
        ${rawJson(data)}`;
    }

    function predictionSection(title, data, includeEvidence = false) {
      const ambiguity = data.ambiguous
        ? `<span class="badge warning">Ambiguous: ${escapeHtml(data.ambiguity_reasons.join(", "))}</span>`
        : `<span class="badge">Confident</span>`;
      const evidence = includeEvidence
        ? `<span class="badge ${data.domain_signal_present ? "" : "warning"}">Domain evidence: ${data.domain_signal_present ? "present" : "not explicit"}</span>`
        : "";
      return `<div class="section">
        <h3>${escapeHtml(title)}</h3>
        <div class="prediction">
          <strong>${escapeHtml(data.predicted)}</strong>
          <span class="badge">${Number(data.confidence).toFixed(4)} confidence</span>
          <span class="badge">Runner-up: ${escapeHtml(data.secondary)} (${Number(data.secondary_confidence).toFixed(4)})</span>
          <span class="badge">Margin: ${Number(data.margin).toFixed(4)}</span>
          ${ambiguity}
          ${evidence}
        </div>
        ${vector(data.scores)}
      </div>`;
    }

    function renderQuery(data) {
      const concerns = data.query_concerns;
      const concernSummary = concerns.detected.length
        ? concerns.detected.map((name) => `<span class="badge">${escapeHtml(name)} (${Number(concerns.scores[name]).toFixed(4)})</span>`).join("")
        : `<span class="badge">No explicit concern</span>`;
      byId("query-results").innerHTML = `
        ${predictionSection("Query-operation vector", data.query_operation)}
        <div class="section">
          <h3>Query concerns (multi-label)</h3>
          <div class="prediction">${concernSummary}</div>
          ${vector(concerns.scores, "secondary")}
        </div>
        ${predictionSection("Query-domain vector", data.query_domain, true)}
        ${rawJson(data)}`;
    }

    function renderRouter(data) {
      byId("router-results").innerHTML = `
        <div class="section">
          <h3>Minimum capable model</h3>
          <div class="prediction">
            <strong>${escapeHtml(data.predicted_label)}</strong>
            <span class="badge">${Number(data.confidence).toFixed(4)} confidence</span>
            <span class="badge">Feature columns: ${escapeHtml(data.feature_count)}</span>
            <span class="badge">Input tokens: ${escapeHtml(data.estimated_input_tokens)}</span>
          </div>
          ${vector(data.probabilities)}
        </div>
        <div class="section">
          <h3>Generated signal summary</h3>
          ${metricGrid({
            "query operation": data.generated_query_operation,
            "query domain": data.generated_query_domain,
            "code domain": data.generated_code_domain,
            "requires code": data.requires_code,
            "requires tools": data.requires_tools
          })}
        </div>
        <div class="section">
          <h3>Probabilities</h3>
          <div class="prediction">${probabilityBadges(data.probabilities)}</div>
        </div>
        ${rawJson(data)}`;
    }

    async function extract(kind) {
      const isCode = kind === "code";
      const button = byId(`${kind}-button`);
      const status = byId(`${kind}-status`);
      const value = byId(`${kind}-input`).value;
      button.disabled = true;
      status.className = "status";
      status.textContent = isCode
        ? "Extracting. CodeRankEmbed can be slow on CPU..."
        : "Extracting query signals...";
      const started = performance.now();
      try {
        const response = await fetch(`/api/${kind}`, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({[kind]: value})
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
        isCode ? renderCode(payload) : renderQuery(payload);
        status.textContent = `Completed in ${((performance.now() - started) / 1000).toFixed(2)}s`;
      } catch (error) {
        status.className = "status error";
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    }

    async function predictRouter() {
      const button = byId("router-button");
      const status = byId("router-status");
      button.disabled = true;
      status.className = "status";
      status.textContent = "Generating signals and scoring LightGBM router...";
      const started = performance.now();
      try {
        const response = await fetch("/api/router-model", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            query: byId("query-input").value,
            code: byId("code-input").value
          })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
        renderRouter(payload);
        status.textContent = `Completed in ${((performance.now() - started) / 1000).toFixed(2)}s`;
      } catch (error) {
        status.className = "status error";
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    }

    byId("code-button").addEventListener("click", () => extract("code"));
    byId("query-button").addEventListener("click", () => extract("query"));
    byId("router-button").addEventListener("click", () => predictRouter());

    document.querySelectorAll("textarea").forEach((textarea) => {
      textarea.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
          extract(textarea.id.startsWith("code") ? "code" : "query");
        }
      });
    });
  </script>
</body>
</html>
""".replace("__DEFAULT_CODE__", DEFAULT_CODE).replace(
    "__DEFAULT_QUERY__",
    DEFAULT_QUERY,
)


def extract_code_payload(code: str) -> dict:
    with _MODEL_LOCK:
        from code_signals import extract_all

        return extract_all(code)


def extract_query_payload(query: str) -> dict:
    with _MODEL_LOCK:
        from query_signals import extract_query_signals

        return extract_query_signals(query)


def extract_route_payload(
    query: str,
    code: str | None = None,
    *,
    context_tokens: int = 0,
    expected_output_tokens: int = 1200,
    quality_threshold: float = 0.75,
) -> dict:
    with _MODEL_LOCK:
        from routing import route_request

        return route_request(
            query,
            code,
            context_tokens=context_tokens,
            expected_output_tokens=expected_output_tokens,
            quality_threshold=quality_threshold,
        ).to_dict()


def _load_router_model_payload() -> dict:
    global _ROUTER_MODEL_CACHE
    if _ROUTER_MODEL_CACHE is None:
        if not _ROUTER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained router model not found: {_ROUTER_MODEL_PATH}"
            )
        import joblib

        _ROUTER_MODEL_CACHE = joblib.load(_ROUTER_MODEL_PATH)
    return _ROUTER_MODEL_CACHE


def extract_router_model_payload(query: str, code: str | None = None) -> dict:
    with _MODEL_LOCK:
        import numpy as np

        from code_signals import extract_all
        from query_signals import extract_query_signals
        from routing.features import build_routing_features

        model_payload = _load_router_model_payload()
        model = model_payload["model"]
        feature_names = model_payload["feature_names"]
        labels = model_payload["labels"]

        query_signals = extract_query_signals(query)
        code_signals = extract_all(code) if code and code.strip() else None
        features = build_routing_features(
            query_signals,
            code_signals,
            query=query,
            code=code,
        )
        row = {
            name: float(features.values.get(name.removeprefix("feature."), 0.0))
            for name in feature_names
        }
        matrix = np.asarray(
            [[row[name] for name in feature_names]],
            dtype=np.float32,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            probabilities = model.predict_proba(matrix)[0]
        best_index = int(np.argmax(probabilities))
        probability_map = {
            label: round(float(probabilities[index]), 6)
            for index, label in enumerate(labels)
        }
        return {
            "predicted_label": labels[best_index],
            "confidence": round(float(probabilities[best_index]), 6),
            "probabilities": probability_map,
            "feature_count": len(feature_names),
            "model_path": str(_ROUTER_MODEL_PATH),
            "generated_query_operation": features.query_operation,
            "generated_query_domain": features.query_domain,
            "generated_code_domain": features.code_domain,
            "estimated_input_tokens": features.estimated_input_tokens,
            "requires_code": features.requires_code,
            "requires_tools": features.requires_tools,
            "schema_version": features.to_dict()["schema_version"],
        }


def warm_code_model() -> bool:
    with _MODEL_LOCK:
        from code_signals.extractor import _domain_classifier

        return _domain_classifier._ensure_embeddings()


class SignalUIHandler(BaseHTTPRequestHandler):
    server_version = "SignalUI/1.0"

    def log_message(self, format_string, *args):
        print(f"{self.address_string()} - {format_string % args}")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in {
            "/api/code",
            "/api/query",
            "/api/route",
            "/api/router-model",
        }:
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid Content-Length"})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request body is empty or too large"})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON"})
            return

        if path == "/api/route":
            query = payload.get("query")
            code = payload.get("code")
            if not isinstance(query, str):
                self._send_json(400, {"error": "'query' must be a string"})
                return
            if code is not None and not isinstance(code, str):
                self._send_json(
                    400,
                    {"error": "'code' must be a string or null"},
                )
                return
            try:
                result = extract_route_payload(
                    query,
                    code,
                    context_tokens=int(payload.get("context_tokens", 0)),
                    expected_output_tokens=int(
                        payload.get("expected_output_tokens", 1200)
                    ),
                    quality_threshold=float(
                        payload.get("quality_threshold", 0.75)
                    ),
                )
            except (TypeError, ValueError) as error:
                self._send_json(400, {"error": str(error)})
                return
            except Exception as error:
                self._send_json(
                    500,
                    {
                        "error": f"Routing failed: "
                        f"{type(error).__name__}: {error}"
                    },
                )
                return
            self._send_json(200, result)
            return

        if path == "/api/router-model":
            query = payload.get("query")
            code = payload.get("code")
            if not isinstance(query, str):
                self._send_json(400, {"error": "'query' must be a string"})
                return
            if code is not None and not isinstance(code, str):
                self._send_json(
                    400,
                    {"error": "'code' must be a string or null"},
                )
                return
            try:
                result = extract_router_model_payload(query, code)
            except Exception as error:
                self._send_json(
                    500,
                    {
                        "error": f"Router model prediction failed: "
                        f"{type(error).__name__}: {error}"
                    },
                )
                return
            self._send_json(200, result)
            return

        key = "code" if path == "/api/code" else "query"
        value = payload.get(key)
        if not isinstance(value, str):
            self._send_json(400, {"error": f"{key!r} must be a string"})
            return

        try:
            result = (
                extract_code_payload(value)
                if key == "code"
                else extract_query_payload(value)
            )
        except Exception as error:
            self._send_json(
                500,
                {
                    "error": f"Signal extraction failed: "
                    f"{type(error).__name__}: {error}"
                },
            )
            return
        self._send_json(200, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local signal extraction UI.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the default browser automatically.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Start the server before loading the code embedding cache.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.skip_warmup:
        print("Loading CodeRankEmbed and Tree-sitter V7 sketch cache...", flush=True)
    if not args.skip_warmup and not warm_code_model():
        print(
            "Warning: CodeRankEmbed failed to load; code-domain predictions "
            "will use the rule fallback.",
            flush=True,
        )
    server = ThreadingHTTPServer((args.host, args.port), SignalUIHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Signal Inspector running at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Signal Inspector.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
