from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.exception_handlers import request_validation_exception_handler

from src.config import PROJECT_ROOT, get_settings
from src.features import ensure_dataframe
from src.logging_utils import configure_logging
from src.schemas import PredictionItem, PredictionRequest, PredictionResponse


LOGGER = logging.getLogger(__name__)


class MetricsStore:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.last_latency_ms = 0.0

    def render(self) -> str:
        uptime = time.time() - self.started_at
        lines = [
            "# HELP app_requests_total Total HTTP requests handled",
            "# TYPE app_requests_total counter",
            f"app_requests_total {self.request_count}",
            "# HELP app_prediction_items_total Total Jira issues scored by the model",
            "# TYPE app_prediction_items_total counter",
            f"app_prediction_items_total {self.prediction_count}",
            "# HELP app_errors_total Total application errors",
            "# TYPE app_errors_total counter",
            f"app_errors_total {self.error_count}",
            "# HELP app_last_request_latency_ms Latency of the last handled request in milliseconds",
            "# TYPE app_last_request_latency_ms gauge",
            f"app_last_request_latency_ms {self.last_latency_ms:.2f}",
            "# HELP app_uptime_seconds Service uptime in seconds",
            "# TYPE app_uptime_seconds gauge",
            f"app_uptime_seconds {uptime:.2f}",
        ]
        return "\n".join(lines) + "\n"


def _risk_bucket(probability: float) -> str:
    if probability >= 0.66:
        return "high"
    if probability >= 0.38:
        return "medium"
    return "low"


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


settings = get_settings()
configure_logging(settings.log_level)
metrics_store = MetricsStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = PROJECT_ROOT / settings.model_path if not settings.model_path.is_absolute() else settings.model_path
    metadata_path = (
        PROJECT_ROOT / settings.model_metadata_path
        if not settings.model_metadata_path.is_absolute()
        else settings.model_metadata_path
    )
    if not model_path.exists() or not metadata_path.exists():
        raise RuntimeError("Model artifacts are missing. Run `python -m src.train` first.")

    app.state.model = joblib.load(model_path)
    app.state.metadata = load_metadata(metadata_path)
    LOGGER.info("Model loaded from %s", model_path)
    LOGGER.info("Service started with model=%s version=%s", app.state.metadata["best_model_name"], settings.model_version)
    yield


app = FastAPI(title="Jira Issue Delay Risk Service", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    metrics_store.error_count += 1
    LOGGER.warning("%s %s validation failed: %s", request.method, request.url.path, exc.errors())
    return await request_validation_exception_handler(request, exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    model_name = getattr(app.state, "metadata", {}).get("best_model_name", "not_loaded")
    threshold = getattr(app.state, "metadata", {}).get("delay_threshold_days", "unknown")
    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Jira Issue Delay Risk Service</title>
        <style>
            body {{
                margin: 0;
                font-family: "Segoe UI", Tahoma, sans-serif;
                background: #f5f7fb;
                color: #111827;
            }}
            main {{
                width: min(1040px, calc(100% - 32px));
                margin: 0 auto;
                padding: 28px 0;
            }}
            section {{
                background: white;
                border: 1px solid #dbe3ef;
                border-radius: 16px;
                padding: 22px;
                margin-bottom: 18px;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            }}
            h1 {{ margin: 0 0 10px; }}
            p {{ color: #4b5563; line-height: 1.55; }}
            textarea, pre {{
                width: 100%;
                min-height: 260px;
                box-sizing: border-box;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 14px;
                font-family: Consolas, monospace;
                font-size: 13px;
            }}
            pre {{ background: #111827; color: #e5e7eb; overflow: auto; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
            .toolbar {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }}
            button, a {{
                border: 0;
                border-radius: 10px;
                padding: 10px 14px;
                background: #2563eb;
                color: white;
                text-decoration: none;
                font-weight: 650;
                cursor: pointer;
            }}
            button.secondary, a.secondary {{ background: #e0e7ff; color: #1e40af; }}
            @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr; }} }}
        </style>
    </head>
    <body>
        <main>
            <section>
                <h1>Jira Issue Delay Risk Service</h1>
                <p>
                    API-сервис прогнозирует риск долгого закрытия Jira issue по признакам,
                    известным на момент создания задачи. Целевая переменная построена
                    по реальному времени между Created и Resolved.
                </p>
                <p>Активная модель: <strong>{model_name}</strong></p>
                <p>Порог долгого закрытия: <strong>{threshold}</strong> дней.</p>
                <div class="toolbar">
                    <a href="/docs">Swagger UI</a>
                    <a class="secondary" href="/health">/health</a>
                    <a class="secondary" href="/metrics">/metrics</a>
                </div>
            </section>
            <section>
                <h2>Проверка /predict</h2>
                <div class="toolbar">
                    <button onclick="loadDemoPayload()">Load demo payload</button>
                    <button class="secondary" onclick="runPrediction()">Run prediction</button>
                    <button class="secondary" onclick="checkHealth()">Check health</button>
                    <button class="secondary" onclick="loadMetrics()">Load metrics</button>
                </div>
                <div class="grid">
                    <textarea id="payloadEditor"></textarea>
                    <pre id="responseViewer">Нажми Load demo payload, затем Run prediction.</pre>
                </div>
            </section>
        </main>
        <script>
            const demoPayload = {{
              "issues": [
                {{
                  "issue_type": "Bug",
                  "priority": "High",
                  "has_priority": true,
                  "component_present": true,
                  "summary_length": 92,
                  "summary_word_count": 12,
                  "description_length": 1840,
                  "description_word_count": 245
                }},
                {{
                  "issue_type": "Suggestion",
                  "priority": "Low",
                  "has_priority": true,
                  "component_present": true,
                  "summary_length": 44,
                  "summary_word_count": 6,
                  "description_length": 380,
                  "description_word_count": 58
                }}
              ]
            }};

            const payloadEditor = document.getElementById("payloadEditor");
            const responseViewer = document.getElementById("responseViewer");

            function renderOutput(title, payload) {{
                responseViewer.textContent = `${{title}}\\n\\n${{typeof payload === "string" ? payload : JSON.stringify(payload, null, 2)}}`;
            }}
            function loadDemoPayload() {{
                payloadEditor.value = JSON.stringify(demoPayload, null, 2);
                renderOutput("Demo payload loaded", demoPayload);
            }}
            async function checkHealth() {{
                const response = await fetch("/health");
                renderOutput("GET /health", await response.json());
            }}
            async function loadMetrics() {{
                const response = await fetch("/metrics");
                renderOutput("GET /metrics", await response.text());
            }}
            async function runPrediction() {{
                try {{
                    const response = await fetch("/predict", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: payloadEditor.value
                    }});
                    renderOutput(`POST /predict [${{response.status}}]`, await response.json());
                }} catch (error) {{
                    renderOutput("Client error", String(error));
                }}
            }}
            loadDemoPayload();
        </script>
    </body>
    </html>
    """


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = time.perf_counter()
    metrics_store.request_count += 1
    try:
        response = await call_next(request)
    except Exception:
        metrics_store.error_count += 1
        LOGGER.exception("Unhandled error while processing %s %s", request.method, request.url.path)
        raise
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    metrics_store.last_latency_ms = elapsed_ms
    LOGGER.info("%s %s -> %s in %.2f ms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "environment": settings.app_env,
        "model_loaded": hasattr(app.state, "model"),
        "model_name": app.state.metadata["best_model_name"],
        "model_version": settings.model_version,
        "dataset": app.state.metadata.get("source_dataset_path"),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return metrics_store.render()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        frame = ensure_dataframe([item.model_dump() for item in payload.issues])
        probabilities = app.state.model.predict_proba(frame)[:, 1]
        predictions = [
            PredictionItem(
                prediction=int(probability >= 0.5),
                probability=round(float(probability), 4),
                is_delayed=int(probability >= 0.5),
                delay_probability=round(float(probability), 4),
                delay_risk=_risk_bucket(float(probability)),
            )
            for probability in probabilities
        ]
        LOGGER.info("/predict scored %s issue(s)", len(predictions))
        metrics_store.prediction_count += len(predictions)
        return PredictionResponse(
            model_name=app.state.metadata["best_model_name"],
            model_version=settings.model_version,
            predictions=predictions,
        )
    except ValueError as exc:
        metrics_store.error_count += 1
        LOGGER.warning("Validation-like error during prediction: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        metrics_store.error_count += 1
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


def main() -> None:
    import uvicorn

    uvicorn.run("src.service:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    main()
