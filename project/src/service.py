from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

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
            "# HELP app_prediction_items_total Total tasks scored by the model",
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
    LOGGER.info("Service started with model=%s", app.state.metadata["best_model_name"])
    yield


app = FastAPI(title="Task Overdue Risk Service", version="1.0.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    model_name = getattr(app.state, "metadata", {}).get("best_model_name", "not_loaded")
    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Task Overdue Risk Service</title>
        <style>
            :root {{
                color-scheme: light;
                --bg: #f3f7f4;
                --card: rgba(255, 255, 255, 0.92);
                --text: #17212b;
                --muted: #5b6472;
                --accent: #0f766e;
                --accent-strong: #115e59;
                --accent-soft: #d8f3ee;
                --secondary: #1d4ed8;
                --secondary-soft: #dbeafe;
                --border: #d9e2ec;
                --shadow: 0 20px 50px rgba(15, 23, 42, 0.10);
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: "Segoe UI", Tahoma, sans-serif;
                background:
                    radial-gradient(circle at top left, rgba(34, 197, 94, 0.18) 0, transparent 25%),
                    radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.18) 0, transparent 30%),
                    linear-gradient(135deg, #f7fbf9 0%, #eef5ff 100%);
                color: var(--text);
            }}
            .wrap {{
                min-height: 100vh;
                padding: 28px;
            }}
            .shell {{
                width: min(1240px, 100%);
                margin: 0 auto;
                display: grid;
                gap: 20px;
            }}
            .hero {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 28px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(14px);
            }}
            .badge {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                background: var(--accent-soft);
                color: var(--accent);
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 16px;
            }}
            h1 {{
                margin: 0 0 12px;
                font-size: clamp(28px, 4vw, 42px);
                line-height: 1.1;
            }}
            p {{
                margin: 0 0 12px;
                color: var(--muted);
                font-size: 17px;
                line-height: 1.6;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
                margin: 24px 0;
            }}
            .tile {{
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 18px;
                background: rgba(251, 253, 255, 0.96);
            }}
            .tile strong {{
                display: block;
                margin-bottom: 8px;
            }}
            .layout {{
                display: grid;
                grid-template-columns: 1.1fr 0.9fr;
                gap: 20px;
            }}
            .panel {{
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 24px;
                box-shadow: var(--shadow);
                backdrop-filter: blur(14px);
            }}
            h2 {{
                margin: 0 0 12px;
                font-size: 24px;
            }}
            .subtle {{
                color: var(--muted);
                margin-bottom: 18px;
            }}
            .toolbar {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-bottom: 16px;
            }}
            button, .link-button {{
                border: none;
                cursor: pointer;
                border-radius: 14px;
                padding: 12px 16px;
                font-size: 15px;
                font-weight: 600;
                transition: transform 0.15s ease, opacity 0.15s ease, background 0.15s ease;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }}
            button:hover, .link-button:hover {{
                transform: translateY(-1px);
            }}
            .primary {{
                background: var(--accent);
                color: white;
            }}
            .primary:hover {{
                background: var(--accent-strong);
            }}
            .secondary {{
                background: var(--secondary-soft);
                color: var(--secondary);
            }}
            .ghost {{
                background: #f8fafc;
                color: var(--text);
                border: 1px solid var(--border);
            }}
            textarea {{
                width: 100%;
                min-height: 340px;
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 16px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 14px;
                resize: vertical;
                background: #fcfefe;
                color: var(--text);
            }}
            pre {{
                margin: 0;
                padding: 18px;
                border-radius: 18px;
                background: #0f172a;
                color: #dbeafe;
                min-height: 340px;
                overflow: auto;
                font-size: 13px;
                line-height: 1.55;
            }}
            code {{
                background: #f3f4f6;
                padding: 2px 6px;
                border-radius: 6px;
            }}
            .endpoint-list {{
                display: grid;
                gap: 12px;
                margin-top: 18px;
            }}
            .endpoint {{
                display: flex;
                gap: 12px;
                align-items: flex-start;
                padding: 14px 16px;
                border: 1px solid var(--border);
                border-radius: 16px;
                background: #fbfdff;
            }}
            .method {{
                min-width: 56px;
                text-align: center;
                font-weight: 700;
                padding: 6px 8px;
                border-radius: 10px;
                color: white;
                background: var(--accent);
                font-size: 13px;
            }}
            .method.get {{
                background: var(--secondary);
            }}
            .note {{
                margin-top: 18px;
                padding: 14px 16px;
                border-radius: 16px;
                background: #fff8e7;
                border: 1px solid #f6d38f;
                color: #7a5a00;
                font-size: 14px;
            }}
            @media (max-width: 980px) {{
                .layout {{
                    grid-template-columns: 1fr;
                }}
                .wrap {{
                    padding: 16px;
                }}
                .hero, .panel {{
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <main class="shell">
                <section class="hero">
                    <div class="badge">Service is running</div>
                    <h1>Task Overdue Risk Service</h1>
                    <p>
                        Учебный API-сервис для прогноза просрочки задач в небольшой команде.
                        Здесь можно сразу проверить состояние сервиса, подставить demo-данные и получить предсказание без перехода в Postman или curl.
                    </p>
                    <p>
                        Активная модель: <strong>{model_name}</strong>
                    </p>
                    <div class="summary-grid">
                        <section class="tile">
                            <strong>Быстрый старт</strong>
                            Нажми <code>Load demo payload</code>, затем <code>Run prediction</code>.
                        </section>
                        <section class="tile">
                            <strong>Проверка сервиса</strong>
                            Кнопка <code>Check health</code> покажет, что модель загружена.
                        </section>
                        <section class="tile">
                            <strong>Документация</strong>
                            Swagger доступен по ссылке <code>/docs</code>.
                        </section>
                    </div>
                </section>

                <section class="layout">
                    <section class="panel">
                        <h2>Интерактивная проверка</h2>
                        <div class="subtle">
                            Слева JSON запроса, справа ответ сервиса. Можно использовать готовый пример или поменять поля вручную.
                        </div>
                        <div class="toolbar">
                            <button class="primary" onclick="loadDemoPayload()">Load demo payload</button>
                            <button class="secondary" onclick="runPrediction()">Run prediction</button>
                            <button class="ghost" onclick="checkHealth()">Check health</button>
                            <button class="ghost" onclick="loadMetrics()">Load metrics</button>
                        </div>
                        <div class="layout" style="grid-template-columns: 1fr 1fr; gap: 16px;">
                            <div>
                                <div class="subtle">Request body for <code>POST /predict</code></div>
                                <textarea id="payloadEditor"></textarea>
                            </div>
                            <div>
                                <div class="subtle">Server response</div>
                                <pre id="responseViewer">Нажми любую кнопку выше, чтобы увидеть ответ сервиса.</pre>
                            </div>
                        </div>
                        <div class="note">
                            Если хочется смотреть endpoints по отдельности, справа оставлены быстрые ссылки и краткие пояснения.
                        </div>
                    </section>

                    <aside class="panel">
                        <h2>Endpoints</h2>
                        <div class="endpoint-list">
                            <div class="endpoint">
                                <span class="method get">GET</span>
                                <div>
                                    <strong>/health</strong><br />
                                    Показывает, что сервис поднят и какая модель загружена.
                                </div>
                            </div>
                            <div class="endpoint">
                                <span class="method">POST</span>
                                <div>
                                    <strong>/predict</strong><br />
                                    Принимает список задач и возвращает вероятность просрочки и уровень риска.
                                </div>
                            </div>
                            <div class="endpoint">
                                <span class="method get">GET</span>
                                <div>
                                    <strong>/metrics</strong><br />
                                    Отдаёт базовые метрики запросов, ошибок, latency и uptime.
                                </div>
                            </div>
                            <div class="endpoint">
                                <span class="method get">GET</span>
                                <div>
                                    <strong>/docs</strong><br />
                                    Swagger UI для ручной проверки всех endpoint'ов.
                                </div>
                            </div>
                        </div>
                        <div class="toolbar" style="margin-top: 18px;">
                            <a class="link-button primary" href="/docs">Open Swagger</a>
                            <a class="link-button ghost" href="/health" target="_blank" rel="noreferrer">Open /health</a>
                            <a class="link-button ghost" href="/metrics" target="_blank" rel="noreferrer">Open /metrics</a>
                        </div>
                    </aside>
                </section>
            </main>
        </div>
        <script>
            const demoPayload = {{
              "tasks": [
                {{
                  "priority": "critical",
                  "assignee_experience": 1.4,
                  "estimated_hours": 42.0,
                  "actual_progress": 28.0,
                  "days_since_created": 19,
                  "comments_count": 11,
                  "status": "blocked",
                  "team_size": 5,
                  "task_type": "bug",
                  "has_blockers": true,
                  "blockers_count": 2,
                  "recent_reassignments": 2,
                  "sprint_phase": "release_week",
                  "priority_score": 4,
                  "workload_ratio": 3.1,
                  "is_customer_facing": true,
                  "requires_review": true
                }},
                {{
                  "priority": "medium",
                  "assignee_experience": 5.8,
                  "estimated_hours": 10.0,
                  "actual_progress": 82.0,
                  "days_since_created": 6,
                  "comments_count": 2,
                  "status": "review",
                  "team_size": 6,
                  "task_type": "documentation",
                  "has_blockers": false,
                  "blockers_count": 0,
                  "recent_reassignments": 0,
                  "sprint_phase": "mid_sprint",
                  "priority_score": 2,
                  "workload_ratio": 1.0,
                  "is_customer_facing": false,
                  "requires_review": true
                }}
              ]
            }};

            const payloadEditor = document.getElementById("payloadEditor");
            const responseViewer = document.getElementById("responseViewer");

            function renderOutput(title, payload) {{
                const body = typeof payload === "string"
                    ? payload
                    : JSON.stringify(payload, null, 2);
                responseViewer.textContent = `${{title}}\\n\\n${{body}}`;
            }}

            function loadDemoPayload() {{
                payloadEditor.value = JSON.stringify(demoPayload, null, 2);
                renderOutput("Demo payload loaded", demoPayload);
            }}

            async function checkHealth() {{
                const response = await fetch("/health");
                const data = await response.json();
                renderOutput("GET /health", data);
            }}

            async function loadMetrics() {{
                const response = await fetch("/metrics");
                const text = await response.text();
                renderOutput("GET /metrics", text);
            }}

            async function runPrediction() {{
                try {{
                    const payload = JSON.parse(payloadEditor.value);
                    const response = await fetch("/predict", {{
                        method: "POST",
                        headers: {{
                            "Content-Type": "application/json"
                        }},
                        body: JSON.stringify(payload)
                    }});
                    const data = await response.json();
                    renderOutput(`POST /predict [${{response.status}}]`, data);
                }} catch (error) {{
                    renderOutput("Client error", {{
                        message: "Не удалось выполнить запрос. Проверь JSON в поле слева.",
                        error: String(error)
                    }});
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
        "model_name": app.state.metadata["best_model_name"],
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return metrics_store.render()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    try:
        frame = ensure_dataframe([item.model_dump() for item in payload.tasks])
        probabilities = app.state.model.predict_proba(frame)[:, 1]
        predictions = [
            PredictionItem(
                is_overdue=int(probability >= 0.5),
                overdue_probability=round(float(probability), 4),
                overdue_risk=_risk_bucket(float(probability)),
            )
            for probability in probabilities
        ]
        metrics_store.prediction_count += len(predictions)
        return PredictionResponse(model_name=app.state.metadata["best_model_name"], predictions=predictions)
    except ValueError as exc:
        metrics_store.error_count += 1
        LOGGER.warning("Validation-like error during prediction: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        metrics_store.error_count += 1
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
