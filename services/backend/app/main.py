"""Service `backend` — orchestrateur (SQUELETTE À COMPLÉTER).

Rôle attendu : exposé au navigateur (via le frontend nginx), il valide
l'entrée avec le **même schéma Pydantic** que le modèle, appelle le service
`model` en interne (`http://model:8000/predict`), et expose `/health`,
`/score`, `/metrics`.

👉 Inspirez-vous du service `model` (déjà fourni) pour le pattern `/metrics`
   et le middleware de logging. Mini-cours : `02_FastAPI_metrics_Prometheus`.
"""
from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    disable_created_metrics,
    generate_latest,
)
from pydantic import ValidationError

from app.middleware import LoggingMiddleware
from app.schemas import HealthResponse, LoanApplication, Prediction

# URL du service model — configurable par variable d'env (dev/staging/prod)
MODEL_URL = os.environ.get("MODEL_URL", "http://model:8000")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8088").split(",")

app = FastAPI(title="Pyrenex Backend Orchestrator", version="1.0.0")
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


# Expose les erreurs cumulées lors de l'appel au model upstream via /metrics
disable_created_metrics()
metrics_registry = CollectorRegistry()
backend_upstream_errors_total = Counter(
    "backend_upstream_errors_total",
    "Nombre d'erreurs rencontrées lors de l'appel au service model upstream depuis le backend.",
    labelnames=("kind",),
    registry=metrics_registry,
)
backend_upstream_errors_total.labels(kind="unavailable")
backend_upstream_errors_total.labels(kind="bad_response")


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose uniquement le compteur des erreurs du service model upstream."""
    return Response(
        content=generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness du backend (ne dépend PAS du model)."""
    return HealthResponse(status="ok")


@app.post("/score", response_model=Prediction)
async def score(application: LoanApplication, request: Request) -> Prediction:
    """Appelle le service model et comptabilise ses erreurs upstream."""
    request_id = getattr(request.state, "request_id", "n/a")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MODEL_URL}/predict",
                json=application.model_dump(),
                headers={"X-Request-ID": request_id},
                timeout=5.0,
            )
    except httpx.RequestError as exc:
        backend_upstream_errors_total.labels(kind="unavailable").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable",
        ) from exc

    if response.is_error:
        backend_upstream_errors_total.labels(kind="bad_response").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Model service returned an error",
        )

    try:
        return Prediction.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        backend_upstream_errors_total.labels(kind="bad_response").inc()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Invalid response from model service",
        ) from exc
