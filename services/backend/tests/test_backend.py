"""Tests du service backend Pyrenex."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).parents[1] / "services" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app import main as backend_main  # noqa: E402


VALID_PAYLOAD = {
    "loan_amnt": 10_000.0,
    "int_rate": 13.5,
    "installment": 340.0,
    "annual_inc": 55_000.0,
    "dti": 18.2,
    "delinq_2yrs": 0,
    "fico_range_low": 690,
    "revol_util": 42.5,
    "term": "36 months",
    "grade": "B",
    "home_ownership": "MORTGAGE",
    "verification_status": "Not Verified",
    "purpose": "debt_consolidation",
    "emp_length": "10+ years",
}

VALID_PREDICTION = {
    "prediction": 0,
    "probability": 0.1234,
    "model_version": "v2.0.0",
    "request_id": "test-request-id",
}


class FakeAsyncClient:
    """Double minimal de httpx.AsyncClient pour isoler le service model."""

    def __init__(self, response: httpx.Response | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.post_kwargs: dict[str, Any] | None = None

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.post_kwargs = {"url": url, **kwargs}
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


@pytest.fixture
def client() -> TestClient:
    return TestClient(backend_main.app)


def metric_value(client: TestClient, kind: str) -> float:
    prefix = f'backend_upstream_errors_total{{kind="{kind}"}} '
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    line = next(line for line in metrics.text.splitlines() if line.startswith(prefix))
    return float(line.removeprefix(prefix))


def total_calls_value(client: TestClient) -> float:
    prefix = "backend_score_calls_total "
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    line = next(line for line in metrics.text.splitlines() if line.startswith(prefix))
    return float(line.removeprefix(prefix))


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: httpx.Response | None = None,
    error: Exception | None = None,
) -> FakeAsyncClient:
    fake = FakeAsyncClient(response=response, error=error)
    monkeypatch.setattr(backend_main.httpx, "AsyncClient", lambda: fake)
    return fake


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]


def test_metrics_exposes_only_backend_counters(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    samples = [line for line in response.text.splitlines() if line and not line.startswith("#")]
    assert samples
    assert all(
        line.startswith(("backend_upstream_errors_total{", "backend_score_calls_total "))
        for line in samples
    )
    assert any('kind="unavailable"' in line for line in samples)
    assert any('kind="bad_response"' in line for line in samples)
    assert any(line.startswith("backend_score_calls_total ") for line in samples)


def test_score_returns_prediction_and_propagates_request_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream_response = httpx.Response(200, json=VALID_PREDICTION)
    fake = install_fake_client(monkeypatch, response=upstream_response)
    before = total_calls_value(client)

    response = client.post(
        "/score",
        json=VALID_PAYLOAD,
        headers={"X-Request-ID": "test-request-id"},
    )

    assert response.status_code == 200
    assert response.json() == VALID_PREDICTION
    assert total_calls_value(client) == before + 1
    assert fake.post_kwargs is not None
    assert fake.post_kwargs["url"] == f"{backend_main.MODEL_URL}/predict"
    assert fake.post_kwargs["headers"] == {"X-Request-ID": "test-request-id"}
    assert fake.post_kwargs["json"] == VALID_PAYLOAD


def test_score_rejects_invalid_application_without_calling_model(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_fake_client(monkeypatch, response=httpx.Response(200, json=VALID_PREDICTION))
    invalid_payload = {**VALID_PAYLOAD, "fico_range_low": 9999}
    before = total_calls_value(client)

    response = client.post("/score", json=invalid_payload)

    assert response.status_code == 422
    assert fake.post_kwargs is None
    assert total_calls_value(client) == before


def test_score_unavailable_returns_503_and_increments_metric(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = httpx.Request("POST", f"{backend_main.MODEL_URL}/predict")
    install_fake_client(monkeypatch, error=httpx.ConnectError("connection failed", request=request))
    before = metric_value(client, "unavailable")

    response = client.post("/score", json=VALID_PAYLOAD)

    assert response.status_code == 503
    assert response.json() == {"detail": "Model service unavailable"}
    assert metric_value(client, "unavailable") == before + 1


def test_score_upstream_error_returns_502_and_increments_metric(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_client(monkeypatch, response=httpx.Response(500, json={"detail": "failure"}))
    before = metric_value(client, "bad_response")

    response = client.post("/score", json=VALID_PAYLOAD)

    assert response.status_code == 502
    assert response.json() == {"detail": "Model service returned an error"}
    assert metric_value(client, "bad_response") == before + 1


def test_score_invalid_upstream_payload_returns_502_and_increments_metric(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_client(monkeypatch, response=httpx.Response(200, json={"unexpected": "value"}))
    before = metric_value(client, "bad_response")

    response = client.post("/score", json=VALID_PAYLOAD)

    assert response.status_code == 502
    assert response.json() == {"detail": "Invalid response from model service"}
    assert metric_value(client, "bad_response") == before + 1
