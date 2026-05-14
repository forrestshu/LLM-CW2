from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "ok"
    assert "tavily_configured" in data


def test_topics_endpoint_bilingual():
    response = client.get("/api/topics?language=zh")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 30
    assert "label" in data[0]
    assert any(item["id"] == "rel-01" for item in data)
    assert any(item["id"] == "rel-02" for item in data)
