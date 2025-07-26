# test_iris_api.py
from fastapi.testclient import TestClient
from app.iris_fastapi import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict/", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert "predicted_class" in response.json()
