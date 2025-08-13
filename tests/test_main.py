from fastapi.testclient import TestClient
# 假设你的主应用文件是app/main.py
from app.main import app
 

 
client = TestClient(app)

def test_health_check():
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
