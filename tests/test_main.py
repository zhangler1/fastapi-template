from fastapi.testclient import TestClient
# 假设你的主应用文件是app/main.py
from app.main import app
 

 
client = TestClient(app)

def test_health_check():
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root_endpoint():
    """测试根路径接口"""
    response = client.get("/generate")
    assert response.status_code == 200
    