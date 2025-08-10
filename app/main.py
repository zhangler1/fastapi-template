import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger

# 配置日志 - 使用loguru
logger.add(
    "app.log",
    rotation="500 MB",  # 日志文件达到500MB时轮转
    retention="10 days",  # 保留10天的日志
    compression="zip",  # 压缩旧日志
    level="INFO",  # 日志级别
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"  # 日志格式
)

# 将标准日志重定向到loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 获取loguru对应的日志级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 找到调用者
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)

# 配置Prometheus指标 - 参考vllm的监控指标
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Duration of HTTP requests in seconds",
    ["method", "endpoint"]
)

ACTIVE_REQUESTS = Gauge(
    "http_active_requests",
    "Number of active HTTP requests",
    ["method", "endpoint"]
)

# 模拟vllm相关指标
MODEL_LOADING_TIME = Gauge(
    "model_loading_time_seconds",
    "Time taken to load the model in seconds"
)

TOKEN_GENERATION_RATE = Gauge(
    "token_generation_rate_tokens_per_second",
    "Rate of token generation in tokens per second"
)

QUEUE_SIZE = Gauge(
    "request_queue_size",
    "Number of requests in the queue"
)

TOTAL_TOKENS_GENERATED = Counter(
    "total_tokens_generated",
    "Total number of tokens generated"
)

# 全局状态管理
app_state: Dict[str, Any] = {
    "model_loaded": False,
    "model_loading_time": 0.0,
    "queue": [],
    "available_models": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时的操作 - startup回调
    logger.info("Application startup started")
    
    # 模拟模型加载过程
    start_time = time.time()
    logger.info("Loading model...")
    
    # 模拟加载多个模型
    for model_name in ["model-small", "model-large"]:
        model_start = time.time()
        logger.info(f"Loading {model_name}...")
        await asyncio.sleep(1)  # 模拟加载延迟
        app_state["available_models"].append(model_name)
        logger.info(f"Loaded {model_name} in {time.time() - model_start:.2f}s")
    
    # 记录总加载时间
    load_time = time.time() - start_time
    app_state["model_loaded"] = True
    app_state["model_loading_time"] = load_time
    MODEL_LOADING_TIME.set(load_time)
    
    logger.info(f"All models loaded successfully in {load_time:.2f} seconds")
    logger.info("Application startup completed")
    
    yield  # 应用运行期间
    
    # 关闭时的操作 - shutdown回调
    logger.info("Application shutdown started")
    logger.info(f"Total requests processed: {REQUEST_COUNT.sum()}")
    logger.info(f"Total tokens generated: {TOTAL_TOKENS_GENERATED.sum()}")
    logger.info("Cleaning up resources...")
    
    # 模拟资源清理
    await asyncio.sleep(1)
    app_state["model_loaded"] = False
    app_state["queue"].clear()
    
    logger.info("Application shutdown completed")

# 创建FastAPI应用
app = FastAPI(lifespan=lifespan, title="LLM Service with Monitoring")

# 中间件 - 用于监控和日志记录
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    
    # 记录活跃请求数
    ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()
    
    # 记录请求开始时间
    start_time = time.time()
    
    # 记录请求日志
    logger.info(f"Received request: {method} {endpoint}")
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 记录请求计数
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code).inc()
        
        # 记录请求延迟
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        
        logger.info(f"Completed request: {method} {endpoint} {response.status_code} in {duration:.4f}s")
        
        return response
    except Exception as e:
        # 处理异常
        status_code = 500
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        
        logger.error(f"Error processing request: {method} {endpoint} {str(e)} in {duration:.4f}s")
        
        return JSONResponse(
            status_code=status_code,
            content={"message": "An error occurred during processing"}
        )
    finally:
        # 减少活跃请求数
        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).dec()

# 健康检查端点
@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Check if the service is running and models are loaded"""
    return {
        "status": "healthy" if app_state["model_loaded"] else "unhealthy",
        "model_loaded": app_state["model_loaded"],
        "model_loading_time": app_state["model_loading_time"],
        "available_models": app_state["available_models"],
        "queue_size": len(app_state["queue"])
    }

# 示例端点 - 模拟文本生成
@app.post("/generate", summary="Generate text from prompt")
async def generate_text(
    prompt: str, 
    model: str = "model-small",
    max_tokens: Optional[int] = 100,
    temperature: float = 0.7
):
    """Generate text based on a prompt using specified model parameters"""
    if not app_state["model_loaded"]:
        return JSONResponse(
            status_code=503,
            content={"message": "Model not loaded yet, please try again later"}
        )
    
    if model not in app_state["available_models"]:
        return JSONResponse(
            status_code=400,
            content={"message": f"Model {model} not available", "available_models": app_state["available_models"]}
        )
    
    # 模拟添加到队列
    request_id = len(app_state["queue"]) + 1
    app_state["queue"].append(request_id)
    QUEUE_SIZE.set(len(app_state["queue"]))
    logger.info(f"Request {request_id} added to queue. Queue size: {len(app_state['queue'])}")
    
    try:
        # 模拟处理延迟 - 基于max_tokens和模型大小
        start_time = time.time()
        model_factor = 1.5 if model == "model-large" else 1.0
        processing_time = (max_tokens / 100) * model_factor
        await asyncio.sleep(processing_time)
        
        # 模拟生成 tokens
        generated_tokens = int(max_tokens * (0.7 + 0.3 * (1 - temperature)))
        TOTAL_TOKENS_GENERATED.inc(generated_tokens)
        
        # 计算生成速率
        generation_rate = generated_tokens / processing_time if processing_time > 0 else 0
        TOKEN_GENERATION_RATE.set(generation_rate)
        
        return {
            "request_id": request_id,
            "model": model,
            "prompt": prompt[:50] + ("..." if len(prompt) > 50 else ""),
            "generated_tokens": generated_tokens,
            "processing_time": round(processing_time, 4),
            "generation_rate": round(generation_rate, 2),
            "temperature": temperature
        }
    finally:
        # 从队列中移除
        if request_id in app_state["queue"]:
            app_state["queue"].remove(request_id)
        QUEUE_SIZE.set(len(app_state["queue"]))
        logger.info(f"Request {request_id} processed. Queue size: {len(app_state['queue'])}")

# 指标端点 - 供Prometheus抓取
@app.get("/metrics", summary="Prometheus metrics endpoint")
async def metrics():
    """Expose metrics for Prometheus monitoring"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
