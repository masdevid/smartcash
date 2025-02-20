# 🔌 API Documentation

## 📋 Overview

SmartCash menyediakan REST API untuk integrasi deteksi nilai mata uang Rupiah ke dalam aplikasi lain.

## 🚀 Quick Start

### 1. Installation
```bash
pip install smartcash[api]
```

### 2. Run Server
```bash
uvicorn smartcash.api:app --host 0.0.0.0 --port 8000
```

### 3. Test API
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@test.jpg"
```

## 🛠️ API Endpoints

### 1. Prediction
```
POST /predict
Content-Type: multipart/form-data

Parameters:
- image: File (required)
- conf_thres: float (optional, default: 0.25)
- iou_thres: float (optional, default: 0.45)

Response:
{
    "predictions": [
        {
            "class_id": 0,
            "class_name": "1000",
            "confidence": 0.95,
            "bbox": [x1, y1, x2, y2]
        }
    ],
    "inference_time": 0.045
}
```

### 2. Batch Prediction
```
POST /predict/batch
Content-Type: multipart/form-data

Parameters:
- images: List[File] (required)
- conf_thres: float (optional, default: 0.25)
- iou_thres: float (optional, default: 0.45)

Response:
{
    "predictions": [
        {
            "image_id": "image1.jpg",
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "1000",
                    "confidence": 0.95,
                    "bbox": [x1, y1, x2, y2]
                }
            ]
        }
    ],
    "batch_inference_time": 0.145
}
```

### 3. Model Info
```
GET /model/info

Response:
{
    "model_name": "smartcash-v1.0",
    "backbone": "efficientnet_b4",
    "input_size": [640, 640],
    "classes": ["1000", "2000", "5000", "10000", "20000", "50000", "100000"],
    "version": "1.0.0"
}
```

### 4. Health Check
```
GET /health

Response:
{
    "status": "healthy",
    "gpu_available": true,
    "memory_usage": "1.2GB",
    "uptime": "2d 3h 45m"
}
```

## 🔒 Authentication

### 1. API Key
```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(
    api_key: str = Depends(api_key_header),
    image: UploadFile = File(...)
):
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    # Process prediction
```

### 2. Rate Limiting
```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request,
    image: UploadFile = File(...)
):
    # Process prediction
```

## 📊 Response Format

### 1. Success Response
```json
{
    "status": "success",
    "data": {
        "predictions": [...],
        "inference_time": 0.045
    },
    "message": null
}
```

### 2. Error Response
```json
{
    "status": "error",
    "data": null,
    "message": "Invalid image format"
}
```

## 🔍 Error Handling

### 1. Input Validation
```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    
    @validator("conf_thres")
    def validate_conf_thres(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("conf_thres must be between 0 and 1")
        return v
```

### 2. Exception Handling
```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "details": exc.errors()
        }
    )
```

## 🔄 Asynchronous Processing

### 1. Background Tasks
```python
from fastapi.background import BackgroundTasks

@app.post("/predict/async")
async def predict_async(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...)
):
    task_id = generate_task_id()
    background_tasks.add_task(
        process_prediction,
        image,
        task_id
    )
    return {"task_id": task_id}
```

### 2. Task Status
```python
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    status = await get_task(task_id)
    return {
        "task_id": task_id,
        "status": status.state,
        "result": status.result
    }
```

## 📊 Monitoring

### 1. Metrics
```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### 2. Logging
```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(
        f"Method={request.method} "
        f"Path={request.url.path} "
        f"Status={response.status_code} "
        f"Duration={duration:.3f}s"
    )
    return response
```

## 🔧 Configuration

### 1. Environment Variables
```env
API_KEY=your_api_key
MODEL_PATH=/path/to/model
MAX_BATCH_SIZE=32
ENABLE_GPU=true
```

### 2. API Settings
```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    api_key: str
    model_path: str
    max_batch_size: int = 32
    enable_gpu: bool = True
    
    class Config:
        env_file = ".env"
```

## 📚 API Documentation

### 1. Swagger UI
```
http://localhost:8000/docs
```

### 2. ReDoc
```
http://localhost:8000/redoc
```

## 🚀 Deployment

### 1. Docker
```dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./app /app
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```

### 2. Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smartcash-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: smartcash-api:latest
        ports:
        - containerPort: 80
```

## 📈 Next Steps

1. [Model Monitoring](MONITORING.md)
2. [Performance Optimization](OPTIMIZATION.md)
3. [Scaling Guide](SCALING.md)
