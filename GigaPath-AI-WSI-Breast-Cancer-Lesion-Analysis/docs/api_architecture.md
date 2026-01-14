# API Architecture (Design Document)

> [!WARNING]
> **This is an architecture specification only. No API code has been implemented.**
> Implementation requires user approval of technology stack and deployment model.

---

## Overview

This document outlines the proposed RESTful API architecture for the GigaPath WSI system to enable:
- Programmatic inference requests
- WSI upload and processing
- Result retrieval
- System health monitoring

---

## Technology Stack (Recommendations)

### Option 1: FastAPI (Recommended)

**Pros**:
- Modern async support
- Automatic OpenAPI documentation
- Type safety with Pydantic
- High performance (ASGI)

**Cons**:
- Newer framework (less mature ecosystem)

### Option 2: Flask

**Pros**:
- Mature ecosystem
- Simple deployment
- Extensive documentation

**Cons**:
- Synchronous by default
- Manual API documentation

### Option 3: Django REST Framework

**Pros**:
- Full-featured ORM
- Built-in admin panel
- Enterprise-ready

**Cons**:
- Heavier framework
- Overkill for simple inference API

---

## API Endpoints

### 1. Health Check

```
GET /api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": true,
  "model_loaded": true,
  "timestamp": "2026-01-12T00:18:32Z"
}
```

---

### 2. Upload WSI

```
POST /api/v1/slides/upload
```

**Request** (multipart/form-data):
- `file`: WSI file (.svs, .tiff)
- `slide_name`: String (optional, defaults to filename)

**Response**:
```json
{
  "slide_id": "abc123",
  "slide_name": "slide_001",
  "status": "uploaded",
  "file_size_mb": 245.3
}
```

---

### 3. Start Preprocessing

```
POST /api/v1/slides/{slide_id}/preprocess
```

**Response**:
```json
{
  "task_id": "task_xyz",
  "status": "processing",
  "estimated_time_minutes": 5
}
```

---

### 4. Run Inference

```
POST /api/v1/slides/{slide_id}/infer
```

**Request**:
```json
{
  "model_checkpoint": "checkpoints/best_model.pth"
}
```

**Response**:
```json
{
  "slide_id": "abc123",
  "predicted_label": 1,
  "confidence_percent": 87.3,
  "interpretation": "Malignant (87.3% confidence)",
  "processing_time_seconds": 12.5
}
```

---

### 5. Get Results

```
GET /api/v1/slides/{slide_id}/results
```

**Response**:
```json
{
  "slide_id": "abc123",
  "slide_name": "slide_001",
  "prediction": {
    "label": 1,
    "confidence_percent": 87.3,
    "interpretation": "Malignant (87.3% confidence)"
  },
  "heatmap_url": "/api/v1/slides/abc123/heatmap",
  "created_at": "2026-01-12T00:18:32Z"
}
```

---

### 6. Generate Heatmap

```
POST /api/v1/slides/{slide_id}/heatmap
```

**Request**:
```json
{
  "mode": "attention",
  "overlay_alpha": 0.5
}
```

**Response**:
```json
{
  "heatmap_id": "heatmap_456",
  "status": "generated",
  "download_url": "/api/v1/heatmaps/heatmap_456/download"
}
```

---

### 7. List Slides

```
GET /api/v1/slides?limit=10&offset=0
```

**Response**:
```json
{
  "total": 42,
  "limit": 10,
  "offset": 0,
  "slides": [
    {
      "slide_id": "abc123",
      "slide_name": "slide_001",
      "status": "completed",
      "uploaded_at": "2026-01-12T00:10:00Z"
    }
  ]
}
```

---

## Data Models

### Slide Status

```python
class SlideStatus(str, Enum):
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    EXTRACTING_FEATURES = "extracting_features"
    SAMPLING = "sampling"
    INFERRING = "inferring"
    COMPLETED = "completed"
    FAILED = "failed"
```

### Prediction Result

```python
class PredictionResult(BaseModel):
    slide_id: str
    predicted_label: int
    probability: float
    confidence_percent: float
    interpretation: str
    logit: float
    processing_time_seconds: float
```

---

## Authentication & Authorization

### Option 1: API Keys

```
GET /api/v1/slides
Authorization: Bearer YOUR_API_KEY
```

### Option 2: JWT Tokens

```
POST /api/v1/auth/login
{
  "username": "researcher",
  "password": "******"
}

Response:
{
  "access_token": "eyJ0eXAi...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## Asynchronous Processing

### Task Queue (Celery + Redis)

For long-running operations:

```python
@app.post("/api/v1/slides/{slide_id}/infer")
async def run_inference(slide_id: str):
    task = inference_task.delay(slide_id)
    return {"task_id": task.id, "status": "processing"}

@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = AsyncResult(task_id)
    return {"status": task.state, "result": task.result}
```

---

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "SLIDE_NOT_FOUND",
    "message": "Slide with ID 'abc123' does not exist",
    "status_code": 404
  }
}
```

### Error Codes

- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (missing/invalid API key)
- `404`: Not Found (slide/model not found)
- `500`: Internal Server Error (inference failed)
- `503`: Service Unavailable (GPU offline)

---

## Rate Limiting

```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/slides/upload")
@limiter.limit("10/minute")
async def upload_slide():
    ...
```

---

## Deployment Architecture

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .


    ports:
      - "8000:8000"
    environment:
      - GPU_DEVICE=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  redis:
    image: redis:latest
  
  celery_worker:
    build: .
    command: celery -A app.celery worker --loglevel=info
```

---

## Security Considerations

1. **HTTPS Only**: Enforce TLS/SSL
2. **Input Validation**: Sanitize all uploads
3. **File Size Limits**: Max 2GB per WSI
4. **Rate Limiting**: Prevent abuse
5. **CORS**: Restrict to allowed origins
6. **PHI Handling**: Do NOT accept patient-identifiable data

---

## Monitoring & Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

inference_requests = Counter('inference_requests_total', 'Total inference requests')
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')

@app.post("/api/v1/slides/{slide_id}/infer")
async def run_inference(slide_id: str):
    inference_requests.inc()
    with inference_duration.time():
        result = model.predict(...)
    return result
```

---

## Implementation Checklist

> [!WARNING]
> **Do NOT implement without user approval**

- [ ] Choose web framework (FastAPI/Flask/Django)
- [ ] Define API schema (OpenAPI 3.0)
- [ ] Implement authentication
- [ ] Set up task queue (Celery/RQ)
- [ ] Add rate limiting
- [ ] Implement file upload handling
- [ ] Add monitoring (Prometheus/Grafana)
- [ ] Write API tests
- [ ] Deploy with Docker
- [ ] Configure HTTPS/SSL
- [ ] Document API endpoints
- [ ] Add usage examples

---

**Last Updated**: 2026-01-12

**Status**: **Architecture Only - Not Implemented**
