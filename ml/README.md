# RecursiaDx ML Module

A comprehensive machine learning module for tumor detection using deep learning with ResNet50 architecture.

## Overview

This ML module provides a complete solution for tumor detection in medical images using a pre-trained ResNet50 model fine-tuned for binary classification (tumor vs non-tumor). The module includes data preprocessing, model training, prediction services, and a REST API.

## Features

- **ResNet50-based Architecture**: Uses pre-trained ResNet50 with custom classification layers
- **Image Enhancement**: Medical image preprocessing with CLAHE, noise reduction, and contrast enhancement
- **REST API**: Flask-based API for predictions
- **Batch Processing**: Support for processing multiple images
- **Database Integration**: SQLite database for storing prediction results
- **Visualization**: Automatic generation of prediction visualizations
- **Risk Assessment**: Categorized risk levels based on prediction confidence

## Directory Structure

```
ml/
├── api/
│   └── app.py                 # Flask REST API
├── config/
│   └── config.py             # Configuration settings
├── models/
│   └── tumor_predictor.py    # Main prediction model
├── utils/
│   ├── image_utils.py        # Image processing utilities
│   └── data_manager.py       # Data management and storage
├── train_model.py            # Model training script
├── predict.py                # Prediction service script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Install Python Dependencies**:
   ```bash
   cd ml
   pip install -r requirements.txt
   ```

2. **Create Required Directories**:
   The system will automatically create necessary directories on first run.

## Quick Start

### 1. Basic Prediction

```python
from models.tumor_predictor import TumorPredictor

# Initialize predictor
predictor = TumorPredictor()
predictor.build_model()

# Make prediction
result = predictor.predict('path/to/image.jpg')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 2. Using the Prediction Service

```bash
# Predict single image
python predict.py path/to/image.jpg --enhance --save-viz

# Predict all images in directory
python predict.py path/to/images/ --enhance --save-viz

# Use custom model
python predict.py image.jpg --model path/to/model.h5
```

### 3. Starting the API Server

```bash
python api/app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /health
```

### Single Image Prediction
```
POST /predict
Content-Type: multipart/form-data

Parameters:
- image: Image file
- user_id: Optional user identifier
- enhance_image: Optional image enhancement (true/false)
- save_result: Optional save to database (true/false)
```

### Base64 Image Prediction
```
POST /predict_base64
Content-Type: application/json

{
  "image": "base64_encoded_image_data",
  "user_id": "optional_user_id",
  "enhance_image": false
}
```

### Batch Prediction
```
POST /batch_predict
Content-Type: multipart/form-data

Parameters:
- images: Multiple image files
- user_id: Optional user identifier
- enhance_images: Optional image enhancement (true/false)
```

### Prediction History
```
GET /history?user_id=optional&limit=100
```

### Statistics
```
GET /stats?user_id=optional
```

### Model Information
```
GET /model_info
```

## Model Training

### Data Preparation

Organize your training data in the following structure:
```
data/
├── train/
│   ├── tumor/
│   │   ├── tumor_001.jpg
│   │   ├── tumor_002.jpg
│   │   └── ...
│   └── non_tumor/
│       ├── normal_001.jpg
│       ├── normal_002.jpg
│       └── ...
└── validation/
    ├── tumor/
    └── non_tumor/
```

### Training Command

```bash
python train_model.py --data_dir path/to/data --epochs 50
```

## Configuration

Edit `config/config.py` to customize:

- Model parameters
- API settings
- File paths
- Thresholds
- Database settings

## Risk Level Categories

The system categorizes predictions into risk levels:

- **High Risk** (≥90% confidence): Immediate medical attention recommended
- **Moderate Risk** (70-89% confidence): Further examination recommended
- **Low-Moderate Risk** (50-69% confidence): Monitoring recommended
- **Low Risk** (<50% confidence): Routine follow-up

## Image Enhancement Features

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Noise Reduction** using Non-local Means Denoising
- **Contrast Enhancement**
- **Sharpness Enhancement**

## Performance Considerations

- **Input Size**: Images are resized to 224x224 pixels
- **Batch Processing**: Supports batch sizes up to 64 images
- **Memory Usage**: Approximately 500MB for base model
- **Processing Time**: ~100-500ms per image on modern hardware

## Database Schema

### Predictions Table
- `id`: Primary key
- `image_path`: Path to analyzed image
- `predicted_class`: Tumor or Non-Tumor
- `confidence`: Prediction confidence (0-1)
- `is_tumor`: Boolean tumor flag
- `risk_level`: Risk category
- `timestamp`: Prediction timestamp
- `user_id`: Optional user identifier
- `processing_time`: Time taken for prediction

## Error Handling

The system includes comprehensive error handling for:
- Invalid image formats
- File size limits
- Model loading errors
- Prediction failures
- Database errors

## Security Features

- File type validation
- File size limits (16MB default)
- Input sanitization
- Error message sanitization in production

## Integration with RecursiaDx Backend

To integrate with the main backend:

1. **Add ML API endpoint to backend routes**:
   ```javascript
   // In backend/routes/samples.js
   app.post('/api/analyze', async (req, res) => {
     // Forward to ML API
     const mlResponse = await fetch('http://localhost:5000/predict', {
       method: 'POST',
       body: formData
     });
     // Process response
   });
   ```

2. **Update frontend to call analysis endpoint**:
   ```javascript
   // In client components
   const analyzeImage = async (imageFile) => {
     const formData = new FormData();
     formData.append('image', imageFile);
     
     const response = await fetch('/api/analyze', {
       method: 'POST',
       body: formData
     });
     
     return response.json();
   };
   ```

## Development Workflow

1. **Setup Development Environment**:
   ```bash
   pip install -r requirements.txt
   export FLASK_ENV=development
   ```

2. **Run Tests** (if test data available):
   ```bash
   python predict.py test_images/ --config-env testing
   ```

3. **Start Development Server**:
   ```bash
   python api/app.py
   ```

## Production Deployment

1. **Use Production Configuration**:
   ```bash
   export FLASK_ENV=production
   ```

2. **Deploy with Gunicorn**:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
   ```

3. **Setup Reverse Proxy** (nginx example):
   ```nginx
   location /ml {
       proxy_pass http://localhost:5000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

## Monitoring and Logging

- Logs are stored in `logs/ml_api.log`
- Prediction results stored in SQLite database
- Performance metrics available via `/stats` endpoint

## Troubleshooting

### Common Issues

1. **Model Not Loading**:
   - Check if model file exists
   - Verify TensorFlow/Keras versions
   - Check file permissions

2. **Memory Errors**:
   - Reduce batch size
   - Use smaller images
   - Increase system memory

3. **Slow Predictions**:
   - Enable GPU acceleration
   - Optimize image preprocessing
   - Use model quantization

4. **API Errors**:
   - Check Flask logs
   - Verify image format
   - Check file size limits

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Ensure compatibility with existing API

## License

This module is part of the RecursiaDx project and follows the same licensing terms.

## Disclaimer

This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.