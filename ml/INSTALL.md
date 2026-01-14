# Quick Installation Guide for RecursiaDx ML Module

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Steps

### 1. Navigate to ML Directory
```bash
cd ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install tensorflow==2.15.0
pip install keras==2.15.0
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install Pillow==10.0.1
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install pandas==2.1.4
pip install flask==3.0.0
pip install flask-cors==4.0.0
pip install requests==2.31.0
pip install python-dotenv==1.0.0
pip install gunicorn==21.2.0
```

### 3. Test Installation
```bash
python test_installation.py
```

### 4. Start ML API Server
```bash
python start_server.py
```

Or use the PowerShell script (Windows):
```powershell
.\start_ml_server.ps1 -InstallDeps
```

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Test the installation
python test_installation.py

# Start the ML API server
python start_server.py

# Make a prediction
python predict.py path/to/image.jpg --enhance --save-viz

# Train a new model (if you have training data)
python train_model.py --data_dir path/to/training/data --epochs 50
```

## API Usage

Once the server is running, you can use the API:

```bash
# Health check
curl http://localhost:5000/health

# Predict tumor from image
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict

# Get model information
curl http://localhost:5000/model_info
```

## Integration with RecursiaDx Backend

1. Copy `integration_example.js` to your backend directory
2. Install Node.js dependencies:
   ```bash
   npm install multer form-data node-fetch
   ```
3. Import and use the integration functions in your routes

## Troubleshooting

### Common Issues:

1. **TensorFlow Installation Issues**: 
   - Ensure you have Python 3.8-3.11
   - Try: `pip install --upgrade tensorflow`

2. **Memory Issues**:
   - Reduce batch size in configuration
   - Use smaller images for testing

3. **API Connection Issues**:
   - Check if ML server is running on port 5000
   - Verify firewall settings

4. **Model Not Found**:
   - The system will use an untrained model initially
   - Train your own model or obtain a pre-trained one