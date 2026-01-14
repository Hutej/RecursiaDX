"""
GigaPath AI - Breast Cancer Lesion Detection API
=================================================

Production-ready inference API for tissue biopsy analysis using 
Multiple Instance Learning (MIL) with Gated Attention.

INFERENCE-ONLY: This module requires a pre-trained checkpoint.
No training or dataset downloads are required.

Usage:
    python gigapath_api.py --checkpoint_path /path/to/best_model.pth --port 5002
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import base64
import io

# Flask imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from PIL import Image
import h5py

# WSI support
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    logger = None  # Will be set later

# ===================================================
# CONFIGURATION
# ===================================================
CHECKPOINT_PATH = None  # Set via command line or environment
DEVICE = None
MODEL = None
FEATURE_EXTRACTOR = None

# Supported image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('GigaPath-API')

# ===================================================
# MODEL ARCHITECTURE (Copied from GigaPath src/mil)
# ===================================================

class GatedAttention(nn.Module):
    """
    Gated attention mechanism for MIL.
    Reference: CLAM (Lu et al., 2021)
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.25
    ):
        super(GatedAttention, self).__init__()
        
        # Attention branch V (what to focus on)
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attn_dim)
        )
        
        # Gate branch U (how much to focus)
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attn_dim),
            nn.Sigmoid()
        )
        
        # Attention weights projection
        self.attention_w = nn.Linear(attn_dim, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = A_V * A_U
        attention_scores = self.attention_w(A)
        attention_weights = F.softmax(attention_scores, dim=0)
        return attention_weights


class AttentionMIL(nn.Module):
    """
    Attention-based MIL model for slide-level classification.
    Architecture: Features (K, 2048) → Gated Attention → Slide Embedding → Classifier
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.25
    ):
        super(AttentionMIL, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.attention = GatedAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
            dropout=dropout
        )
        
        # Binary classifier (single logit for BCEWithLogitsLoss)
        self.classifier = nn.Linear(input_dim, 1)
        
        logger.info(f"AttentionMIL initialized: input_dim={input_dim}")
    
    def forward(self, features: torch.Tensor, return_attention: bool = False):
        features = features.float()
        attention_weights = self.attention(features)
        slide_embedding = torch.sum(attention_weights * features, dim=0)
        logit = self.classifier(slide_embedding)
        
        if return_attention:
            return logit, attention_weights.squeeze()
        return logit, None
    
    def predict_slide(self, features: torch.Tensor, return_attention: bool = True) -> dict:
        """Predict slide-level label with attention weights."""
        self.eval()
        with torch.no_grad():
            logit, attention_weights = self.forward(features, return_attention=True)
            prob = torch.sigmoid(logit).item()
            prediction = 1 if prob > 0.5 else 0
        
        result = {
            'prediction': prediction,
            'probability': prob,
            'logit': logit.item()
        }
        
        if return_attention and attention_weights is not None:
            result['attention_weights'] = attention_weights.cpu().numpy()
        
        return result


class FeatureExtractor:
    """ResNet50-based feature extractor for histopathology patches."""
    
    def __init__(self, device):
        self.device = device
        
        # Load ResNet50 without final FC layer
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Identity()  # Remove classifier, keep 2048-dim features
        self.model = self.model.to(device)
        self.model.eval()
        
        # ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("FeatureExtractor initialized with ResNet50")
    
    def extract(self, image: Image.Image) -> torch.Tensor:
        """Extract 2048-dim features from a single image."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tensor)
        
        return features.squeeze()  # (2048,)
    
    def extract_batch(self, images: list) -> torch.Tensor:
        """Extract features from multiple images."""
        tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensors)
        
        return features  # (N, 2048)


# ===================================================
# WSI TIFF PROCESSING
# ===================================================

def process_wsi_tiff(file_path: str, feature_extractor, tile_size: int = 256, max_tiles: int = 100) -> torch.Tensor:
    """
    Process a WSI TIFF file by extracting tiles from pyramid levels.
    
    Args:
        file_path: Path to the TIFF file
        feature_extractor: FeatureExtractor instance
        tile_size: Size of each tile
        max_tiles: Maximum number of tiles to extract
        
    Returns:
        Tensor of features (N, 2048) where N is number of tiles
    """
    if not TIFFFILE_AVAILABLE:
        raise ImportError("tifffile is required for WSI processing. Install with: pip install tifffile imagecodecs")
    
    logger.info(f"Processing WSI TIFF: {file_path}")
    
    with tifffile.TiffFile(file_path) as tif:
        # Find a suitable pyramid level (not too big, not too small)
        # Aim for a page where we can extract at least max_tiles tiles
        best_page = None
        best_page_idx = 0
        
        for i, page in enumerate(tif.pages):
            h, w = page.shape[:2]
            if h >= tile_size * 5 and w >= tile_size * 5:
                # This page is large enough for multiple tiles
                if h * w < 100_000_000:  # Less than 100M pixels to avoid memory issues
                    best_page = page
                    best_page_idx = i
                    break
        
        if best_page is None:
            # Use the smallest valid page
            for i in range(len(tif.pages) - 1, -1, -1):
                page = tif.pages[i]
                h, w = page.shape[:2]
                if h >= tile_size and w >= tile_size:
                    best_page = page
                    best_page_idx = i
                    break
        
        if best_page is None:
            raise ValueError("No suitable pyramid level found in TIFF")
        
        h, w = best_page.shape[:2]
        logger.info(f"Using pyramid level {best_page_idx}: {w}x{h}")
        
        # Read the page data
        data = best_page.asarray()
        logger.info(f"Loaded image data: {data.shape}")
        
        # Extract tiles
        tiles = []
        n_tiles_x = w // tile_size
        n_tiles_y = h // tile_size
        
        # Calculate step to get roughly max_tiles tiles evenly distributed
        total_possible = n_tiles_x * n_tiles_y
        step = max(1, total_possible // max_tiles)
        
        tile_coords = []
        for y in range(0, n_tiles_y, max(1, int(np.sqrt(step)))):
            for x in range(0, n_tiles_x, max(1, int(np.sqrt(step)))):
                tile_coords.append((x * tile_size, y * tile_size))
                if len(tile_coords) >= max_tiles:
                    break
            if len(tile_coords) >= max_tiles:
                break
        
        logger.info(f"Extracting {len(tile_coords)} tiles of size {tile_size}x{tile_size}")
        
        for x, y in tile_coords:
            tile_data = data[y:y+tile_size, x:x+tile_size]
            if tile_data.shape[0] == tile_size and tile_data.shape[1] == tile_size:
                # Convert to PIL Image
                tile_img = Image.fromarray(tile_data)
                if tile_img.mode != 'RGB':
                    tile_img = tile_img.convert('RGB')
                tiles.append(tile_img)
        
        logger.info(f"Extracted {len(tiles)} valid tiles")
        
        if len(tiles) == 0:
            raise ValueError("No valid tiles could be extracted from TIFF")
        
        # Extract features from all tiles
        features = feature_extractor.extract_batch(tiles)
        logger.info(f"Extracted features: {features.shape}")
        
        return features


# ===================================================
# CHECKPOINT VALIDATION
# ===================================================

def validate_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate that the checkpoint file exists and is loadable.
    Returns True if valid, exits with clear error if not.
    """
    if not checkpoint_path:
        logger.error("=" * 60)
        logger.error("CHECKPOINT PATH NOT PROVIDED")
        logger.error("=" * 60)
        logger.error("This project requires a pre-trained best_model.pth")
        logger.error("generated during training.")
        logger.error("")
        logger.error("Usage:")
        logger.error("  python gigapath_api.py --checkpoint_path /path/to/best_model.pth")
        logger.error("")
        logger.error("No dataset is required for inference.")
        logger.error("=" * 60)
        return False
    
    if not os.path.exists(checkpoint_path):
        logger.error("=" * 60)
        logger.error("TRAINED CHECKPOINT NOT FOUND")
        logger.error("=" * 60)
        logger.error(f"Expected checkpoint at: {checkpoint_path}")
        logger.error("")
        logger.error("This project requires a pre-trained best_model.pth")
        logger.error("generated during training.")
        logger.error("")
        logger.error("No dataset is required for inference.")
        logger.error("Place best_model.pth at the specified path and restart.")
        logger.error("=" * 60)
        return False
    
    # Verify file size (should be ~27MB for this model)
    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1000000:  # Less than 1MB is suspicious
        logger.warning(f"Checkpoint file is unusually small ({file_size} bytes)")
    
    logger.info(f"✅ Checkpoint validated: {checkpoint_path} ({file_size / 1024 / 1024:.1f} MB)")
    return True


def load_model(checkpoint_path: str, device: torch.device) -> AttentionMIL:
    """Load trained MIL model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = AttentionMIL(
        input_dim=2048,
        hidden_dim=512,
        attn_dim=256,
        num_classes=2,
        dropout=0.25
    )
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded from epoch {checkpoint.get('epoch', 'N/A')}, AUC: {checkpoint.get('best_auc', 'N/A')}")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info("✅ Model loaded successfully in EVALUATION MODE")
    return model


# ===================================================
# FLASK APPLICATION
# ===================================================

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = Path(__file__).parent / 'uploads' / 'gigapath'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'GigaPath-AttentionMIL',
        'version': '1.0.0',
        'device': str(DEVICE),
        'checkpoint_loaded': MODEL is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict tumor presence in tissue biopsy image.
    
    Accepts:
        - multipart/form-data with 'image' file
        
    Returns:
        JSON with prediction results
    """
    start_time = datetime.now()
    
    try:
        # Validate model is loaded
        if MODEL is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Checkpoint may be missing.',
                'message': 'Please ensure best_model.pth is available.'
            }), 503
        
        # Check for file
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'message': 'Please upload an image file with key "image"'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Load and process image
        logger.info(f"Processing image: {file.filename}")
        
        image = Image.open(file.stream).convert('RGB')
        original_size = image.size
        
        # Extract features
        features = FEATURE_EXTRACTOR.extract(image)
        features = features.unsqueeze(0)  # Add batch dimension for single image
        
        # Run MIL inference
        result = MODEL.predict_slide(features, return_attention=True)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine risk level
        prob = result['probability']
        if result['prediction'] == 0:
            risk_level = 'Low Risk'
            predicted_class = 'Non-Tumor'  # Match backend expectations
        else:
            if prob >= 0.9:
                risk_level = 'High Risk'
            elif prob >= 0.7:
                risk_level = 'Medium Risk'
            else:
                risk_level = 'Low Risk'
            predicted_class = 'Tumor'
        
        # Build response
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'is_tumor': result['prediction'] == 1,
                'confidence': float(prob if result['prediction'] == 1 else 1 - prob),
                'probability': float(prob),
                'risk_level': risk_level
            },
            'probabilities': {
                'non_tumor': float(1 - prob),
                'tumor': float(prob)
            },
            'risk_assessment': risk_level.lower().replace(' risk', ''),  # 'low', 'medium', 'high'
            'metadata': {
                'model': 'GigaPath-AttentionMIL',
                'model_version': '1.0.0',
                'image_size': list(original_size),
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Prediction: {predicted_class} ({prob:.4f}) in {processing_time:.0f}ms")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500


@app.route('/predict_wsi', methods=['POST'])
def predict_wsi():
    """
    Predict tumor presence in a WSI TIFF file.
    
    Accepts JSON with:
        - file_path: Path to the TIFF file on disk
        - max_tiles: (optional) Maximum number of tiles to extract (default: 100)
        
    Returns:
        JSON with prediction results aggregated from all tiles
    """
    start_time = datetime.now()
    
    try:
        if MODEL is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        data = request.get_json() or {}
        file_path = data.get('file_path')
        max_tiles = data.get('max_tiles', 100)
        
        if not file_path:
            return jsonify({
                'success': False,
                'error': 'file_path is required'
            }), 400
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': f'File not found: {file_path}'
            }), 404
        
        logger.info(f"Processing WSI TIFF: {file_path}")
        
        # Process WSI and extract features from tiles
        features = process_wsi_tiff(file_path, FEATURE_EXTRACTOR, max_tiles=max_tiles)
        
        # Run MIL inference on all tile features
        result = MODEL.predict_slide(features, return_attention=True)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine risk level
        prob = result['probability']
        if result['prediction'] == 0:
            risk_level = 'Low Risk'
            predicted_class = 'Non-Tumor'
        else:
            if prob >= 0.9:
                risk_level = 'High Risk'
            elif prob >= 0.7:
                risk_level = 'Medium Risk'
            else:
                risk_level = 'Low Risk'
            predicted_class = 'Tumor'
        
        response = {
            'success': True,
            'prediction': {
                'class': predicted_class,
                'predicted_class': predicted_class,  # Backend compatibility
                'is_tumor': result['prediction'] == 1,
                'confidence': float(prob if result['prediction'] == 1 else 1 - prob),
                'probability': float(prob),
                'risk_level': risk_level
            },
            'probabilities': {
                'non_tumor': float(1 - prob),
                'tumor': float(prob)
            },
            'risk_assessment': risk_level.lower().replace(' risk', ''),
            'metadata': {
                'model': 'GigaPath-AttentionMIL',
                'model_version': '1.0.0',
                'wsi_file': os.path.basename(file_path),
                'tiles_processed': features.shape[0],
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"WSI Prediction: {predicted_class} ({prob:.4f}) from {features.shape[0]} tiles in {processing_time:.0f}ms")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"WSI prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during WSI prediction'
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict on multiple images.
    Accepts multiple files with key 'images[]'.
    """
    start_time = datetime.now()
    
    try:
        if MODEL is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503
        
        if 'images[]' not in request.files and 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        files = request.files.getlist('images[]') or request.files.getlist('images')
        
        results = []
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    image = Image.open(file.stream).convert('RGB')
                    features = FEATURE_EXTRACTOR.extract(image)
                    features = features.unsqueeze(0)
                    
                    result = MODEL.predict_slide(features)
                    prob = result['probability']
                    
                    results.append({
                        'filename': secure_filename(file.filename),
                        'prediction': 'Tumor' if result['prediction'] == 1 else 'Normal',
                        'is_tumor': result['prediction'] == 1,
                        'confidence': float(prob if result['prediction'] == 1 else 1 - prob),
                        'probability': float(prob)
                    })
                except Exception as e:
                    results.append({
                        'filename': secure_filename(file.filename),
                        'error': str(e)
                    })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results),
            'processing_time_ms': round(processing_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model architecture and configuration info."""
    return jsonify({
        'model': 'GigaPath-AttentionMIL',
        'architecture': {
            'backbone': 'ResNet50 (ImageNet pretrained)',
            'feature_dim': 2048,
            'hidden_dim': 512,
            'attention_dim': 256,
            'classifier': 'Binary (Tumor/Normal)'
        },
        'input': {
            'format': list(ALLOWED_EXTENSIONS),
            'preprocessing': 'Resize 256 → CenterCrop 224 → ImageNet normalization'
        },
        'output': {
            'classes': ['Normal', 'Tumor'],
            'type': 'Binary classification with confidence score'
        },
        'device': str(DEVICE),
        'ready': MODEL is not None
    })


# ===================================================
# MAIN ENTRY POINT
# ===================================================

def main():
    global CHECKPOINT_PATH, DEVICE, MODEL, FEATURE_EXTRACTOR
    
    parser = argparse.ArgumentParser(
        description='GigaPath AI - Tissue Biopsy Cancer Detection API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INFERENCE-ONLY SETUP:
  This API requires a pre-trained checkpoint (best_model.pth).
  No dataset or training is required.

EXAMPLE:
  python gigapath_api.py --checkpoint_path checkpoints/best_model.pth --port 5002
        """
    )
    
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=os.environ.get('GIGAPATH_CHECKPOINT', None),
        help='Path to trained model checkpoint (best_model.pth)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.environ.get('GIGAPATH_PORT', 5002)),
        help='Port to run the API (default: 5002)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Try to find checkpoint in default locations if not provided
    if not args.checkpoint_path:
        default_paths = [
            Path(__file__).parent.parent.parent / 'GigaPath-AI-WSI-Breast-Cancer-Lesion-Analysis' / 'checkpoints' / 'best_model.pth',
            Path(__file__).parent / 'checkpoints' / 'best_model.pth',
            Path('checkpoints') / 'best_model.pth',
        ]
        for path in default_paths:
            if path.exists():
                args.checkpoint_path = str(path)
                logger.info(f"Found checkpoint at default location: {path}")
                break
    
    # Validate checkpoint
    if not validate_checkpoint(args.checkpoint_path):
        sys.exit(1)
    
    CHECKPOINT_PATH = args.checkpoint_path
    
    # Setup device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")
    
    if DEVICE.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load model
    try:
        MODEL = load_model(CHECKPOINT_PATH, DEVICE)
        FEATURE_EXTRACTOR = FeatureExtractor(DEVICE)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Print startup banner
    print("\n" + "=" * 60)
    print("   GigaPath AI - Breast Cancer Lesion Detection API")
    print("=" * 60)
    print(f"   🚀 Server: http://{args.host}:{args.port}")
    print(f"   📊 Model: AttentionMIL (Gated Attention)")
    print(f"   🔧 Device: {DEVICE}")
    print(f"   ✅ Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)
    print("   Endpoints:")
    print("   - GET  /health      - Health check")
    print("   - POST /predict     - Single image prediction")
    print("   - POST /batch_predict - Batch prediction")
    print("   - GET  /model_info  - Model information")
    print("=" * 60 + "\n")
    
    # Run Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
