from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import logging
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
from PIL import Image
import io
import base64
import traceback

# ===================================================
# 🔒 DETERMINISTIC SETUP - MUST BE BEFORE TORCH IMPORT
# ===================================================
import random
random.seed(42)

import numpy as np
np.random.seed(42)

# Set environment variables for deterministic behavior
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Import our ML modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.malaria_predictor import MalariaPredictor
from models.platelet_counter import PlateletCounter
from utils.image_utils import validate_image_format, get_image_info, enhance_medical_image
from utils.data_manager import DataManager, save_prediction_report, validate_prediction_data
from config.config import get_config, ERROR_MESSAGES
import requests as http_requests  # For proxying to GigaPath API

# GigaPath API configuration
GIGAPATH_API_URL = os.environ.get('GIGAPATH_API_URL', 'http://localhost:5002')
# from pipeline import HistopathologyPipeline  # Commented out - not needed for predictions
# from classifier import PatchClassifier  # Commented out - not needed for predictions
# from aggregation import HeatmapGenerator  # Commented out - not needed for predictions

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = get_config(os.getenv('FLASK_ENV', 'development'))
config.create_directories()

# Configure Flask app
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOADS_DIR

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'ml_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components (tumor detection removed - now handled by GigaPath API)
malaria_predictor = None
platelet_counter = None
pipeline = None
data_manager = DataManager(str(config.DATABASE_PATH))

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # NumPy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # NumPy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def initialize_models():
    """Initialize all ML models: tumor, malaria, and platelet detection."""
    global tumor_predictor, malaria_predictor, platelet_counter, pipeline
    try:
        # ===================================================
        # 🔒 PYTORCH DETERMINISTIC SETUP
        # ===================================================
        import torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)  # Disable gradients for inference
        
        # Use deterministic algorithms when available
        try:
            torch.use_deterministic_algorithms(True)
            logger.info("✅ PyTorch deterministic algorithms enabled")
        except Exception as det_error:
            logger.warning(f"⚠️ Could not enable deterministic algorithms: {det_error}")
        
        # ===================================================
        # TISSUE DETECTION: Handled by GigaPath API (port 5002)
        # ===================================================
        logger.info(f"🔗 Tissue analysis will be proxied to GigaPath API at: {GIGAPATH_API_URL}")
        try:
            gigapath_health = http_requests.get(f"{GIGAPATH_API_URL}/health", timeout=2)
            if gigapath_health.status_code == 200:
                logger.info("✅ GigaPath API is available")
            else:
                logger.warning("⚠️ GigaPath API returned non-200 status")
        except Exception:
            logger.warning("⚠️ GigaPath API not reachable - tissue analysis may fail")
            logger.warning(f"   Start GigaPath API: python gigapath_api.py --port 5002")
        
        # ===================================================
        # INITIALIZE MALARIA DETECTION MODEL (for blood smear images)
        # ===================================================
        try:
            malaria_model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Malaria-Disease-Detection-Using-Transfer-Learning', 'InceptionV3_Malaria_PyTorch.pth')
            malaria_predictor = MalariaPredictor(model_path=malaria_model_path)
            if malaria_predictor.model is not None:
                logger.info("✅ Malaria detection model loaded successfully")
            else:
                logger.warning("⚠️ Malaria detection model not available")
        except Exception as mal_error:
            logger.warning(f"⚠️ Could not load malaria model: {mal_error}")
            malaria_predictor = None
        
        # ===================================================
        # INITIALIZE PLATELET COUNTER (for blood smear images)
        # ===================================================
        try:
            platelet_counter = PlateletCounter()
            if platelet_counter.model is not None:
                logger.info("✅ Platelet counting model loaded successfully")
            else:
                logger.warning("⚠️ Platelet counting model not available")
        except Exception as plat_error:
            logger.warning(f"⚠️ Could not load platelet counter: {plat_error}")
            platelet_counter = None
        
        logger.info("="*70)
        logger.info("✅ Server initialization complete")
        logger.info(f"   - Tissue Detection: Proxied to GigaPath API ({GIGAPATH_API_URL})")
        logger.info(f"   - Malaria Detection: {'✓' if malaria_predictor and malaria_predictor.model else '✗'}")
        logger.info(f"   - Platelet Counting: {'✓' if platelet_counter and platelet_counter.model else '✗'}")
        logger.info("="*70)
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': ERROR_MESSAGES['image_too_large']
    }), 413

@app.errorhandler(500)
def handle_internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'success': False,
        'error': ERROR_MESSAGES['unknown_error']
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global malaria_predictor, platelet_counter, pipeline
    
    # Check GigaPath API status
    gigapath_available = False
    try:
        response = http_requests.get(f"{GIGAPATH_API_URL}/health", timeout=2)
        gigapath_available = response.status_code == 200
    except:
        pass
    
    return jsonify({
        'status': 'healthy',
        'models': {
            'tissue_detection': gigapath_available,
            'malaria_detection': malaria_predictor is not None and malaria_predictor.model is not None,
            'platelet_counting': platelet_counter is not None and platelet_counter.model is not None,
        },
        'gigapath_api': GIGAPATH_API_URL,
        'pipeline_loaded': pipeline is not None,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict_tumor():
    """Predict from uploaded image with routing based on imageType."""
    global malaria_predictor, platelet_counter
    
    try:
        # Get imageType parameter to route to correct model
        image_type = request.form.get('imageType', 'tissue').lower()
        
        logger.info(f"📋 Single predict request for imageType: {image_type}")
        
        # Route to appropriate model
        if image_type == 'tissue':
            # Tissue analysis is handled by GigaPath API proxy (see /batch_predict)
            model_name = "Tissue Detection (GigaPath-AttentionMIL)"
        elif image_type == 'blood':
            if malaria_predictor is None or malaria_predictor.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Malaria detection model not loaded'
                }), 500
            if platelet_counter is None or platelet_counter.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Platelet counting model not loaded'
                }), 500
            model_name = "Blood Analysis (Malaria + Platelet Count)"
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid imageType: {image_type}. Must be "tissue" or "blood"'
            }), 400
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['invalid_image_format']
            }), 400
        
        # Get optional parameters
        user_id = request.form.get('user_id')
        enhance_image = request.form.get('enhance_image', 'false').lower() == 'true'
        save_result = request.form.get('save_result', 'true').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOADS_DIR, filename)
        file.save(filepath)
        
        logger.info(f"Processing image: {filename}")
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = Image.open(filepath)
            image_array = np.array(image.convert('RGB'))
            
            # Apply enhancement if requested
            if enhance_image:
                image_array = enhance_medical_image(image_array)
            
            # Make prediction based on image type
            if image_type == 'tissue':
                # Proxy to GigaPath API for tissue analysis
                try:
                    # Re-open file for sending to GigaPath API
                    with open(filepath, 'rb') as img_file:
                        files = {'image': (filename, img_file, 'image/png')}
                        gigapath_response = http_requests.post(
                            f"{GIGAPATH_API_URL}/predict",
                            files=files,
                            timeout=60
                        )
                    
                    if gigapath_response.status_code == 200:
                        gigapath_data = gigapath_response.json()
                        if gigapath_data.get('success'):
                            # Transform GigaPath response to match expected format
                            prediction_result = {
                                'predicted_class': gigapath_data['prediction']['class'],
                                'confidence': gigapath_data['prediction']['confidence'],
                                'is_tumor': gigapath_data['prediction']['is_tumor'],
                                'probabilities': gigapath_data['probabilities'],
                                'risk_level': gigapath_data['prediction']['risk_level'],
                                'risk_assessment': gigapath_data.get('risk_assessment', 'medium')
                            }
                        else:
                            raise Exception(gigapath_data.get('error', 'GigaPath prediction failed'))
                    else:
                        raise Exception(f"GigaPath API returned status {gigapath_response.status_code}")
                        
                except http_requests.exceptions.ConnectionError:
                    return jsonify({
                        'success': False,
                        'error': 'GigaPath API is not available',
                        'message': 'Please start the GigaPath API: python gigapath_api.py --port 5002'
                    }), 503
                except http_requests.exceptions.Timeout:
                    return jsonify({
                        'success': False,
                        'error': 'GigaPath API request timed out'
                    }), 504
                    
            elif image_type == 'blood':
                # Run both malaria and platelet detection on same image
                malaria_result = malaria_predictor.predict(image_array)
                platelet_result = platelet_counter.predict(image_array)
                
                # Combine results
                prediction_result = {
                    'malaria_detection': malaria_result,
                    'blood_cell_count': platelet_result,
                    'predicted_class': 'Blood Analysis Complete',
                    'confidence': (malaria_result['confidence'] + platelet_result['confidence']) / 2
                }
            
            processing_time = time.time() - start_time
            
            # Validate prediction result
            if not validate_prediction_data(prediction_result):
                raise ValueError("Invalid prediction result")
            
            # Get image info
            image_info = get_image_info(filepath)
            
            # Prepare response
            response_data = {
                'success': True,
                'prediction': prediction_result,
                'image_type': image_type,
                'model_used': model_name,
                'image_info': {
                    'filename': filename,
                    'size': image_info.get('size', 'Unknown'),
                    'format': image_info.get('format', 'Unknown')
                },
                'processing_time': round(processing_time, 3),
                'timestamp': time.time(),
                'model_version': model_name
            }
            
            # Save to database if requested
            if save_result:
                try:
                    record_id = data_manager.save_prediction(
                        filepath, 
                        prediction_result, 
                        user_id=user_id,
                        processing_time=processing_time
                    )
                    response_data['record_id'] = record_id
                except Exception as e:
                    logger.warning(f"Failed to save prediction to database: {str(e)}")
            
            logger.info(f"Prediction completed: {prediction_result['predicted_class']} "
                       f"({prediction_result['confidence']:.3f}) in {processing_time:.3f}s")
            
            # Convert NumPy types to JSON-serializable types
            response_data = convert_numpy_types(response_data)
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['prediction_failed'],
                'details': str(e) if app.debug else None
            }), 500
            
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': ERROR_MESSAGES['unknown_error'],
            'details': str(e) if app.debug else None
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_tumor_base64():
    """Predict tumor from base64 encoded image."""
    global predictor
    
    try:
        # Check if model is loaded
        if predictor is None:
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['model_not_loaded']
            }), 500
        
        # Get JSON data
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No base64 image data provided'
            }), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image.convert('RGB'))
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Invalid base64 image data'
            }), 400
        
        # Get optional parameters
        user_id = data.get('user_id')
        enhance_image = data.get('enhance_image', False)
        
        logger.info("Processing base64 image")
        start_time = time.time()
        
        # Apply enhancement if requested
        if enhance_image:
            image_array = enhance_medical_image(image_array)
        
        # Make prediction
        prediction_result = predictor.predict(image_array)
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            'success': True,
            'prediction': prediction_result,
            'processing_time': round(processing_time, 3),
            'timestamp': time.time(),
            'model_version': 'ResNet50_v1'
        }
        
        logger.info(f"Base64 prediction completed: {prediction_result['predicted_class']} "
                   f"({prediction_result['confidence']:.3f}) in {processing_time:.3f}s")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Base64 prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': ERROR_MESSAGES['prediction_failed'],
            'details': str(e) if app.debug else None
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple images at once with routing based on imageType."""
    global malaria_predictor, platelet_counter
    
    try:
        # Get imageType parameter to route to correct model
        image_type = request.form.get('imageType', 'tissue').lower()
        
        logger.info(f"📋 Batch predict request for imageType: {image_type}")
        
        # Route to appropriate model
        if image_type == 'tissue':
            # Tissue analysis uses GigaPath API
            model_name = "Tissue Detection (GigaPath-AttentionMIL)"
        elif image_type == 'blood':
            # For blood smear, we'll run both malaria and platelet detection
            if malaria_predictor is None or malaria_predictor.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Malaria detection model not loaded'
                }), 500
            if platelet_counter is None or platelet_counter.model is None:
                return jsonify({
                    'success': False,
                    'error': 'Platelet counting model not loaded'
                }), 500
            model_name = "Blood Analysis (Malaria + Platelet Count)"
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid imageType: {image_type}. Must be "tissue" or "blood"'
            }), 400
        
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image files provided'
            }), 400
        
        files = request.files.getlist('images')
        if len(files) > config.MAX_BATCH_SIZE:
            return jsonify({
                'success': False,
                'error': f'Too many files. Maximum batch size: {config.MAX_BATCH_SIZE}'
            }), 400
        
        user_id = request.form.get('user_id')
        enhance_images = request.form.get('enhance_images', 'false').lower() == 'true'
        
        results = []
        start_time = time.time()
        
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Invalid file'
                })
                continue
            
            try:
                # Process individual image
                image = Image.open(file.stream)
                image_array = np.array(image.convert('RGB'))
                
                if enhance_images:
                    image_array = enhance_medical_image(image_array)
                
                # Route based on image type
                if image_type == 'tissue':
                    # Proxy to GigaPath API for tissue analysis
                    try:
                        # Save image temporarily to send to GigaPath
                        import io
                        img_bytes = io.BytesIO()
                        Image.fromarray(image_array).save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        
                        files_to_send = {'image': (file.filename, img_bytes, 'image/png')}
                        gigapath_response = http_requests.post(
                            f"{GIGAPATH_API_URL}/predict",
                            files=files_to_send,
                            timeout=60
                        )
                        
                        if gigapath_response.status_code == 200:
                            gigapath_data = gigapath_response.json()
                            if gigapath_data.get('success'):
                                prediction_result = {
                                    'predicted_class': gigapath_data['prediction']['class'],
                                    'confidence': gigapath_data['prediction']['confidence'],
                                    'is_tumor': gigapath_data['prediction']['is_tumor'],
                                    'probabilities': gigapath_data['probabilities'],
                                    'risk_level': gigapath_data['prediction']['risk_level'],
                                    'risk_assessment': gigapath_data.get('risk_assessment', 'medium')
                                }
                                results.append({
                                    'filename': file.filename,
                                    'success': True,
                                    'prediction': prediction_result,
                                    'model_used': model_name
                                })
                            else:
                                results.append({
                                    'filename': file.filename,
                                    'success': False,
                                    'error': gigapath_data.get('error', 'GigaPath prediction failed')
                                })
                        else:
                            results.append({
                                'filename': file.filename,
                                'success': False,
                                'error': f'GigaPath API returned status {gigapath_response.status_code}'
                            })
                    except http_requests.exceptions.ConnectionError:
                        return jsonify({
                            'success': False,
                            'error': 'GigaPath API is not available',
                            'message': 'Please start the GigaPath API: python gigapath_api.py --port 5002'
                        }), 503
                    except http_requests.exceptions.Timeout:
                        results.append({
                            'filename': file.filename,
                            'success': False,
                            'error': 'GigaPath API request timed out'
                        })
                    
                elif image_type == 'blood':
                    # Run both malaria and platelet detection on same image
                    malaria_result = malaria_predictor.predict(image_array)
                    platelet_result = platelet_counter.predict(image_array)
                    
                    # Combine results
                    combined_result = {
                        'malaria_detection': convert_numpy_types(malaria_result),
                        'blood_cell_count': convert_numpy_types(platelet_result),
                        'predicted_class': 'Blood Analysis Complete',
                        'confidence': (malaria_result['confidence'] + platelet_result['confidence']) / 2
                    }
                    
                    results.append({
                        'filename': file.filename,
                        'success': True,
                        'prediction': combined_result,
                        'model_used': model_name
                    })
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'results': results,
            'image_type': image_type,
            'model_used': model_name,
            'total_images': len(files),
            'successful_predictions': sum(1 for r in results if r['success']),
            'total_processing_time': round(total_time, 3)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': ERROR_MESSAGES['unknown_error'],
            'details': str(e) if app.debug else None
        }), 500

@app.route('/history', methods=['GET'])
def get_prediction_history():
    """Get prediction history."""
    try:
        user_id = request.args.get('user_id')
        limit = int(request.args.get('limit', 100))
        
        predictions = data_manager.get_predictions(user_id=user_id, limit=limit)
        stats = data_manager.get_prediction_stats(user_id=user_id)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get history: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve prediction history'
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get prediction statistics."""
    try:
        user_id = request.args.get('user_id')
        stats = data_manager.get_prediction_stats(user_id=user_id)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve statistics'
        }), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get model information."""
    global predictor, pipeline
    
    try:
        if predictor is None:
            return jsonify({
                'success': False,
                'error': ERROR_MESSAGES['model_not_loaded']
            }), 500
        
        info = {
            'success': True,
            'model_type': 'ResNet50',
            'input_shape': config.MODEL_INPUT_SIZE,
            'classes': config.MODEL_CLASSES,
            'num_classes': config.NUM_CLASSES,
            'version': 'v1.0',
            'description': 'Pre-trained ResNet50 model fine-tuned for tumor detection',
            'pipeline_available': pipeline is not None
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve model information'
        }), 500

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    """Generate real heatmap from image using patch-based prediction."""
    global tumor_predictor
    
    try:
        # Check if model is loaded
        if tumor_predictor is None or tumor_predictor.model is None:
            return jsonify({
                'success': False,
                'error': 'Tumor detection model not available'
            }), 503
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
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
                'error': ERROR_MESSAGES['invalid_image_format']
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(config.UPLOADS_DIR, filename)
        file.save(filepath)
        
        # Also save heatmap output
        heatmap_filename = f"heatmap_{filename}"
        heatmap_filepath = os.path.join(config.UPLOADS_DIR, heatmap_filename)
        
        logger.info(f"Generating heatmap for: {filename}")
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(filepath)
            image_array = np.array(image.convert('RGB'))
            img_height, img_width = image_array.shape[:2]
            
            # Get overall prediction first
            prediction_result = tumor_predictor.predict(image_array)
            
            # Create heatmap using patch-based prediction
            patch_size = 112  # Smaller patches for more detail
            stride = 56  # 50% overlap for smoother heatmap
            
            # Initialize probability map
            prob_map = np.zeros((img_height, img_width), dtype=np.float32)
            count_map = np.zeros((img_height, img_width), dtype=np.float32)
            
            # Process patches
            for y in range(0, img_height - patch_size + 1, stride):
                for x in range(0, img_width - patch_size + 1, stride):
                    # Extract patch
                    patch = image_array[y:y+patch_size, x:x+patch_size]
                    
                    # Resize patch to model input size if needed
                    patch_resized = np.array(Image.fromarray(patch).resize((224, 224)))
                    
                    # Predict on patch
                    try:
                        patch_pred = tumor_predictor.predict(patch_resized)
                        tumor_prob = patch_pred.get('probabilities', {}).get('tumor', 0.5)
                        if hasattr(tumor_prob, 'item'):
                            tumor_prob = tumor_prob.item()
                    except Exception as patch_err:
                        logger.warning(f"Patch prediction failed: {patch_err}")
                        tumor_prob = 0.5
                    
                    # Add to probability map with Gaussian weighting
                    prob_map[y:y+patch_size, x:x+patch_size] += tumor_prob
                    count_map[y:y+patch_size, x:x+patch_size] += 1
            
            # Avoid division by zero
            count_map[count_map == 0] = 1
            prob_map = prob_map / count_map
            
            # Apply Gaussian smoothing
            try:
                from scipy.ndimage import gaussian_filter
                prob_map = gaussian_filter(prob_map, sigma=5)
            except ImportError:
                pass  # Skip smoothing if scipy not available
            
            # Normalize to 0-255 for colormap
            prob_map_normalized = (prob_map * 255).astype(np.uint8)
            
            # Apply colormap (blue=low, red=high)
            try:
                import cv2
                colored_heatmap = cv2.applyColorMap(prob_map_normalized, cv2.COLORMAP_JET)
                colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            except ImportError:
                # Fallback: simple red-blue colormap
                colored_heatmap = np.zeros((img_height, img_width, 3), dtype=np.uint8)
                colored_heatmap[:, :, 0] = prob_map_normalized  # Red channel
                colored_heatmap[:, :, 2] = 255 - prob_map_normalized  # Blue channel
            
            # Create overlay
            alpha = 0.5
            overlay = (image_array * (1 - alpha) + colored_heatmap * alpha).astype(np.uint8)
            
            # Save heatmap overlay
            heatmap_pil = Image.fromarray(overlay)
            heatmap_pil.save(heatmap_filepath, 'PNG')
            
            # Also create base64 version
            import io
            import base64
            buffer = io.BytesIO()
            heatmap_pil.save(buffer, format='PNG')
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response_data = {
                'success': True,
                'prediction': convert_numpy_types(prediction_result),
                'heatmap': {
                    'url': f'/uploads/{heatmap_filename}',
                    'base64': f'data:image/png;base64,{heatmap_base64}',
                    'type': 'tumor_probability',
                    'colormap': 'jet',
                    'analytics': {
                        'min_value': float(prob_map.min()),
                        'max_value': float(prob_map.max()),
                        'mean_value': float(prob_map.mean()),
                        'std_value': float(prob_map.std())
                    }
                },
                'processing_time': round(processing_time, 3),
                'timestamp': time.time()
            }
            
            logger.info(f"Heatmap generated in {processing_time:.3f}s")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': 'Heatmap generation failed',
                'details': str(e)
            }), 500
            
        finally:
            # Clean up original uploaded file (keep heatmap)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Request processing failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': ERROR_MESSAGES['unknown_error'],
            'details': str(e)
        }), 500

if __name__ == '__main__':
    logger.info("Starting RecursiaDx ML API Server")
    logger.info("=" * 70)
    
    # Initialize model
    if initialize_models():
        logger.info("✅ Server initialization successful")
        logger.info("=" * 70)
        
        # Run Flask app
        port = int(os.getenv('ML_API_PORT', 5000))
        logger.info(f"🚀 Starting server on port {port}...")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True
        )
    else:
        logger.error("❌ Server initialization failed - Model not available")
        logger.error("Please ensure trained model file exists:")
        logger.error("  ml/models/__pycache__/best_resnet50_model.pth")
        sys.exit(1)