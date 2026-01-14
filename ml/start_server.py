#!/usr/bin/env python3
"""
Startup script for RecursiaDx ML API server.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging(log_level='INFO'):
    """Setup logging configuration."""
    from config.config import get_config
    
    config = get_config()
    config.create_directories()
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOGS_DIR / 'ml_startup.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_pytorch_device():
    """Check PyTorch device availability."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"[GPU] Device: {device}")
            logger.info(f"[GPU] Memory: {memory:.1f} GB")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("[MPS] Using Apple Metal Performance Shaders")
            return 'mps'
        else:
            logger.info("[CPU] Using CPU device")
            return 'cpu'
            
    except Exception as e:
        logger.warning(f"[WARNING] Device check failed: {e}")
        return 'cpu'

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'opencv-python',
        'Pillow',
        'flask',
        'flask-cors'
    ]
    
    optional_packages = [
        'scikit-learn'
    ]
    
    missing_packages = []
    
    # Check required packages
    for package in required_packages:
        try:
            # Handle special cases for package imports
            if package == 'opencv-python':
                import cv2
            elif package == 'torch':
                import torch
            elif package == 'torchvision':
                import torchvision
            elif package == 'Pillow':
                import PIL
            elif package == 'flask-cors':
                import flask_cors
            else:
                __import__(package.replace('-', '_'))
            logger.info(f"[OK] {package} is available")
        except ImportError as e:
            missing_packages.append(package)
            logger.error(f"[MISSING] {package} is missing: {e}")
    
    # Check optional packages
    for package in optional_packages:
        try:
            if package == 'scikit-learn':
                # Handle NumPy compatibility issues with scikit-learn
                try:
                    import sklearn
                    logger.info(f"[OK] {package} is available")
                except ImportError as sklearn_error:
                    if "NumPy" in str(sklearn_error):
                        logger.warning(f"[WARNING] {package} has NumPy compatibility issues")
                        logger.warning(f"[SUGGESTION] Consider: pip install 'numpy<2' 'scikit-learn'")
                        logger.info(f"[SKIP] {package} skipped (optional)")
                    else:
                        logger.warning(f"[WARNING] {package} not available: {sklearn_error}")
            else:
                __import__(package.replace('-', '_'))
                logger.info(f"[OK] {package} is available")
        except ImportError as e:
            logger.warning(f"[WARNING] Optional package {package} not available: {e}")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages using:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("All dependencies are available")
    return True

def test_model_loading():
    """Test if the model can be loaded."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        
        # Check PyTorch installation and CUDA availability
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("[WARNING] CUDA not available, using CPU")
        
        from models.tumor_predictor import TumorPredictor
        from config.config import get_config
        
        config = get_config()
        predictor = TumorPredictor()
        
        # Check if pre-trained model exists
        if config.PRETRAINED_MODEL_PATH.exists():
            predictor.load_model(str(config.PRETRAINED_MODEL_PATH))
            logger.info("[OK] Pre-trained model loaded successfully")
        else:
            predictor.build_model()
            logger.warning("[WARNING] Using untrained model - training recommended")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Model loading failed: {e}")
        return False

def start_api_server(host='0.0.0.0', port=5000, debug=False, workers=1):
    """Start the ML API server."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import the Flask app and initialization function
        from api.app import app, initialize_models
        
        # Initialize all models (tumor, malaria, platelet)
        logger.info("[INIT] Initializing all ML models...")
        if not initialize_models():
            logger.error("[ERROR] Failed to initialize models")
            return False
        
        if workers > 1:
            # Use Gunicorn for production
            logger.info(f"Starting ML API server with Gunicorn ({workers} workers)")
            logger.info(f"Server will be available at http://{host}:{port}")
            
            os.system(f"gunicorn -w {workers} -b {host}:{port} api.app:app")
        else:
            # Use Flask development server
            logger.info(f"Starting ML API development server")
            logger.info(f"Server will be available at http://{host}:{port}")
            
            app.run(host=host, port=port, debug=debug)
            
    except Exception as e:
        logger.error(f"[ERROR] Failed to start API server: {e}")
        return False
    
    return True

def main():
    """Main startup function."""
    parser = argparse.ArgumentParser(description='RecursiaDx ML API Startup Script')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    parser.add_argument('--skip-checks', action='store_true', help='Skip dependency and model checks')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("[STARTUP] Starting RecursiaDx ML API Server (PyTorch)")
    logger.info("=" * 50)
    
    # Perform startup checks
    if not args.skip_checks:
        logger.info("[CHECKS] Performing startup checks...")
        
        # Check PyTorch device
        device = check_pytorch_device()
        logger.info(f"[DEVICE] Selected device: {device}")
        
        # Check dependencies
        if not check_dependencies():
            logger.error("[ERROR] Dependency check failed")
            return 1
        
        # Test model loading
        if not test_model_loading():
            logger.error("[ERROR] Model loading test failed")
            return 1
        
        logger.info("[OK] All startup checks passed")
    else:
        logger.info("[SKIP] Skipping startup checks")
    
    # Start the server
    logger.info("[SERVER] Starting API server...")
    
    try:
        start_api_server(
            host=args.host,
            port=args.port,
            debug=args.debug,
            workers=args.workers
        )
        return 0
    except KeyboardInterrupt:
        logger.info("[STOP] Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"[ERROR] Server startup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())