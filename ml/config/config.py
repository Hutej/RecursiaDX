import os
from pathlib import Path

# Base configuration
class Config:
    """Base configuration class for the ML module."""
    
    # Model configuration
    MODEL_INPUT_SIZE = (224, 224, 3)
    MODEL_CLASSES = ['Non-Tumor', 'Tumor']
    NUM_CLASSES = 2
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / 'models'
    UTILS_DIR = BASE_DIR / 'utils'
    API_DIR = BASE_DIR / 'api'
    DATA_DIR = BASE_DIR / 'data'
    UPLOADS_DIR = BASE_DIR / 'uploads'
    RESULTS_DIR = BASE_DIR / 'results'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Model files
    PRETRAINED_MODEL_PATH = MODELS_DIR / '__pycache__' / 'best_resnet50_model.pth'
    MODEL_WEIGHTS_PATH = MODELS_DIR / 'model_weights.h5'
    MODEL_CONFIG_PATH = MODELS_DIR / 'model_config.json'
    
    # Database
    DATABASE_PATH = BASE_DIR / 'tumor_predictions.db'
    
    # Image processing
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.dcm']
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Prediction thresholds
    HIGH_RISK_THRESHOLD = 0.9
    MODERATE_RISK_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.6
    
    # API configuration
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    API_DEBUG = False
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance
    BATCH_SIZE = 32
    MAX_BATCH_SIZE = 64
    PREDICTION_TIMEOUT = 30  # seconds
    
    # Security
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.UPLOADS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR,
            cls.MODELS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration."""
    API_DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    API_DEBUG = False
    LOG_LEVEL = 'WARNING'
    API_HOST = '127.0.0.1'

class TestingConfig(Config):
    """Testing configuration."""
    API_DEBUG = True
    LOG_LEVEL = 'DEBUG'
    DATABASE_PATH = Config.BASE_DIR / 'test_predictions.db'

# Configuration mapping
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(env_name='default'):
    """Get configuration based on environment name."""
    return config_map.get(env_name, DevelopmentConfig)

# Model hyperparameters
MODEL_HYPERPARAMETERS = {
    'learning_rate': 0.0001,
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
    'dropout_rate': 0.5,
    'l2_regularization': 0.001,
    'data_augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'horizontal_flip': True,
        'fill_mode': 'nearest'
    }
}

# Image preprocessing parameters
IMAGE_PREPROCESSING = {
    'target_size': (224, 224),
    'normalization_range': (0, 1),
    'enhance_contrast': True,
    'apply_clahe': True,
    'noise_reduction': True,
    'edge_enhancement': False
}

# Model evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc',
    'specificity',
    'sensitivity'
]

# Risk level definitions
RISK_LEVELS = {
    'high': {
        'threshold': 0.9,
        'color': '#ff0000',
        'description': 'High probability of tumor presence - immediate medical attention recommended'
    },
    'moderate': {
        'threshold': 0.7,
        'color': '#ff8800',
        'description': 'Moderate probability of tumor presence - further examination recommended'
    },
    'low_moderate': {
        'threshold': 0.5,
        'color': '#ffff00',
        'description': 'Low-moderate probability - monitoring recommended'
    },
    'low': {
        'threshold': 0.0,
        'color': '#00ff00',
        'description': 'Low probability of tumor presence - routine follow-up'
    }
}

# Error messages
ERROR_MESSAGES = {
    'invalid_image_format': 'Invalid image format. Supported formats: JPG, PNG, TIFF, BMP',
    'image_too_large': f'Image file too large. Maximum size: {Config.MAX_IMAGE_SIZE / (1024*1024):.1f}MB',
    'model_not_loaded': 'Model not loaded. Please initialize the model first.',
    'prediction_failed': 'Prediction failed. Please try again with a different image.',
    'invalid_image_path': 'Invalid image path or file does not exist.',
    'processing_timeout': 'Image processing timeout. Please try with a smaller image.',
    'insufficient_memory': 'Insufficient memory for processing. Please try with a smaller image.',
    'unknown_error': 'An unknown error occurred during processing.'
}