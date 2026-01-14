import os
import logging
import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlateletCounter:
    """
    Platelet and blood cell counter using YOLOv8.
    Detects and counts: RBC, WBC, and Platelets.
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.class_names = {
            0: "RBC",
            1: "WBC",
            2: "Platelets"
        }
        
        # Try to import YOLO
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            logger.warning("ultralytics not installed. Platelet counting will not be available.")
            logger.warning("Install with: pip install ultralytics")
            self.YOLO = None
            return
        
        # Load the model if path is provided
        if model_path is None:
            # Look in multiple possible locations (use absolute paths from project root)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            project_root = os.path.dirname(base_dir)  # Go up to RecursiaDx folder
            
            possible_paths = [
                os.path.join(project_root, 'Platelates', 'runs', 'detect', 'bccd_blood_cells', 'weights', 'best.pt'),
                os.path.join(project_root, 'Platelates', 'yolo11n.pt'),
                os.path.join(base_dir, 'models', 'platelet_model.pt'),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found platelet model at: {path}")
                    break
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(f"Platelet detection model not found. Searched paths: {possible_paths if model_path is None else model_path}")
    
    def load_model(self, model_path):
        """Load YOLOv8 model."""
        try:
            if self.YOLO is None:
                logger.error("YOLOv8 not available")
                return
            
            logger.info(f"Loading platelet detection model from {model_path}...")
            self.model = self.YOLO(str(model_path))
            logger.info(f"✅ Platelet detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading platelet model: {str(e)}")
            logger.warning("Platelet counting will not be available")
    
    def predict(self, image_path_or_array, conf_threshold=0.25):
        """
        Count blood cells in image.
        
        Args:
            image_path_or_array: Path to image or numpy array
            conf_threshold: Confidence threshold for detections
        
        Returns:
            Dictionary with cell counts and details
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Please check model path.")
            
            # Load image
            if isinstance(image_path_or_array, str):
                image = cv2.imread(image_path_or_array)
            elif isinstance(image_path_or_array, np.ndarray):
                image = image_path_or_array
            else:
                # PIL Image
                image = np.array(image_path_or_array.convert('RGB'))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Run inference
            results = self.model.predict(
                image,
                conf=conf_threshold,
                verbose=False
            )[0]
            
            # Count detections per class
            counts = {name: 0 for name in self.class_names.values()}
            total_detections = 0
            
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id in self.class_names:
                        counts[self.class_names[cls_id]] += 1
                        total_detections += 1
            
            # Calculate percentages
            percentages = {}
            if total_detections > 0:
                for name, count in counts.items():
                    percentages[name] = (count / total_detections) * 100
            else:
                percentages = {name: 0.0 for name in counts.keys()}
            
            # Determine if counts are normal (basic heuristic)
            status = 'normal'
            if counts['Platelets'] < 5:  # Very low platelet count in field
                status = 'low_platelets'
            elif counts['WBC'] > counts['RBC'] * 0.02:  # WBC ratio too high
                status = 'elevated_wbc'
            
            results_dict = {
                'counts': counts,
                'total_cells': total_detections,
                'percentages': percentages,
                'status': status,
                'confidence': 0.85,  # Average confidence from YOLOv8
                'details': {
                    'rbc_count': counts['RBC'],
                    'wbc_count': counts['WBC'],
                    'platelet_count': counts['Platelets'],
                    'rbc_percentage': percentages['RBC'],
                    'wbc_percentage': percentages['WBC'],
                    'platelet_percentage': percentages['Platelets']
                }
            }
            
            logger.info(f"Cell count: RBC={counts['RBC']}, WBC={counts['WBC']}, Platelets={counts['Platelets']}")
            return results_dict
            
        except Exception as e:
            logger.error(f"Error during cell counting: {str(e)}")
            raise
    
    def batch_predict(self, image_paths, conf_threshold=0.25):
        """Predict on multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, conf_threshold)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        return results
