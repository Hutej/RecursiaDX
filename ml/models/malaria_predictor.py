import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
import cv2
from PIL import Image
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MalariaPredictor:
    """
    Malaria detection model using InceptionV3.
    Binary classification: Parasitized vs Uninfected cells.
    """
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['Uninfected', 'Parasitized']
        self.model = None
        
        # Define image preprocessing transforms (128x128 for malaria model)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load the model if path is provided
        if model_path is None:
            # Look in multiple possible locations
            possible_paths = [
                # Absolute path from project root
                os.path.join(os.path.dirname(__file__), '..', '..', 'Malaria-Disease-Detection-Using-Transfer-Learning', 'InceptionV3_Malaria_PyTorch.pth'),
                # Direct relative path
                os.path.join('Malaria-Disease-Detection-Using-Transfer-Learning', 'InceptionV3_Malaria_PyTorch.pth')
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning(f"Malaria model not found. Please provide model path.")
    
    def build_model(self):
        """Build InceptionV3-based model for malaria detection."""
        try:
            # Load InceptionV3 model
            self.model = inception_v3(pretrained=False, aux_logits=False)
            
            # Modify final layer for binary classification
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid()
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("Malaria model built successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error building malaria model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path_or_array):
        """Preprocess image for malaria prediction."""
        try:
            if isinstance(image_path_or_array, str):
                image = Image.open(image_path_or_array).convert('RGB')
            elif isinstance(image_path_or_array, np.ndarray):
                if image_path_or_array.max() > 1.0:
                    image_path_or_array = image_path_or_array / 255.0
                image = Image.fromarray((image_path_or_array * 255).astype(np.uint8))
            else:
                image = image_path_or_array.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image_path_or_array):
        """
        Predict malaria presence in blood smear image.
        
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Preprocess the image
            processed_image = self.preprocess_image(image_path_or_array)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(processed_image)
                parasitized_probability = float(prediction.cpu().numpy()[0][0])
            
            # Binary classification
            uninfected_prob = 1.0 - parasitized_probability
            
            # Determine predicted class
            predicted_class_idx = 1 if parasitized_probability >= 0.5 else 0
            predicted_class = self.class_names[predicted_class_idx]
            confidence = parasitized_probability if predicted_class_idx == 1 else uninfected_prob
            
            # Create detailed results
            results = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_parasitized': predicted_class_idx == 1,
                'probabilities': {
                    'uninfected': uninfected_prob,
                    'parasitized': parasitized_probability
                },
                'risk_level': 'High Risk' if predicted_class_idx == 1 else 'Low Risk'
            }
            
            logger.info(f"Malaria prediction: {predicted_class} ({confidence:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"Error during malaria prediction: {str(e)}")
            raise
    
    def load_model(self, model_path):
        """Load a saved PyTorch model."""
        try:
            # Build model architecture
            if self.model is None:
                self.build_model()
            
            logger.info(f"Loading malaria model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            logger.info(f"✅ Malaria model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Error loading malaria model: {str(e)}")
            logger.warning("Malaria detection will not be available")
