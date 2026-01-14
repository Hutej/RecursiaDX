"""
PyTorch Model Inference - Test on Sample Images
Tests the trained malaria detection model on images
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

# ============ CONFIGURATION ============
MODEL_PATH = "InceptionV3_Malaria_PyTorch.pth"
IMG_SIZE = 299  # InceptionV3 input size
CLASSES = ['Uninfected', 'Parasitized']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============ LOAD MODEL ============
def load_model():
    """Load the trained model"""
    print("\nLoading model...")
    
    # Build model architecture (EXACTLY as in training)
    model = models.inception_v3(pretrained=False, aux_logits=True)
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    
    # Replace auxiliary classifier
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully!")
    return model

# ============ IMAGE PREPROCESSING ============
def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

# ============ PREDICTION ============
def predict_image(model, image_path):
    """Make prediction on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing image: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Preprocess
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    # Results
    predicted_label = CLASSES[predicted_class.item()]
    confidence_score = confidence.item() * 100
    
    print(f"\n🔬 Prediction Results:")
    print(f"  Predicted Class: {predicted_label}")
    print(f"  Confidence: {confidence_score:.2f}%")
    print(f"\n  Probability Distribution:")
    print(f"    Uninfected:  {probabilities[0][0].item()*100:.2f}%")
    print(f"    Parasitized: {probabilities[0][1].item()*100:.2f}%")
    
    # Interpretation
    print(f"\n💡 Interpretation:")
    if predicted_label == "Parasitized":
        print(f"  ⚠️  MALARIA DETECTED - This cell appears to be infected")
    else:
        print(f"  ✓ HEALTHY - This cell appears to be uninfected")
    
    return predicted_label, confidence_score

# ============ MAIN ============
def main():
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path>")
        print("Example: python test_model.py 'Test Images/sample.png'")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Load model
    model = load_model()
    
    # Make prediction
    predicted_label, confidence = predict_image(model, image_path)
    
    print(f"\n{'='*60}")
    print("PREDICTION COMPLETE!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
