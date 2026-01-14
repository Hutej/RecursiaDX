import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image for model input.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image array
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        raise

def enhance_medical_image(image: np.ndarray) -> np.ndarray:
    """
    Apply medical image enhancement techniques.
    
    Args:
        image: Input image array
        
    Returns:
        Enhanced image array
    """
    try:
        # Convert to PIL for enhancement
        if image.dtype != np.uint8:
            image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            image_pil = Image.fromarray(image)
        
        # Apply contrast enhancement
        enhancer = ImageEnhance.Contrast(image_pil)
        image_pil = enhancer.enhance(1.2)
        
        # Apply sharpness enhancement
        enhancer = ImageEnhance.Sharpness(image_pil)
        image_pil = enhancer.enhance(1.1)
        
        # Convert back to numpy
        enhanced_image = np.array(image_pil)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(enhanced_image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced_image = cv2.merge([l, a, b])
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)
        
        return enhanced_image.astype(np.float32) / 255.0
        
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image

def apply_noise_reduction(image: np.ndarray) -> np.ndarray:
    """
    Apply noise reduction to medical images.
    
    Args:
        image: Input image array
        
    Returns:
        Denoised image array
    """
    try:
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = image
        
        # Apply Non-local Means Denoising
        if len(img_uint8.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(img_uint8, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(img_uint8, None, 10, 7, 21)
        
        return denoised.astype(np.float32) / 255.0
        
    except Exception as e:
        logger.error(f"Error applying noise reduction: {str(e)}")
        return image

def create_image_patches(image: np.ndarray, patch_size: int = 224, overlap: int = 56) -> List[np.ndarray]:
    """
    Create overlapping patches from a large image for analysis.
    
    Args:
        image: Input image array
        patch_size: Size of each patch
        overlap: Overlap between patches
        
    Returns:
        List of image patches
    """
    try:
        patches = []
        h, w = image.shape[:2]
        stride = patch_size - overlap
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[y:y + patch_size, x:x + patch_size]
                patches.append(patch)
        
        return patches
        
    except Exception as e:
        logger.error(f"Error creating patches: {str(e)}")
        return []

def validate_image_format(image_path: str) -> bool:
    """
    Validate if the image format is supported.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if format is supported, False otherwise
    """
    supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.dcm']
    file_ext = os.path.splitext(image_path)[1].lower()
    return file_ext in supported_formats

def get_image_info(image_path: str) -> dict:
    """
    Get image information including dimensions, format, etc.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    try:
        with Image.open(image_path) as img:
            info = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'width': img.width,
                'height': img.height,
                'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
            return info
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return {}

def save_prediction_visualization(image: np.ndarray, prediction_result: dict, output_path: str):
    """
    Save visualization of prediction results.
    
    Args:
        image: Original image array
        prediction_result: Prediction results dictionary
        output_path: Path to save the visualization
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Display prediction results
        labels = list(prediction_result['probabilities'].keys())
        values = list(prediction_result['probabilities'].values())
        colors = ['green' if label == 'non_tumor' else 'red' for label in labels]
        
        bars = ax2.bar(labels, values, color=colors, alpha=0.7)
        ax2.set_title(f'Prediction: {prediction_result["predicted_class"]}\n'
                     f'Confidence: {prediction_result["confidence"]:.2%}\n'
                     f'Risk Level: {prediction_result["risk_level"]}')
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}")

def batch_process_images(image_folder: str, output_folder: str, processor_func) -> List[str]:
    """
    Process multiple images in a folder.
    
    Args:
        image_folder: Folder containing input images
        output_folder: Folder to save processed images
        processor_func: Function to process each image
        
    Returns:
        List of processed image paths
    """
    try:
        processed_paths = []
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        for filename in os.listdir(image_folder):
            if validate_image_format(filename):
                input_path = os.path.join(image_folder, filename)
                output_path = os.path.join(output_folder, f"processed_{filename}")
                
                try:
                    processed_image = processor_func(input_path)
                    # Save processed image
                    if processed_image is not None:
                        cv2.imwrite(output_path, (processed_image * 255).astype(np.uint8))
                        processed_paths.append(output_path)
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
        
        return processed_paths
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return []