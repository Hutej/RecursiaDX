import fetch from 'node-fetch';
import FormData from 'form-data';
import fs from 'fs';

const ML_API_URL = 'http://localhost:5000';

class MLService {
  static async predictImage(imagePath, filename) {
    try {
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath), filename);

      const response = await fetch(`${ML_API_URL}/predict`, {
        method: 'POST',
        body: formData,
        timeout: 30000 // 30 second timeout
      });

      if (!response.ok) {
        throw new Error(`ML API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'ML prediction failed');
      }

      return {
        success: true,
        prediction: result.prediction,
        metadata: result.metadata
      };
    } catch (error) {
      console.error('ML prediction error:', error);
      return {
        success: false,
        error: error.message,
        prediction: null
      };
    }
  }

  static async batchPredict(imagePaths, imageType = 'tissue') {
    try {
      const formData = new FormData();

      // Add imageType to form data for ML API routing
      formData.append('imageType', imageType);

      imagePaths.forEach(({ path, filename }) => {
        formData.append('images', fs.createReadStream(path), filename);
      });

      const response = await fetch(`${ML_API_URL}/batch_predict`, {
        method: 'POST',
        body: formData,
        timeout: 60000 // 60 second timeout for batch
      });

      if (!response.ok) {
        throw new Error(`ML API error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Batch ML prediction failed');
      }

      return {
        success: true,
        predictions: result.results, // ML server returns 'results' not 'predictions'
        imageType: result.image_type, // Return imageType from ML response
        modelUsed: result.model_used, // Return which model was used
        summary: {
          total_images: result.total_images,
          successful_predictions: result.successful_predictions,
          total_processing_time: result.total_processing_time
        },
        metadata: result.metadata || {}
      };
    } catch (error) {
      console.error('Batch ML prediction error:', error);
      return {
        success: false,
        error: error.message,
        predictions: []
      };
    }
  }

  static async checkHealth() {
    try {
      const response = await fetch(`${ML_API_URL}/health`, {
        timeout: 5000
      });

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      const result = await response.json();
      return {
        healthy: true,
        details: result
      };
    } catch (error) {
      return {
        healthy: false,
        error: error.message
      };
    }
  }

  static async getModelInfo() {
    try {
      const response = await fetch(`${ML_API_URL}/model_info`, {
        timeout: 5000
      });

      if (!response.ok) {
        throw new Error(`Model info failed: ${response.status}`);
      }

      const result = await response.json();
      return {
        success: true,
        modelInfo: result
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  static async generateHeatmap(imagePath, filename) {
    try {
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath), filename);

      const response = await fetch(`${ML_API_URL}/generate_heatmap`, {
        method: 'POST',
        body: formData,
        timeout: 120000 // 2 minute timeout for heatmap generation
      });

      if (!response.ok) {
        throw new Error(`Heatmap generation failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Heatmap generation failed');
      }

      return {
        success: true,
        heatmap: result.heatmap,
        prediction: result.prediction,
        processingTime: result.processing_time
      };
    } catch (error) {
      console.error('Heatmap generation error:', error);
      return {
        success: false,
        error: error.message,
        heatmap: null
      };
    }
  }
}

export default MLService;