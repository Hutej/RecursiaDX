/**
 * Integration example for RecursiaDx ML module
 * This file shows how to integrate the ML API with the existing Node.js backend
 */

const express = require('express');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

// Configuration
const ML_API_URL = process.env.ML_API_URL || 'http://localhost:5000';
const UPLOAD_DIR = path.join(__dirname, '../uploads');

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Multer configuration for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, UPLOAD_DIR);
    },
    filename: (req, file, cb) => {
        const timestamp = Date.now();
        const originalName = file.originalname.replace(/[^a-zA-Z0-9.]/g, '_');
        cb(null, `${timestamp}_${originalName}`);
    }
});

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 16 * 1024 * 1024 // 16MB limit
    },
    fileFilter: (req, file, cb) => {
        // Allow only image files
        const allowedTypes = ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp'];
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only JPEG, PNG, TIFF, and BMP are allowed.'));
        }
    }
});

/**
 * Helper function to call ML API
 */
async function callMLAPI(endpoint, formData) {
    try {
        const response = await fetch(`${ML_API_URL}${endpoint}`, {
            method: 'POST',
            body: formData,
            headers: formData.getHeaders ? formData.getHeaders() : {}
        });

        if (!response.ok) {
            throw new Error(`ML API error: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('ML API call failed:', error);
        throw error;
    }
}

/**
 * Route: Analyze single image for tumor detection
 * POST /api/analyze
 */
const analyzeImage = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                error: 'No image file provided'
            });
        }

        const { userId, enhanceImage = 'false', saveResult = 'true' } = req.body;

        // Prepare form data for ML API
        const formData = new FormData();
        formData.append('image', fs.createReadStream(req.file.path));
        
        if (userId) {
            formData.append('user_id', userId);
        }
        formData.append('enhance_image', enhanceImage);
        formData.append('save_result', saveResult);

        // Call ML API
        const mlResult = await callMLAPI('/predict', formData);

        // Clean up uploaded file
        fs.unlink(req.file.path, (err) => {
            if (err) console.error('Failed to delete uploaded file:', err);
        });

        // Return result
        res.json({
            success: true,
            analysis: mlResult.prediction,
            metadata: {
                filename: req.file.originalname,
                processingTime: mlResult.processing_time,
                modelVersion: mlResult.model_version,
                timestamp: new Date().toISOString()
            }
        });

    } catch (error) {
        console.error('Analysis failed:', error);
        
        // Clean up uploaded file on error
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlink(req.file.path, (err) => {
                if (err) console.error('Failed to delete uploaded file:', err);
            });
        }

        res.status(500).json({
            success: false,
            error: 'Analysis failed',
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};

/**
 * Route: Analyze multiple images
 * POST /api/analyze/batch
 */
const analyzeBatch = async (req, res) => {
    try {
        if (!req.files || req.files.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'No image files provided'
            });
        }

        const { userId, enhanceImages = 'false' } = req.body;

        // Prepare form data for ML API
        const formData = new FormData();
        
        req.files.forEach(file => {
            formData.append('images', fs.createReadStream(file.path));
        });
        
        if (userId) {
            formData.append('user_id', userId);
        }
        formData.append('enhance_images', enhanceImages);

        // Call ML API
        const mlResult = await callMLAPI('/batch_predict', formData);

        // Clean up uploaded files
        req.files.forEach(file => {
            fs.unlink(file.path, (err) => {
                if (err) console.error('Failed to delete uploaded file:', err);
            });
        });

        // Process results
        const processedResults = mlResult.results.map(result => ({
            filename: result.filename,
            success: result.success,
            analysis: result.prediction,
            error: result.error
        }));

        res.json({
            success: true,
            results: processedResults,
            summary: {
                totalImages: mlResult.total_images,
                successfulPredictions: mlResult.successful_predictions,
                totalProcessingTime: mlResult.total_processing_time
            }
        });

    } catch (error) {
        console.error('Batch analysis failed:', error);
        
        // Clean up uploaded files on error
        if (req.files) {
            req.files.forEach(file => {
                if (fs.existsSync(file.path)) {
                    fs.unlink(file.path, (err) => {
                        if (err) console.error('Failed to delete uploaded file:', err);
                    });
                }
            });
        }

        res.status(500).json({
            success: false,
            error: 'Batch analysis failed',
            details: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};

/**
 * Route: Get analysis history
 * GET /api/analyze/history
 */
const getAnalysisHistory = async (req, res) => {
    try {
        const { userId, limit = 50 } = req.query;

        const queryParams = new URLSearchParams();
        if (userId) queryParams.append('user_id', userId);
        queryParams.append('limit', limit);

        const response = await fetch(`${ML_API_URL}/history?${queryParams}`);
        
        if (!response.ok) {
            throw new Error(`ML API error: ${response.status}`);
        }

        const result = await response.json();
        res.json(result);

    } catch (error) {
        console.error('Failed to get analysis history:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve analysis history'
        });
    }
};

/**
 * Route: Get analysis statistics
 * GET /api/analyze/stats
 */
const getAnalysisStats = async (req, res) => {
    try {
        const { userId } = req.query;

        const queryParams = new URLSearchParams();
        if (userId) queryParams.append('user_id', userId);

        const response = await fetch(`${ML_API_URL}/stats?${queryParams}`);
        
        if (!response.ok) {
            throw new Error(`ML API error: ${response.status}`);
        }

        const result = await response.json();
        res.json(result);

    } catch (error) {
        console.error('Failed to get analysis stats:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to retrieve analysis statistics'
        });
    }
};

/**
 * Route: Check ML service health
 * GET /api/analyze/health
 */
const checkMLHealth = async (req, res) => {
    try {
        const response = await fetch(`${ML_API_URL}/health`);
        const result = await response.json();
        
        res.json({
            success: true,
            mlService: {
                status: response.ok ? 'healthy' : 'unhealthy',
                details: result
            }
        });

    } catch (error) {
        console.error('ML health check failed:', error);
        res.json({
            success: false,
            mlService: {
                status: 'unavailable',
                error: error.message
            }
        });
    }
};

// Export routes for use in main application
module.exports = {
    // Middleware
    upload,
    
    // Route handlers
    analyzeImage,
    analyzeBatch,
    getAnalysisHistory,
    getAnalysisStats,
    checkMLHealth,
    
    // Helper functions
    callMLAPI
};

// Example usage in main app.js or routes file:
/*
const mlIntegration = require('./ml-integration');

// Single image analysis
app.post('/api/analyze', 
    mlIntegration.upload.single('image'), 
    mlIntegration.analyzeImage
);

// Batch analysis
app.post('/api/analyze/batch', 
    mlIntegration.upload.array('images', 10), 
    mlIntegration.analyzeBatch
);

// History and stats
app.get('/api/analyze/history', mlIntegration.getAnalysisHistory);
app.get('/api/analyze/stats', mlIntegration.getAnalysisStats);
app.get('/api/analyze/health', mlIntegration.checkMLHealth);
*/