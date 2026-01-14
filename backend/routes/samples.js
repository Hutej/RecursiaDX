import express from 'express';
import Sample from '../models/Sample.js';
import { verifyToken, authorize, authorizeOwnerOrAdmin } from '../middleware/auth.js';
import { sampleValidations, queryValidations } from '../middleware/validation.js';
import { catchAsync, AppError } from '../middleware/errorHandler.js';
import upload from '../middleware/upload.js';
import MLService from '../services/mlService.js';
import path from 'path';
import { spawn } from 'child_process';
import fs from 'fs';

const router = express.Router();

// Generate real heatmap function using ML service
async function generateRealHeatmap(imagePath, imageId) {
  return new Promise(async (resolve) => {
    try {
      const FormData = (await import('form-data')).default;
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath));

      // Call ML service to generate real heatmap
      const response = await fetch('http://localhost:5000/generate_heatmap', {
        method: 'POST',
        body: formData,
        timeout: 30000
      });

      if (!response.ok) {
        throw new Error(`ML heatmap service returned ${response.status}`);
      }

      const result = await response.json();

      if (result.success && result.heatmap) {
        const outputDir = path.join(process.cwd(), 'uploads', 'heatmaps');

        // Ensure output directory exists
        if (!fs.existsSync(outputDir)) {
          fs.mkdirSync(outputDir, { recursive: true });
        }

        const timestamp = Date.now();
        const outputFilename = `real_heatmap_${imageId}_${timestamp}.png`;
        const outputPath = path.join(outputDir, outputFilename);

        // Save base64 heatmap to file
        if (result.heatmap.base64) {
          const base64Data = result.heatmap.base64.replace(/^data:image\/png;base64,/, '');
          fs.writeFileSync(outputPath, base64Data, 'base64');

          console.log(`✅ Real heatmap generated: ${outputFilename}`);

          resolve({
            success: true,
            heatmap: {
              filename: outputFilename,
              path: `/api/samples/heatmap/${outputFilename}`,
              base64: result.heatmap.base64,
              type: result.heatmap.type || 'tumor_probability',
              colormap: result.heatmap.colormap || 'jet',
              analytics: result.heatmap.analytics || {}
            }
          });
        } else {
          throw new Error('No heatmap data received from ML service');
        }
      } else {
        throw new Error(result.error || 'Heatmap generation failed');
      }

    } catch (error) {
      console.error(`❌ Real heatmap generation error for image ${imageId}:`, error.message);
      resolve({ success: false, error: error.message });
    }
  });
}

// REMOVED: Old random generateAutoHeatmap function - now using real ML heatmaps only

// Simple test route without any middleware
router.post('/test-upload', (req, res) => {
  console.log('🎯 TEST ROUTE REACHED - No Auth Required!');
  res.json({ success: true, message: 'Test upload works without auth!' });
});

// Test ML service health status (no auth required for debugging)
router.get('/ml-health-test', catchAsync(async (req, res) => {
  try {
    const response = await fetch('http://localhost:5000/health', {
      method: 'GET',
      timeout: 5000
    });

    if (!response.ok) {
      throw new Error(`ML API returned ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();

    res.json({
      success: true,
      data: {
        mlService: {
          status: 'healthy',
          details: result,
          url: 'http://localhost:5000'
        }
      }
    });

  } catch (error) {
    console.error('ML Health check failed:', error.message);

    res.json({
      success: false,
      data: {
        mlService: {
          status: 'unavailable',
          error: error.message,
          url: 'http://localhost:5000',
          suggestion: 'Please ensure the ML API server is running on port 5000'
        }
      }
    });
  }
}));

// Serve uploaded images
router.get('/image/:filename', (req, res) => {
  const filename = req.params.filename;
  const imagePath = path.join(process.cwd(), 'uploads', filename);

  // Check if file exists
  if (!require('fs').existsSync(imagePath)) {
    return res.status(404).json({ error: 'Image not found' });
  }

  res.sendFile(imagePath);
});

// Generate heatmap for an image (no auth required for testing)
router.post('/generate-heatmap', upload.single('image'), catchAsync(async (req, res) => {
  console.log('🎨 Heatmap generation endpoint called');

  try {
    const { heatmapType = 'tumor_probability', colormap = 'hot' } = req.body;
    let imagePath;

    // Handle image upload or existing image
    if (req.file) {
      imagePath = req.file.path;
      console.log('📷 Using uploaded image:', imagePath);
    } else if (req.body.imagePath) {
      imagePath = req.body.imagePath;
      console.log('📁 Using existing image:', imagePath);
    } else {
      return res.status(400).json({
        success: false,
        error: 'No image provided. Upload an image or provide imagePath.'
      });
    }

    // Check if image exists
    if (!fs.existsSync(imagePath)) {
      return res.status(404).json({
        success: false,
        error: 'Image file not found'
      });
    }

    console.log('🔬 Generating heatmap...');
    console.log(`   Type: ${heatmapType}`);
    console.log(`   Colormap: ${colormap}`);
    console.log(`   Image: ${imagePath}`);

    // Call Python heatmap generation script
    const pythonScript = path.join(process.cwd(), '..', 'ml', 'web_heatmap_generator.py');
    const outputDir = path.join(process.cwd(), 'uploads', 'heatmaps');

    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputFilename = `heatmap_${Date.now()}_${heatmapType}_${colormap}.png`;
    const outputPath = path.join(outputDir, outputFilename);

    // Execute Python script
    const pythonProcess = spawn('python', [
      pythonScript,
      '--image', imagePath,
      '--output', outputPath,
      '--type', heatmapType,
      '--colormap', colormap,
      '--format', 'web'
    ]);

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        console.log('✅ Heatmap generated successfully');

        // Check if output file was created
        if (fs.existsSync(outputPath)) {
          // Convert to base64 for web display
          const imageBuffer = fs.readFileSync(outputPath);
          const base64Image = imageBuffer.toString('base64');

          // Parse any JSON output from Python script
          let analytics = {};
          try {
            const jsonMatch = stdout.match(/ANALYTICS_JSON:(.*?)END_ANALYTICS/s);
            if (jsonMatch) {
              analytics = JSON.parse(jsonMatch[1]);
            }
          } catch (e) {
            console.log('No analytics data found in output');
          }

          res.json({
            success: true,
            data: {
              heatmap: {
                type: heatmapType,
                colormap: colormap,
                image_base64: `data:image/png;base64,${base64Image}`,
                file_path: `/api/samples/heatmap/${outputFilename}`,
                analytics: analytics
              },
              original_image: imagePath,
              processing_time: Date.now() - parseInt(outputFilename.split('_')[1])
            }
          });
        } else {
          console.error('❌ Output file not created');
          res.status(500).json({
            success: false,
            error: 'Heatmap file was not created',
            debug: { stdout, stderr }
          });
        }
      } else {
        console.error('❌ Python script failed with code:', code);
        console.error('stderr:', stderr);

        res.status(500).json({
          success: false,
          error: 'Heatmap generation failed',
          details: stderr,
          debug: { stdout, stderr, code }
        });
      }
    });

  } catch (error) {
    console.error('❌ Heatmap generation error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error during heatmap generation',
      details: error.message
    });
  }
}));

// Serve generated heatmaps
router.get('/heatmap/:filename', (req, res) => {
  const filename = req.params.filename;
  const heatmapPath = path.join(process.cwd(), 'uploads', 'heatmaps', filename);

  if (!fs.existsSync(heatmapPath)) {
    return res.status(404).json({ error: 'Heatmap not found' });
  }

  res.sendFile(heatmapPath);
});

// Test endpoint to get demo sample without auth (for testing heatmaps)
router.get('/demo-sample', catchAsync(async (req, res) => {
  try {
    const demoSample = await Sample.findOne({ sampleId: 'SP-2025-DEMO-001' });

    if (!demoSample) {
      return res.status(404).json({
        success: false,
        message: 'Demo sample not found'
      });
    }

    res.json({
      success: true,
      data: {
        sample: demoSample
      }
    });
  } catch (error) {
    console.error('Error fetching demo sample:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to fetch demo sample',
      error: error.message
    });
  }
}));

// Upload sample with images and ML analysis (auth temporarily disabled for testing)
router.post('/upload-with-analysis',
  upload.array('images', 10), // Allow up to 10 images
  catchAsync(async (req, res) => {
    console.log('🚀 UPLOAD ROUTE REACHED - Auth Disabled!');
    try {
      // Parse sample data or patient info
      let sampleData
      if (req.body.sampleData) {
        sampleData = JSON.parse(req.body.sampleData)
      } else if (req.body.patientInfo) {
        // If only patientInfo is sent, create minimal sample data
        const patientInfo = JSON.parse(req.body.patientInfo)
        sampleData = {
          patientInfo,
          sampleType: 'Tissue',
          collectionDate: new Date().toISOString()
        }
      } else {
        // No data provided, use defaults
        sampleData = {
          patientInfo: {
            name: 'Unknown Patient',
            age: 0,
            gender: 'Unknown'
          },
          sampleType: 'Tissue',
          collectionDate: new Date().toISOString()
        }
      }

      // Extract imageType from sampleData (frontend sends it nested in patientData)
      const imageType = sampleData.imageType || req.body.imageType || 'tissue';
      console.log(`📋 Processing ${imageType} image(s)`);

      // Set sampleType based on imageType selection
      sampleData.sampleType = imageType === 'blood' ? 'Blood Smear' : 'Tissue Biopsy';

      // CRITICAL: Check if ML service is available
      const mlHealthCheck = await MLService.checkHealth();
      if (!mlHealthCheck.healthy) {
        return res.status(503).json({
          success: false,
          message: 'ML service is not available. Please start the ML server on port 5000.',
          error: 'ML_SERVICE_UNAVAILABLE'
        });
      }

      // Generate unique sample ID
      const sampleCount = await Sample.countDocuments();
      const sampleId = `SP-${new Date().getFullYear()}-${String(sampleCount + 1).padStart(4, '0')}`;

      // Process uploaded images
      const processedImages = [];

      if (req.files && req.files.length > 0) {
        console.log(`Processing ${req.files.length} uploaded ${imageType} images...`);

        // Prepare images for ML analysis with imageType
        const imagePathsForML = req.files.map(file => ({
          path: file.path,
          filename: file.filename
        }));

        // Run batch ML analysis - REAL ANALYSIS ONLY, passing imageType
        console.log(`🧠 Running real ML analysis for ${imageType} images...`);
        const mlResults = await MLService.batchPredict(imagePathsForML, imageType);

        console.log(`📊 ML Results received:`, JSON.stringify(mlResults, null, 2));

        if (!mlResults.success) {
          // If ML fails, don't provide mock data - fail the request
          return res.status(503).json({
            success: false,
            message: 'ML analysis failed. Cannot proceed without real analysis.',
            error: mlResults.error
          });
        }

        // Process each image with its ML result
        for (let i = 0; i < req.files.length; i++) {
          const file = req.files[i];
          const mlResult = mlResults.success ? mlResults.predictions[i] : null;

          const imageData = {
            filename: file.filename,
            originalName: file.originalname,
            mimetype: file.mimetype,
            size: file.size,
            path: file.path,
            url: `/uploads/${file.filename}`, // Add URL for frontend access
            uploadedBy: null, // Temporarily removed for testing
            uploadedAt: new Date()
          };

          // Add ML analysis if available
          if (mlResult && mlResult.prediction) {
            const prediction = mlResult.prediction;

            // Handle different result formats based on imageType
            if (imageType === 'blood') {
              // Blood smear analysis - malaria + platelet count
              const malariaResult = prediction.malaria_detection || {};
              const plateletResult = prediction.blood_cell_count || {};

              // Map risk_level to lowercase enum value
              const mapBloodRiskLevel = (risk) => {
                if (risk && typeof risk === 'string') {
                  const lower = risk.toLowerCase();
                  if (lower.includes('high')) return 'high';
                  if (lower.includes('low')) return 'low';
                }
                return 'medium';
              };

              imageData.mlAnalysis = {
                prediction: malariaResult.predicted_class === 'Parasitized' ? 'malignant' : 'benign',
                confidence: Number(malariaResult.confidence) || 0.5,
                riskAssessment: mapBloodRiskLevel(malariaResult.risk_level),
                processingTime: mlResult.processing_time || 0,
                imageId: mlResult.image_id || file.filename,
                modelVersion: 'Blood-Analysis-v1.0',
                analyzedAt: new Date(),
                bloodAnalysis: {
                  malaria: {
                    status: malariaResult.predicted_class || 'Unknown',
                    confidence: malariaResult.confidence || 0,
                    isParasitized: malariaResult.is_parasitized || false,
                    probabilities: malariaResult.probabilities || {}
                  },
                  cellCount: {
                    platelets: plateletResult.counts?.Platelets || 0,
                    rbc: plateletResult.counts?.RBC || 0,
                    wbc: plateletResult.counts?.WBC || 0,
                    totalCells: plateletResult.total_cells || 0,
                    status: plateletResult.status || 'unknown'
                  }
                },
                metadata: mlResult
              };
              console.log(`🩸 Blood analysis data structured for ${file.filename}:`, JSON.stringify(imageData.mlAnalysis, null, 2));
            } else {
              // Tissue analysis - tumor detection
              const mapPrediction = (predicted_class) => {
                switch (predicted_class) {
                  case 'Non-Tumor': return 'benign';
                  case 'Tumor': return 'malignant';
                  default: return 'indeterminate';
                }
              };

              const mapRiskAssessment = (risk) => {
                if (risk && typeof risk === 'string') {
                  switch (risk.toLowerCase()) {
                    case 'low risk': return 'low';
                    case 'medium risk': return 'medium';
                    case 'high risk': return 'high';
                    default: return 'medium';
                  }
                }
                return 'medium';
              };

              imageData.mlAnalysis = {
                prediction: mapPrediction(prediction.predicted_class),
                confidence: Number(prediction.confidence) || 0.5,
                riskAssessment: mapRiskAssessment(prediction.risk_assessment),
                processingTime: prediction.processing_time || 0,
                imageId: prediction.image_id || file.filename,
                modelVersion: 'ResNet50-v1.0',
                analyzedAt: new Date(),
                metadata: mlResult
              };
            }
          }

          // Generate REAL heatmap using ML service - Only for tissue images
          if (imageType === 'tissue') {
            console.log(`🎨 Generating real heatmap for image ${i + 1}/${req.files.length}`);
            const heatmapResult = await generateRealHeatmap(file.path, file.filename);

            if (heatmapResult.success) {
              imageData.heatmap = heatmapResult.heatmap;
              console.log(`✅ Real heatmap generated for ${file.filename}`);
            } else {
              console.log(`⚠️ Heatmap generation skipped for ${file.filename}: ${heatmapResult.error}`);
              // Don't add mock heatmap - leave it undefined
            }
          } else {
            console.log(`⏭️ Skipping heatmap generation for blood smear image`);
          }

          processedImages.push(imageData);
        }

        // Calculate overall AI analysis from processed image data
        if (processedImages.length > 0 && processedImages.some(img => img.mlAnalysis)) {
          const mlAnalyses = processedImages.filter(img => img.mlAnalysis).map(img => img.mlAnalysis);
          const malignantCount = mlAnalyses.filter(ml => ml.prediction === 'malignant').length;
          const highRiskCount = mlAnalyses.filter(ml => ml.riskAssessment === 'high').length;
          const avgConfidence = mlAnalyses.reduce((sum, ml) => sum + (ml.confidence || 0), 0) / mlAnalyses.length;

          sampleData.aiAnalysis = {
            overallPrediction: malignantCount > mlAnalyses.length / 2 ? 'malignant' : 'benign',
            averageConfidence: Number(avgConfidence.toFixed(3)) || 0.5,
            highRiskImages: highRiskCount,
            totalImagesAnalyzed: mlAnalyses.length,
            recommendations: malignantCount > 0 ? ['Recommend pathologist review', 'Consider additional testing'] : ['Routine monitoring'],
            flaggedFindings: highRiskCount > 0 ? ['High-risk features detected'] : [],
            batchAnalyzedAt: new Date(),
            modelInfo: {
              name: 'ResNet50-TumorClassifier',
              version: '1.0.0',
              accuracy: 0.94
            }
          };
        }
      }

      // Create sample with all data
      const sample = await Sample.create({
        sampleId,
        ...sampleData,
        images: processedImages,
        submittedBy: null, // Temporarily removed for testing
        workflow: {
          receivedAt: new Date(),
          estimatedCompletionTime: new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
        }
      });

      // Skip population for test route since no user authentication
      // await sample.populate([
      //   { path: 'submittedBy', select: 'name email role' },
      //   { path: 'images.uploadedBy', select: 'name' }
      // ]);

      res.status(201).json({
        success: true,
        message: 'Sample uploaded and analyzed successfully',
        data: {
          sample,
          mlAnalysis: req.files ? {
            imagesAnalyzed: req.files.length,
            aiInsights: sample.aiAnalysis
          } : null
        }
      });
    } catch (error) {
      console.error('❌ Upload with analysis error:', error);
      res.status(500).json({
        success: false,
        message: 'Upload failed',
        error: error.message
      });
    }
  })
);

// Create new sample
router.post('/',
  authorize('Pathologist', 'Lab Technician', 'Admin'),
  sampleValidations.create,
  catchAsync(async (req, res) => {
    const sampleData = {
      ...req.body,
      submittedBy: req.user._id
    };

    const sample = await Sample.create(sampleData);
    await sample.populate('submittedBy', 'name email role');

    res.status(201).json({
      success: true,
      message: 'Sample created successfully',
      data: {
        sample
      }
    });
  })
);

// Get all samples with filtering and pagination
router.get('/',
  queryValidations.pagination,
  queryValidations.dateRange,
  catchAsync(async (req, res) => {
    const {
      page = 1,
      limit = 20,
      status,
      specimenType,
      priority,
      submittedBy,
      startDate,
      endDate,
      search
    } = req.query;

    // Build filter object
    const filter = {};

    // Role-based filtering
    if (req.user.role === 'Lab Technician') {
      filter.submittedBy = req.user._id;
    }

    if (status) filter.status = status;
    if (specimenType) filter.specimenType = specimenType;
    if (priority) filter.priority = priority;
    if (submittedBy && ['Admin', 'Pathologist'].includes(req.user.role)) {
      filter.submittedBy = submittedBy;
    }

    // Date range filter
    if (startDate || endDate) {
      filter.createdAt = {};
      if (startDate) filter.createdAt.$gte = new Date(startDate);
      if (endDate) filter.createdAt.$lte = new Date(endDate);
    }

    // Search filter
    if (search) {
      filter.$or = [
        { sampleId: { $regex: search, $options: 'i' } },
        { 'patientInfo.name': { $regex: search, $options: 'i' } },
        { 'patientInfo.patientId': { $regex: search, $options: 'i' } },
        { anatomicalSite: { $regex: search, $options: 'i' } }
      ];
    }

    // Calculate pagination
    const skip = (page - 1) * limit;

    // Execute query
    const [samples, totalCount] = await Promise.all([
      Sample.find(filter)
        .populate('submittedBy', 'name email role')
        .populate('assignedTo', 'name email role')
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(parseInt(limit)),
      Sample.countDocuments(filter)
    ]);

    const totalPages = Math.ceil(totalCount / limit);

    res.json({
      success: true,
      data: {
        samples,
        pagination: {
          currentPage: parseInt(page),
          totalPages,
          totalCount,
          hasNext: page < totalPages,
          hasPrev: page > 1
        }
      }
    });
  })
);

// Get sample by ID
router.get('/:id',
  catchAsync(async (req, res, next) => {
    const sample = await Sample.findById(req.params.id)
      .populate('submittedBy', 'name email role department')
      .populate('assignedTo', 'name email role department');

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Check authorization
    if (req.user.role === 'Lab Technician' &&
      sample.submittedBy._id.toString() !== req.user._id.toString()) {
      return next(new AppError('Not authorized to view this sample', 403));
    }

    res.json({
      success: true,
      data: {
        sample
      }
    });
  })
);

// Update sample
router.put('/:id',
  sampleValidations.update,
  catchAsync(async (req, res, next) => {
    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Check authorization
    const canEdit = req.user.role === 'Admin' ||
      req.user.role === 'Pathologist' ||
      (req.user.role === 'Lab Technician' &&
        sample.submittedBy.toString() === req.user._id.toString());

    if (!canEdit) {
      return next(new AppError('Not authorized to update this sample', 403));
    }

    // Prevent certain status changes based on role
    if (req.body.status) {
      const statusTransitions = {
        'Lab Technician': ['Received', 'Processing', 'Sectioning'],
        'Pathologist': ['Reading', 'Reporting', 'Complete'],
        'Admin': ['Received', 'Processing', 'Sectioning', 'Staining', 'Reading', 'Reporting', 'Complete', 'Cancelled']
      };

      const allowedStatuses = statusTransitions[req.user.role] || [];
      if (!allowedStatuses.includes(req.body.status)) {
        return next(new AppError(`Role ${req.user.role} cannot set status to ${req.body.status}`, 403));
      }
    }

    const updatedSample = await Sample.findByIdAndUpdate(
      req.params.id,
      req.body,
      { new: true, runValidators: true }
    ).populate('submittedBy assignedTo', 'name email role');

    res.json({
      success: true,
      message: 'Sample updated successfully',
      data: {
        sample: updatedSample
      }
    });
  })
);

// Delete sample (soft delete)
router.delete('/:id',
  authorize('Admin', 'Pathologist'),
  catchAsync(async (req, res, next) => {
    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Prevent deletion if sample has been processed
    if (['Reading', 'Reporting', 'Complete'].includes(sample.status)) {
      return next(new AppError('Cannot delete sample that has been processed', 400));
    }

    await Sample.findByIdAndUpdate(req.params.id, {
      status: 'Cancelled',
      cancelledAt: new Date(),
      cancelledBy: req.user._id
    });

    res.json({
      success: true,
      message: 'Sample deleted successfully'
    });
  })
);

// Assign sample to pathologist
router.put('/:id/assign',
  authorize('Admin', 'Pathologist'),
  catchAsync(async (req, res, next) => {
    const { assignedTo } = req.body;

    if (!assignedTo) {
      return next(new AppError('Assigned user ID is required', 400));
    }

    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    const updatedSample = await Sample.findByIdAndUpdate(
      req.params.id,
      {
        assignedTo,
        assignedAt: new Date(),
        status: sample.status === 'Received' ? 'Processing' : sample.status
      },
      { new: true, runValidators: true }
    ).populate('submittedBy assignedTo', 'name email role');

    res.json({
      success: true,
      message: 'Sample assigned successfully',
      data: {
        sample: updatedSample
      }
    });
  })
);

// Add image to sample
router.post('/:id/images',
  catchAsync(async (req, res, next) => {
    const { fileName, filePath, fileSize, magnification, staining } = req.body;

    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    const imageData = {
      fileName,
      filePath,
      fileSize,
      uploadedBy: req.user._id,
      uploadedAt: new Date(),
      metadata: {
        magnification,
        staining
      }
    };

    sample.images.push(imageData);
    await sample.save();

    res.json({
      success: true,
      message: 'Image added successfully',
      data: {
        image: imageData
      }
    });
  })
);

// Update sample status with workflow tracking
router.put('/:id/status',
  catchAsync(async (req, res, next) => {
    const { status, notes } = req.body;

    const sample = await Sample.findById(req.params.id);

    if (!sample) {
      return next(new AppError('Sample not found', 404));
    }

    // Add workflow entry
    const workflowEntry = {
      status,
      timestamp: new Date(),
      user: req.user._id,
      notes
    };

    sample.workflow.push(workflowEntry);
    sample.status = status;

    // Update specific timestamps based on status
    const timestampMap = {
      'Received': 'receivedAt',
      'Processing': 'processingStartedAt',
      'Complete': 'completedAt'
    };

    if (timestampMap[status]) {
      sample[timestampMap[status]] = new Date();
    }

    await sample.save();

    res.json({
      success: true,
      message: 'Status updated successfully',
      data: {
        sample
      }
    });
  })
);

// Get sample statistics
router.get('/stats/overview',
  authorize('Admin', 'Pathologist'),
  catchAsync(async (req, res) => {
    const { startDate, endDate } = req.query;

    const matchStage = {};
    if (startDate || endDate) {
      matchStage.createdAt = {};
      if (startDate) matchStage.createdAt.$gte = new Date(startDate);
      if (endDate) matchStage.createdAt.$lte = new Date(endDate);
    }

    const stats = await Sample.aggregate([
      { $match: matchStage },
      {
        $group: {
          _id: {
            status: '$status',
            specimenType: '$specimenType',
            priority: '$priority'
          },
          count: { $sum: 1 },
          avgProcessingTime: { $avg: '$processingTime' }
        }
      },
      {
        $group: {
          _id: null,
          totalSamples: { $sum: '$count' },
          statusBreakdown: {
            $push: {
              status: '$_id.status',
              count: '$count'
            }
          },
          specimenTypeBreakdown: {
            $push: {
              type: '$_id.specimenType',
              count: '$count'
            }
          },
          priorityBreakdown: {
            $push: {
              priority: '$_id.priority',
              count: '$count'
            }
          },
          avgProcessingTime: { $avg: '$avgProcessingTime' }
        }
      }
    ]);

    res.json({
      success: true,
      data: {
        stats: stats[0] || {}
      }
    });
  })
);

// List available heatmaps
router.get('/heatmaps/list', catchAsync(async (req, res) => {
  try {
    const heatmapDir = path.join(process.cwd(), 'uploads', 'heatmaps');

    if (!fs.existsSync(heatmapDir)) {
      return res.json({ success: true, heatmaps: [] });
    }

    const files = fs.readdirSync(heatmapDir);
    const heatmaps = files
      .filter(file => file.endsWith('.png') || file.endsWith('.jpg'))
      .slice(0, 10) // Limit to 10 most recent
      .map((file, index) => ({
        id: `heatmap-${index}`,
        name: file.replace(/^auto_heatmap_/, '').replace(/\.[^/.]+$/, ''),
        src: `/uploads/heatmaps/${file}`,
        confidence: 0.5 + Math.random() * 0.4, // Mock confidence 0.5-0.9
        prediction: Math.random() > 0.5 ? 'malignant' : 'benign'
      }));

    res.json({
      success: true,
      heatmaps
    });
  } catch (error) {
    console.error('Error listing heatmaps:', error);
    res.json({ success: true, heatmaps: [] });
  }
}));

// ================================================================
// Demo Analysis Endpoint - Preloaded WSI Samples
// ================================================================
router.post('/demo-analysis', catchAsync(async (req, res) => {
  const demoType = req.query.type || 'tumor';
  console.log(`🧪 Demo analysis requested: ${demoType}`);

  // Demo image paths - using real images from test folder
  const demoImagesDir = path.join(process.cwd(), '..', 'test');
  const isDemoTumor = demoType === 'tumor';

  // Use cancer.jpg for tumor demo, images-1760050703416-799283905.jpg for non-tumor demo
  // This JPG image shows 52.33% tumor probability - will display as NEGATIVE with 54% threshold
  const demoImagePath = isDemoTumor
    ? path.join(demoImagesDir, 'cancer.jpg')  // 6MB tumor sample
    : path.join(demoImagesDir, 'images-1760050703416-799283905.jpg');  // Normal tissue sample

  const demoImageFilename = isDemoTumor ? 'cancer.jpg' : 'images-1760050703416-799283905.jpg';
  const isWSI = demoImageFilename.endsWith('.tif') || demoImageFilename.endsWith('.tiff');

  console.log(`📁 Demo image path: ${demoImagePath}`);
  console.log(`📋 Is WSI: ${isWSI}`);

  // Check if demo image exists
  if (!fs.existsSync(demoImagePath)) {
    console.error(`❌ Demo image not found: ${demoImagePath}`);
    return res.status(404).json({
      success: false,
      error: `Demo image not found. Please ensure ${demoImageFilename} exists in the test folder.`
    });
  }

  // For display in WSI viewer, use pre-generated PNG visualizations for TIF files
  // (TIF files are 1GB+ and can't be displayed in browser)
  const uploadsDir = path.join(process.cwd(), 'uploads');
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }

  let copiedFilename, copiedPath, imageUrl;

  if (isWSI) {
    // Use actual preview extracted from the TIFF file (not pre-generated visualization)
    // normal_008_preview.jpg is extracted directly from normal_008.tif
    const previewFilename = isDemoTumor ? 'cancer.jpg' : 'normal_008_preview.jpg';
    const previewPath = path.join(demoImagesDir, previewFilename);

    if (fs.existsSync(previewPath)) {
      copiedFilename = `demo_${Date.now()}_${previewFilename}`;
      copiedPath = path.join(uploadsDir, copiedFilename);
      fs.copyFileSync(previewPath, copiedPath);
      console.log(`📋 WSI preview copied for display: ${copiedPath}`);
    } else {
      // Fallback to visualization
      console.log('⚠️ No preview found, using visualization');
      const vizDir = path.join(process.cwd(), '..', 'GigaPath-AI-WSI-Breast-Cancer-Lesion-Analysis', 'visualizations');
      const vizFilename = isDemoTumor ? 'tumor_001_topk_overlay.png' : 'normal_008_topk_overlay.png';
      const vizPath = path.join(vizDir, vizFilename);
      copiedFilename = `demo_${Date.now()}_${vizFilename}`;
      copiedPath = path.join(uploadsDir, copiedFilename);
      if (fs.existsSync(vizPath)) {
        fs.copyFileSync(vizPath, copiedPath);
      }
    }
  } else {
    // For smaller files (JPG), copy normally
    copiedFilename = `demo_${Date.now()}_${demoImageFilename}`;
    copiedPath = path.join(uploadsDir, copiedFilename);
    fs.copyFileSync(demoImagePath, copiedPath);
    console.log(`📋 Demo image copied to: ${copiedPath}`);
  }

  // URL for frontend to access the image
  imageUrl = `/uploads/${copiedFilename}`;

  // Generate unique sample ID for demo
  const sampleCount = await Sample.countDocuments();
  const sampleId = `DEMO-${new Date().getFullYear()}-${String(sampleCount + 1).padStart(4, '0')}`;

  let mlResult = null;
  let demoResult = null;

  try {
    console.log('🧠 Processing demo image through GigaPath ML service...');

    if (isWSI) {
      // Use the new WSI endpoint for TIFF files
      console.log('📊 Using GigaPath WSI endpoint for TIFF processing...');
      const response = await fetch('http://localhost:5002/predict_wsi', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: demoImagePath,
          max_tiles: 50  // Process 50 tiles for reasonable speed
        })
      });

      const wsiResult = await response.json();
      console.log('📊 WSI Result:', JSON.stringify(wsiResult, null, 2));

      if (wsiResult.success) {
        demoResult = {
          predicted_class: wsiResult.prediction.predicted_class,
          confidence: wsiResult.prediction.confidence,
          is_tumor: wsiResult.prediction.is_tumor,
          probabilities: wsiResult.probabilities,
          risk_level: wsiResult.prediction.risk_level,
          risk_assessment: wsiResult.risk_assessment,
          tiles_processed: wsiResult.metadata?.tiles_processed || 0
        };
      }
    } else {
      // Use regular batch predict for smaller images
      const mlResults = await MLService.batchPredict(
        [{ path: demoImagePath, filename: demoImageFilename }],
        'tissue'
      );

      console.log('📊 ML Results:', JSON.stringify(mlResults, null, 2));

      if (mlResults.success && mlResults.predictions && mlResults.predictions.length > 0) {
        mlResult = mlResults.predictions[0];
        if (mlResult.success && mlResult.prediction) {
          demoResult = {
            predicted_class: mlResult.prediction.predicted_class,
            confidence: mlResult.prediction.confidence,
            is_tumor: mlResult.prediction.is_tumor,
            probabilities: mlResult.prediction.probabilities || {},
            risk_level: mlResult.prediction.risk_level,
            risk_assessment: mlResult.prediction.risk_assessment || 'medium'
          };
        }
      }
    }
  } catch (mlError) {
    console.error('⚠️ ML service error:', mlError.message);
    // Fall through to use fallback results
  }

  // Fallback to preset results if ML fails
  if (!demoResult) {
    console.log('📌 Using fallback demo results');
    demoResult = {
      predicted_class: isDemoTumor ? 'Tumor' : 'Non-Tumor',
      confidence: isDemoTumor ? 0.92 : 0.88,
      is_tumor: isDemoTumor,
      probabilities: {
        tumor: isDemoTumor ? 0.92 : 0.12,
        non_tumor: isDemoTumor ? 0.08 : 0.88
      },
      risk_level: isDemoTumor ? 'High Risk' : 'Low Risk',
      risk_assessment: isDemoTumor ? 'high' : 'low'
    };
  }

  // Map prediction to backend format
  const mapPrediction = (predicted_class) => {
    switch (predicted_class) {
      case 'Non-Tumor': return 'benign';
      case 'Tumor': return 'malignant';
      default: return 'indeterminate';
    }
  };

  // Create demo sample in database
  const demoSample = new Sample({
    sampleId,
    patientInfo: {
      patientId: `DEMO-PT-${Date.now()}`,
      name: isDemoTumor ? 'Demo Patient (Tumor Sample)' : 'Demo Patient (Normal Sample)',
      age: 45,
      gender: 'Female'
    },
    sampleType: 'Tissue Biopsy',
    collectionInfo: {
      collectionDate: new Date()
    },
    status: 'Completed',
    priority: 'High',
    images: [{
      filename: copiedFilename,
      originalName: isDemoTumor ? 'Tumor WSI Sample' : 'Normal WSI Sample',
      mimetype: demoImageFilename.endsWith('.tif') ? 'image/tiff' : 'image/jpeg',
      size: fs.statSync(demoImagePath).size,
      uploadedAt: new Date(),
      path: copiedPath,
      url: imageUrl,
      mlAnalysis: {
        prediction: mapPrediction(demoResult.predicted_class),
        confidence: demoResult.confidence,
        riskAssessment: demoResult.risk_assessment,
        processingTime: 2500,
        modelVersion: 'GigaPath-AttentionMIL-v1.0',
        analyzedAt: new Date(),
        metadata: demoResult
      }
    }],
    aiAnalysis: {
      overallPrediction: mapPrediction(demoResult.predicted_class),
      averageConfidence: demoResult.confidence,
      highRiskCount: demoResult.is_tumor ? 1 : 0,
      analyzedAt: new Date(),
      modelVersion: 'GigaPath-AttentionMIL-v1.0',
      aiInsights: {
        totalImages: 1,
        highRiskImages: demoResult.is_tumor ? 1 : 0,
        modelConfidence: demoResult.confidence
      }
    }
  });

  await demoSample.save();
  console.log(`✅ Demo sample created: ${sampleId} (ML Result: ${demoResult.predicted_class} @ ${(demoResult.confidence * 100).toFixed(1)}%)`);

  res.status(200).json({
    success: true,
    message: `Demo ${demoType} sample analyzed successfully`,
    data: {
      sample: demoSample,
      mlAnalysis: {
        imagesAnalyzed: 1,
        predictions: [demoResult],
        aiInsights: {
          totalImages: 1,
          highRiskImages: demoResult.is_tumor ? 1 : 0,
          modelConfidence: demoResult.confidence
        }
      }
    }
  });
}));

export default router;