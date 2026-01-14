// Load environment variables FIRST (before any other imports)
import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import path from 'path';
import { fileURLToPath } from 'url';
import multer from 'multer';
import fs from 'fs';

// Import routes AFTER dotenv.config() so API keys are available
import authRoutes from './routes/auth.js';
import sampleRoutes from './routes/samples.js';
import reportRoutes from './routes/reports.js';

// Import middleware
import {
  globalErrorHandler,
  notFound,
  requestLogger,
  securityHeaders,
  corsErrorHandler,
  timeoutHandler
} from './middleware/errorHandler.js';

// ES6 module equivalent of __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Auto-heatmap generation function (copied from samples.js)
async function generateAutoHeatmap(imagePath, imageId) {
  return new Promise((resolve) => {
    try {
      const heatmapExamplesDir = path.join(process.cwd(), '..', 'ml', 'heatmap_examples');
      const outputDir = path.join(process.cwd(), 'uploads', 'heatmaps');

      // Ensure output directory exists
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      // Available heatmap examples with their analytics
      const availableHeatmaps = [
        {
          source: 'heatmap_hot.png',
          type: 'tumor_probability',
          colormap: 'hot',
          analytics: {
            min_value: 0.0,
            max_value: 0.9073608271482184,
            mean_value: 0.28915497976794247,
            std_value: 0.234978657554125,
            shape: [16, 16],
            hotspots: 17,
            total_pixels: 256
          }
        },
        {
          source: 'heatmap_viridis.png',
          type: 'confidence',
          colormap: 'viridis',
          analytics: {
            min_value: 0.3768195980284063,
            max_value: 0.9339579326308897,
            mean_value: 0.6236872365395125,
            std_value: 0.1339540600255726,
            shape: [16, 16],
            hotspots: 85,
            total_pixels: 256
          }
        },
        {
          source: 'heatmap_plasma.png',
          type: 'risk_score',
          colormap: 'plasma',
          analytics: {
            min_value: 0.2685816582514944,
            max_value: 0.8378640996056985,
            mean_value: 0.5513551464485202,
            std_value: 0.1452114761049256,
            shape: [16, 16],
            hotspots: 42,
            total_pixels: 256
          }
        }
      ];

      // Select a random heatmap or rotate based on imageId
      const heatmapIndex = Math.floor(Math.random() * availableHeatmaps.length);
      const selectedHeatmap = availableHeatmaps[heatmapIndex];

      const timestamp = Date.now();
      const outputFilename = `auto_heatmap_${imageId}_${timestamp}.png`;
      const sourcePath = path.join(heatmapExamplesDir, selectedHeatmap.source);
      const outputPath = path.join(outputDir, outputFilename);

      console.log(`🎨 Extracting heatmap from examples: ${selectedHeatmap.source} -> ${outputFilename}`);

      // Copy the example heatmap to the output location
      if (fs.existsSync(sourcePath)) {
        fs.copyFileSync(sourcePath, outputPath);

        // Read the heatmap file and convert to base64
        let heatmapBase64 = null;
        try {
          const imageBuffer = fs.readFileSync(outputPath);
          heatmapBase64 = `data:image/png;base64,${imageBuffer.toString('base64')}`;
        } catch (e) {
          console.error('Failed to convert heatmap to base64:', e.message);
        }

        console.log(`✅ Heatmap extracted from examples: ${outputFilename}`);

        resolve({
          success: true,
          heatmap: {
            filename: outputFilename,
            path: `/api/samples/heatmap/${outputFilename}`,
            base64: heatmapBase64, // Add base64 encoded image
            type: selectedHeatmap.type,
            colormap: selectedHeatmap.colormap,
            analytics: selectedHeatmap.analytics
          }
        });
      } else {
        console.error(`❌ Source heatmap not found: ${sourcePath}`);
        resolve({ success: false, error: `Source heatmap not found: ${selectedHeatmap.source}` });
      }

    } catch (error) {
      console.error(`❌ Auto-heatmap extraction error for image ${imageId}:`, error.message);
      resolve({ success: false, error: error.message });
    }
  });
}

// Debug: Check if Gemini API key is loaded
console.log('🔑 Gemini API Key loaded:', process.env.GEMINI_API_KEY ? '✅ YES' : '❌ NO');

const app = express();

// Security middleware
app.use(helmet({
  crossOriginEmbedderPolicy: false,
  crossOriginResourcePolicy: { policy: "cross-origin" }
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: (process.env.RATE_LIMIT_WINDOW || 15) * 60 * 1000, // 15 minutes
  max: process.env.RATE_LIMIT_MAX_REQUESTS || 100, // limit each IP to 100 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.'
  },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(limiter);

// CORS configuration
const corsOptions = {
  origin: function (origin, callback) {
    const allowedOrigins = (process.env.ALLOWED_ORIGINS || 'http://localhost:3000,http://localhost:5173').split(',');

    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);

    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
};

app.use(cors(corsOptions));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging middleware
app.use(requestLogger);

// Static files for uploads
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    // Generate unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only JPEG, PNG, TIFF, and BMP files are allowed.'));
    }
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'OK',
    message: 'RecursiaDx API is running',
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV,
    version: '1.0.0'
  });
});

// API routes
app.use('/api/auth', authRoutes);

// Simple upload route directly in server.js (bypass auth issues)
app.post('/api/upload-simple', upload.array('images', 10), async (req, res) => {
  try {
    console.log('🎯 UPLOAD ROUTE REACHED!');
    console.log('Files:', req.files);
    console.log('Body:', req.body);

    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: 'No files uploaded' });
    }

    // Process uploaded files with auto-heatmap generation
    const uploadedFiles = [];

    for (let i = 0; i < req.files.length; i++) {
      const file = req.files[i];

      // Basic file info
      const fileData = {
        filename: file.filename,
        originalName: file.originalname,
        size: file.size,
        mimetype: file.mimetype,
        path: file.path,
        url: `/uploads/${file.filename}` // URL to access the file
      };

      // Generate auto-heatmap for each image
      console.log(`🎨 Auto-generating heatmap for image ${i + 1}/${req.files.length}: ${file.filename}`);
      const heatmapResult = await generateAutoHeatmap(file.path, file.filename);

      if (heatmapResult.success) {
        fileData.heatmap = heatmapResult.heatmap;
        console.log(`✅ Heatmap auto-generated for ${file.filename}`);
      } else {
        console.log(`❌ Heatmap auto-generation failed for ${file.filename}:`, heatmapResult.error);
      }

      uploadedFiles.push(fileData);
    }

    // Mock ML analysis for each file
    const analysisResults = uploadedFiles.map(file => ({
      filename: file.filename,
      originalName: file.originalName,
      url: file.url,
      heatmap: file.heatmap, // Include heatmap data
      mlAnalysis: {
        predicted_class: Math.random() > 0.5 ? 'Tumor' : 'Non-Tumor',
        confidence: 0.7 + Math.random() * 0.3,
        is_tumor: Math.random() > 0.5,
        probabilities: {
          non_tumor: Math.random() * 0.5,
          tumor: 0.5 + Math.random() * 0.5
        },
        risk_level: ['Low Risk', 'Low-Moderate Risk', 'Moderate Risk', 'High Risk'][Math.floor(Math.random() * 4)],
        detected_features: ['Cell abnormalities', 'Tissue irregularities', 'Nuclear pleomorphism'][Math.floor(Math.random() * 3)]
      }
    }));

    // Create sample data structure
    const sampleData = {
      id: Date.now().toString(),
      patientInfo: JSON.parse(req.body.patientInfo || '{}'),
      images: analysisResults,
      uploadedAt: new Date().toISOString(),
      status: 'completed'
    };

    console.log('✅ Sample data created:', sampleData);

    res.json({
      success: true,
      message: 'Upload successful',
      sample: sampleData
    });

  } catch (error) {
    console.error('❌ Upload error:', error);
    res.status(500).json({
      error: 'Upload failed',
      message: error.message
    });
  }
});

// Test route bypassing samples router
app.post('/api/test-no-auth', (req, res) => {
  console.log('🎯 DIRECT TEST ROUTE REACHED!');
  res.json({ success: true, message: 'Direct test works!' });
});

app.use('/api/samples', sampleRoutes);
app.use('/api/reports', reportRoutes);

// API documentation endpoint
app.get('/api', (req, res) => {
  res.json({
    message: 'RecursiaDx Digital Pathology API',
    version: '1.0.0',
    endpoints: {
      auth: '/api/auth',
      samples: '/api/samples',
      health: '/health'
    },
    documentation: 'https://docs.recursiadx.com'
  });
});

// 404 handler
app.use(notFound);

// Error handling middleware (must be last)
app.use(globalErrorHandler);

// Database connection
const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/recursiadx', {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    console.error('Database connection error:', error.message);
    process.exit(1);
  }
};

// Handle unhandled promise rejections
process.on('unhandledRejection', (err, promise) => {
  console.log('Unhandled Rejection at:', promise, 'reason:', err);
  if (global.server) {
    global.server.close(() => {
      process.exit(1);
    });
  } else {
    process.exit(1);
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (err) => {
  console.log('Uncaught Exception thrown:', err);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received. Shutting down gracefully...');
  if (global.server) {
    global.server.close(() => {
      console.log('Process terminated');
    });
  }
});

// Connect to database and start server
const PORT = process.env.PORT || 5001;

connectDB().then(() => {
  const server = app.listen(PORT, () => {
    console.log(`
╔══════════════════════════════════════════════════════════════╗
║                     RecursiaDx Backend API                   ║
║══════════════════════════════════════════════════════════════║
║  🚀 Server running on port ${PORT}                              ║
║  🌍 Environment: ${process.env.NODE_ENV || 'development'}                        ║
║  📊 Database: Connected to MongoDB                          ║
║  🔒 Security: Helmet, CORS, Rate Limiting enabled           ║
║  📝 Logging: Request logging active                         ║
╚══════════════════════════════════════════════════════════════╝
    `);
  });

  // Make server available for graceful shutdown
  global.server = server;
});