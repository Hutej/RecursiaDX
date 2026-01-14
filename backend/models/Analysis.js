import mongoose from 'mongoose';

const analysisSchema = new mongoose.Schema({
  sampleId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Sample',
    required: [true, 'Sample ID is required']
  },
  analysisId: {
    type: String,
    unique: true,
    required: true
  },
  analysisType: {
    type: String,
    enum: {
      values: ['Blood Analysis', 'Tissue Analysis', 'Cancer Detection', 'Cell Count', 'Morphology Analysis', 'Immunohistochemistry'],
      message: 'Analysis type must be one of the predefined values'
    },
    required: [true, 'Analysis type is required']
  },
  aiModel: {
    name: {
      type: String,
      required: [true, 'AI model name is required']
    },
    version: {
      type: String,
      required: [true, 'AI model version is required']
    },
    algorithm: String,
    trainingData: String,
    accuracy: Number,
    sensitivity: Number,
    specificity: Number
  },
  inputData: {
    images: [{
      imageId: String,
      path: String,
      preprocessing: [String],
      roi: { // Region of Interest
        x: Number,
        y: Number,
        width: Number,
        height: Number
      }
    }],
    parameters: {
      magnification: String,
      staining: String,
      scanResolution: String,
      colorSpace: String
    }
  },
  results: {
    overall: {
      diagnosis: String,
      confidence: {
        type: Number,
        min: 0,
        max: 100
      },
      category: {
        type: String,
        enum: ['Normal', 'Abnormal', 'Suspicious', 'Malignant', 'Benign', 'Inconclusive']
      }
    },
    detailed: [{
      parameter: String,
      value: mongoose.Schema.Types.Mixed,
      unit: String,
      referenceRange: String,
      status: {
        type: String,
        enum: ['Normal', 'Low', 'High', 'Critical Low', 'Critical High', 'Abnormal']
      },
      confidence: Number,
      notes: String
    }],
    cellCounts: [{
      cellType: String,
      count: Number,
      percentage: Number,
      morphology: String,
      abnormalities: [String]
    }],
    measurements: [{
      feature: String,
      value: Number,
      unit: String,
      mean: Number,
      stdDev: Number,
      min: Number,
      max: Number
    }],
    annotations: [{
      type: String, // 'region', 'point', 'measurement'
      coordinates: [{
        x: Number,
        y: Number
      }],
      label: String,
      confidence: Number,
      description: String
    }]
  },
  quality: {
    imageQuality: {
      score: {
        type: Number,
        min: 0,
        max: 100
      },
      issues: [String],
      recommendations: [String]
    },
    analysisQuality: {
      completeness: Number,
      reliability: Number,
      consistency: Number
    }
  },
  processing: {
    startTime: {
      type: Date,
      default: Date.now
    },
    endTime: Date,
    duration: Number, // in milliseconds
    resources: {
      cpuUsage: Number,
      memoryUsage: Number,
      gpuUsage: Number
    },
    errors: [String],
    warnings: [String]
  },
  validation: {
    isValidated: {
      type: Boolean,
      default: false
    },
    validatedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    validatedAt: Date,
    validationNotes: String,
    corrections: [{
      field: String,
      originalValue: String,
      correctedValue: String,
      reason: String,
      correctedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
      },
      correctedAt: { type: Date, default: Date.now }
    }],
    approvalStatus: {
      type: String,
      enum: ['Pending', 'Approved', 'Rejected', 'Needs Review'],
      default: 'Pending'
    }
  },
  flags: [{
    type: {
      type: String,
      enum: ['Critical', 'Abnormal', 'Quality Issue', 'Review Required', 'Technical Error']
    },
    description: String,
    severity: {
      type: String,
      enum: ['Low', 'Medium', 'High', 'Critical'],
      default: 'Medium'
    },
    flaggedBy: String, // 'AI' or user ID
    flaggedAt: { type: Date, default: Date.now },
    resolved: { type: Boolean, default: false },
    resolvedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    resolvedAt: Date,
    resolutionNotes: String
  }],
  metadata: {
    version: {
      type: Number,
      default: 1
    },
    isLatest: {
      type: Boolean,
      default: true
    },
    parentAnalysis: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Analysis'
    },
    batchId: String,
    priority: {
      type: String,
      enum: ['Low', 'Normal', 'High', 'Urgent'],
      default: 'Normal'
    },
    tags: [String],
    notes: [String]
  },
  auditTrail: [{
    action: String,
    performedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    performedAt: { type: Date, default: Date.now },
    details: String,
    ipAddress: String,
    userAgent: String
  }]
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for processing time in a readable format
analysisSchema.virtual('processingTimeFormatted').get(function() {
  if (!this.processing.duration) return 'N/A';
  
  const seconds = Math.floor(this.processing.duration / 1000);
  if (seconds < 60) return `${seconds}s`;
  
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
});

// Virtual for overall confidence level
analysisSchema.virtual('confidenceLevel').get(function() {
  const confidence = this.results.overall.confidence;
  if (confidence >= 90) return 'Very High';
  if (confidence >= 80) return 'High';
  if (confidence >= 70) return 'Medium';
  if (confidence >= 60) return 'Low';
  return 'Very Low';
});

// Indexes for performance
analysisSchema.index({ sampleId: 1 });
analysisSchema.index({ analysisId: 1 });
analysisSchema.index({ analysisType: 1 });
analysisSchema.index({ 'results.overall.category': 1 });
analysisSchema.index({ 'validation.approvalStatus': 1 });
analysisSchema.index({ createdAt: -1 });
analysisSchema.index({ 'metadata.isLatest': 1 });
analysisSchema.index({ 'flags.type': 1 });

// Pre-save middleware to generate analysis ID
analysisSchema.pre('save', async function(next) {
  if (!this.analysisId) {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    // Find the count of analyses created today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    const count = await this.constructor.countDocuments({
      createdAt: { $gte: today, $lt: tomorrow }
    });
    
    const sequence = String(count + 1).padStart(6, '0');
    this.analysisId = `ANL-${year}${month}${day}-${sequence}`;
  }
  
  // Set processing duration if end time is set
  if (this.processing.endTime && this.processing.startTime) {
    this.processing.duration = this.processing.endTime - this.processing.startTime;
  }
  
  next();
});

// Pre-save middleware to handle versioning
analysisSchema.pre('save', async function(next) {
  if (this.isNew) {
    // Check if there are existing analyses for this sample
    const existingAnalyses = await this.constructor.find({
      sampleId: this.sampleId,
      analysisType: this.analysisType
    });
    
    if (existingAnalyses.length > 0) {
      // Mark all existing analyses as not latest
      await this.constructor.updateMany(
        { sampleId: this.sampleId, analysisType: this.analysisType },
        { 'metadata.isLatest': false }
      );
      
      this.metadata.version = existingAnalyses.length + 1;
      this.metadata.parentAnalysis = existingAnalyses[existingAnalyses.length - 1]._id;
    }
  }
  next();
});

// Static method to get analysis statistics
analysisSchema.statics.getStats = async function(startDate, endDate) {
  const matchStage = {};
  if (startDate || endDate) {
    matchStage.createdAt = {};
    if (startDate) matchStage.createdAt.$gte = new Date(startDate);
    if (endDate) matchStage.createdAt.$lte = new Date(endDate);
  }

  return await this.aggregate([
    { $match: matchStage },
    {
      $group: {
        _id: {
          type: '$analysisType',
          category: '$results.overall.category'
        },
        count: { $sum: 1 },
        avgConfidence: { $avg: '$results.overall.confidence' },
        avgProcessingTime: { $avg: '$processing.duration' }
      }
    },
    {
      $group: {
        _id: '$_id.type',
        totalAnalyses: { $sum: '$count' },
        categories: {
          $push: {
            category: '$_id.category',
            count: '$count',
            avgConfidence: '$avgConfidence'
          }
        },
        avgProcessingTime: { $avg: '$avgProcessingTime' }
      }
    }
  ]);
};

// Static method to find analyses requiring review
analysisSchema.statics.findRequiringReview = function() {
  return this.find({
    $or: [
      { 'validation.approvalStatus': 'Needs Review' },
      { 'results.overall.confidence': { $lt: 70 } },
      { 'flags.resolved': false }
    ],
    'metadata.isLatest': true
  }).populate('sampleId').sort({ 'results.overall.confidence': 1 });
};

// Instance method to add flag
analysisSchema.methods.addFlag = function(flagData, userId) {
  this.flags.push({
    ...flagData,
    flaggedBy: userId || 'AI',
    flaggedAt: new Date()
  });
  return this.save();
};

// Instance method to resolve flag
analysisSchema.methods.resolveFlag = function(flagId, resolutionNotes, userId) {
  const flag = this.flags.id(flagId);
  if (flag) {
    flag.resolved = true;
    flag.resolvedBy = userId;
    flag.resolvedAt = new Date();
    flag.resolutionNotes = resolutionNotes;
  }
  return this.save();
};

// Instance method to add correction
analysisSchema.methods.addCorrection = function(correctionData, userId) {
  this.validation.corrections.push({
    ...correctionData,
    correctedBy: userId,
    correctedAt: new Date()
  });
  return this.save();
};

const Analysis = mongoose.model('Analysis', analysisSchema);

export default Analysis;