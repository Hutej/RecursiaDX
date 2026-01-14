import mongoose from 'mongoose';

const reportSchema = new mongoose.Schema({
  reportId: {
    type: String,
    unique: true,
    required: true
  },
  sampleId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Sample',
    required: [true, 'Sample ID is required']
  },
  analysisId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Analysis',
    required: [true, 'Analysis ID is required']
  },
  reportType: {
    type: String,
    enum: {
      values: ['Preliminary', 'Final', 'Amended', 'Addendum', 'Consultation'],
      message: 'Report type must be one of the predefined values'
    },
    required: [true, 'Report type is required']
  },
  template: {
    name: String,
    version: String,
    sections: [String]
  },
  content: {
    clinicalHistory: String,
    grossDescription: String,
    microscopicDescription: String,
    diagnosis: {
      primary: String,
      secondary: [String],
      differential: [String],
      staging: String,
      grade: String,
      margin: String,
      lymphNodes: String
    },
    immunohistochemistry: [{
      marker: String,
      result: String,
      interpretation: String,
      staining: String
    }],
    molecularFindings: [{
      test: String,
      result: String,
      significance: String,
      method: String
    }],
    specialStains: [{
      stain: String,
      result: String,
      interpretation: String
    }],
    comment: String,
    recommendations: [String],
    followUp: String,
    prognosticFactors: [String],
    therapeuticImplications: [String]
  },
  findings: {
    keyFindings: [String],
    abnormalities: [{
      type: String,
      location: String,
      description: String,
      significance: String,
      grading: String
    }],
    measurements: [{
      structure: String,
      dimension: String,
      value: Number,
      unit: String
    }],
    cellularDetails: [{
      cellType: String,
      count: Number,
      percentage: Number,
      morphology: String,
      abnormalities: [String]
    }]
  },
  images: [{
    imageId: String,
    caption: String,
    magnification: String,
    staining: String,
    annotations: [{
      type: String,
      coordinates: [Number],
      label: String,
      description: String
    }]
  }],
  validation: {
    status: {
      type: String,
      enum: ['Draft', 'Under Review', 'Approved', 'Published', 'Revised'],
      default: 'Draft'
    },
    reviewedBy: [{
      user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
      },
      reviewedAt: Date,
      status: {
        type: String,
        enum: ['Approved', 'Rejected', 'Needs Changes']
      },
      comments: String,
      signature: String
    }],
    approvedBy: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    approvedAt: Date,
    digitalSignature: String,
    amendments: [{
      amendmentNumber: Number,
      reason: String,
      changes: String,
      amendedBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
      },
      amendedAt: Date
    }]
  },
  distribution: {
    recipients: [{
      type: {
        type: String,
        enum: ['Physician', 'Patient', 'Laboratory', 'External Consultant']
      },
      name: String,
      email: String,
      department: String,
      deliveryMethod: {
        type: String,
        enum: ['Email', 'Portal', 'Print', 'Fax', 'API'],
        default: 'Email'
      },
      sentAt: Date,
      deliveryStatus: {
        type: String,
        enum: ['Pending', 'Sent', 'Delivered', 'Failed', 'Bounced'],
        default: 'Pending'
      },
      trackingId: String
    }],
    isConfidential: {
      type: Boolean,
      default: true
    },
    accessLevel: {
      type: String,
      enum: ['Public', 'Internal', 'Restricted', 'Confidential'],
      default: 'Confidential'
    }
  },
  turnaroundTime: {
    received: Date,
    reported: Date,
    duration: Number // in hours
  },
  quality: {
    completeness: {
      type: Number,
      min: 0,
      max: 100
    },
    accuracy: {
      type: Number,
      min: 0,
      max: 100
    },
    clarity: {
      type: Number,
      min: 0,
      max: 100
    },
    clinicalRelevance: {
      type: Number,
      min: 0,
      max: 100
    },
    overallScore: {
      type: Number,
      min: 0,
      max: 100
    }
  },
  compliance: {
    guidelines: [String],
    standards: [String],
    certifications: [String],
    auditTrail: [{
      action: String,
      user: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
      },
      timestamp: { type: Date, default: Date.now },
      details: String,
      ipAddress: String
    }]
  },
  metadata: {
    version: {
      type: Number,
      default: 1
    },
    isLatest: {
      type: Boolean,
      default: true
    },
    parentReport: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Report'
    },
    language: {
      type: String,
      default: 'en'
    },
    format: {
      type: String,
      enum: ['PDF', 'HTML', 'DOCX', 'HL7'],
      default: 'PDF'
    },
    size: Number, // in bytes
    checksum: String,
    tags: [String],
    priority: {
      type: String,
      enum: ['Low', 'Normal', 'High', 'Urgent'],
      default: 'Normal'
    }
  },
  billing: {
    codes: [{
      type: String, // CPT, ICD-10, etc.
      code: String,
      description: String,
      modifier: String,
      units: Number,
      amount: Number
    }],
    totalAmount: Number,
    currency: {
      type: String,
      default: 'USD'
    },
    billingDate: Date,
    paymentStatus: {
      type: String,
      enum: ['Pending', 'Paid', 'Partial', 'Overdue', 'Cancelled'],
      default: 'Pending'
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual for formatted turnaround time
reportSchema.virtual('turnaroundTimeFormatted').get(function() {
  if (!this.turnaroundTime.duration) return 'N/A';
  
  const hours = this.turnaroundTime.duration;
  if (hours < 24) return `${hours}h`;
  
  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;
  return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
});

// Virtual for overall quality score
reportSchema.virtual('qualityScore').get(function() {
  const scores = [
    this.quality.completeness,
    this.quality.accuracy,
    this.quality.clarity,
    this.quality.clinicalRelevance
  ].filter(score => score !== undefined);
  
  if (scores.length === 0) return 0;
  
  const average = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  return Math.round(average * 100) / 100;
});

// Indexes for performance
reportSchema.index({ reportId: 1 });
reportSchema.index({ sampleId: 1 });
reportSchema.index({ analysisId: 1 });
reportSchema.index({ reportType: 1 });
reportSchema.index({ 'validation.status': 1 });
reportSchema.index({ createdAt: -1 });
reportSchema.index({ 'metadata.isLatest': 1 });
reportSchema.index({ 'turnaroundTime.reported': -1 });

// Pre-save middleware to generate report ID
reportSchema.pre('save', async function(next) {
  if (!this.reportId) {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    // Find the count of reports created today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    
    const count = await this.constructor.countDocuments({
      createdAt: { $gte: today, $lt: tomorrow }
    });
    
    const sequence = String(count + 1).padStart(6, '0');
    this.reportId = `RPT-${year}${month}${day}-${sequence}`;
  }
  
  // Calculate turnaround time if both dates are available
  if (this.turnaroundTime.reported && this.turnaroundTime.received) {
    const diffMs = this.turnaroundTime.reported - this.turnaroundTime.received;
    this.turnaroundTime.duration = Math.floor(diffMs / (1000 * 60 * 60)); // in hours
  }
  
  // Calculate overall quality score
  if (this.quality.completeness !== undefined || 
      this.quality.accuracy !== undefined || 
      this.quality.clarity !== undefined || 
      this.quality.clinicalRelevance !== undefined) {
    this.quality.overallScore = this.qualityScore;
  }
  
  next();
});

// Pre-save middleware to handle versioning
reportSchema.pre('save', async function(next) {
  if (this.isNew) {
    // Check if there are existing reports for this sample
    const existingReports = await this.constructor.find({
      sampleId: this.sampleId,
      reportType: this.reportType
    });
    
    if (existingReports.length > 0) {
      // Mark all existing reports as not latest
      await this.constructor.updateMany(
        { sampleId: this.sampleId, reportType: this.reportType },
        { 'metadata.isLatest': false }
      );
      
      this.metadata.version = existingReports.length + 1;
      this.metadata.parentReport = existingReports[existingReports.length - 1]._id;
    }
  }
  next();
});

// Static method to get report statistics
reportSchema.statics.getStats = async function(startDate, endDate) {
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
          type: '$reportType',
          status: '$validation.status'
        },
        count: { $sum: 1 },
        avgTurnaroundTime: { $avg: '$turnaroundTime.duration' },
        avgQualityScore: { $avg: '$quality.overallScore' }
      }
    },
    {
      $group: {
        _id: '$_id.type',
        totalReports: { $sum: '$count' },
        statusBreakdown: {
          $push: {
            status: '$_id.status',
            count: '$count'
          }
        },
        avgTurnaroundTime: { $avg: '$avgTurnaroundTime' },
        avgQualityScore: { $avg: '$avgQualityScore' }
      }
    }
  ]);
};

// Static method to find pending reports
reportSchema.statics.findPending = function() {
  return this.find({
    'validation.status': { $in: ['Draft', 'Under Review'] },
    'metadata.isLatest': true
  }).populate('sampleId analysisId').sort({ createdAt: 1 });
};

// Instance method to add review
reportSchema.methods.addReview = function(reviewData, userId) {
  this.validation.reviewedBy.push({
    ...reviewData,
    user: userId,
    reviewedAt: new Date()
  });
  return this.save();
};

// Instance method to approve report
reportSchema.methods.approve = function(userId, signature) {
  this.validation.status = 'Approved';
  this.validation.approvedBy = userId;
  this.validation.approvedAt = new Date();
  this.validation.digitalSignature = signature;
  return this.save();
};

// Instance method to add amendment
reportSchema.methods.addAmendment = function(amendmentData, userId) {
  const amendmentNumber = this.validation.amendments.length + 1;
  this.validation.amendments.push({
    ...amendmentData,
    amendmentNumber,
    amendedBy: userId,
    amendedAt: new Date()
  });
  this.validation.status = 'Revised';
  return this.save();
};

const Report = mongoose.model('Report', reportSchema);

export default Report;