import { body, param, query, validationResult } from 'express-validator';
import mongoose from 'mongoose';

// Validation error handler
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  
  if (!errors.isEmpty()) {
    const errorMessages = errors.array().map(error => ({
      field: error.path,
      message: error.msg,
      value: error.value
    }));
    
    return res.status(400).json({
      success: false,
      message: 'Validation errors',
      errors: errorMessages
    });
  }
  
  next();
};

// Common validation rules
const commonValidations = {
  email: body('email')
    .isEmail()
    .withMessage('Please provide a valid email address')
    .normalizeEmail(),
    
  password: body('password')
    .isLength({ min: 8 })
    .withMessage('Password must be at least 8 characters long')
    .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/)
    .withMessage('Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'),
    
  name: body('name')
    .trim()
    .isLength({ min: 2, max: 100 })
    .withMessage('Name must be between 2 and 100 characters'),
    
  objectId: (field) => 
    body(field)
      .custom((value) => {
        if (!mongoose.Types.ObjectId.isValid(value)) {
          throw new Error(`${field} must be a valid ObjectId`);
        }
        return true;
      }),
      
  objectIdParam: (param) =>
    param(param)
      .custom((value) => {
        if (!mongoose.Types.ObjectId.isValid(value)) {
          throw new Error(`${param} must be a valid ObjectId`);
        }
        return true;
      })
};

// User validation rules
const userValidations = {
  register: [
    commonValidations.name,
    commonValidations.email,
    commonValidations.password,
    body('confirmPassword')
      .custom((value, { req }) => {
        if (value !== req.body.password) {
          throw new Error('Password confirmation does not match password');
        }
        return true;
      }),
    body('role')
      .optional()
      .isIn(['Administrator', 'Pathologist', 'Technician', 'Resident'])
      .withMessage('Invalid role specified'),
    body('department')
      .optional()
      .trim()
      .isLength({ max: 100 })
      .withMessage('Department name cannot exceed 100 characters'),
    body('licenseNumber')
      .optional()
      .trim()
      .isLength({ max: 50 })
      .withMessage('License number cannot exceed 50 characters'),
    handleValidationErrors
  ],
  
  login: [
    commonValidations.email,
    body('password')
      .notEmpty()
      .withMessage('Password is required'),
    handleValidationErrors
  ],
  
  updateProfile: [
    body('name')
      .optional()
      .trim()
      .isLength({ min: 2, max: 100 })
      .withMessage('Name must be between 2 and 100 characters'),
    body('department')
      .optional()
      .trim()
      .isLength({ max: 100 })
      .withMessage('Department name cannot exceed 100 characters'),
    body('phone')
      .optional()
      .isMobilePhone()
      .withMessage('Please provide a valid phone number'),
    body('preferences.theme')
      .optional()
      .isIn(['light', 'dark', 'system'])
      .withMessage('Theme must be light, dark, or system'),
    body('preferences.language')
      .optional()
      .isLength({ min: 2, max: 5 })
      .withMessage('Language code must be 2-5 characters'),
    handleValidationErrors
  ],
  
  changePassword: [
    body('currentPassword')
      .notEmpty()
      .withMessage('Current password is required'),
    commonValidations.password.withMessage('New password must meet security requirements'),
    body('confirmPassword')
      .custom((value, { req }) => {
        if (value !== req.body.password) {
          throw new Error('Password confirmation does not match new password');
        }
        return true;
      }),
    handleValidationErrors
  ]
};

// Sample validation rules
const sampleValidations = {
  create: [
    body('patientInfo.patientId')
      .notEmpty()
      .withMessage('Patient ID is required')
      .isLength({ max: 50 })
      .withMessage('Patient ID cannot exceed 50 characters'),
    body('patientInfo.name')
      .notEmpty()
      .withMessage('Patient name is required')
      .trim()
      .isLength({ min: 2, max: 100 })
      .withMessage('Patient name must be between 2 and 100 characters'),
    body('patientInfo.dateOfBirth')
      .isDate()
      .withMessage('Valid date of birth is required'),
    body('patientInfo.gender')
      .isIn(['Male', 'Female', 'Other', 'Unknown'])
      .withMessage('Gender must be Male, Female, Other, or Unknown'),
    body('specimenType')
      .notEmpty()
      .withMessage('Specimen type is required')
      .isIn(['Biopsy', 'Resection', 'Cytology', 'Frozen Section', 'Bone Marrow', 'Other'])
      .withMessage('Invalid specimen type'),
    body('anatomicalSite')
      .notEmpty()
      .withMessage('Anatomical site is required')
      .trim()
      .isLength({ max: 200 })
      .withMessage('Anatomical site cannot exceed 200 characters'),
    body('clinicalHistory')
      .optional()
      .trim()
      .isLength({ max: 1000 })
      .withMessage('Clinical history cannot exceed 1000 characters'),
    body('priority')
      .optional()
      .isIn(['Routine', 'Urgent', 'STAT'])
      .withMessage('Priority must be Routine, Urgent, or STAT'),
    handleValidationErrors
  ],
  
  update: [
    body('patientInfo.name')
      .optional()
      .trim()
      .isLength({ min: 2, max: 100 })
      .withMessage('Patient name must be between 2 and 100 characters'),
    body('clinicalHistory')
      .optional()
      .trim()
      .isLength({ max: 1000 })
      .withMessage('Clinical history cannot exceed 1000 characters'),
    body('status')
      .optional()
      .isIn(['Received', 'Processing', 'Sectioning', 'Staining', 'Reading', 'Reporting', 'Complete', 'Cancelled'])
      .withMessage('Invalid status'),
    body('priority')
      .optional()
      .isIn(['Routine', 'Urgent', 'STAT'])
      .withMessage('Priority must be Routine, Urgent, or STAT'),
    handleValidationErrors
  ]
};

// Analysis validation rules
const analysisValidations = {
  create: [
    commonValidations.objectId('sampleId'),
    body('analysisType')
      .notEmpty()
      .withMessage('Analysis type is required')
      .isIn(['H&E', 'IHC', 'ISH', 'Molecular', 'Cytology', 'Frozen Section', 'Special Stains'])
      .withMessage('Invalid analysis type'),
    body('aiModel.name')
      .notEmpty()
      .withMessage('AI model name is required'),
    body('aiModel.version')
      .notEmpty()
      .withMessage('AI model version is required'),
    body('confidence')
      .optional()
      .isFloat({ min: 0, max: 1 })
      .withMessage('Confidence must be between 0 and 1'),
    handleValidationErrors
  ],
  
  update: [
    body('status')
      .optional()
      .isIn(['Pending', 'In Progress', 'Complete', 'Under Review', 'Approved', 'Rejected'])
      .withMessage('Invalid status'),
    body('confidence')
      .optional()
      .isFloat({ min: 0, max: 1 })
      .withMessage('Confidence must be between 0 and 1'),
    body('reviewedBy')
      .optional()
      .custom((value) => {
        if (!mongoose.Types.ObjectId.isValid(value)) {
          throw new Error('ReviewedBy must be a valid user ID');
        }
        return true;
      }),
    handleValidationErrors
  ]
};

// Report validation rules
const reportValidations = {
  create: [
    commonValidations.objectId('sampleId'),
    commonValidations.objectId('analysisId'),
    body('reportType')
      .notEmpty()
      .withMessage('Report type is required')
      .isIn(['Preliminary', 'Final', 'Amended', 'Addendum', 'Consultation'])
      .withMessage('Invalid report type'),
    body('content.diagnosis.primary')
      .optional()
      .trim()
      .isLength({ max: 500 })
      .withMessage('Primary diagnosis cannot exceed 500 characters'),
    handleValidationErrors
  ],
  
  update: [
    body('reportType')
      .optional()
      .isIn(['Preliminary', 'Final', 'Amended', 'Addendum', 'Consultation'])
      .withMessage('Invalid report type'),
    body('validation.status')
      .optional()
      .isIn(['Draft', 'Under Review', 'Approved', 'Published', 'Revised'])
      .withMessage('Invalid validation status'),
    handleValidationErrors
  ],
  
  review: [
    body('status')
      .notEmpty()
      .withMessage('Review status is required')
      .isIn(['Approved', 'Rejected', 'Needs Changes'])
      .withMessage('Invalid review status'),
    body('comments')
      .optional()
      .trim()
      .isLength({ max: 1000 })
      .withMessage('Comments cannot exceed 1000 characters'),
    handleValidationErrors
  ]
};

// Query validation rules
const queryValidations = {
  pagination: [
    query('page')
      .optional()
      .isInt({ min: 1 })
      .withMessage('Page must be a positive integer'),
    query('limit')
      .optional()
      .isInt({ min: 1, max: 100 })
      .withMessage('Limit must be between 1 and 100'),
    handleValidationErrors
  ],
  
  dateRange: [
    query('startDate')
      .optional()
      .isISO8601()
      .withMessage('Start date must be in ISO 8601 format'),
    query('endDate')
      .optional()
      .isISO8601()
      .withMessage('End date must be in ISO 8601 format')
      .custom((value, { req }) => {
        if (req.query.startDate && new Date(value) < new Date(req.query.startDate)) {
          throw new Error('End date must be after start date');
        }
        return true;
      }),
    handleValidationErrors
  ]
};

// File upload validation
const fileValidations = {
  image: [
    body('fileType')
      .optional()
      .isIn(['image/jpeg', 'image/png', 'image/tiff', 'image/dicom'])
      .withMessage('Invalid file type. Only JPEG, PNG, TIFF, and DICOM files are allowed'),
    body('fileName')
      .notEmpty()
      .withMessage('File name is required')
      .matches(/^[a-zA-Z0-9._-]+$/)
      .withMessage('File name contains invalid characters'),
    handleValidationErrors
  ]
};

export {
  handleValidationErrors,
  userValidations,
  sampleValidations,
  analysisValidations,
  reportValidations,
  queryValidations,
  fileValidations
};