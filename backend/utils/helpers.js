import jwt from 'jsonwebtoken';
import crypto from 'crypto';

// JWT utilities
export const jwtUtils = {
  // Generate access token
  generateAccessToken: (payload) => {
    return jwt.sign(payload, process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_EXPIRE || '7d'
    });
  },

  // Generate refresh token
  generateRefreshToken: (payload) => {
    return jwt.sign(payload, process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET, {
      expiresIn: process.env.JWT_REFRESH_EXPIRE || '30d'
    });
  },

  // Verify token
  verifyToken: (token, secret = process.env.JWT_SECRET) => {
    return jwt.verify(token, secret);
  },

  // Decode token without verification
  decodeToken: (token) => {
    return jwt.decode(token);
  }
};

// Encryption utilities
export const encryptionUtils = {
  // Generate random token
  generateRandomToken: (length = 32) => {
    return crypto.randomBytes(length).toString('hex');
  },

  // Hash string with SHA-256
  hashString: (string) => {
    return crypto.createHash('sha256').update(string).digest('hex');
  },

  // Generate secure random password
  generatePassword: (length = 12) => {
    const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*';
    let password = '';
    for (let i = 0; i < length; i++) {
      password += charset.charAt(Math.floor(Math.random() * charset.length));
    }
    return password;
  },

  // Encrypt data (AES-256-GCM)
  encrypt: (text, key = process.env.ENCRYPTION_KEY) => {
    if (!key) throw new Error('Encryption key not provided');
    
    const algorithm = 'aes-256-gcm';
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(algorithm, key);
    
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  },

  // Decrypt data (AES-256-GCM)
  decrypt: (encryptedData, key = process.env.ENCRYPTION_KEY) => {
    if (!key) throw new Error('Encryption key not provided');
    
    const algorithm = 'aes-256-gcm';
    const decipher = crypto.createDecipher(algorithm, key);
    
    decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
    
    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }
};

// Date utilities
export const dateUtils = {
  // Get formatted date string
  formatDate: (date, format = 'YYYY-MM-DD') => {
    const d = new Date(date);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');

    const formatMap = {
      'YYYY-MM-DD': `${year}-${month}-${day}`,
      'DD-MM-YYYY': `${day}-${month}-${year}`,
      'MM/DD/YYYY': `${month}/${day}/${year}`,
      'YYYY-MM-DD HH:mm:ss': `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`,
      'DD/MM/YYYY HH:mm': `${day}/${month}/${year} ${hours}:${minutes}`
    };

    return formatMap[format] || formatMap['YYYY-MM-DD'];
  },

  // Get time difference in human-readable format
  getTimeAgo: (date) => {
    const now = new Date();
    const diffMs = now - new Date(date);
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    const diffMonths = Math.floor(diffDays / 30);
    const diffYears = Math.floor(diffDays / 365);

    if (diffYears > 0) return `${diffYears} year${diffYears > 1 ? 's' : ''} ago`;
    if (diffMonths > 0) return `${diffMonths} month${diffMonths > 1 ? 's' : ''} ago`;
    if (diffDays > 0) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    if (diffHours > 0) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffMins > 0) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    return 'Just now';
  },

  // Add/subtract time from date
  addTime: (date, amount, unit = 'days') => {
    const d = new Date(date);
    const unitMap = {
      seconds: 1000,
      minutes: 60 * 1000,
      hours: 60 * 60 * 1000,
      days: 24 * 60 * 60 * 1000,
      weeks: 7 * 24 * 60 * 60 * 1000,
      months: 30 * 24 * 60 * 60 * 1000,
      years: 365 * 24 * 60 * 60 * 1000
    };

    d.setTime(d.getTime() + (amount * unitMap[unit]));
    return d;
  },

  // Check if date is within range
  isWithinRange: (date, startDate, endDate) => {
    const d = new Date(date);
    const start = new Date(startDate);
    const end = new Date(endDate);
    return d >= start && d <= end;
  }
};

// Validation utilities
export const validationUtils = {
  // Validate email format
  isValidEmail: (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  },

  // Validate phone number
  isValidPhone: (phone) => {
    const phoneRegex = /^\+?[\d\s\-\(\)]{10,}$/;
    return phoneRegex.test(phone);
  },

  // Validate password strength
  validatePassword: (password) => {
    const minLength = 8;
    const hasLowerCase = /[a-z]/.test(password);
    const hasUpperCase = /[A-Z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    const errors = [];
    
    if (password.length < minLength) {
      errors.push(`Password must be at least ${minLength} characters long`);
    }
    if (!hasLowerCase) {
      errors.push('Password must contain at least one lowercase letter');
    }
    if (!hasUpperCase) {
      errors.push('Password must contain at least one uppercase letter');
    }
    if (!hasNumbers) {
      errors.push('Password must contain at least one number');
    }
    if (!hasSpecialChar) {
      errors.push('Password must contain at least one special character');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  },

  // Sanitize input string
  sanitizeString: (str, maxLength = 1000) => {
    if (!str || typeof str !== 'string') return '';
    
    return str
      .trim()
      .slice(0, maxLength)
      .replace(/[<>]/g, '') // Remove potential HTML tags
      .replace(/javascript:/gi, '') // Remove javascript: protocol
      .replace(/on\w+=/gi, ''); // Remove event handlers
  },

  // Validate MongoDB ObjectId
  isValidObjectId: (id) => {
    return /^[0-9a-fA-F]{24}$/.test(id);
  }
};

// Response utilities
export const responseUtils = {
  // Success response
  success: (res, message = 'Success', data = null, statusCode = 200) => {
    const response = {
      success: true,
      message
    };
    
    if (data !== null) {
      response.data = data;
    }
    
    return res.status(statusCode).json(response);
  },

  // Error response
  error: (res, message = 'An error occurred', statusCode = 500, errors = null) => {
    const response = {
      success: false,
      message
    };
    
    if (errors) {
      response.errors = errors;
    }
    
    return res.status(statusCode).json(response);
  },

  // Paginated response
  paginated: (res, data, pagination, message = 'Success') => {
    return res.json({
      success: true,
      message,
      data,
      pagination
    });
  }
};

// File utilities
export const fileUtils = {
  // Get file extension
  getFileExtension: (filename) => {
    return filename.split('.').pop().toLowerCase();
  },

  // Check if file type is allowed
  isAllowedFileType: (filename, allowedTypes = []) => {
    const extension = fileUtils.getFileExtension(filename);
    return allowedTypes.includes(extension);
  },

  // Format file size
  formatFileSize: (bytes) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  // Generate unique filename
  generateUniqueFilename: (originalName) => {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2);
    const extension = fileUtils.getFileExtension(originalName);
    return `${timestamp}_${random}.${extension}`;
  }
};

// Database utilities
export const dbUtils = {
  // Build MongoDB filter from query parameters
  buildFilter: (query, allowedFields = []) => {
    const filter = {};
    
    allowedFields.forEach(field => {
      if (query[field] !== undefined) {
        filter[field] = query[field];
      }
    });
    
    return filter;
  },

  // Build sort object from query
  buildSort: (sortBy, sortOrder = 'desc') => {
    if (!sortBy) return { createdAt: -1 };
    
    const order = sortOrder.toLowerCase() === 'asc' ? 1 : -1;
    return { [sortBy]: order };
  },

  // Calculate pagination
  calculatePagination: (page = 1, limit = 20, totalCount = 0) => {
    const currentPage = parseInt(page);
    const itemsPerPage = parseInt(limit);
    const totalPages = Math.ceil(totalCount / itemsPerPage);
    const skip = (currentPage - 1) * itemsPerPage;
    
    return {
      currentPage,
      totalPages,
      totalCount,
      itemsPerPage,
      skip,
      hasNext: currentPage < totalPages,
      hasPrev: currentPage > 1
    };
  }
};

// Logging utilities
export const logUtils = {
  // Log levels
  levels: {
    ERROR: 'error',
    WARN: 'warn',
    INFO: 'info',
    DEBUG: 'debug'
  },

  // Format log message
  formatLog: (level, message, meta = {}) => {
    return {
      timestamp: new Date().toISOString(),
      level,
      message,
      ...meta
    };
  },

  // Log to console (in development)
  log: (level, message, meta = {}) => {
    if (process.env.NODE_ENV === 'development') {
      const formatted = logUtils.formatLog(level, message, meta);
      console.log(JSON.stringify(formatted, null, 2));
    }
  }
};