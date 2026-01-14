import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const config = {
  // Server configuration
  server: {
    port: process.env.PORT || 5000,
    nodeEnv: process.env.NODE_ENV || 'development',
    trustProxy: process.env.TRUST_PROXY === 'true'
  },

  // Database configuration
  database: {
    uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/recursia-dx',
    testUri: process.env.MONGODB_TEST_URI || 'mongodb://localhost:27017/recursia-dx-test',
    options: {
      useNewUrlParser: true,
      useUnifiedTopology: true,
      maxPoolSize: 10,
      serverSelectionTimeoutMS: 5000,
      socketTimeoutMS: 45000
    }
  },

  // JWT configuration
  jwt: {
    secret: process.env.JWT_SECRET || 'fallback-secret-change-in-production',
    expire: process.env.JWT_EXPIRE || '7d',
    refreshSecret: process.env.JWT_REFRESH_SECRET || process.env.JWT_SECRET || 'fallback-refresh-secret',
    refreshExpire: process.env.JWT_REFRESH_EXPIRE || '30d'
  },

  // Security configuration
  security: {
    bcryptSaltRounds: parseInt(process.env.BCRYPT_SALT_ROUNDS) || 12,
    maxLoginAttempts: parseInt(process.env.MAX_LOGIN_ATTEMPTS) || 5,
    accountLockTime: parseInt(process.env.ACCOUNT_LOCK_TIME) || 30 * 60 * 1000, // 30 minutes
    encryptionKey: process.env.ENCRYPTION_KEY,
    sessionSecret: process.env.SESSION_SECRET || 'fallback-session-secret',
    sessionExpire: parseInt(process.env.SESSION_EXPIRE) || 24 * 60 * 60 * 1000 // 24 hours
  },

  // CORS configuration
  cors: {
    origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
    credentials: process.env.CORS_CREDENTIALS === 'true' || true,
    optionsSuccessStatus: 200
  },

  // Rate limiting
  rateLimit: {
    windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 15 * 60 * 1000, // 15 minutes
    maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 100,
    skipSuccessfulRequests: false,
    skipFailedRequests: false
  },

  // File upload configuration
  upload: {
    maxFileSize: parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024, // 10MB
    uploadPath: process.env.UPLOAD_PATH || 'uploads/',
    allowedTypes: process.env.ALLOWED_FILE_TYPES?.split(',') || ['jpg', 'jpeg', 'png', 'tiff', 'dcm', 'pdf'],
    storageType: process.env.STORAGE_TYPE || 'local'
  },

  // Email configuration
  email: {
    from: process.env.EMAIL_FROM || 'noreply@recursiadx.com',
    host: process.env.EMAIL_HOST || 'smtp.gmail.com',
    port: parseInt(process.env.EMAIL_PORT) || 587,
    secure: false,
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS
    }
  },

  // AWS configuration
  aws: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION || 'us-east-1',
    bucket: process.env.AWS_BUCKET || 'recursiadx-files'
  },

  // Redis configuration
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT) || 6379,
    password: process.env.REDIS_PASSWORD,
    db: 0,
    keyPrefix: 'recursiadx:'
  },

  // API configuration
  api: {
    version: process.env.API_VERSION || 'v1',
    prefix: process.env.API_PREFIX || '/api'
  },

  // Logging configuration
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: process.env.LOG_FORMAT || 'combined'
  },

  // AI/ML service configuration
  ai: {
    serviceUrl: process.env.AI_SERVICE_URL || 'http://localhost:8080',
    apiKey: process.env.AI_SERVICE_API_KEY,
    timeout: parseInt(process.env.AI_MODEL_TIMEOUT) || 30000
  },

  // Feature flags
  features: {
    enableRegistration: process.env.ENABLE_REGISTRATION !== 'false',
    enableEmailVerification: process.env.ENABLE_EMAIL_VERIFICATION === 'true',
    enableTwoFactorAuth: process.env.ENABLE_TWO_FACTOR_AUTH === 'true',
    enableAuditLogs: process.env.ENABLE_AUDIT_LOGS !== 'false',
    enableCompression: process.env.ENABLE_COMPRESSION !== 'false',
    enableHelmet: process.env.ENABLE_HELMET !== 'false'
  },

  // Monitoring configuration
  monitoring: {
    enablePrometheus: process.env.ENABLE_PROMETHEUS === 'true',
    prometheusPort: parseInt(process.env.PROMETHEUS_PORT) || 9090
  },

  // Webhook configuration
  webhook: {
    secret: process.env.WEBHOOK_SECRET
  }
};

// Validation function to check required environment variables
export const validateConfig = () => {
  const required = [];
  
  if (!config.jwt.secret || config.jwt.secret === 'fallback-secret-change-in-production') {
    required.push('JWT_SECRET');
  }
  
  if (!config.database.uri.includes('localhost') && process.env.NODE_ENV === 'production') {
    if (!process.env.MONGODB_URI) {
      required.push('MONGODB_URI');
    }
  }
  
  if (config.upload.storageType === 'aws') {
    if (!config.aws.accessKeyId) required.push('AWS_ACCESS_KEY_ID');
    if (!config.aws.secretAccessKey) required.push('AWS_SECRET_ACCESS_KEY');
  }
  
  if (config.features.enableEmailVerification) {
    if (!config.email.auth.user) required.push('EMAIL_USER');
    if (!config.email.auth.pass) required.push('EMAIL_PASS');
  }
  
  if (required.length > 0) {
    console.error('Missing required environment variables:', required.join(', '));
    console.error('Please check your .env file or environment configuration.');
    
    if (process.env.NODE_ENV === 'production') {
      process.exit(1);
    } else {
      console.warn('Running in development mode with default values.');
    }
  }
  
  return true;
};

// Get configuration for specific environment
export const getEnvConfig = () => {
  const env = config.server.nodeEnv;
  
  const envConfigs = {
    development: {
      ...config,
      logging: { ...config.logging, level: 'debug' },
      security: { ...config.security, bcryptSaltRounds: 10 } // Faster in dev
    },
    
    production: {
      ...config,
      logging: { ...config.logging, level: 'warn' },
      rateLimit: { ...config.rateLimit, maxRequests: 50 } // Stricter in prod
    },
    
    test: {
      ...config,
      database: { ...config.database, uri: config.database.testUri },
      jwt: { ...config.jwt, expire: '1h' },
      logging: { ...config.logging, level: 'error' }
    }
  };
  
  return envConfigs[env] || envConfigs.development;
};

export default config;