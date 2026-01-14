import jwt from 'jsonwebtoken';
import User from '../models/User.js';

// Verify JWT token
const verifyToken = async (req, res, next) => {
  try {
    let token;
    
    // Check for token in header
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    } else if (req.cookies && req.cookies.token) {
      // Check for token in cookies
      token = req.cookies.token;
    }

    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'Access denied. No token provided.'
      });
    }

    // Verify token
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    
    // Check if user still exists
    const user = await User.findById(decoded.userId).select('-password');
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid token. User not found.'
      });
    }

    // Check if user is active
    if (!user.isActive) {
      return res.status(401).json({
        success: false,
        message: 'Account is deactivated.'
      });
    }

    // Check if password was changed after token was issued
    if (user.passwordChangedAt && 
        new Date(decoded.iat * 1000) < user.passwordChangedAt) {
      return res.status(401).json({
        success: false,
        message: 'Password recently changed. Please login again.'
      });
    }

    req.user = user;
    next();
  } catch (error) {
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({
        success: false,
        message: 'Invalid token.'
      });
    } else if (error.name === 'TokenExpiredError') {
      return res.status(401).json({
        success: false,
        message: 'Token expired. Please login again.'
      });
    }
    
    return res.status(500).json({
      success: false,
      message: 'Token verification failed.'
    });
  }
};

// Authorize based on roles
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Access denied. Please login.'
      });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        success: false,
        message: `Access denied. Required roles: ${roles.join(', ')}`
      });
    }

    next();
  };
};

// Check if user owns resource or has admin access
const authorizeOwnerOrAdmin = (resourceUserField = 'userId') => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        message: 'Access denied. Please login.'
      });
    }

    // Admin can access everything
    if (req.user.role === 'Admin' || req.user.role === 'SuperAdmin') {
      return next();
    }

    // Check if user owns the resource
    const resourceUserId = req.body[resourceUserField] || 
                          req.params[resourceUserField] || 
                          req.query[resourceUserField];

    if (resourceUserId && resourceUserId.toString() === req.user._id.toString()) {
      return next();
    }

    return res.status(403).json({
      success: false,
      message: 'Access denied. You can only access your own resources.'
    });
  };
};

// Optional authentication - sets user if token exists but doesn't require it
const optionalAuth = async (req, res, next) => {
  try {
    let token;
    
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    } else if (req.cookies && req.cookies.token) {
      token = req.cookies.token;
    }

    if (token) {
      try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        const user = await User.findById(decoded.userId).select('-password');
        
        if (user && user.isActive) {
          req.user = user;
        }
      } catch (error) {
        // Ignore token errors in optional auth
      }
    }

    next();
  } catch (error) {
    next();
  }
};

// Check account lockout
const checkAccountLockout = async (req, res, next) => {
  try {
    const { email } = req.body;
    
    if (!email) {
      return next();
    }

    const user = await User.findOne({ email });
    
    if (user && user.isLocked) {
      return res.status(423).json({
        success: false,
        message: `Account is locked. Try again after ${user.lockUntil || 'some time'}.`
      });
    }

    next();
  } catch (error) {
    next(error);
  }
};

// Rate limiting for authentication attempts
const authRateLimit = (maxAttempts = 5, windowMs = 15 * 60 * 1000) => {
  const attempts = new Map();
  
  return (req, res, next) => {
    const ip = req.ip || req.connection.remoteAddress;
    const now = Date.now();
    
    // Clean up old attempts
    for (const [key, data] of attempts.entries()) {
      if (now - data.firstAttempt > windowMs) {
        attempts.delete(key);
      }
    }
    
    const userAttempts = attempts.get(ip) || { count: 0, firstAttempt: now };
    
    if (userAttempts.count >= maxAttempts) {
      const timeLeft = windowMs - (now - userAttempts.firstAttempt);
      return res.status(429).json({
        success: false,
        message: `Too many authentication attempts. Try again in ${Math.ceil(timeLeft / 1000)} seconds.`
      });
    }
    
    // Increment attempt count on failed login
    req.incrementAuthAttempt = () => {
      userAttempts.count++;
      attempts.set(ip, userAttempts);
    };
    
    // Reset attempts on successful login
    req.resetAuthAttempts = () => {
      attempts.delete(ip);
    };
    
    next();
  };
};

// Validate JWT structure without verification (for logout)
const validateTokenStructure = (req, res, next) => {
  let token;
  
  if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
    token = req.headers.authorization.split(' ')[1];
  } else if (req.cookies && req.cookies.token) {
    token = req.cookies.token;
  }

  if (!token) {
    return res.status(400).json({
      success: false,
      message: 'No token provided.'
    });
  }

  req.token = token;
  next();
};

export {
  verifyToken,
  authorize,
  authorizeOwnerOrAdmin,
  optionalAuth,
  checkAccountLockout,
  authRateLimit,
  validateTokenStructure
};