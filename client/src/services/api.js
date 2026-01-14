// API configuration and utility functions
const API_BASE_URL = 'http://localhost:5001/api';

// Cookie expiry time (1 hour in seconds)
const COOKIE_EXPIRY_SECONDS = 60 * 60; // 1 hour

// Cookie utility functions
const cookieUtils = {
  // Set a cookie with expiry
  setCookie: (name, value, maxAgeSeconds = COOKIE_EXPIRY_SECONDS) => {
    const expires = new Date(Date.now() + maxAgeSeconds * 1000).toUTCString();
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Strict`;
  },

  // Get a cookie value by name
  getCookie: (name) => {
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
      const [cookieName, cookieValue] = cookie.trim().split('=');
      if (cookieName === name) {
        return decodeURIComponent(cookieValue);
      }
    }
    return null;
  },

  // Delete a cookie
  deleteCookie: (name) => {
    document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Strict`;
  }
};

// API response handler
const handleApiResponse = async (response) => {
  const data = await response.json();

  if (!response.ok) {
    throw {
      status: response.status,
      message: data.message || 'An error occurred',
      errors: data.errors || null
    };
  }

  return data;
};

// Generic API request function
const apiRequest = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  // Check cookies first, then localStorage/sessionStorage
  const token = cookieUtils.getCookie('authToken') ||
    localStorage.getItem('authToken') ||
    sessionStorage.getItem('authToken');
  if (token) {
    defaultOptions.headers['Authorization'] = `Bearer ${token}`;
  }

  const config = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...options.headers,
    },
  };

  try {
    const response = await fetch(url, config);
    return await handleApiResponse(response);
  } catch (error) {
    // Network errors or other fetch-related errors
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw {
        status: 0,
        message: 'Network error. Please check your connection and try again.',
        errors: null
      };
    }
    throw error;
  }
};

// Authentication API functions
export const authAPI = {
  // Register new user
  register: async (userData) => {
    const payload = {
      name: `${userData.firstName} ${userData.lastName}`.trim(),
      email: userData.email,
      password: userData.password,
      confirmPassword: userData.confirmPassword,
      role: userData.role === 'technician' ? 'Technician' :
        userData.role === 'pathologist' ? 'Pathologist' :
          userData.role === 'administrator' ? 'Administrator' :
            userData.role === 'resident' ? 'Resident' : 'Technician',
      department: userData.department || userData.institution,
      licenseNumber: userData.license,
      phone: userData.phone
    };

    return await apiRequest('/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  // Login user
  login: async (email, password, rememberMe = false) => {
    const response = await apiRequest('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });

    if (response.success && response.data.token) {
      // Always set cookie for 1-hour session persistence (survives tab close)
      cookieUtils.setCookie('authToken', response.data.token, COOKIE_EXPIRY_SECONDS);
      cookieUtils.setCookie('userData', JSON.stringify(response.data.user), COOKIE_EXPIRY_SECONDS);

      // Also store in localStorage/sessionStorage based on rememberMe preference
      if (rememberMe) {
        localStorage.setItem('authToken', response.data.token);
        localStorage.setItem('userData', JSON.stringify(response.data.user));
      } else {
        sessionStorage.setItem('authToken', response.data.token);
        sessionStorage.setItem('userData', JSON.stringify(response.data.user));
      }
    }

    return response;
  },

  // Logout user
  logout: async () => {
    try {
      await apiRequest('/auth/logout', {
        method: 'POST',
      });
    } catch (error) {
      // Even if logout fails on backend, clear local data
      console.warn('Logout request failed, but clearing local data:', error);
    } finally {
      // Clear cookies
      cookieUtils.deleteCookie('authToken');
      cookieUtils.deleteCookie('userData');
      // Clear local/session storage
      localStorage.removeItem('authToken');
      localStorage.removeItem('userData');
      sessionStorage.removeItem('authToken');
      sessionStorage.removeItem('userData');
    }
  },

  // Get current user
  getCurrentUser: async () => {
    return await apiRequest('/auth/me', {
      method: 'GET',
    });
  },

  // Update user profile
  updateProfile: async (profileData) => {
    return await apiRequest('/auth/profile', {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  },

  // Change password
  changePassword: async (currentPassword, newPassword) => {
    return await apiRequest('/auth/change-password', {
      method: 'PUT',
      body: JSON.stringify({
        currentPassword,
        password: newPassword,
        confirmPassword: newPassword
      }),
    });
  },

  // Forgot password
  forgotPassword: async (email) => {
    return await apiRequest('/auth/forgot-password', {
      method: 'POST',
      body: JSON.stringify({ email }),
    });
  },

  // Reset password
  resetPassword: async (token, password) => {
    return await apiRequest(`/auth/reset-password/${token}`, {
      method: 'POST',
      body: JSON.stringify({
        password,
        confirmPassword: password
      }),
    });
  },

  // Refresh token
  refreshToken: async () => {
    return await apiRequest('/auth/refresh-token', {
      method: 'POST',
    });
  },

  // Upload avatar
  uploadAvatar: async (file) => {
    const formData = new FormData();
    formData.append('avatar', file);

    const token = cookieUtils.getCookie('authToken') ||
      localStorage.getItem('authToken') ||
      sessionStorage.getItem('authToken');

    const response = await fetch(`${API_BASE_URL}/auth/avatar`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      },
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw {
        status: response.status,
        message: data.message || 'Failed to upload avatar',
        errors: data.errors || null
      };
    }

    return data;
  },
};

// Samples API functions
export const samplesAPI = {
  // Get all samples
  getSamples: async (params = {}) => {
    const queryParams = new URLSearchParams();

    Object.keys(params).forEach(key => {
      if (params[key] !== undefined && params[key] !== '') {
        queryParams.append(key, params[key]);
      }
    });

    const queryString = queryParams.toString();
    const endpoint = `/samples${queryString ? `?${queryString}` : ''}`;

    return await apiRequest(endpoint, {
      method: 'GET',
    });
  },

  // Get sample by ID
  getSample: async (id) => {
    return await apiRequest(`/samples/${id}`, {
      method: 'GET',
    });
  },

  // Create new sample
  createSample: async (sampleData) => {
    return await apiRequest('/samples', {
      method: 'POST',
      body: JSON.stringify(sampleData),
    });
  },

  // Update sample
  updateSample: async (id, sampleData) => {
    return await apiRequest(`/samples/${id}`, {
      method: 'PUT',
      body: JSON.stringify(sampleData),
    });
  },

  // Update sample status
  updateSampleStatus: async (id, status, notes = '') => {
    return await apiRequest(`/samples/${id}/status`, {
      method: 'PUT',
      body: JSON.stringify({ status, notes }),
    });
  },

  // Assign sample
  assignSample: async (id, assignedTo) => {
    return await apiRequest(`/samples/${id}/assign`, {
      method: 'PUT',
      body: JSON.stringify({ assignedTo }),
    });
  },

  // Add image to sample
  addSampleImage: async (id, imageData) => {
    return await apiRequest(`/samples/${id}/images`, {
      method: 'POST',
      body: JSON.stringify(imageData),
    });
  },

  // Delete sample (soft delete)
  deleteSample: async (id) => {
    return await apiRequest(`/samples/${id}`, {
      method: 'DELETE',
    });
  },

  // Get sample statistics
  getSampleStats: async (startDate, endDate) => {
    const params = new URLSearchParams();
    if (startDate) params.append('startDate', startDate);
    if (endDate) params.append('endDate', endDate);

    return await apiRequest(`/samples/stats/overview?${params.toString()}`, {
      method: 'GET',
    });
  },
};

// Utility functions
export const apiUtils = {
  // Check if user is authenticated (checks cookies first)
  isAuthenticated: () => {
    const token = cookieUtils.getCookie('authToken') ||
      localStorage.getItem('authToken') ||
      sessionStorage.getItem('authToken');
    return !!token;
  },

  // Get stored user data (checks cookies first)
  getStoredUserData: () => {
    const userData = cookieUtils.getCookie('userData') ||
      localStorage.getItem('userData') ||
      sessionStorage.getItem('userData');
    return userData ? JSON.parse(userData) : null;
  },

  // Get stored auth token (checks cookies first)
  getAuthToken: () => {
    return cookieUtils.getCookie('authToken') ||
      localStorage.getItem('authToken') ||
      sessionStorage.getItem('authToken');
  },

  // Clear all auth data (including cookies)
  clearAuthData: () => {
    cookieUtils.deleteCookie('authToken');
    cookieUtils.deleteCookie('userData');
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
    sessionStorage.removeItem('authToken');
    sessionStorage.removeItem('userData');
  },

  // Check if token is expired (basic check)
  isTokenExpired: () => {
    const token = apiUtils.getAuthToken();
    if (!token) return true;

    try {
      // Decode JWT token to check expiration
      const payload = JSON.parse(atob(token.split('.')[1]));
      const currentTime = Date.now() / 1000;
      return payload.exp < currentTime;
    } catch (error) {
      // If we can't decode the token, consider it expired
      return true;
    }
  },

  // Auto-refresh token if needed
  refreshTokenIfNeeded: async () => {
    if (apiUtils.isAuthenticated() && apiUtils.isTokenExpired()) {
      try {
        await authAPI.refreshToken();
        return true;
      } catch (error) {
        // If refresh fails, clear auth data
        apiUtils.clearAuthData();
        return false;
      }
    }
    return true;
  }
};

// HTTP status codes for reference
export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  TOO_MANY_REQUESTS: 429,
  INTERNAL_SERVER_ERROR: 500,
  SERVICE_UNAVAILABLE: 503,
};

export default {
  authAPI,
  samplesAPI,
  apiUtils,
  HTTP_STATUS
};