import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { authAPI, apiUtils } from '../services/api.js';
import { toast } from 'sonner';

// Authentication action types
const AUTH_ACTIONS = {
  LOGIN_START: 'LOGIN_START',
  LOGIN_SUCCESS: 'LOGIN_SUCCESS',
  LOGIN_FAILURE: 'LOGIN_FAILURE',
  LOGOUT: 'LOGOUT',
  REGISTER_START: 'REGISTER_START',
  REGISTER_SUCCESS: 'REGISTER_SUCCESS',
  REGISTER_FAILURE: 'REGISTER_FAILURE',
  UPDATE_USER: 'UPDATE_USER',
  SET_LOADING: 'SET_LOADING',
  CLEAR_ERROR: 'CLEAR_ERROR',
  TOKEN_REFRESH: 'TOKEN_REFRESH'
};

// Initial authentication state
const initialState = {
  user: null,
  isAuthenticated: false,
  isLoading: true,
  error: null,
  isLoginLoading: false,
  isRegisterLoading: false
};

// Authentication reducer
const authReducer = (state, action) => {
  switch (action.type) {
    case AUTH_ACTIONS.LOGIN_START:
      return {
        ...state,
        isLoginLoading: true,
        error: null
      };

    case AUTH_ACTIONS.LOGIN_SUCCESS:
      return {
        ...state,
        user: action.payload.user,
        isAuthenticated: true,
        isLoginLoading: false,
        isLoading: false,
        error: null
      };

    case AUTH_ACTIONS.LOGIN_FAILURE:
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        isLoginLoading: false,
        isLoading: false,
        error: action.payload
      };

    case AUTH_ACTIONS.REGISTER_START:
      return {
        ...state,
        isRegisterLoading: true,
        error: null
      };

    case AUTH_ACTIONS.REGISTER_SUCCESS:
      return {
        ...state,
        user: action.payload.user,
        isAuthenticated: true,
        isRegisterLoading: false,
        error: null
      };

    case AUTH_ACTIONS.REGISTER_FAILURE:
      return {
        ...state,
        isRegisterLoading: false,
        error: action.payload
      };

    case AUTH_ACTIONS.LOGOUT:
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        isLoading: false,
        error: null
      };

    case AUTH_ACTIONS.UPDATE_USER:
      return {
        ...state,
        user: { ...state.user, ...action.payload }
      };

    case AUTH_ACTIONS.SET_LOADING:
      return {
        ...state,
        isLoading: action.payload
      };

    case AUTH_ACTIONS.CLEAR_ERROR:
      return {
        ...state,
        error: null
      };

    case AUTH_ACTIONS.TOKEN_REFRESH:
      return {
        ...state,
        user: action.payload.user
      };

    default:
      return state;
  }
};

// Create authentication context
const AuthContext = createContext();

// Authentication provider component
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Initialize authentication state on app start
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        // Check if user is already authenticated
        if (apiUtils.isAuthenticated() && !apiUtils.isTokenExpired()) {
          // Get current user data
          const response = await authAPI.getCurrentUser();
          dispatch({
            type: AUTH_ACTIONS.LOGIN_SUCCESS,
            payload: { user: response.data.user }
          });
        } else {
          // Clear any invalid auth data
          apiUtils.clearAuthData();
          dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
        }
      } catch (error) {
        // If getting current user fails, clear auth data
        apiUtils.clearAuthData();
        dispatch({ type: AUTH_ACTIONS.SET_LOADING, payload: false });
      }
    };

    initializeAuth();
  }, []);

  // Login function
  const login = async (email, password, rememberMe = false) => {
    dispatch({ type: AUTH_ACTIONS.LOGIN_START });

    try {
      const response = await authAPI.login(email, password, rememberMe);
      
      if (response.success) {
        dispatch({
          type: AUTH_ACTIONS.LOGIN_SUCCESS,
          payload: { user: response.data.user }
        });
        
        toast.success('Login successful! Welcome back.');
        return { success: true, user: response.data.user };
      } else {
        throw new Error(response.message || 'Login failed');
      }
    } catch (error) {
      const errorMessage = error.message || 'Login failed. Please try again.';
      dispatch({
        type: AUTH_ACTIONS.LOGIN_FAILURE,
        payload: errorMessage
      });
      
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  // Register function
  const register = async (userData) => {
    dispatch({ type: AUTH_ACTIONS.REGISTER_START });

    try {
      const response = await authAPI.register(userData);
      
      if (response.success) {
        dispatch({
          type: AUTH_ACTIONS.REGISTER_SUCCESS,
          payload: { user: response.data.user }
        });
        
        toast.success('Account created successfully! Welcome to RecursiaDx.');
        return { success: true, user: response.data.user };
      } else {
        throw new Error(response.message || 'Registration failed');
      }
    } catch (error) {
      let errorMessage = 'Registration failed. Please try again.';
      
      // Handle specific error types
      if (error.status === 400 && error.errors) {
        // Validation errors
        const validationErrors = error.errors.map(err => err.message).join(', ');
        errorMessage = validationErrors;
      } else if (error.message.includes('already exists')) {
        errorMessage = 'An account with this email already exists. Please use a different email or try logging in.';
      } else {
        errorMessage = error.message || errorMessage;
      }

      dispatch({
        type: AUTH_ACTIONS.REGISTER_FAILURE,
        payload: errorMessage
      });
      
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  // Logout function
  const logout = async () => {
    try {
      await authAPI.logout();
      dispatch({ type: AUTH_ACTIONS.LOGOUT });
      toast.success('Logged out successfully.');
    } catch (error) {
      // Even if logout API fails, clear local state
      dispatch({ type: AUTH_ACTIONS.LOGOUT });
      console.warn('Logout API call failed, but cleared local state:', error);
    }
  };

  // Update user profile
  const updateUser = async (profileData) => {
    try {
      const response = await authAPI.updateProfile(profileData);
      
      if (response.success) {
        dispatch({
          type: AUTH_ACTIONS.UPDATE_USER,
          payload: response.data.user
        });
        
        toast.success('Profile updated successfully.');
        return { success: true };
      }
    } catch (error) {
      const errorMessage = error.message || 'Failed to update profile.';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  // Change password
  const changePassword = async (currentPassword, newPassword) => {
    try {
      const response = await authAPI.changePassword(currentPassword, newPassword);
      
      if (response.success) {
        toast.success('Password changed successfully.');
        return { success: true };
      }
    } catch (error) {
      const errorMessage = error.message || 'Failed to change password.';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  // Forgot password
  const forgotPassword = async (email) => {
    try {
      const response = await authAPI.forgotPassword(email);
      
      if (response.success) {
        toast.success('Password reset instructions sent to your email.');
        return { success: true };
      }
    } catch (error) {
      const errorMessage = error.message || 'Failed to send reset email.';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  // Reset password
  const resetPassword = async (token, newPassword) => {
    try {
      const response = await authAPI.resetPassword(token, newPassword);
      
      if (response.success) {
        dispatch({
          type: AUTH_ACTIONS.LOGIN_SUCCESS,
          payload: { user: response.data.user }
        });
        
        toast.success('Password reset successfully. You are now logged in.');
        return { success: true };
      }
    } catch (error) {
      const errorMessage = error.message || 'Failed to reset password.';
      toast.error(errorMessage);
      return { success: false, error: errorMessage };
    }
  };

  // Refresh token
  const refreshToken = async () => {
    try {
      const response = await authAPI.refreshToken();
      
      if (response.success) {
        dispatch({
          type: AUTH_ACTIONS.TOKEN_REFRESH,
          payload: { user: response.data.user }
        });
        return true;
      }
    } catch (error) {
      // If refresh fails, logout user
      dispatch({ type: AUTH_ACTIONS.LOGOUT });
      return false;
    }
  };

  // Clear error
  const clearError = () => {
    dispatch({ type: AUTH_ACTIONS.CLEAR_ERROR });
  };

  // Check if user has specific role
  const hasRole = (role) => {
    if (!state.user) return false;
    
    // Convert frontend role names to backend role names
    const roleMap = {
      'technician': 'Lab Technician',
      'pathologist': 'Pathologist',
      'admin': 'Admin',
      'resident': 'Resident',
      'researcher': 'Researcher'
    };
    
    const backendRole = roleMap[role] || role;
    return state.user.role === backendRole;
  };

  // Check if user has any of the specified roles
  const hasAnyRole = (roles) => {
    return roles.some(role => hasRole(role));
  };

  // Get user display name
  const getUserDisplayName = () => {
    if (!state.user) return '';
    return state.user.name || 'User';
  };

  // Get user initials for avatar
  const getUserInitials = () => {
    if (!state.user || !state.user.name) return 'U';
    
    const names = state.user.name.split(' ');
    if (names.length >= 2) {
      return `${names[0][0]}${names[1][0]}`.toUpperCase();
    }
    return state.user.name[0].toUpperCase();
  };

  // Context value
  const value = {
    // State
    ...state,
    
    // Actions
    login,
    register,
    logout,
    updateUser,
    changePassword,
    forgotPassword,
    resetPassword,
    refreshToken,
    clearError,
    
    // Utility functions
    hasRole,
    hasAnyRole,
    getUserDisplayName,
    getUserInitials,
    
    // Direct access to utilities
    isAuthenticated: state.isAuthenticated,
    user: state.user,
    loading: state.isLoading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use authentication context
export const useAuth = () => {
  const context = useContext(AuthContext);
  
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
};

// Higher-order component for route protection
export const withAuth = (Component, allowedRoles = []) => {
  return (props) => {
    const { isAuthenticated, hasAnyRole, isLoading } = useAuth();
    
    if (isLoading) {
      return (
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
        </div>
      );
    }
    
    if (!isAuthenticated) {
      // Redirect to login or show login page
      return <div>Please log in to access this page.</div>;
    }
    
    if (allowedRoles.length > 0 && !hasAnyRole(allowedRoles)) {
      return <div>You don't have permission to access this page.</div>;
    }
    
    return <Component {...props} />;
  };
};

export default AuthContext;