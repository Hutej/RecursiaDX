import React, { createContext, useContext, useState, useEffect, useMemo, useCallback } from 'react'
import { 
  preloadThemeStyles, 
  cacheThemePreferences, 
  getCachedThemePreferences,
  debounce,
  getSystemThemePreference,
  applyThemeWithTransition,
  isValidTheme,
  measureThemePerformance
} from '../utils/themeUtils'

const ThemeContext = createContext()

// Theme definitions - moved outside component to prevent recreating on each render
const THEMES = Object.freeze({
  default: {
    name: 'Default Blue',
    id: 'default',
    description: 'Cool blue theme',
    color: '#91C5FD'
  },
  coral: {
    name: 'Coral',
    id: 'coral',
    description: 'Warm coral and orange theme',
    color: '#FF6678'
  },
  rose: {
    name: 'Rose',
    id: 'rose', 
    description: 'Pink and rose theme',
    color: '#FC64AB'
  },
  purple: {
    name: 'Purple',
    id: 'purple',
    description: 'Purple and violet theme',
    color: '#A985FF'
  },
  ocean: {
    name: 'Ocean',
    id: 'ocean',
    description: 'Blue ocean theme',
    color: '#0099FF'
  },
  emerald: {
    name: 'Emerald',
    id: 'emerald',
    description: 'Green emerald theme',
    color: '#00D696'
  },
  sunshine: {
    name: 'Sunshine',
    id: 'sunshine',
    description: 'Yellow sunshine theme',
    color: '#FFBF00'
  }
})

// Theme storage keys
const STORAGE_KEYS = {
  theme: 'recursia-theme',
  darkMode: 'recursia-dark-mode'
}

export function ThemeProvider({ children }) {
  const [currentTheme, setCurrentTheme] = useState('default')
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [isTransitioning, setIsTransitioning] = useState(false)

  // Initialize theme state from storage and system preferences
  useEffect(() => {
    let mounted = true
    
    const initializeTheme = () => {
      measureThemePerformance('init', () => {
        try {
          // Preload theme styles for smooth transitions
          preloadThemeStyles()
          
          // Try to load from cache first
          const cached = getCachedThemePreferences()
          if (cached && mounted) {
            if (isValidTheme(cached.theme, Object.values(THEMES))) {
              setCurrentTheme(cached.theme)
            }
            setIsDarkMode(cached.darkMode)
            return
          }
          
          // Fallback to localStorage
          const savedTheme = localStorage.getItem(STORAGE_KEYS.theme)
          const savedDarkMode = localStorage.getItem(STORAGE_KEYS.darkMode)
          
          if (mounted) {
            // Set theme
            if (savedTheme && THEMES[savedTheme]) {
              setCurrentTheme(savedTheme)
            }
            
            // Set dark mode
            if (savedDarkMode !== null) {
              setIsDarkMode(savedDarkMode === 'true')
            } else {
              // Check system preference
              const prefersDark = getSystemThemePreference()
              setIsDarkMode(prefersDark)
            }
          }
        } catch (error) {
          console.warn('Failed to load theme preferences:', error)
        }
      })
    }

    // Debounced system theme change handler
    const handleSystemThemeChange = debounce((e) => {
      if (mounted && localStorage.getItem(STORAGE_KEYS.darkMode) === null) {
        setIsDarkMode(e.matches)
      }
    }, 100)

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    
    initializeTheme()
    mediaQuery.addEventListener('change', handleSystemThemeChange)
    
    return () => {
      mounted = false
      mediaQuery.removeEventListener('change', handleSystemThemeChange)
    }
  }, [])

  // Apply theme classes to document with smooth transitions and caching
  useEffect(() => {
    const root = document.documentElement
    
    const applyTheme = async () => {
      setIsTransitioning(true)
      
      try {
        await measureThemePerformance('apply', () => 
          applyThemeWithTransition(root, currentTheme, isDarkMode)
        )
        
        // Cache the theme preferences
        cacheThemePreferences(currentTheme, isDarkMode)
        
      } catch (error) {
        console.warn('Failed to apply theme:', error)
      } finally {
        setIsTransitioning(false)
      }
    }

    applyTheme()
  }, [currentTheme, isDarkMode])

  // Optimized theme change function with validation, caching, and performance monitoring
  const changeTheme = useCallback((themeId) => {
    return measureThemePerformance('change', () => {
      if (!themeId || !THEMES[themeId] || themeId === currentTheme) {
        return false
      }
      
      try {
        setCurrentTheme(themeId)
        localStorage.setItem(STORAGE_KEYS.theme, themeId)
        return true
      } catch (error) {
        console.warn('Failed to save theme preference:', error)
        return false
      }
    })
  }, [currentTheme])

  // Optimized dark mode toggle
  const toggleDarkMode = useCallback(() => {
    try {
      const newDarkMode = !isDarkMode
      setIsDarkMode(newDarkMode)
      localStorage.setItem(STORAGE_KEYS.darkMode, newDarkMode.toString())
      return newDarkMode
    } catch (error) {
      console.warn('Failed to save dark mode preference:', error)
      return isDarkMode
    }
  }, [isDarkMode])

  // Memoized theme list to prevent unnecessary re-renders
  const themeList = useMemo(() => Object.values(THEMES), [])

  // Memoized current theme object
  const currentThemeObj = useMemo(() => THEMES[currentTheme], [currentTheme])

  // Memoized context value
  const contextValue = useMemo(() => ({
    currentTheme,
    isDarkMode,
    isTransitioning,
    themes: themeList,
    changeTheme,
    toggleDarkMode,
    getCurrentTheme: () => currentThemeObj,
    getThemeColor: (themeId) => THEMES[themeId]?.color || THEMES.default.color
  }), [currentTheme, isDarkMode, isTransitioning, themeList, changeTheme, toggleDarkMode, currentThemeObj])

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}