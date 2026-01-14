
export function preloadThemeStyles() {
  return Promise.resolve()
}

/**
 * Cache theme preferences in memory for fast access
 */
const themeCache = new Map()

export function cacheThemePreferences(theme, isDarkMode) {
  const preferences = { theme, darkMode: isDarkMode, timestamp: Date.now() }
  themeCache.set('current', preferences)
  
  // Also save to localStorage as backup
  try {
    localStorage.setItem('theme-cache', JSON.stringify(preferences))
  } catch (error) {
    console.warn('Failed to cache theme preferences:', error)
  }
}

export function getCachedThemePreferences() {
  // Try memory cache first
  if (themeCache.has('current')) {
    return themeCache.get('current')
  }
  
  // Fallback to localStorage
  try {
    const cached = localStorage.getItem('theme-cache')
    if (cached) {
      const preferences = JSON.parse(cached)
      themeCache.set('current', preferences)
      return preferences
    }
  } catch (error) {
    console.warn('Failed to get cached theme preferences:', error)
  }
  
  return null
}

/**
 * Debounce function for performance optimization
 */
export function debounce(func, wait) {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

/**
 * Get system theme preference
 */
export function getSystemThemePreference() {
  try {
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  } catch (error) {
    console.warn('Failed to get system theme preference:', error)
    return false
  }
}

/**
 * Apply theme with smooth transition
 */
export function applyThemeWithTransition(root, themeId, isDarkMode) {
  return new Promise((resolve) => {
    // Remove existing theme classes
    const existingClasses = Array.from(root.classList).filter(cls => 
      cls.startsWith('theme-') || cls === 'dark'
    )
    existingClasses.forEach(cls => root.classList.remove(cls))
    
    // Add new theme classes
    if (themeId !== 'default') {
      root.classList.add(`theme-${themeId}`)
    }
    
    if (isDarkMode) {
      root.classList.add('dark')
    }
    
    // Allow transition to complete
    setTimeout(resolve, 100)
  })
}

/**
 * Validate theme ID
 */
export function isValidTheme(themeId, availableThemes) {
  return availableThemes.some(theme => theme.id === themeId)
}

/**
 * Measure theme performance for debugging
 */
export function measureThemePerformance(operation, callback) {
  const startTime = performance.now()
  
  const result = callback()
  
  const endTime = performance.now()
  const duration = endTime - startTime
  
  if (duration > 100) {
    console.warn(`Theme ${operation} took ${duration.toFixed(2)}ms - consider optimization`)
  }
  
  return result
}