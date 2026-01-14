import React, { useState, useCallback, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Badge } from '@/components/ui/badge'
import { useTheme } from '../contexts/ThemeContext'
import { Palette, Check, Moon, Sun, Loader2 } from 'lucide-react'

export function ThemeSwitcher({ className = "" }) {
  const { 
    currentTheme, 
    isDarkMode, 
    isTransitioning,
    themes, 
    changeTheme, 
    toggleDarkMode, 
    getCurrentTheme,
    getThemeColor 
  } = useTheme()
  
  const [isOpen, setIsOpen] = useState(false)

  // Memoized current theme color
  const currentThemeColor = useMemo(() => 
    getThemeColor(currentTheme), 
    [currentTheme, getThemeColor]
  )

  // Optimized theme change handler
  const handleThemeChange = useCallback((themeId) => {
    const success = changeTheme(themeId)
    if (success) {
      setIsOpen(false) // Close dropdown on successful change
    }
  }, [changeTheme])

  // Optimized dark mode toggle handler
  const handleDarkModeToggle = useCallback(() => {
    toggleDarkMode()
    setIsOpen(false) // Close dropdown after toggle
  }, [toggleDarkMode])

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="outline" 
          size="sm" 
          className={`gap-2 ${className} ${isTransitioning ? 'opacity-75' : ''}`}
          disabled={isTransitioning}
        >
          {isTransitioning ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Palette className="h-4 w-4" />
          )}
          <div 
            className="w-3 h-3 rounded-full border border-gray-300 transition-colors duration-300"
            style={{ backgroundColor: currentThemeColor }}
          />
          <span className="hidden sm:inline">
            {isTransitioning ? 'Applying...' : 'Theme'}
          </span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel className="flex items-center gap-2">
          <Palette className="h-4 w-4" />
          Color Themes
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        
        {themes.map((theme) => (
          <DropdownMenuItem
            key={theme.id}
            onClick={() => handleThemeChange(theme.id)}
            className={`flex items-center gap-3 cursor-pointer transition-colors ${
              currentTheme === theme.id ? 'bg-primary/10' : ''
            }`}
            disabled={isTransitioning}
          >
            <div 
              className="w-4 h-4 rounded-full border border-gray-300 transition-colors duration-200"
              style={{ backgroundColor: theme.color }}
            />
            <div className="flex-1">
              <div className="font-medium text-sm">{theme.name}</div>
              <div className="text-xs text-muted-foreground">{theme.description}</div>
            </div>
            {currentTheme === theme.id && (
              <Check className="h-4 w-4 text-primary" />
            )}
          </DropdownMenuItem>
        ))}
        
        <DropdownMenuSeparator />
        
        <DropdownMenuItem
          onClick={handleDarkModeToggle}
          className="flex items-center gap-3 cursor-pointer"
          disabled={isTransitioning}
        >
          {isDarkMode ? (
            <>
              <Sun className="h-4 w-4" />
              <span>Switch to Light Mode</span>
            </>
          ) : (
            <>
              <Moon className="h-4 w-4" />
              <span>Switch to Dark Mode</span>
            </>
          )}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function ThemePreview({ themeId, isSelected, onClick, disabled = false }) {
  const { getThemeColor } = useTheme()

  const handleClick = useCallback(() => {
    if (!disabled && onClick) {
      onClick(themeId)
    }
  }, [themeId, onClick, disabled])

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={`
        relative p-3 rounded-lg border-2 transition-all duration-200 hover:shadow-md focus:outline-none focus:ring-2 focus:ring-primary/50
        ${isSelected ? 'border-primary shadow-lg' : 'border-gray-200 hover:border-primary/50'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
    >
      <div 
        className="w-8 h-8 rounded-full mx-auto mb-2 transition-transform duration-200 hover:scale-110"
        style={{ backgroundColor: getThemeColor(themeId) }}
      />
      {isSelected && (
        <div className="absolute -top-1 -right-1 bg-primary text-white rounded-full p-0.5">
          <Check className="h-3 w-3" />
        </div>
      )}
    </button>
  )
}