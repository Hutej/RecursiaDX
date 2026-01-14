import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Palette } from 'lucide-react'

const themes = [
  { name: 'default', label: 'Blue', color: '#91C5FD' },
  { name: 'coral', label: 'Coral', color: '#FB7185' },
  { name: 'purple', label: 'Purple', color: '#A78BFA' },
  { name: 'emerald', label: 'Emerald', color: '#34D399' },
  { name: 'orange', label: 'Orange', color: '#FB923C' }
]

export function ColorThemeChanger() {
  const [currentTheme, setCurrentTheme] = useState('default')

  useEffect(() => {
    // Load saved theme
    const saved = localStorage.getItem('color-theme') || 'default'
    setCurrentTheme(saved)
    applyTheme(saved)
  }, [])

  const applyTheme = (theme) => {
    // Remove all theme classes
    themes.forEach(t => {
      document.documentElement.removeAttribute('data-theme')
    })
    
    // Apply new theme
    if (theme !== 'default') {
      document.documentElement.setAttribute('data-theme', theme)
    }
  }

  const handleThemeChange = (theme) => {
    setCurrentTheme(theme)
    localStorage.setItem('color-theme', theme)
    applyTheme(theme)
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm">
          <Palette className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {themes.map((theme) => (
          <DropdownMenuItem 
            key={theme.name}
            onClick={() => handleThemeChange(theme.name)}
            className="flex items-center gap-2"
          >
            <div 
              className="w-4 h-4 rounded-full border"
              style={{ backgroundColor: theme.color }}
            />
            {theme.label}
            {currentTheme === theme.name && (
              <div className="ml-auto w-2 h-2 bg-foreground rounded-full" />
            )}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}