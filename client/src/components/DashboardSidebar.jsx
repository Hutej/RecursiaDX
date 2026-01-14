import React, { useState, useCallback } from "react"
import { ChevronDown, LogOut, Activity, Settings } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import * as CollapsiblePrimitive from "@radix-ui/react-collapsible"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Button } from '@/components/ui/button'
import Icon from './AppIcon'

const Collapsible = CollapsiblePrimitive.Root
const CollapsibleTrigger = CollapsiblePrimitive.CollapsibleTrigger
const CollapsibleContent = CollapsiblePrimitive.CollapsibleContent

export function DashboardSidebar({ activeTab, onTabChange, user, onLogout }) {
  const [isHovered, setIsHovered] = useState(false)
  const [openSubmenus, setOpenSubmenus] = useState(new Set())

  const menuItems = [
    {
      id: 'upload',
      title: 'Sample Upload',
      icon: 'Upload',
      description: 'Upload blood smears and tissue slides'
    },
    {
      id: 'viewer',
      title: 'Image Preview',
      icon: 'Microscope',
      description: 'View and inspect uploaded images'
    },
    {
      id: 'analysis',
      title: 'AI Analysis',
      icon: 'Brain',
      description: 'AI-powered pathology analysis'
    },
    {
      id: 'review',
      title: 'Results Review',
      icon: 'CheckCircle',
      description: 'Technician verification and review'
    },
    {
      id: 'report',
      title: 'Generate Report',
      icon: 'FileText',
      description: 'Create professional pathology reports'
    }
  ]

  const isActive = useCallback((tabId) => activeTab === tabId, [activeTab])
  const hasActiveSubmenu = useCallback((submenu) => 
    submenu?.some(sub => isActive(sub.id)), [isActive])

  const toggleSubmenu = useCallback((title) => {
    setOpenSubmenus(prev => {
      const newSet = new Set(prev)
      if (newSet.has(title)) {
        newSet.delete(title)
      } else {
        newSet.add(title)
      }
      return newSet
    })
  }, [])

  const sidebarVariants = {
    collapsed: { width: 64 },
    expanded: { width: 280 }
  }

  const itemVariants = {
    collapsed: { opacity: 0, x: -10 },
    expanded: { opacity: 1, x: 0 }
  }

  return (
    <motion.div
      className="fixed left-0 top-0 pt-16 h-screen z-10 bg-background/95 backdrop-blur-md border-r border-border shadow-lg flex flex-col"
      variants={sidebarVariants}
      animate={isHovered ? "expanded" : "collapsed"}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {isHovered ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="flex-1 flex flex-col h-full"
        >
          {/* Header */}
          <div className="p-4 border-b border-border">
            <motion.div
              variants={itemVariants}
              initial="collapsed"
              animate="expanded"
              transition={{ delay: 0.15 }}
              className="flex items-center space-x-3"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center shadow-md">
                <Activity className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <h3 className="font-semibold text-foreground">RecursiaDx</h3>
                <p className="text-xs text-muted-foreground">Digital Pathology Platform</p>
              </div>
            </motion.div>
          </div>

          {/* Navigation */}
          <ScrollArea className="flex-1 px-3 py-4">
            <nav className="space-y-1">
              <motion.div
                variants={itemVariants}
                initial="collapsed"
                animate="expanded"
                transition={{ delay: 0.1 }}
                className="px-2 py-2"
              >
                <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  Pathology Workflow
                </h2>
              </motion.div>
              
              {menuItems.map((item, index) => {
                const isItemActive = isActive(item.id)
                const hasActiveChild = hasActiveSubmenu(item.submenu)
                
                if (item.submenu) {
                  return (
                    <motion.div
                      key={item.title}
                      variants={itemVariants}
                      initial="collapsed"
                      animate="expanded"
                      transition={{ delay: 0.1 + index * 0.02 }}
                    >
                      <Collapsible
                        open={openSubmenus.has(item.title)}
                        onOpenChange={() => toggleSubmenu(item.title)}
                      >
                        <CollapsibleTrigger asChild>
                          <button
                            className={`flex items-center justify-between w-full h-11 px-3 rounded-lg transition-all duration-200 group ${
                              hasActiveChild || isItemActive
                                ? "bg-primary/10 text-primary shadow-sm"
                                : "hover:bg-accent text-muted-foreground hover:text-foreground"
                            }`}
                            onClick={() => !item.submenu && onTabChange(item.id)}
                          >
                            <div className="flex items-center space-x-3">
                              <Icon name={item.icon} className="h-5 w-5 shrink-0" />
                              <span className="text-sm font-medium">{item.title}</span>
                            </div>
                            <motion.div
                              animate={{ rotate: openSubmenus.has(item.title) ? 180 : 0 }}
                              transition={{ duration: 0.2 }}
                            >
                              <ChevronDown className="h-4 w-4" />
                            </motion.div>
                          </button>
                        </CollapsibleTrigger>
                        <AnimatePresence>
                          {openSubmenus.has(item.title) && (
                            <CollapsibleContent asChild>
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: 0.2 }}
                                className="space-y-1 mt-1 ml-8"
                              >
                                {item.submenu.map((subItem) => (
                                  <button
                                    key={subItem.title}
                                    onClick={() => onTabChange(subItem.id)}
                                    className={`flex items-center space-x-3 h-9 px-3 rounded-lg transition-all duration-200 w-full ${
                                      isActive(subItem.id)
                                        ? "bg-primary/10 text-primary border-r-2 border-primary"
                                        : "hover:bg-accent text-muted-foreground hover:text-foreground"
                                    }`}
                                  >
                                    <Icon name={subItem.icon} className="h-4 w-4 shrink-0" />
                                    <span className="text-sm font-medium">{subItem.title}</span>
                                  </button>
                                ))}
                              </motion.div>
                            </CollapsibleContent>
                          )}
                        </AnimatePresence>
                      </Collapsible>
                    </motion.div>
                  )
                }

                return (
                  <motion.div
                    key={item.title}
                    variants={itemVariants}
                    initial="collapsed"
                    animate="expanded"
                    transition={{ delay: 0.1 + index * 0.02 }}
                  >
                    <button
                      onClick={() => onTabChange(item.id)}
                      className={`flex items-center space-x-3 h-11 px-3 rounded-lg transition-all duration-200 group w-full ${
                        isItemActive
                          ? "bg-primary/10 text-primary shadow-sm border-r-2 border-primary"
                          : "hover:bg-accent text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      <Icon name={item.icon} className="h-5 w-5 shrink-0" />
                      <span className="text-sm font-medium">{item.title}</span>
                    </button>
                  </motion.div>
                )
              })}

              {/* System Section */}
              <motion.div
                variants={itemVariants}
                initial="collapsed"
                animate="expanded"
                transition={{ delay: 0.2 }}
                className="px-2 py-4"
              >
                <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
                  System
                </h2>
                <button 
                  onClick={() => onTabChange('settings')}
                  className={`flex items-center space-x-3 h-11 px-3 rounded-lg transition-all duration-200 group w-full ${
                    activeTab === 'settings'
                      ? "bg-primary/10 text-primary shadow-sm border-r-2 border-primary"
                      : "hover:bg-accent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  <Settings className="h-5 w-5 shrink-0" />
                  <span className="text-sm font-medium">Settings</span>
                </button>
              </motion.div>
            </nav>
          </ScrollArea>

          {/* User Profile Section at the bottom */}
          <div className="p-3 border-t border-border mt-auto">
            <motion.div
              variants={itemVariants}
              initial="collapsed"
              animate="expanded"
              transition={{ delay: 0.25 }}
              className="flex items-center space-x-3 p-3 rounded-lg bg-accent/50"
            >
              <Avatar className="h-8 w-8">
                <AvatarImage src={user?.avatar ? `http://localhost:5001${user.avatar}` : undefined} />
                <AvatarFallback className="text-xs">
                  {user?.name ? user.name.split(' ').map(n => n[0]).join('').toUpperCase() : 'U'}
                </AvatarFallback>
              </Avatar>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {user?.name || 'User'}
                </p>
                <p className="text-xs text-muted-foreground truncate">
                  {user?.email || 'user@recursiaDx.com'}
                </p>
              </div>
              <Button variant="ghost" size="sm" onClick={onLogout} className="h-8 w-8 p-0">
                <LogOut className="h-4 w-4" />
              </Button>
            </motion.div>
          </div>
        </motion.div>
      ) : (
        <div className="pt-6 px-3 flex flex-col h-full">
          <nav className="flex flex-col space-y-2 flex-1">
            {menuItems.map((item, index) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className="flex justify-center"
                title={item.title}
              >
                {item.submenu ? (
                  <button
                    className={`flex items-center justify-center w-10 h-10 rounded-lg transition-all duration-200 ${
                      hasActiveSubmenu(item.submenu) || isActive(item.id)
                        ? "bg-primary/10 text-primary shadow-sm"
                        : "hover:bg-accent text-muted-foreground hover:text-foreground"
                    }`}
                    onClick={() => onTabChange(item.id)}
                  >
                    <Icon name={item.icon} className="h-5 w-5 shrink-0" />
                  </button>
                ) : (
                  <button
                    onClick={() => onTabChange(item.id)}
                    className={`flex items-center justify-center w-10 h-10 rounded-lg transition-all duration-200 ${
                      isActive(item.id)
                        ? "bg-primary/10 text-primary shadow-sm"
                        : "hover:bg-accent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Icon name={item.icon} className="h-5 w-5 shrink-0" />
                  </button>
                )}
              </motion.div>
            ))}
          </nav>

          {/* Collapsed Settings Button */}
          <div className="pb-2 pt-2 flex justify-center">
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.15 }}
              title="Settings"
            >
              <button
                onClick={() => onTabChange('settings')}
                className={`flex items-center justify-center w-10 h-10 rounded-lg transition-all duration-200 ${
                  isActive('settings')
                    ? "bg-primary/10 text-primary shadow-sm"
                    : "hover:bg-accent text-muted-foreground hover:text-foreground"
                }`}
              >
                <Settings className="h-5 w-5 shrink-0" />
              </button>
            </motion.div>
          </div>

          {/* Collapsed User Avatar at bottom */}
          <div className="pb-3 pt-2 border-t border-border mt-auto">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="flex justify-center"
            >
              <Avatar className="h-10 w-10">
                <AvatarImage src={user?.avatar ? `http://localhost:5001${user.avatar}` : undefined} />
                <AvatarFallback className="text-xs">
                  {user?.name ? user.name.split(' ').map(n => n[0]).join('').toUpperCase() : 'U'}
                </AvatarFallback>
              </Avatar>
            </motion.div>
          </div>
        </div>
      )}
    </motion.div>
  )
}