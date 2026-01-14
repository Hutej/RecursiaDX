import React, { useState } from 'react'
import { 
  User, 
  Bell, 
  Shield, 
  Palette, 
  Globe, 
  Database,
  Key,
  Monitor,
  Moon,
  Sun,
  Volume2,
  Mail,
  Smartphone,
  Eye,
  EyeOff,
  Save,
  RefreshCw,
  Trash2,
  Download,
  Upload,
  LogOut,
  AlertTriangle
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { useTheme } from '../contexts/ThemeContext'
import { useAuth } from '../contexts/AuthContext'
import { authAPI } from '../services/api'
import { toast } from 'sonner'

export function Settings() {
  const { isDarkMode, toggleDarkMode } = useTheme()
  const { user, updateUser, changePassword, logout } = useAuth()
  
  // Profile settings - initialize from user data
  const [profileData, setProfileData] = useState({
    name: user?.name || '',
    phone: user?.phone || '',
    department: user?.department || '',
    licenseNumber: user?.licenseNumber || ''
  })
  
  // Update profile data when user changes
  React.useEffect(() => {
    if (user) {
      setProfileData({
        name: user.name || '',
        phone: user.phone || '',
        department: user.department || '',
        licenseNumber: user.licenseNumber || ''
      })
      // Update notification settings from user preferences
      if (user.preferences?.notifications) {
        setNotifications(prev => ({
          ...prev,
          emailNotifications: user.preferences.notifications.email ?? true,
          pushNotifications: user.preferences.notifications.push ?? true
        }))
      }
      // Update display settings from user preferences
      if (user.preferences) {
        setDisplaySettings(prev => ({
          ...prev,
          language: user.preferences.language || 'en'
        }))
      }
    }
  }, [user])
  
  // Password change
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  })
  const [showPasswords, setShowPasswords] = useState(false)
  
  // Notification settings
  const [notifications, setNotifications] = useState({
    emailNotifications: user?.preferences?.notifications?.email ?? true,
    pushNotifications: user?.preferences?.notifications?.push ?? true,
    analysisComplete: true,
    reportReady: true,
    systemAlerts: true,
    weeklyDigest: false,
    soundEnabled: true
  })
  
  // Display settings
  const [displaySettings, setDisplaySettings] = useState({
    language: user?.preferences?.language || 'en',
    timezone: 'Asia/Kolkata',
    dateFormat: 'DD/MM/YYYY',
    imageQuality: 'high',
    autoSaveInterval: 5
  })
  
  // Privacy settings
  const [privacySettings, setPrivacySettings] = useState({
    shareAnalytics: true,
    showOnlineStatus: true,
    allowDataExport: true
  })
  
  const [isSaving, setIsSaving] = useState(false)
  
  // Save profile to database
  const handleSaveProfile = async () => {
    setIsSaving(true)
    try {
      const result = await updateUser(profileData)
      if (result?.success !== false) {
        toast.success('Profile updated successfully')
      }
    } catch (error) {
      toast.error(error.message || 'Failed to update profile')
    } finally {
      setIsSaving(false)
    }
  }
  
  // Change password
  const handleChangePassword = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      toast.error('New passwords do not match')
      return
    }
    if (passwordData.newPassword.length < 8) {
      toast.error('Password must be at least 8 characters')
      return
    }
    
    setIsSaving(true)
    try {
      const result = await changePassword(passwordData.currentPassword, passwordData.newPassword)
      if (result?.success !== false) {
        setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' })
        toast.success('Password changed successfully')
      }
    } catch (error) {
      toast.error(error.message || 'Failed to change password')
    } finally {
      setIsSaving(false)
    }
  }
  
  // Save notification preferences to database
  const handleSaveNotifications = async () => {
    setIsSaving(true)
    try {
      const result = await updateUser({
        preferences: {
          notifications: {
            email: notifications.emailNotifications,
            push: notifications.pushNotifications,
            sms: false
          }
        }
      })
      if (result?.success !== false) {
        toast.success('Notification settings saved')
      }
    } catch (error) {
      toast.error(error.message || 'Failed to save notification settings')
    } finally {
      setIsSaving(false)
    }
  }
  
  // Save display preferences to database
  const handleSaveDisplaySettings = async () => {
    setIsSaving(true)
    try {
      // Save language preference to database
      const result = await updateUser({
        preferences: {
          language: displaySettings.language,
          theme: isDarkMode ? 'dark' : 'light'
        }
      })
      if (result?.success !== false) {
        toast.success('Display settings saved')
      }
    } catch (error) {
      toast.error(error.message || 'Failed to save display settings')
    } finally {
      setIsSaving(false)
    }
  }
  
  const handleExportData = () => {
    toast.info('Preparing data export...')
    // Simulate export
    setTimeout(() => {
      toast.success('Data export ready for download')
    }, 2000)
  }
  
  const handleClearCache = () => {
    // Only clear cache, not auth data
    const authToken = localStorage.getItem('authToken')
    const userData = localStorage.getItem('userData')
    localStorage.clear()
    sessionStorage.clear()
    // Restore auth data
    if (authToken) localStorage.setItem('authToken', authToken)
    if (userData) localStorage.setItem('userData', userData)
    toast.success('Cache cleared successfully')
  }
  
  // Avatar upload
  const fileInputRef = React.useRef(null)
  const [isUploadingAvatar, setIsUploadingAvatar] = useState(false)
  
  const handleAvatarClick = () => {
    fileInputRef.current?.click()
  }
  
  const handleAvatarChange = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
    if (!validTypes.includes(file.type)) {
      toast.error('Please select a valid image file (JPEG, PNG, GIF, WebP)')
      return
    }
    
    // Validate file size (2MB)
    if (file.size > 2 * 1024 * 1024) {
      toast.error('Image must be less than 2MB')
      return
    }
    
    setIsUploadingAvatar(true)
    try {
      const result = await authAPI.uploadAvatar(file)
      if (result.success) {
        // Refresh user data to get new avatar
        toast.success('Avatar uploaded successfully')
        // Force reload to update avatar everywhere
        window.location.reload()
      }
    } catch (error) {
      toast.error(error.message || 'Failed to upload avatar')
    } finally {
      setIsUploadingAvatar(false)
      // Reset file input
      event.target.value = ''
    }
  }
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-muted-foreground">
            Manage your account preferences and application settings
          </p>
        </div>
        <Badge variant="outline" className="text-sm">
          <User className="h-4 w-4 mr-2" />
          {user?.role || 'User'}
        </Badge>
      </div>
      
      {/* Settings Tabs */}
      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid">
          <TabsTrigger value="profile" className="flex items-center gap-2">
            <User className="h-4 w-4" />
            <span className="hidden sm:inline">Profile</span>
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            <span className="hidden sm:inline">Notifications</span>
          </TabsTrigger>
          <TabsTrigger value="display" className="flex items-center gap-2">
            <Palette className="h-4 w-4" />
            <span className="hidden sm:inline">Display</span>
          </TabsTrigger>
          <TabsTrigger value="privacy" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            <span className="hidden sm:inline">Privacy</span>
          </TabsTrigger>
          <TabsTrigger value="data" className="flex items-center gap-2">
            <Database className="h-4 w-4" />
            <span className="hidden sm:inline">Data</span>
          </TabsTrigger>
        </TabsList>
        
        {/* Profile Tab */}
        <TabsContent value="profile" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Profile Information */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="h-5 w-5" />
                  Profile Information
                </CardTitle>
                <CardDescription>
                  Update your personal information and contact details
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4 mb-6">
                  <Avatar className="h-20 w-20">
                    <AvatarImage src={user?.avatar ? `http://localhost:5001${user.avatar}` : undefined} />
                    <AvatarFallback className="text-xl">
                      {user?.name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'U'}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleAvatarChange}
                      accept="image/jpeg,image/jpg,image/png,image/gif,image/webp"
                      className="hidden"
                    />
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={handleAvatarClick}
                      disabled={isUploadingAvatar}
                    >
                      {isUploadingAvatar ? (
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      ) : (
                        <Upload className="h-4 w-4 mr-2" />
                      )}
                      {isUploadingAvatar ? 'Uploading...' : 'Change Photo'}
                    </Button>
                    <p className="text-xs text-muted-foreground mt-1">
                      JPG, PNG, GIF, WebP - max 2MB
                    </p>
                  </div>
                </div>
                
                <div className="grid gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      value={profileData.name}
                      onChange={(e) => setProfileData({ ...profileData, name: e.target.value })}
                      placeholder="Dr. John Doe"
                    />
                  </div>
                  
                  <div className="grid gap-2">
                    <Label htmlFor="email">Email Address</Label>
                    <Input
                      id="email"
                      type="email"
                      value={user?.email || ''}
                      disabled
                      className="bg-muted cursor-not-allowed"
                    />
                    <p className="text-xs text-muted-foreground">Email cannot be changed</p>
                  </div>
                  
                  <div className="grid gap-2">
                    <Label htmlFor="phone">Phone Number</Label>
                    <Input
                      id="phone"
                      value={profileData.phone}
                      onChange={(e) => setProfileData({ ...profileData, phone: e.target.value })}
                      placeholder="+91 98765 43210"
                    />
                  </div>
                  
                  <div className="grid gap-2">
                    <Label htmlFor="department">Department</Label>
                    <Input
                      id="department"
                      value={profileData.department}
                      onChange={(e) => setProfileData({ ...profileData, department: e.target.value })}
                      placeholder="Pathology Department"
                    />
                  </div>
                  
                  <div className="grid gap-2">
                    <Label htmlFor="license">License Number</Label>
                    <Input
                      id="license"
                      value={profileData.licenseNumber}
                      onChange={(e) => setProfileData({ ...profileData, licenseNumber: e.target.value })}
                      placeholder="MED-2024-XXXXX"
                    />
                  </div>
                </div>
                
                <Button onClick={handleSaveProfile} disabled={isSaving} className="w-full">
                  {isSaving ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  Save Changes
                </Button>
              </CardContent>
            </Card>
            
            {/* Change Password */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Key className="h-5 w-5" />
                  Change Password
                </CardTitle>
                <CardDescription>
                  Update your password to keep your account secure
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-4">
                  <div className="grid gap-2">
                    <Label htmlFor="currentPassword">Current Password</Label>
                    <div className="relative">
                      <Input
                        id="currentPassword"
                        type={showPasswords ? 'text' : 'password'}
                        value={passwordData.currentPassword}
                        onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
                        placeholder="Enter current password"
                      />
                      <Button
                        variant="ghost"
                        size="sm"
                        type="button"
                        className="absolute right-0 top-0 h-full px-3"
                        onClick={() => setShowPasswords(!showPasswords)}
                      >
                        {showPasswords ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                    </div>
                  </div>
                  
                  <div className="grid gap-2">
                    <Label htmlFor="newPassword">New Password</Label>
                    <Input
                      id="newPassword"
                      type={showPasswords ? 'text' : 'password'}
                      value={passwordData.newPassword}
                      onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
                      placeholder="Enter new password"
                    />
                  </div>
                  
                  <div className="grid gap-2">
                    <Label htmlFor="confirmPassword">Confirm New Password</Label>
                    <Input
                      id="confirmPassword"
                      type={showPasswords ? 'text' : 'password'}
                      value={passwordData.confirmPassword}
                      onChange={(e) => setPasswordData({ ...passwordData, confirmPassword: e.target.value })}
                      placeholder="Confirm new password"
                    />
                  </div>
                </div>
                
                <div className="text-sm text-muted-foreground">
                  <p className="font-medium mb-2">Password Requirements:</p>
                  <ul className="list-disc list-inside space-y-1">
                    <li>At least 8 characters long</li>
                    <li>Include uppercase and lowercase letters</li>
                    <li>Include at least one number</li>
                    <li>Include at least one special character</li>
                  </ul>
                </div>
                
                <Button onClick={handleChangePassword} disabled={isSaving} variant="outline" className="w-full">
                  {isSaving ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Key className="h-4 w-4 mr-2" />
                  )}
                  Update Password
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Notifications Tab */}
        <TabsContent value="notifications" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Notification Preferences
              </CardTitle>
              <CardDescription>
                Configure how and when you receive notifications
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Email Notifications */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Mail className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">Email Notifications</p>
                    <p className="text-sm text-muted-foreground">Receive updates via email</p>
                  </div>
                </div>
                <Switch
                  checked={notifications.emailNotifications}
                  onCheckedChange={(checked) => setNotifications({ ...notifications, emailNotifications: checked })}
                />
              </div>
              
              <Separator />
              
              {/* Push Notifications */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Smartphone className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">Push Notifications</p>
                    <p className="text-sm text-muted-foreground">Receive browser push notifications</p>
                  </div>
                </div>
                <Switch
                  checked={notifications.pushNotifications}
                  onCheckedChange={(checked) => setNotifications({ ...notifications, pushNotifications: checked })}
                />
              </div>
              
              <Separator />
              
              {/* Analysis Complete */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Monitor className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">Analysis Complete</p>
                    <p className="text-sm text-muted-foreground">Notify when AI analysis is complete</p>
                  </div>
                </div>
                <Switch
                  checked={notifications.analysisComplete}
                  onCheckedChange={(checked) => setNotifications({ ...notifications, analysisComplete: checked })}
                />
              </div>
              
              <Separator />
              
              {/* Report Ready */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Download className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">Report Ready</p>
                    <p className="text-sm text-muted-foreground">Notify when reports are ready for download</p>
                  </div>
                </div>
                <Switch
                  checked={notifications.reportReady}
                  onCheckedChange={(checked) => setNotifications({ ...notifications, reportReady: checked })}
                />
              </div>
              
              <Separator />
              
              {/* Sound */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Volume2 className="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p className="font-medium">Sound Notifications</p>
                    <p className="text-sm text-muted-foreground">Play sound for important alerts</p>
                  </div>
                </div>
                <Switch
                  checked={notifications.soundEnabled}
                  onCheckedChange={(checked) => setNotifications({ ...notifications, soundEnabled: checked })}
                />
              </div>
              
              <Button onClick={handleSaveNotifications} disabled={isSaving} className="w-full mt-4">
                {isSaving ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
                Save Notification Settings
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Display Tab */}
        <TabsContent value="display" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Theme Settings */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Palette className="h-5 w-5" />
                  Appearance
                </CardTitle>
                <CardDescription>
                  Customize how RecursiaDx looks on your device
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <Label>Theme</Label>
                  <div className="grid grid-cols-2 gap-4">
                    <button
                      onClick={() => { if (isDarkMode) toggleDarkMode(); }}
                      className={`flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all ${
                        !isDarkMode ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <Sun className="h-6 w-6" />
                      <span className="text-sm font-medium">Light</span>
                    </button>
                    <button
                      onClick={() => { if (!isDarkMode) toggleDarkMode(); }}
                      className={`flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all ${
                        isDarkMode ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'
                      }`}
                    >
                      <Moon className="h-6 w-6" />
                      <span className="text-sm font-medium">Dark</span>
                    </button>
                  </div>
                </div>
                
                <Separator />
                
                <div className="space-y-4">
                  <Label>Image Quality</Label>
                  <Select 
                    value={displaySettings.imageQuality} 
                    onValueChange={(value) => setDisplaySettings({ ...displaySettings, imageQuality: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select quality" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low (Faster loading)</SelectItem>
                      <SelectItem value="medium">Medium (Balanced)</SelectItem>
                      <SelectItem value="high">High (Best quality)</SelectItem>
                      <SelectItem value="original">Original (Full resolution)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>
            
            {/* Regional Settings */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Globe className="h-5 w-5" />
                  Regional Settings
                </CardTitle>
                <CardDescription>
                  Set your language, timezone, and date format
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-3">
                  <Label>Language</Label>
                  <Select 
                    value={displaySettings.language} 
                    onValueChange={(value) => setDisplaySettings({ ...displaySettings, language: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select language" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="hi">हिंदी (Hindi)</SelectItem>
                      <SelectItem value="es">Español (Spanish)</SelectItem>
                      <SelectItem value="fr">Français (French)</SelectItem>
                      <SelectItem value="de">Deutsch (German)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-3">
                  <Label>Timezone</Label>
                  <Select 
                    value={displaySettings.timezone} 
                    onValueChange={(value) => setDisplaySettings({ ...displaySettings, timezone: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select timezone" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Asia/Kolkata">India Standard Time (IST)</SelectItem>
                      <SelectItem value="America/New_York">Eastern Time (ET)</SelectItem>
                      <SelectItem value="America/Los_Angeles">Pacific Time (PT)</SelectItem>
                      <SelectItem value="Europe/London">Greenwich Mean Time (GMT)</SelectItem>
                      <SelectItem value="Europe/Paris">Central European Time (CET)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-3">
                  <Label>Date Format</Label>
                  <Select 
                    value={displaySettings.dateFormat} 
                    onValueChange={(value) => setDisplaySettings({ ...displaySettings, dateFormat: value })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select format" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="DD/MM/YYYY">DD/MM/YYYY (31/12/2024)</SelectItem>
                      <SelectItem value="MM/DD/YYYY">MM/DD/YYYY (12/31/2024)</SelectItem>
                      <SelectItem value="YYYY-MM-DD">YYYY-MM-DD (2024-12-31)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <Button onClick={handleSaveDisplaySettings} disabled={isSaving} className="w-full">
                  {isSaving ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
                  Save Display Settings
                </Button>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Privacy Tab */}
        <TabsContent value="privacy" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Privacy & Security
              </CardTitle>
              <CardDescription>
                Manage your privacy settings and data sharing preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Share Analytics Data</p>
                  <p className="text-sm text-muted-foreground">Help improve RecursiaDx by sharing anonymous usage data</p>
                </div>
                <Switch
                  checked={privacySettings.shareAnalytics}
                  onCheckedChange={(checked) => {
                    setPrivacySettings({ ...privacySettings, shareAnalytics: checked })
                    toast.success(checked ? 'Analytics sharing enabled' : 'Analytics sharing disabled')
                  }}
                />
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Show Online Status</p>
                  <p className="text-sm text-muted-foreground">Let other team members see when you're active</p>
                </div>
                <Switch
                  checked={privacySettings.showOnlineStatus}
                  onCheckedChange={(checked) => {
                    setPrivacySettings({ ...privacySettings, showOnlineStatus: checked })
                    toast.success(checked ? 'Online status visible' : 'Online status hidden')
                  }}
                />
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Allow Data Export</p>
                  <p className="text-sm text-muted-foreground">Enable downloading your data and reports</p>
                </div>
                <Switch
                  checked={privacySettings.allowDataExport}
                  onCheckedChange={(checked) => {
                    setPrivacySettings({ ...privacySettings, allowDataExport: checked })
                    toast.success(checked ? 'Data export enabled' : 'Data export disabled')
                  }}
                />
              </div>
              
              <Separator />
              
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-amber-600">
                  <AlertTriangle className="h-5 w-5" />
                  <p className="font-medium">Danger Zone</p>
                </div>
                
                <div className="p-4 rounded-lg border border-red-200 bg-red-50/50 dark:bg-red-950/20 dark:border-red-800">
                  <p className="font-medium text-red-600 dark:text-red-400 mb-2">Delete Account</p>
                  <p className="text-sm text-muted-foreground mb-4">
                    Permanently delete your account and all associated data. This action cannot be undone.
                  </p>
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button variant="destructive" size="sm">
                        <Trash2 className="h-4 w-4 mr-2" />
                        Delete Account
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This action cannot be undone. This will permanently delete your account
                          and remove all your data from our servers including analysis history,
                          reports, and settings.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          onClick={() => {
                            toast.error('Account deletion requested. Please contact support to complete this process.')
                          }}
                        >
                          Yes, delete my account
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        {/* Data Tab */}
        <TabsContent value="data" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Data Management */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Data Management
                </CardTitle>
                <CardDescription>
                  Export, backup, and manage your data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Button onClick={handleExportData} variant="outline" className="w-full justify-start">
                  <Download className="h-4 w-4 mr-2" />
                  Export All Data
                </Button>
                
                <Button variant="outline" className="w-full justify-start">
                  <Download className="h-4 w-4 mr-2" />
                  Export Reports Only
                </Button>
                
                <Button variant="outline" className="w-full justify-start">
                  <Download className="h-4 w-4 mr-2" />
                  Export Analysis History
                </Button>
                
                <Separator />
                
                <p className="text-sm text-muted-foreground">
                  Data exports are generated as encrypted ZIP files and will be available for download for 24 hours.
                </p>
              </CardContent>
            </Card>
            
            {/* Storage & Cache */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <RefreshCw className="h-5 w-5" />
                  Storage & Cache
                </CardTitle>
                <CardDescription>
                  Manage local storage and cached data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Browser Cache</span>
                    <span className="text-muted-foreground">~12.5 MB</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary w-1/3 rounded-full" />
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Offline Data</span>
                    <span className="text-muted-foreground">~4.2 MB</span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500 w-1/5 rounded-full" />
                  </div>
                </div>
                
                <Separator />
                
                <Button onClick={handleClearCache} variant="outline" className="w-full">
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear Cache
                </Button>
                
                <p className="text-xs text-muted-foreground">
                  Clearing cache will log you out and remove all offline data. You will need to log in again.
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
