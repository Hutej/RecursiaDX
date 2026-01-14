import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { toast } from 'sonner'
import { useAuth } from '../contexts/AuthContext'
import { 
  Activity,
  Eye, 
  EyeOff, 
  Mail, 
  Lock, 
  ArrowLeft,
  Shield,
  CheckCircle2,
  AlertCircle,
  Loader2
} from 'lucide-react'

export function LoginPage({ onLogin, onBackToHome, onGoToSignup }) {
  const { login, isLoginLoading, error: authError } = useAuth()
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false
  })
  const [showPassword, setShowPassword] = useState(false)


  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }



  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!formData.email || !formData.password) {
      toast.error('Please fill in all required fields')
      return
    }

    if (!formData.email.includes('@')) {
      toast.error('Please enter a valid email address')
      return
    }

    try {
      const result = await login(formData.email, formData.password, formData.rememberMe)
      
      if (result.success) {
        // Login successful, call onLogin callback
        setTimeout(() => {
          onLogin({
            email: formData.email,
            role: result.user.role,
            name: result.user.name,
            rememberMe: formData.rememberMe,
            user: result.user
          })
        }, 1500)
      }
    } catch (error) {
      console.error('Login error:', error)
      // Error handling is managed by the AuthContext
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        
        {/* Left Side - Branding & Info */}
        <div className="hidden lg:block">
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary">
                <Activity className="h-7 w-7 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">RecursiaDx</h1>
                <Badge variant="outline" className="text-xs">
                  Digital Pathology Platform
                </Badge>
              </div>
            </div>
            
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Welcome back to the future of pathology
            </h2>
            <p className="text-gray-600 text-lg">
              Access your digital pathology workspace and continue transforming healthcare with AI-powered analysis.
            </p>
          </div>



          {/* Trust Indicators */}
          <div className="mt-8 grid grid-cols-3 gap-4 text-center">
            <div className="flex flex-col items-center gap-2">
              <Shield className="h-8 w-8 text-green-600" />
              <span className="text-sm font-medium">HIPAA Compliant</span>
            </div>
            <div className="flex flex-col items-center gap-2">
              <CheckCircle2 className="h-8 w-8 text-blue-600" />
              <span className="text-sm font-medium">CAP Certified</span>
            </div>
            <div className="flex flex-col items-center gap-2">
              <Lock className="h-8 w-8 text-purple-600" />
              <span className="text-sm font-medium">Secure Login</span>
            </div>
          </div>
        </div>

        {/* Right Side - Login Form */}
        <div className="w-full max-w-md mx-auto">
          <Card className="shadow-xl border-0">
            <CardHeader className="text-center">
              <div className="lg:hidden flex items-center justify-center gap-3 mb-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                  <Activity className="h-6 w-6 text-primary-foreground" />
                </div>
                <div>
                  <h1 className="text-xl font-bold">RecursiaDx</h1>
                </div>
              </div>
              
              <CardTitle className="text-2xl">Sign in to your account</CardTitle>
              <CardDescription>
                Access your digital pathology workspace
              </CardDescription>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                {authError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{authError}</AlertDescription>
                  </Alert>
                )}

                <div className="space-y-2">
                  <Label htmlFor="email">Email address</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="Enter your email"
                      value={formData.email}
                      onChange={(e) => handleInputChange('email', e.target.value)}
                      className="pl-10"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your password"
                      value={formData.password}
                      onChange={(e) => handleInputChange('password', e.target.value)}
                      className="pl-10 pr-10"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="rememberMe"
                      checked={formData.rememberMe}
                      onCheckedChange={(checked) => handleInputChange('rememberMe', checked)}
                    />
                    <Label htmlFor="rememberMe" className="text-sm">
                      Remember me
                    </Label>
                  </div>
                  <a href="#" className="text-sm text-primary hover:underline">
                    Forgot password?
                  </a>
                </div>

                <Button type="submit" className="w-full" disabled={isLoginLoading}>
                  {isLoginLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {isLoginLoading ? 'Signing in...' : 'Sign in'}
                </Button>
              </form>

              <div className="mt-6 text-center">
                <p className="text-sm text-gray-600">
                  Don't have an account?{' '}
                  <button
                    onClick={onGoToSignup}
                    className="text-primary hover:underline font-medium"
                  >
                    Sign up here
                  </button>
                </p>
              </div>


            </CardContent>
          </Card>

          <div className="mt-6 text-center">
            <Button variant="ghost" onClick={onBackToHome} className="text-gray-600">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to home
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}