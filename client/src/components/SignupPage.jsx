import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Checkbox } from '@/components/ui/checkbox'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { toast } from 'sonner'
import { useAuth } from '../contexts/AuthContext'
import { 
  Activity,
  Eye, 
  EyeOff, 
  Mail, 
  Lock, 
  User,
  Building2,
  Phone,
  MapPin,
  ArrowLeft,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Shield,
  Award,
  Users
} from 'lucide-react'

export function SignupPage({ onSignup, onBackToHome, onGoToLogin }) {
  const { register, isRegisterLoading, error: authError } = useAuth()
  const [currentStep, setCurrentStep] = useState(1)
  const [formData, setFormData] = useState({
    // Personal Info
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    
    // Professional Info
    role: '',
    license: '',
    institution: '',
    department: '',
    phone: '',
    
    // Address
    address: '',
    city: '',
    state: '',
    zipCode: '',
    country: '',
    
    // Agreements
    termsAccepted: false,
    privacyAccepted: false,
    marketingOptIn: false
  })
  
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [errors, setErrors] = useState({})

  const roles = [
    { value: 'technician', label: 'Medical Technologist', icon: Users },
    { value: 'pathologist', label: 'Pathologist', icon: User },
    { value: 'administrator', label: 'Laboratory Director/Administrator', icon: Building2 },
    { value: 'resident', label: 'Resident/Fellow', icon: Award }
  ]

  const countries = [
    { value: 'us', label: 'United States' },
    { value: 'ca', label: 'Canada' },
    { value: 'uk', label: 'United Kingdom' },
    { value: 'de', label: 'Germany' },
    { value: 'fr', label: 'France' },
    { value: 'au', label: 'Australia' },
    { value: 'other', label: 'Other' }
  ]

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
    
    // Clear specific field error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }))
    }
  }

  const validateStep = (step) => {
    const newErrors = {}
    
    if (step === 1) {
      if (!formData.firstName.trim()) newErrors.firstName = 'First name is required'
      if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required'
      if (!formData.email.trim()) newErrors.email = 'Email is required'
      else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
        newErrors.email = 'Please enter a valid email address'
      }
      if (!formData.password) newErrors.password = 'Password is required'
      else if (formData.password.length < 8) {
        newErrors.password = 'Password must be at least 8 characters'
      } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/.test(formData.password)) {
        newErrors.password = 'Password must contain at least one uppercase letter, lowercase letter, number, and special character'
      }
      if (formData.password !== formData.confirmPassword) {
        newErrors.confirmPassword = 'Passwords do not match'
      }
    }
    
    if (step === 2) {
      if (!formData.role) newErrors.role = 'Please select your role'
      if (!formData.institution.trim()) newErrors.institution = 'Institution name is required'
      if (!formData.phone.trim()) newErrors.phone = 'Phone number is required'
    }
    
    if (step === 3) {
      if (!formData.country) newErrors.country = 'Please select your country'
      if (!formData.city.trim()) newErrors.city = 'City is required'
    }
    
    if (step === 4) {
      if (!formData.termsAccepted) newErrors.termsAccepted = 'You must accept the terms of service'
      if (!formData.privacyAccepted) newErrors.privacyAccepted = 'You must accept the privacy policy'
    }
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleNext = () => {
    if (validateStep(currentStep)) {
      setCurrentStep(prev => Math.min(prev + 1, 4))
    }
  }

  const handleBack = () => {
    setCurrentStep(prev => Math.max(prev - 1, 1))
  }

  const handleSubmit = async () => {
    if (!validateStep(4)) return
    
    try {
      const result = await register(formData)
      
      if (result.success) {
        // Registration successful, call onSignup callback
        setTimeout(() => {
          onSignup({
            email: formData.email,
            name: `${formData.firstName} ${formData.lastName}`,
            role: formData.role,
            institution: formData.institution,
            user: result.user
          })
        }, 1500)
      }
    } catch (error) {
      console.error('Registration error:', error)
      // Error handling is managed by the AuthContext
    }
  }

  const renderStep = () => {
    switch (currentStep) {
      case 1:
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="firstName">First Name *</Label>
                <div className="relative">
                  <User className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="firstName"
                    placeholder="Enter your first name"
                    value={formData.firstName}
                    onChange={(e) => handleInputChange('firstName', e.target.value)}
                    className="pl-10"
                  />
                </div>
                {errors.firstName && <p className="text-sm text-red-600">{errors.firstName}</p>}
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="lastName">Last Name *</Label>
                <Input
                  id="lastName"
                  placeholder="Enter your last name"
                  value={formData.lastName}
                  onChange={(e) => handleInputChange('lastName', e.target.value)}
                />
                {errors.lastName && <p className="text-sm text-red-600">{errors.lastName}</p>}
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">Email Address *</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email address"
                  value={formData.email}
                  onChange={(e) => handleInputChange('email', e.target.value)}
                  className="pl-10"
                />
              </div>
              {errors.email && <p className="text-sm text-red-600">{errors.email}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password *</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Create a strong password"
                  value={formData.password}
                  onChange={(e) => handleInputChange('password', e.target.value)}
                  className="pl-10 pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {errors.password && <p className="text-sm text-red-600">{errors.password}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Confirm Password *</Label>
              <div className="relative">
                <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  id="confirmPassword"
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="Confirm your password"
                  value={formData.confirmPassword}
                  onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                  className="pl-10 pr-10"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                >
                  {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {errors.confirmPassword && <p className="text-sm text-red-600">{errors.confirmPassword}</p>}
            </div>
          </div>
        )

      case 2:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Professional Role *</Label>
              <Select onValueChange={(value) => handleInputChange('role', value)}>
                <SelectTrigger>
                  <SelectValue placeholder="Select your professional role" />
                </SelectTrigger>
                <SelectContent>
                  {roles.map((role) => {
                    const Icon = role.icon
                    return (
                      <SelectItem key={role.value} value={role.value}>
                        <div className="flex items-center gap-2">
                          <Icon className="h-4 w-4" />
                          {role.label}
                        </div>
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
              {errors.role && <p className="text-sm text-red-600">{errors.role}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="license">License Number (Optional)</Label>
              <Input
                id="license"
                placeholder="Enter your professional license number"
                value={formData.license}
                onChange={(e) => handleInputChange('license', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="institution">Institution/Hospital *</Label>
              <div className="relative">
                <Building2 className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  id="institution"
                  placeholder="Enter your institution or hospital name"
                  value={formData.institution}
                  onChange={(e) => handleInputChange('institution', e.target.value)}
                  className="pl-10"
                />
              </div>
              {errors.institution && <p className="text-sm text-red-600">{errors.institution}</p>}
            </div>

            <div className="space-y-2">
              <Label htmlFor="department">Department (Optional)</Label>
              <Input
                id="department"
                placeholder="e.g., Pathology, Clinical Laboratory"
                value={formData.department}
                onChange={(e) => handleInputChange('department', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="phone">Phone Number *</Label>
              <div className="relative">
                <Phone className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  id="phone"
                  type="tel"
                  placeholder="Enter your phone number"
                  value={formData.phone}
                  onChange={(e) => handleInputChange('phone', e.target.value)}
                  className="pl-10"
                />
              </div>
              {errors.phone && <p className="text-sm text-red-600">{errors.phone}</p>}
            </div>
          </div>
        )

      case 3:
        return (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="address">Street Address</Label>
              <div className="relative">
                <MapPin className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  id="address"
                  placeholder="Enter your street address"
                  value={formData.address}
                  onChange={(e) => handleInputChange('address', e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="city">City *</Label>
                <Input
                  id="city"
                  placeholder="Enter your city"
                  value={formData.city}
                  onChange={(e) => handleInputChange('city', e.target.value)}
                />
                {errors.city && <p className="text-sm text-red-600">{errors.city}</p>}
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="state">State/Province</Label>
                <Input
                  id="state"
                  placeholder="Enter your state or province"
                  value={formData.state}
                  onChange={(e) => handleInputChange('state', e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="zipCode">ZIP/Postal Code</Label>
                <Input
                  id="zipCode"
                  placeholder="Enter your ZIP or postal code"
                  value={formData.zipCode}
                  onChange={(e) => handleInputChange('zipCode', e.target.value)}
                />
              </div>
              
              <div className="space-y-2">
                <Label>Country *</Label>
                <Select onValueChange={(value) => handleInputChange('country', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your country" />
                  </SelectTrigger>
                  <SelectContent>
                    {countries.map((country) => (
                      <SelectItem key={country.value} value={country.value}>
                        {country.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {errors.country && <p className="text-sm text-red-600">{errors.country}</p>}
              </div>
            </div>
          </div>
        )

      case 4:
        return (
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-lg font-semibold mb-2">Almost there!</h3>
              <p className="text-gray-600">Please review and accept our terms to complete your registration.</p>
            </div>

            {/* Display authentication error */}
            {authError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{authError}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <Checkbox
                  id="termsAccepted"
                  checked={formData.termsAccepted}
                  onCheckedChange={(checked) => handleInputChange('termsAccepted', checked)}
                />
                <div className="text-sm">
                  <label htmlFor="termsAccepted" className="font-medium cursor-pointer">
                    I agree to the{' '}
                    <a href="#" className="text-primary hover:underline">Terms of Service</a>
                  </label>
                  {errors.termsAccepted && <p className="text-red-600 mt-1">{errors.termsAccepted}</p>}
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <Checkbox
                  id="privacyAccepted"
                  checked={formData.privacyAccepted}
                  onCheckedChange={(checked) => handleInputChange('privacyAccepted', checked)}
                />
                <div className="text-sm">
                  <label htmlFor="privacyAccepted" className="font-medium cursor-pointer">
                    I agree to the{' '}
                    <a href="#" className="text-primary hover:underline">Privacy Policy</a>
                  </label>
                  {errors.privacyAccepted && <p className="text-red-600 mt-1">{errors.privacyAccepted}</p>}
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <Checkbox
                  id="marketingOptIn"
                  checked={formData.marketingOptIn}
                  onCheckedChange={(checked) => handleInputChange('marketingOptIn', checked)}
                />
                <div className="text-sm">
                  <label htmlFor="marketingOptIn" className="cursor-pointer">
                    I would like to receive updates about new features and product announcements (optional)
                  </label>
                </div>
              </div>
            </div>

            <Alert>
              <Shield className="h-4 w-4" />
              <AlertDescription>
                <strong>Your data is secure:</strong> We use enterprise-grade encryption and never share your personal information with third parties.
              </AlertDescription>
            </Alert>
          </div>
        )

      default:
        return null
    }
  }

  const steps = [
    { number: 1, title: 'Account Info', description: 'Basic account details' },
    { number: 2, title: 'Professional Info', description: 'Role and institution' },
    { number: 3, title: 'Location', description: 'Address information' },
    { number: 4, title: 'Terms & Finish', description: 'Review and complete' }
  ]

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
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
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Create your account
          </h2>
          <p className="text-gray-600">
            Join the future of digital pathology
          </p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-4 mb-4">
            {steps.map((step, index) => (
              <div key={step.number} className="flex items-center">
                <div className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                  currentStep >= step.number 
                    ? 'bg-primary border-primary text-white' 
                    : 'border-gray-300 text-gray-400'
                }`}>
                  {currentStep > step.number ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    <span className="text-sm font-medium">{step.number}</span>
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div className={`w-12 h-1 mx-2 ${
                    currentStep > step.number ? 'bg-primary' : 'bg-gray-200'
                  }`} />
                )}
              </div>
            ))}
          </div>
          
          <div className="text-center">
            <h3 className="font-semibold">{steps[currentStep - 1].title}</h3>
            <p className="text-sm text-gray-600">{steps[currentStep - 1].description}</p>
          </div>
        </div>

        {/* Form Card */}
        <Card className="shadow-xl border-0 max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle>Step {currentStep} of {steps.length}</CardTitle>
          </CardHeader>
          <CardContent>
            {renderStep()}
            
            <div className="flex justify-between mt-8">
              <Button 
                variant="outline" 
                onClick={currentStep === 1 ? onBackToHome : handleBack}
                className="flex items-center gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                {currentStep === 1 ? 'Back to Home' : 'Previous'}
              </Button>
              
              {currentStep < 4 ? (
                <Button onClick={handleNext}>
                  Next
                  <ArrowLeft className="h-4 w-4 ml-2 rotate-180" />
                </Button>
              ) : (
                <Button onClick={handleSubmit} disabled={isRegisterLoading}>
                  {isRegisterLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {isRegisterLoading ? 'Creating Account...' : 'Create Account'}
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Login Link */}
        <div className="text-center mt-6">
          <p className="text-sm text-gray-600">
            Already have an account?{' '}
            <button
              onClick={onGoToLogin}
              className="text-primary hover:underline font-medium"
            >
              Sign in here
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}