import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { ThemeSwitcher } from './ThemeSwitcher'
import { 
  Activity,
  Brain, 
  Microscope, 
  Shield, 
  Zap, 
  Users, 
  CheckCircle2, 
  ArrowRight,
  Star,
  Award,
  Clock,
  Globe,
  ChevronDown,
  Play,
  Quote
} from 'lucide-react'

export function HomePage({ onLogin, onSignup }) {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms for accurate pathology detection with 96%+ confidence rates.",
      color: "text-blue-600"
    },
    {
      icon: Microscope,
      title: "WSI Processing",
      description: "Handle gigapixel Whole Slide Images with multi-scale attention and tiling technology.",
      color: "text-green-600"
    },
    {
      icon: Zap,
      title: "Real-time Results",
      description: "Get analysis results in minutes, not hours. Streamlined workflow from upload to report.",
      color: "text-yellow-600"
    },
    {
      icon: Shield,
      title: "HIPAA Compliant",
      description: "Enterprise-grade security with full HIPAA compliance and encrypted data handling.",
      color: "text-purple-600"
    },
    {
      icon: Users,
      title: "Collaborative Review",
      description: "Seamless technician verification and pathologist review workflow integration.",
      color: "text-red-600"
    },
    {
      icon: Globe,
      title: "Cloud-Based Platform",
      description: "Access your pathology lab from anywhere with our secure cloud infrastructure.",
      color: "text-indigo-600"
    }
  ]

  const testimonials = [
    {
      name: "Dr. Sarah Chen",
      role: "Chief Pathologist, Metro Medical Center",
      content: "RecursiaDx has revolutionized our lab workflow. The AI accuracy is remarkable and saves us hours daily.",
      avatar: "SC"
    },
    {
      name: "Tech. Michael Rodriguez",
      role: "Senior Medical Technologist",
      content: "The user interface is intuitive and the verification process is seamless. Best pathology platform I've used.",
      avatar: "MR"
    },
    {
      name: "Dr. Jennifer Kim",
      role: "Laboratory Director",
      content: "Integration was smooth and the reporting features are exactly what we needed for our accreditation.",
      avatar: "JK"
    }
  ]

  const stats = [
    { number: "96.8%", label: "AI Accuracy Rate", sublabel: "Cancer Detection" },
    { number: "2.3M+", label: "Samples Analyzed", sublabel: "Since Launch" },
    { number: "150+", label: "Medical Facilities", sublabel: "Trust RecursiaDx" },
    { number: "< 5min", label: "Average Processing", sublabel: "Time per Sample" }
  ]

  return (
    <div className="min-h-screen">
      {/* Navigation Header */}
      <nav className="bg-white/95 backdrop-blur-sm border-b sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                  <Activity className="h-6 w-6 text-primary-foreground" />
                </div>
                <div>
                  <h1 className="text-xl font-bold">RecursiaDx</h1>
                  <Badge variant="outline" className="text-xs">
                    Digital Pathology Platform
                  </Badge>
                </div>
              </div>
            </div>
            
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-gray-600 hover:text-gray-900 transition-colors">
                Features
              </a>
              <ThemeSwitcher />
              <Button variant="outline" onClick={onLogin}>
                Login
              </Button>
              <Button onClick={onSignup}>
                Get Started
              </Button>
            </div>

            <div className="md:hidden flex items-center gap-2">
              <ThemeSwitcher />
              <Button variant="outline" size="sm" onClick={onLogin}>
                Login
              </Button>
              <Button size="sm" onClick={onSignup}>
                Sign Up
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
              <Badge className="mb-4">
                <Award className="h-4 w-4 mr-2" />
                RecuriaDx - Digital Pathology Diagnostics
              </Badge>
              
              <h1 className="text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
                Revolutionary
                <span className="text-primary block">Digital Pathology</span>
                Platform
              </h1>
              
              <p className="text-xl text-gray-600 mb-8 leading-relaxed">
                Transform your pathology lab with AI-powered analysis, automated workflows, 
                and professional reporting. Process blood tests and tissue biopsies with 
                unprecedented speed and accuracy.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4">
                <Button size="lg" onClick={onSignup} className="text-lg px-8 py-4">
                  Start Free Trial
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
                <Button variant="outline" size="lg" className="text-lg px-8 py-4">
                  <Play className="mr-2 h-5 w-5" />
                  Watch Demo
                </Button>
              </div>
              
              <div className="mt-8 flex items-center gap-6 text-sm text-gray-600">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <span>HIPAA Compliant</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <span>FDA Cleared</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <span>CAP/CLIA Certified</span>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <div className="bg-white rounded-2xl shadow-2xl p-6 border">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Sample Analysis Dashboard</h3>
                  <Badge variant="secondary">Live Demo</Badge>
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-200">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                      <span className="text-sm font-medium">Blood Analysis Complete</span>
                    </div>
                    <Badge className="bg-green-100 text-green-800 border-green-300">
                      96.8% Confidence
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse" />
                      <span className="text-sm font-medium">Tissue Biopsy Processing</span>
                    </div>
                    <Badge variant="secondary">
                      <Clock className="h-3 w-3 mr-1" />
                      2m 15s
                    </Badge>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-3 mt-4">
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-lg font-bold text-blue-600">4.2M</div>
                      <div className="text-xs text-gray-600">RBC Count</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-lg font-bold text-green-600">Normal</div>
                      <div className="text-xs text-gray-600">Hemoglobin</div>
                    </div>
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-lg font-bold text-purple-600">8.9K</div>
                      <div className="text-xs text-gray-600">WBC Count</div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Floating Elements */}
              <div className="absolute -top-4 -right-4 bg-white rounded-lg shadow-lg p-3 border animate-float">
                <div className="flex items-center gap-2">
                  <Brain className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium">AI Processing</span>
                </div>
              </div>
              
              <div className="absolute -bottom-4 -left-4 bg-white rounded-lg shadow-lg p-3 border animate-float" style={{ animationDelay: '1s' }}>
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-medium">Secure & Compliant</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="bg-primary text-primary-foreground py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-4xl lg:text-5xl font-bold mb-2">{stat.number}</div>
                <div className="text-lg font-semibold mb-1">{stat.label}</div>
                <div className="text-sm opacity-80">{stat.sublabel}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 ">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <Badge className="mb-4">Core Features</Badge>
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              Everything you need for
              <span className="text-primary block">modern pathology</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              From AI-powered analysis to professional reporting, RecursiaDx provides 
              a complete digital pathology solution for modern healthcare facilities.
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <Card key={index} className="hover:shadow-lg transition-all duration-300 border-0 shadow-md bg-gray-50">
                  <CardHeader>
                    <div className={`w-12 h-12 rounded-lg bg-gray-100 flex items-center justify-center mb-4`}>
                      <Icon className={`h-6 w-6 ${feature.color}`} />
                    </div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-gray-600 leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <Badge className="mb-4">Testimonials</Badge>
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              Trusted by healthcare
              <span className="text-primary block">professionals worldwide</span>
            </h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <Card key={index} className="border-0 shadow-lg">
                <CardContent className="p-6">
                  <Quote className="h-8 w-8 text-primary mb-4" />
                  <p className="text-gray-700 mb-6 leading-relaxed">
                    "{testimonial.content}"
                  </p>
                  <div className="flex items-center gap-3">
                    <Avatar>
                      <AvatarFallback>{testimonial.avatar}</AvatarFallback>
                    </Avatar>
                    <div>
                      <div className="font-semibold text-gray-900">{testimonial.name}</div>
                      <div className="text-sm text-gray-600">{testimonial.role}</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <div className="flex items-center gap-3 mb-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                  <Activity className="h-6 w-6 text-primary-foreground" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">RecursiaDx</h3>
                  <Badge variant="outline" className="text-xs border-gray-600 text-gray-300">
                    Digital Pathology Platform
                  </Badge>
                </div>
              </div>
              <p className="text-gray-400 mb-4 max-w-md">
                Revolutionizing pathology with AI-powered analysis and automated workflows 
                for modern healthcare facilities worldwide.
              </p>
              <div className="flex items-center gap-4 text-sm text-gray-400">
                <span>© 2024 RecursiaDx. All rights reserved.</span>
              </div>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Pricing</a></li>
                <li><a href="#" className="hover:text-white transition-colors">API</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Documentation</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-400">
                <li><a href="#" className="hover:text-white transition-colors">Help Center</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Contact Us</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-white transition-colors">Terms of Service</a></li>
              </ul>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}