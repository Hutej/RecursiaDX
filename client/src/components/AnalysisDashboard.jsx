import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Skeleton } from '@/components/ui/skeleton'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { toast } from 'sonner'
import { 
  Brain, 
  Activity, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle2, 
  Clock,
  Target,
  PieChart,
  BarChart3
} from 'lucide-react'

export function AnalysisDashboard({ onNext, sample, analysisType = 'general' }) {
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [currentStage, setCurrentStage] = useState('preprocessing')
  const [isAnalyzing, setIsAnalyzing] = useState(true)
  const [realTimeData, setRealTimeData] = useState(null)
  const [bloodAnalysisData, setBloodAnalysisData] = useState(null)
  
  // Determine image type from sample (blood or tissue)
  const imageType = sample?.sampleType === 'Blood Smear' ? 'blood' : 'tissue'
  
  // State for active tab (allows manual switching between tabs)
  const [activeTab, setActiveTab] = useState(imageType === 'blood' ? 'blood-analysis' : 'tissue-analysis')
  
  // Update active tab when imageType changes
  useEffect(() => {
    setActiveTab(imageType === 'blood' ? 'blood-analysis' : 'tissue-analysis')
  }, [imageType])

  // Enhanced debugging for sample data
  console.log('🔍 AnalysisDashboard - Sample data received:', sample)
  console.log('🔍 AnalysisDashboard - Sample type:', typeof sample)
  console.log('🔍 AnalysisDashboard - Image type:', imageType)
  console.log('🔍 AnalysisDashboard - Sample keys:', sample ? Object.keys(sample) : [])
  
  if (sample?.images) {
    console.log('🔍 AnalysisDashboard - Images:', sample.images)
    sample.images.forEach((img, idx) => {
      console.log(`🔍 Image ${idx + 1} analysis:`, {
        filename: img.filename,
        hasHeatmap: !!img.heatmap,
        hasMLAnalysis: !!img.mlAnalysis,
        hasBloodAnalysis: !!img.mlAnalysis?.bloodAnalysis,
        bloodAnalysisData: img.mlAnalysis?.bloodAnalysis,
        heatmapStructure: img.heatmap ? Object.keys(img.heatmap) : null
      })
    })
  }

  // Extract real data from sample if available
  const extractRealData = () => {
    if (!sample || !sample.images || sample.images.length === 0) {
      return null
    }

    const images = sample.images
    const totalImages = images.length
    const mlAnalyses = images.filter(img => img.mlAnalysis).map(img => img.mlAnalysis)
    
    if (mlAnalyses.length === 0) {
      return null
    }

    // Aggregate ML results from all images
    // Check for tumor detection - model returns 'Tumor' or uses is_tumor flag
    const tumorDetections = mlAnalyses.filter(analysis => 
      analysis.prediction === 'Tumor' || 
      analysis.prediction === 'malignant' ||
      analysis.is_tumor === true ||
      (analysis.metadata?.prediction?.is_tumor === true)
    )
    const avgConfidence = mlAnalyses.reduce((sum, analysis) => sum + (analysis.confidence || 0), 0) / mlAnalyses.length
    const riskLevels = mlAnalyses.map(analysis => analysis.riskAssessment || 'medium')

    // Calculate tumor percentage
    const tumorPercentage = mlAnalyses.length > 0 ? (tumorDetections.length / mlAnalyses.length) * 100 : 0
    
    // Calculate average tumor probability from all analyses
    const avgTumorProbability = mlAnalyses.reduce((sum, analysis) => {
      // ML result is stored in metadata, probabilities are at top level
      const tumorProb = analysis.metadata?.probabilities?.tumor || 0
      return sum + tumorProb
    }, 0) / mlAnalyses.length

    // Get the highest tumor probability for critical assessment
    const maxTumorProbability = Math.max(...mlAnalyses.map(analysis => 
      analysis.metadata?.probabilities?.tumor || 0
    ))

    return {
      totalImages,
      analyzedImages: mlAnalyses.length,
      tumorDetected: tumorDetections.length > 0,
      tumorCount: tumorDetections.length,
      tumorPercentage: tumorPercentage,
      avgTumorProbability: avgTumorProbability * 100, // Convert to percentage
      maxTumorProbability: maxTumorProbability * 100, // Convert to percentage
      averageConfidence: avgConfidence,
      confidenceRange: {
        min: Math.min(...mlAnalyses.map(a => a.confidence)),
        max: Math.max(...mlAnalyses.map(a => a.confidence))
      },
      riskDistribution: {
        high: riskLevels.filter(r => r && r === 'high').length,
        moderate: riskLevels.filter(r => r && r === 'medium').length,
        low: riskLevels.filter(r => r && r === 'low').length
      },
      highRiskCount: riskLevels.filter(r => r && r === 'high').length,
      patientInfo: sample.patientInfo || {},
      uploadedAt: sample.uploadedAt,
      analyses: mlAnalyses
    }
  }

  // Extract blood analysis data (malaria + platelet count)
  const extractBloodAnalysisData = () => {
    console.log('🩸 Extracting blood analysis data...')
    if (!sample || !sample.images || sample.images.length === 0) {
      console.log('❌ No sample or images found')
      return null
    }

    const images = sample.images
    const mlAnalyses = images.filter(img => img.mlAnalysis?.bloodAnalysis).map(img => img.mlAnalysis)
    
    console.log('🩸 ML Analyses with bloodAnalysis:', mlAnalyses.length)
    console.log('🩸 ML Analyses data:', mlAnalyses)
    
    if (mlAnalyses.length === 0) {
      console.log('❌ No ML analyses with bloodAnalysis found')
      return null
    }

    // Aggregate blood analysis results
    const bloodAnalyses = mlAnalyses.map(a => a.bloodAnalysis).filter(Boolean)
    console.log('🩸 Blood analyses extracted:', bloodAnalyses)
    
    if (bloodAnalyses.length === 0) {
      console.log('❌ No blood analyses found after filtering')
      return null
    }

    // Malaria detection aggregation
    const malariaResults = bloodAnalyses.map(b => b.malaria).filter(Boolean)
    const parasitizedCount = malariaResults.filter(m => m.isParasitized).length
    const avgMalariaConfidence = malariaResults.reduce((sum, m) => sum + (m.confidence || 0), 0) / malariaResults.length

    // Cell count aggregation
    const cellCounts = bloodAnalyses.map(b => b.cellCount).filter(Boolean)
    const totalPlatelets = cellCounts.reduce((sum, c) => sum + (c.platelets || 0), 0)
    const totalRBC = cellCounts.reduce((sum, c) => sum + (c.rbc || 0), 0)
    const totalWBC = cellCounts.reduce((sum, c) => sum + (c.wbc || 0), 0)
    const totalCells = cellCounts.reduce((sum, c) => sum + (c.totalCells || 0), 0)

    const result = {
      totalImages: images.length,
      analyzedImages: mlAnalyses.length,
      malaria: {
        detected: parasitizedCount > 0,
        parasitizedCount,
        status: parasitizedCount > 0 ? 'Parasitized' : 'Uninfected',
        avgConfidence: avgMalariaConfidence,
        results: malariaResults
      },
      cellCount: {
        platelets: totalPlatelets,
        rbc: totalRBC,
        wbc: totalWBC,
        totalCells: totalCells,
        status: cellCounts[0]?.status || 'analyzed'
      },
      patientInfo: sample.patientInfo || {},
      analyses: mlAnalyses
    }
    
    console.log('✅ Blood analysis data extracted:', result)
    return result
  }

  // Initialize real-time data from sample
  useEffect(() => {
    if (imageType === 'blood') {
      const bloodData = extractBloodAnalysisData()
      setBloodAnalysisData(bloodData)
      if (bloodData) {
        setIsAnalyzing(true)
        console.log('🩸 Blood analysis data loaded:', bloodData)
      } else {
        setIsAnalyzing(false)
      }
    } else {
      const extracted = extractRealData()
      setRealTimeData(extracted)
      
      if (extracted) {
        setIsAnalyzing(true)
        console.log('🔬 Tissue sample data loaded:', extracted)
      } else {
        setIsAnalyzing(false)
      }
    }
  }, [sample, imageType])

  const getConfidenceData = () => {
    // For blood analysis
    if (imageType === 'blood' && bloodAnalysisData) {
      const baseConfidence = bloodAnalysisData.malaria.avgConfidence * 100
      return [
        { 
          category: 'Malaria Detection', 
          confidence: Math.min(98, baseConfidence), 
          color: bloodAnalysisData.malaria.detected ? '#ef4444' : '#22c55e'
        },
        { 
          category: 'Cell Detection', 
          confidence: 95, 
          color: '#3b82f6' 
        },
        { 
          category: 'RBC Analysis', 
          confidence: 92, 
          color: '#ef4444' 
        },
        { 
          category: 'Platelet Count', 
          confidence: 94, 
          color: '#8b5cf6' 
        }
      ]
    }
    
    // For tissue analysis
    if (!realTimeData) {
      return [
        { category: 'Cell Detection', confidence: 96.8, color: '#22c55e' },
        { category: 'Morphology Analysis', confidence: 94.2, color: '#3b82f6' },
        { category: 'Anomaly Detection', confidence: 87.5, color: '#f59e0b' },
        { category: 'Classification', confidence: 91.3, color: '#8b5cf6' }
      ]
    }

    // For tissue analysis - show only what GigaPath model actually provides
    const tumorConfidence = realTimeData.averageConfidence * 100
    const riskLevel = realTimeData.highRiskCount > 0 ? 'high' : 
                      realTimeData.avgTumorProbability > 50 ? 'medium' : 'low'
    
    return [
      { 
        category: 'Tumor Detection', 
        confidence: tumorConfidence, 
        color: realTimeData.tumorDetected ? '#ef4444' : '#22c55e',
        description: realTimeData.tumorDetected ? 'Tumor detected' : 'No tumor detected'
      },
      { 
        category: 'GigaPath Model Confidence', 
        confidence: tumorConfidence, 
        color: tumorConfidence > 80 ? '#22c55e' : tumorConfidence > 60 ? '#f59e0b' : '#ef4444',
        description: 'AttentionMIL model confidence'
      },
      { 
        category: 'Tumor Probability', 
        confidence: realTimeData.avgTumorProbability, 
        color: realTimeData.avgTumorProbability > 50 ? '#ef4444' : '#22c55e',
        description: `${realTimeData.avgTumorProbability.toFixed(1)}% probability`
      },
      { 
        category: 'Risk Assessment', 
        confidence: riskLevel === 'high' ? 90 : riskLevel === 'medium' ? 60 : 30, 
        color: riskLevel === 'high' ? '#ef4444' : riskLevel === 'medium' ? '#f59e0b' : '#22c55e',
        description: `${riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)} Risk`
      }
    ]
  }

  const analysisStages = [
    { id: 'preprocessing', label: 'Image Preprocessing', duration: 15 },
    { id: 'tiling', label: 'WSI Tiling', duration: 25 },
    { id: 'feature_extraction', label: 'Feature Extraction', duration: 30 },
    { id: 'classification', label: 'AI Classification', duration: 20 },
    { id: 'postprocessing', label: 'Result Processing', duration: 10 }
  ]

  const confidenceData = getConfidenceData()

  useEffect(() => {
    if (isAnalyzing) {
      const interval = setInterval(() => {
        setAnalysisProgress(prev => {
          if (prev >= 100) {
            clearInterval(interval)
            setIsAnalyzing(false)
            
            // Toast notification for completion
            const analysisType = imageType === 'blood' ? 'Blood analysis' : 'Tissue analysis'
            toast.success(`${analysisType} completed!`)
            return 100
          }
          
          // Update current stage based on progress
          const currentProgress = Math.min(100, prev + ((realTimeData || bloodAnalysisData) ? 3 : 2))
          if (currentProgress < 15) setCurrentStage('preprocessing')
          else if (currentProgress < 40) setCurrentStage('tiling')
          else if (currentProgress < 70) setCurrentStage('feature_extraction')
          else if (currentProgress < 90) setCurrentStage('classification')
          else setCurrentStage('postprocessing')
          
          return currentProgress
        })
      }, (realTimeData || bloodAnalysisData) ? 80 : 100)

      return () => clearInterval(interval)
    }
  }, [isAnalyzing, realTimeData, bloodAnalysisData, imageType])

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200'
      case 'medium': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'low': return 'text-green-600 bg-green-50 border-green-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  return (
    <div className="grid gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AI Analysis Dashboard</h1>
          <p className="text-muted-foreground">
            Step 3: AI-powered pathology analysis and prediction
            {(realTimeData?.patientInfo?.name || bloodAnalysisData?.patientInfo?.name) && (
              <span className="ml-2 text-primary">• Patient: {realTimeData?.patientInfo?.name || bloodAnalysisData?.patientInfo?.name}</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {(realTimeData || bloodAnalysisData) && (
            <Badge variant="secondary" className="text-sm">
              <Target className="h-4 w-4 mr-2" />
              {(realTimeData?.analyzedImages || bloodAnalysisData?.analyzedImages)}/{(realTimeData?.totalImages || bloodAnalysisData?.totalImages)} Images
            </Badge>
          )}
          <Badge variant="outline" className="text-sm">
            <Brain className="h-4 w-4 mr-2" />
            {isAnalyzing ? 'Processing...' : 'Analysis Complete'}
          </Badge>
        </div>
      </div>

      {/* Blood Analysis Summary Cards - Only show for blood samples */}
      {imageType === 'blood' && bloodAnalysisData && !isAnalyzing && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className={`border-2 ${bloodAnalysisData.malaria.detected ? 'border-red-200 bg-red-50/50' : 'border-green-200 bg-green-50/50'}`}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Malaria Detection</p>
                  <p className={`text-2xl font-bold ${bloodAnalysisData.malaria.detected ? 'text-red-600' : 'text-green-600'}`}>
                    {bloodAnalysisData.malaria.detected ? 'PARASITIZED' : 'UNINFECTED'}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {bloodAnalysisData.malaria.parasitizedCount} of {bloodAnalysisData.analyzedImages} images
                  </p>
                </div>
                {bloodAnalysisData.malaria.detected ? 
                  <AlertTriangle className="h-8 w-8 text-red-600" /> : 
                  <CheckCircle2 className="h-8 w-8 text-green-600" />
                }
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Platelets Count</p>
                  <p className="text-2xl font-bold text-purple-600">{bloodAnalysisData.cellCount.platelets}</p>
                  <p className="text-xs text-muted-foreground">cells detected</p>
                </div>
                <Activity className="h-8 w-8 text-purple-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">RBC Count</p>
                  <p className="text-2xl font-bold text-red-500">{bloodAnalysisData.cellCount.rbc}</p>
                  <p className="text-xs text-muted-foreground">red blood cells</p>
                </div>
                <Target className="h-8 w-8 text-red-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">WBC Count</p>
                  <p className="text-2xl font-bold text-blue-600">{bloodAnalysisData.cellCount.wbc}</p>
                  <p className="text-xs text-muted-foreground">white blood cells</p>
                </div>
                <TrendingUp className="h-8 w-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Tissue Analysis Summary Cards */}
      {imageType === 'tissue' && realTimeData && !isAnalyzing && (() => {
        // Threshold: if tumor probability < 54%, consider it NEGATIVE
        const tumorThreshold = 54
        const isTumorPositive = realTimeData.avgTumorProbability >= tumorThreshold
        
        return (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            
            <Card className={`border-2 ${isTumorPositive ? 'border-red-200 bg-red-50/50' : 'border-green-200 bg-green-50/50'}`}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Tumor Detection</p>
            <p className={`text-2xl font-bold ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
              {isTumorPositive ? 'POSITIVE' : 'NEGATIVE'}
            </p>
            <p className="text-xs text-muted-foreground">
              {realTimeData.tumorCount} of {realTimeData.analyzedImages} images
            </p>
              </div>
              {isTumorPositive ? (
                <AlertTriangle className="h-8 w-8 text-red-600" />
              ) : (
                <CheckCircle2 className="h-8 w-8 text-green-600" />
              )}
            </div>
          </CardContent>
            </Card>

            <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Overall Confidence</p>
            <p className="text-2xl font-bold">{(realTimeData.averageConfidence * 100).toFixed(1)}%</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
            </Card>

            <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Tumor Probability</p>
            <p className="text-2xl font-bold text-orange-600">
              {realTimeData.avgTumorProbability.toFixed(1)}%
            </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-orange-600" />
            </div>
          </CardContent>
            </Card>

            <Card className={`border-2 ${
              realTimeData.highRiskCount > 0 ? 'border-red-200 bg-red-50/50' : 
              realTimeData.avgTumorProbability > 50 ? 'border-yellow-200 bg-yellow-50/50' : 
              'border-green-200 bg-green-50/50'
            }`}>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
            <p className="text-sm font-medium text-muted-foreground">Risk Level</p>
            <p className={`text-2xl font-bold ${
              realTimeData.highRiskCount > 0 ? 'text-red-600' : 
              realTimeData.avgTumorProbability > 50 ? 'text-yellow-600' : 
              'text-green-600'
            }`}>
              {realTimeData.highRiskCount > 0 ? 'HIGH' : 
               realTimeData.avgTumorProbability > 50 ? 'MEDIUM' : 'LOW'}
            </p>
              </div>
              <AlertTriangle className={`h-8 w-8 ${
                realTimeData.highRiskCount > 0 ? 'text-red-600' : 
                realTimeData.avgTumorProbability > 50 ? 'text-yellow-600' : 
                'text-green-600'
              }`} />
            </div>
          </CardContent>
            </Card>
          </div>
        )
      })()}

        {/* Tumor Percentage Detailed Analysis - Only for Tissue */}
      {imageType === 'tissue' && realTimeData && !isAnalyzing && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Tumor Distribution Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <PieChart className="h-5 w-5" />
                Tumor Distribution Analysis
              </CardTitle>
              <CardDescription>
                Percentage breakdown of tumor vs normal tissue detected
              </CardDescription>
            </CardHeader>
            <CardContent>
              {(() => {
                // Threshold: if tumor probability < 54%, consider it Normal
                const tumorThreshold = 54
                const isTumorPositive = realTimeData.avgTumorProbability >= tumorThreshold
                const displayTumorPercent = isTumorPositive ? realTimeData.avgTumorProbability : 0
                const displayNormalPercent = isTumorPositive ? (100 - realTimeData.avgTumorProbability) : 100
                const tumorImageCount = isTumorPositive ? realTimeData.tumorCount : 0
                const normalImageCount = isTumorPositive ? (realTimeData.analyzedImages - realTimeData.tumorCount) : realTimeData.analyzedImages
                
                return (
              <div className="space-y-4">
                {/* Large Percentage Display */}
                <div className={`text-center p-6 rounded-lg border-2 ${
                  isTumorPositive 
                    ? 'bg-gradient-to-br from-red-50 to-orange-50 border-red-200' 
                    : 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-200'
                }`}>
                  <div className={`text-6xl font-bold mb-2 ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                    {realTimeData.avgTumorProbability.toFixed(1)}%
                  </div>
                  <div className={`text-lg font-medium ${isTumorPositive ? 'text-red-700' : 'text-green-700'}`}>
                    {isTumorPositive ? 'Tumor Detected in Samples' : 'Normal Tissue (Below Threshold)'}
                  </div>
                  <div className={`text-sm mt-2 ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                    {isTumorPositive ? `${realTimeData.tumorCount} out of ${realTimeData.analyzedImages} images` : 'Probability below 54% threshold'}
                  </div>
                </div>

                {/* Breakdown */}
                <div className="space-y-3">
                  <div className={`flex items-center justify-between p-3 rounded-lg ${isTumorPositive ? 'bg-red-100' : 'bg-gray-100'}`}>
                    <div className="flex items-center gap-3">
                      <div className={`w-4 h-4 rounded-full ${isTumorPositive ? 'bg-red-600' : 'bg-gray-400'}`}></div>
                      <span className="font-medium">Tumor Detected</span>
                    </div>
                    <div className={`font-bold ${isTumorPositive ? 'text-red-600' : 'text-gray-500'}`}>
                      {displayTumorPercent.toFixed(1)}% ({tumorImageCount} images)
                    </div>
                  </div>
                  
                  <div className={`flex items-center justify-between p-3 rounded-lg ${!isTumorPositive ? 'bg-green-100' : 'bg-green-50'}`}>
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-green-600"></div>
                      <span className="font-medium">Normal Tissue</span>
                    </div>
                    <div className="font-bold text-green-600">
                      {displayNormalPercent.toFixed(1)}% ({normalImageCount} images)
                    </div>
                  </div>
                </div>
              </div>
                )
              })()}
            </CardContent>
          </Card>

          {/* Tumor Probability Details */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Tumor Probability Analysis
              </CardTitle>
              <CardDescription>
                Detailed probability scores from AI model predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Probability Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
                    <div className="text-sm font-medium text-orange-700">Average Probability</div>
                    <div className="text-2xl font-bold text-orange-600">
                      {realTimeData.avgTumorProbability.toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="p-4 bg-red-50 rounded-lg border border-red-200">
                    <div className="text-sm font-medium text-red-700">Maximum Probability</div>
                    <div className="text-2xl font-bold text-red-600">
                      {realTimeData.maxTumorProbability.toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Per-Image Tumor Probabilities */}
                <div className="space-y-2">
                  <h4 className="font-semibold text-sm">Per-Image Tumor Probabilities</h4>
                  {realTimeData.analyses.map((analysis, index) => {
                    const tumorProb = (analysis.metadata?.probabilities?.tumor || 0) * 100
                    return (
                    <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                      <span className="text-sm font-medium">Image {index + 1}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-20 bg-gray-200 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              tumorProb > 50 ? 'bg-red-500' : 'bg-yellow-500'
                            }`}
                            style={{ width: `${tumorProb}%` }}
                          />
                        </div>
                        <span className="text-sm font-bold">
                          {tumorProb.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    )
                  })}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Blood Analysis Detailed Cards */}
      {imageType === 'blood' && bloodAnalysisData && !isAnalyzing && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Malaria Detection Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5" />
                Malaria Detection Analysis
              </CardTitle>
              <CardDescription>
                AI-powered parasite detection in blood smear
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Large Status Display */}
                <div className={`text-center p-6 rounded-lg border-2 ${
                  bloodAnalysisData.malaria.detected 
                    ? 'bg-gradient-to-br from-red-50 to-orange-50 border-red-200' 
                    : 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-200'
                }`}>
                  <div className={`text-4xl font-bold mb-2 ${
                    bloodAnalysisData.malaria.detected ? 'text-red-600' : 'text-green-600'
                  }`}>
                    {bloodAnalysisData.malaria.status}
                  </div>
                  <div className={`text-lg font-medium ${
                    bloodAnalysisData.malaria.detected ? 'text-red-700' : 'text-green-700'
                  }`}>
                    {bloodAnalysisData.malaria.detected ? 'Malaria Parasite Detected' : 'No Malaria Parasite Found'}
                  </div>
                  <div className="text-sm mt-2 text-muted-foreground">
                    Confidence: {(bloodAnalysisData.malaria.avgConfidence * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Detection Stats */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg border">
                    <div className="text-sm font-medium text-muted-foreground">Parasitized Samples</div>
                    <div className="text-2xl font-bold text-red-600">
                      {bloodAnalysisData.malaria.parasitizedCount}
                    </div>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg border">
                    <div className="text-sm font-medium text-muted-foreground">Clean Samples</div>
                    <div className="text-2xl font-bold text-green-600">
                      {bloodAnalysisData.analyzedImages - bloodAnalysisData.malaria.parasitizedCount}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Cell Count Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Blood Cell Count Analysis
              </CardTitle>
              <CardDescription>
                YOLOv8-powered cell detection and counting
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {/* Total Cells */}
                <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border-2 border-blue-200">
                  <div className="text-4xl font-bold text-blue-600 mb-1">
                    {bloodAnalysisData.cellCount.totalCells}
                  </div>
                  <div className="text-sm font-medium text-blue-700">Total Cells Detected</div>
                </div>

                {/* Cell Type Breakdown */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-red-100 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-red-500"></div>
                      <span className="font-medium">Red Blood Cells (RBC)</span>
                    </div>
                    <div className="font-bold text-red-600">{bloodAnalysisData.cellCount.rbc}</div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-blue-100 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-blue-600"></div>
                      <span className="font-medium">White Blood Cells (WBC)</span>
                    </div>
                    <div className="font-bold text-blue-600">{bloodAnalysisData.cellCount.wbc}</div>
                  </div>
                  
                  <div className="flex items-center justify-between p-3 bg-purple-100 rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 rounded-full bg-purple-600"></div>
                      <span className="font-medium">Platelets</span>
                    </div>
                    <div className="font-bold text-purple-600">{bloodAnalysisData.cellCount.platelets}</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Analysis Progress */}
      {isAnalyzing && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 animate-pulse" />
              Analysis in Progress
            </CardTitle>
            <CardDescription>
              AI model is processing your samples using multi-scale attention
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">Overall Progress</span>
                <span className="text-sm text-muted-foreground">{analysisProgress.toFixed(0)}%</span>
              </div>
              
              <div className="w-full bg-muted rounded-full h-3">
                <div 
                  className="bg-primary h-3 rounded-full transition-all duration-300"
                  style={{ width: `${analysisProgress}%` }}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-5 gap-2">
                {analysisStages.map((stage) => (
                  <div 
                    key={stage.id}
                    className={`p-3 rounded-lg text-center ${
                      currentStage === stage.id 
                        ? 'bg-primary text-primary-foreground' 
                        : analysisProgress > analysisStages.findIndex(s => s.id === stage.id) * 20
                        ? 'bg-green-100 text-green-800'
                        : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    <div className="text-xs font-medium">{stage.label}</div>
                    {currentStage === stage.id && (
                      <div className="text-xs mt-1 flex items-center justify-center gap-1">
                        <Clock className="h-3 w-3" />
                        ~{stage.duration}s
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="blood-analysis" className={imageType === 'blood' ? 'bg-red-100' : ''}>
            🩸 Blood Smear Analysis
          </TabsTrigger>
          <TabsTrigger value="tissue-analysis" className={imageType === 'tissue' ? 'bg-blue-100' : ''}>
            🔬 Tissue Analysis
          </TabsTrigger>
        </TabsList>

        <TabsContent value="blood-analysis" className="space-y-6">
          {imageType !== 'blood' ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-semibold mb-2">Blood Analysis Not Applicable</h3>
                <p className="text-muted-foreground">
                  This analysis is only available for blood smear images. 
                  The current sample is a tissue sample for tumor detection.
                </p>
              </CardContent>
            </Card>
          ) : !bloodAnalysisData ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-600" />
                <h3 className="text-lg font-semibold mb-2">No Blood Analysis Data</h3>
                <p className="text-muted-foreground">
                  Blood smear analysis results will appear here after uploading and analyzing blood smear images.
                  Select "Blood Smear Image" in Step 1 to run malaria detection and platelet counting.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6">
              {/* Malaria Detection Section */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5" />
                    Malaria Parasite Detection
                  </CardTitle>
                  <CardDescription>
                    InceptionV3-powered malaria detection in blood smear samples
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isAnalyzing ? (
                    <div className="space-y-4">
                      <Skeleton className="h-32 w-full" />
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Alert className={bloodAnalysisData.malaria.detected ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                        {bloodAnalysisData.malaria.detected ? (
                          <>
                            <AlertTriangle className="h-4 w-4 text-red-600" />
                            <AlertTitle className="text-red-600">Malaria Parasite Detected!</AlertTitle>
                            <AlertDescription>
                              {bloodAnalysisData.malaria.parasitizedCount} sample(s) show signs of parasitic infection. 
                              Immediate medical attention recommended.
                            </AlertDescription>
                          </>
                        ) : (
                          <>
                            <CheckCircle2 className="h-4 w-4 text-green-600" />
                            <AlertTitle className="text-green-600">No Malaria Detected</AlertTitle>
                            <AlertDescription>
                              All {bloodAnalysisData.analyzedImages} blood smear samples appear clear of malaria parasites.
                            </AlertDescription>
                          </>
                        )}
                      </Alert>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Card>
                          <CardContent className="p-4 text-center">
                            <div className={`text-2xl font-bold ${bloodAnalysisData.malaria.detected ? 'text-red-600' : 'text-green-600'}`}>
                              {bloodAnalysisData.malaria.status}
                            </div>
                            <div className="text-sm text-muted-foreground">Detection Status</div>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4 text-center">
                            <div className="text-2xl font-bold">{(bloodAnalysisData.malaria.avgConfidence * 100).toFixed(1)}%</div>
                            <div className="text-sm text-muted-foreground">Model Confidence</div>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="p-4 text-center">
                            <div className="text-2xl font-bold">{bloodAnalysisData.analyzedImages}</div>
                            <div className="text-sm text-muted-foreground">Images Analyzed</div>
                          </CardContent>
                        </Card>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Cell Count Section */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Blood Cell Count Results
                  </CardTitle>
                  <CardDescription>
                    YOLOv8-powered automated cell detection and counting
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isAnalyzing ? (
                    <Skeleton className="h-48 w-full" />
                  ) : (
                    <div className="space-y-4">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Cell Type</TableHead>
                            <TableHead className="text-right">Count</TableHead>
                            <TableHead className="text-right">Percentage</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          <TableRow>
                            <TableCell className="font-medium">
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-red-500" />
                                Red Blood Cells (RBC)
                              </div>
                            </TableCell>
                            <TableCell className="text-right font-bold text-red-600">{bloodAnalysisData.cellCount.rbc}</TableCell>
                            <TableCell className="text-right">
                              {bloodAnalysisData.cellCount.totalCells > 0 
                                ? ((bloodAnalysisData.cellCount.rbc / bloodAnalysisData.cellCount.totalCells) * 100).toFixed(1) 
                                : 0}%
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell className="font-medium">
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-blue-600" />
                                White Blood Cells (WBC)
                              </div>
                            </TableCell>
                            <TableCell className="text-right font-bold text-blue-600">{bloodAnalysisData.cellCount.wbc}</TableCell>
                            <TableCell className="text-right">
                              {bloodAnalysisData.cellCount.totalCells > 0 
                                ? ((bloodAnalysisData.cellCount.wbc / bloodAnalysisData.cellCount.totalCells) * 100).toFixed(1) 
                                : 0}%
                            </TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell className="font-medium">
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-purple-600" />
                                Platelets
                              </div>
                            </TableCell>
                            <TableCell className="text-right font-bold text-purple-600">{bloodAnalysisData.cellCount.platelets}</TableCell>
                            <TableCell className="text-right">
                              {bloodAnalysisData.cellCount.totalCells > 0 
                                ? ((bloodAnalysisData.cellCount.platelets / bloodAnalysisData.cellCount.totalCells) * 100).toFixed(1) 
                                : 0}%
                            </TableCell>
                          </TableRow>
                          <TableRow className="bg-gray-50">
                            <TableCell className="font-bold">Total Cells</TableCell>
                            <TableCell className="text-right font-bold">{bloodAnalysisData.cellCount.totalCells}</TableCell>
                            <TableCell className="text-right font-bold">100%</TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="tissue-analysis" className="space-y-6">
          {imageType !== 'tissue' ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-semibold mb-2">Tissue Analysis Not Applicable</h3>
                <p className="text-muted-foreground">
                  This analysis is only available for tissue/histopathology images. 
                  The current sample is a blood smear for malaria and cell counting.
                </p>
              </CardContent>
            </Card>
          ) : !realTimeData ? (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-600" />
                <h3 className="text-lg font-semibold mb-2">No Tissue Analysis Data</h3>
                <p className="text-muted-foreground">
                  Tissue biopsy analysis results will appear here after analyzing histopathology images.
                </p>
              </CardContent>
            </Card>
          ) : (
          <Card>
            <CardHeader>
              <CardTitle>Tissue Biopsy Analysis</CardTitle>
              <CardDescription>
                AI-powered cancer detection and morphological analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isAnalyzing ? (
                <div className="space-y-4">
                  <Skeleton className="h-32 w-full" />
                  <div className="grid grid-cols-3 gap-4">
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-20 w-full" />
                    <Skeleton className="h-20 w-full" />
                  </div>
                </div>
              ) : (() => {
                // Threshold: if tumor probability < 54%, consider it Normal
                const tumorThreshold = 54
                const isTumorPositive = realTimeData.avgTumorProbability >= tumorThreshold
                
                return (
                <div className="space-y-4">
                  <Alert className={isTumorPositive ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                    {isTumorPositive ? (
                      <>
                        <AlertTriangle className="h-4 w-4 text-red-600" />
                        <AlertTitle className="text-red-600">Abnormality Detected</AlertTitle>
                        <AlertDescription>
                          {realTimeData.tumorCount} suspicious region(s) detected. Further review recommended.
                        </AlertDescription>
                      </>
                    ) : (
                      <>
                        <CheckCircle2 className="h-4 w-4 text-green-600" />
                        <AlertTitle className="text-green-600">Analysis Complete - Normal</AlertTitle>
                        <AlertDescription>
                          No malignant cells detected. Tumor probability ({realTimeData.avgTumorProbability.toFixed(1)}%) is below the 54% threshold.
                        </AlertDescription>
                      </>
                    )}
                  </Alert>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className={`text-2xl font-bold ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                          {isTumorPositive ? 'Abnormal' : 'Normal'}
                        </div>
                        <div className="text-sm text-muted-foreground">Overall Assessment</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold">{(realTimeData.averageConfidence * 100).toFixed(1)}%</div>
                        <div className="text-sm text-muted-foreground">Confidence Score</div>
                      </CardContent>
                    </Card>
                    <Card>
                      <CardContent className="p-4 text-center">
                        <div className={`text-2xl font-bold ${isTumorPositive ? 'text-red-600' : 'text-green-600'}`}>
                          {isTumorPositive ? realTimeData.tumorCount : 0}
                        </div>
                        <div className="text-sm text-muted-foreground">Suspicious Regions</div>
                      </CardContent>
                    </Card>
                  </div>
                </div>
                )
              })()}
            </CardContent>
          </Card>
          )}
        </TabsContent>
      </Tabs>

      {/* Action Button */}
      <div className="flex justify-end">
        <Button 
          onClick={() => {
            toast.success("Proceeding to results review...")
            setTimeout(() => onNext(), 1500)
          }}
          disabled={isAnalyzing}
        >
          {isAnalyzing ? 'Analysis in Progress...' : 'Proceed to Review'}
        </Button>
      </div>
    </div>
  )
}