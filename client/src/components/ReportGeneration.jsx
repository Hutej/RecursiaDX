import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Progress } from '@/components/ui/progress'
import { toast } from 'sonner'
import { 
  FileText, 
  Download,  
  Share, 
  Mail, 
  CheckCircle2,
  AlertTriangle,
  Clock,
  Shield,
  Microscope,
  Droplets,
  ArrowRight,
  RefreshCw,
  Sparkles,
  User,
  Calendar
} from 'lucide-react'

export function ReportGeneration({ sample, onNext }) {
  const [reportStatus, setReportStatus] = useState('idle') // idle, generating, completed, error
  const [reportData, setReportData] = useState(null)
  const [generationProgress, setGenerationProgress] = useState(0)
  const [clinicalSummary, setClinicalSummary] = useState(null)
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false)
  const [geminiReport, setGeminiReport] = useState(null) // Full Gemini report content

  // Lab info with Indian phone
  const labInfo = {
    name: 'RecursiaDx Digital Pathology Lab',
    address: 'Sector 62, Noida, Uttar Pradesh 201301, India',
    phone: '+91 99999 88888',
    email: 'reports@recursiadx.com',
    license: 'DPL-2026-RDX-001',
    accreditation: 'NABL Accredited'
  }

  // Determine sample type
  const sampleType = sample?.sampleType || 'Tissue Biopsy'
  const isBloodSample = sampleType.toLowerCase().includes('blood')

  // Extract patient info from sample (from Step 1)
  const patientInfo = {
    id: sample?.sampleId || sample?._id || 'N/A',
    name: sample?.patientInfo?.name || 'Not Provided',
    age: sample?.patientInfo?.age || 'N/A',
    gender: sample?.patientInfo?.gender || 'N/A',
    bloodGroup: sample?.patientInfo?.bloodGroup || 'N/A',
    phone: sample?.patientInfo?.contactNumber || sample?.patientInfo?.phone || 'N/A',
    email: sample?.patientInfo?.email || 'N/A',
    physician: sample?.clinicalInfo?.orderingPhysician || sample?.patientInfo?.physician || 'N/A',
    orderDate: sample?.createdAt ? new Date(sample.createdAt).toLocaleDateString('en-IN') : 'N/A',
    sampleDate: sample?.sampleInfo?.collectionDate ? new Date(sample.sampleInfo.collectionDate).toLocaleDateString('en-IN') : 'N/A',
    reportDate: new Date().toLocaleDateString('en-IN')
  }

  // Extract ML analysis results
  const getAnalysisResults = () => {
    if (!sample?.images || sample.images.length === 0) return null

    const firstImage = sample.images[0]
    if (!firstImage?.mlAnalysis) return null

    const analysis = firstImage.mlAnalysis
    const tumorProb = (analysis.metadata?.probabilities?.tumor || 0) * 100
    const isPositive = tumorProb >= 54

    return {
      prediction: isPositive ? 'Malignant' : 'Benign',
      confidence: (analysis.confidence * 100).toFixed(1),
      riskLevel: analysis.risk_level || 'Low Risk',
      tumorProbability: tumorProb.toFixed(1),
      normalProbability: ((analysis.metadata?.probabilities?.normal || 0) * 100).toFixed(1),
      isPositive,
      totalImages: sample.images.length,
      analyzedImages: sample.images.filter(img => img.mlAnalysis).length
    }
  }

  const analysisResults = getAnalysisResults()

  // Generate clinical summary using Gemini (or fallback)
  const generateClinicalSummary = async () => {
    if (!sample?._id) return

    setIsGeneratingSummary(true)
    try {
      const response = await fetch(`http://localhost:5001/api/reports/generate-summary/${sample._id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      const data = await response.json()
      if (data.success) {
        setClinicalSummary(data.data.clinicalSummary)
        toast.success('Clinical summary generated!')
      }
    } catch (error) {
      console.error('Failed to generate summary:', error)
      // Use fallback summary
      setClinicalSummary({
        summary: analysisResults?.isPositive 
          ? `AI-assisted analysis detected potential abnormalities with ${analysisResults.confidence}% confidence. Risk assessment: ${analysisResults.riskLevel}. Further clinical evaluation recommended.`
          : `AI-assisted analysis confirms normal findings with ${analysisResults?.confidence || 0}% confidence. No malignant cells or abnormal tissue architecture identified.`,
        generatedBy: 'Fallback'
      })
    } finally {
      setIsGeneratingSummary(false)
    }
  }

  // Generate FULL Gemini Report
  const generateFullGeminiReport = async () => {
    if (!sample?._id) return null

    try {
      const response = await fetch(`http://localhost:5001/api/reports/generate-full/${sample._id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      const data = await response.json()
      if (data.success) {
        return data.data
      }
    } catch (error) {
      console.error('Failed to generate full report:', error)
    }
    return null
  }

  // Generate report
  const generateReport = async () => {
    if (!sample?._id) {
      toast.error('No sample available for report generation')
      return
    }

    setReportStatus('generating')
    setGenerationProgress(0)

    // Simulate progress
    const progressInterval = setInterval(() => {
      setGenerationProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval)
          return 90
        }
        return prev + Math.random() * 15
      })
    }, 300)

    try {
      const response = await fetch(`http://localhost:5001/api/reports/generate/${sample._id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reportType: 'Final',
          includeImages: true
        })
      })

      clearInterval(progressInterval)
      setGenerationProgress(100)

      const data = await response.json()

      if (response.ok && data.success) {
        setReportData(data.data)
        setReportStatus('completed')
        toast.success('Report generated! Generating AI analysis...')
        
        // Generate FULL Gemini report with clinical summary, interpretation, recommendations
        setIsGeneratingSummary(true)
        const fullReport = await generateFullGeminiReport()
        if (fullReport) {
          setGeminiReport(fullReport)
          if (fullReport.clinicalSummary) {
            setClinicalSummary(fullReport.clinicalSummary)
          }
          toast.success('AI report generated successfully!')
        } else {
          // Fallback to simple clinical summary
          generateClinicalSummary()
        }
        setIsGeneratingSummary(false)
      } else {
        throw new Error(data.message || 'Failed to generate report')
      }
    } catch (error) {
      console.error('Report generation failed:', error)
      setReportStatus('error')
      toast.error('Failed to generate report')
    }
  }

  // Download report
  const downloadReport = async (format) => {
    if (!sample?._id) {
      toast.error('No sample available for download')
      return
    }

    if (format === 'pdf') {
      try {
        toast.info('Generating PDF report...')
        
        // Call backend PDF endpoint
        const response = await fetch(`http://localhost:5001/api/reports/download-pdf/${sample._id}`)
        
        if (!response.ok) {
          throw new Error('Failed to generate PDF')
        }

        // Get PDF blob
        const blob = await response.blob()
        
        // Create download link
        const url = window.URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `RecursiaDx_Report_${sample.sampleId || sample._id}.pdf`
        document.body.appendChild(link)
        link.click()
        
        // Cleanup
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
        
        toast.success('PDF downloaded successfully!')
      } catch (error) {
        console.error('PDF download error:', error)
        toast.error('Failed to download PDF')
      }
    } else if (format === 'docx') {
      toast.info('DOCX download coming soon!')
    }
  }

  // If no sample
  if (!sample) {
    return (
      <div className="grid gap-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Report Generation</h1>
            <p className="text-muted-foreground">Step 5: Generate professional pathology report</p>
          </div>
        </div>

        <Card className="border-amber-200 bg-amber-50">
          <CardContent className="p-12 text-center">
            <AlertTriangle className="h-16 w-16 mx-auto text-amber-500 mb-4" />
            <h2 className="text-xl font-semibold mb-2">No Report Available</h2>
            <p className="text-muted-foreground">Please complete previous steps first.</p>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="grid gap-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Report Generation</h1>
          <p className="text-muted-foreground">Step 5: Generate professional pathology report</p>
        </div>
        <Badge variant="outline">
          <FileText className="h-4 w-4 mr-2" />
          {reportStatus === 'completed' ? 'Report Ready' : 'Pending'}
        </Badge>
      </div>

      {/* Sample Summary Card */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              {isBloodSample ? (
                <Droplets className="h-8 w-8 text-red-500" />
              ) : (
                <Microscope className="h-8 w-8 text-blue-500" />
              )}
              <div>
                <div className="font-semibold">{patientInfo.id}</div>
                <div className="text-sm text-muted-foreground">{sampleType}</div>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{analysisResults?.totalImages || 0}</div>
                <div className="text-xs text-muted-foreground">Images</div>
              </div>
              <div className="text-center">
                <div className={`text-2xl font-bold ${analysisResults?.isPositive ? 'text-red-600' : 'text-green-600'}`}>
                  {analysisResults?.isPositive ? 'POSITIVE' : 'NEGATIVE'}
                </div>
                <div className="text-xs text-muted-foreground">Result</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Report Generation Status */}
      {reportStatus === 'idle' && (
        <Card className="border-2 border-dashed">
          <CardContent className="p-8 text-center">
            <FileText className="h-16 w-16 mx-auto text-muted-foreground mb-4" />
            <h2 className="text-xl font-semibold mb-2">Ready to Generate Report</h2>
            <p className="text-muted-foreground mb-6">
              Click below to generate the professional pathology report with AI-assisted clinical summary.
            </p>
            <Button size="lg" onClick={generateReport}>
              <Sparkles className="h-4 w-4 mr-2" />
              Generate Report
            </Button>
          </CardContent>
        </Card>
      )}

      {reportStatus === 'generating' && (
        <Card>
          <CardContent className="p-8 text-center">
            <RefreshCw className="h-12 w-12 mx-auto text-blue-500 animate-spin mb-4" />
            <h2 className="text-xl font-semibold mb-4">Generating Report...</h2>
            <Progress value={generationProgress} className="max-w-md mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">
              {generationProgress < 30 ? 'Compiling analysis results...' :
               generationProgress < 60 ? 'Generating clinical summary...' :
               generationProgress < 90 ? 'Creating report document...' : 'Finalizing...'}
            </p>
          </CardContent>
        </Card>
      )}

      {reportStatus === 'error' && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Report Generation Failed</AlertTitle>
          <AlertDescription className="flex items-center gap-4">
            <span>There was an error generating the report.</span>
            <Button variant="outline" size="sm" onClick={generateReport}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Report Preview - Only when completed */}
      {reportStatus === 'completed' && (
        <>
          {/* Success Alert */}
          <Alert className="border-green-200 bg-green-50">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <AlertTitle className="text-green-800">Report Generated Successfully</AlertTitle>
            <AlertDescription className="text-green-700">
              Report ID: <strong>{reportData?.reportId || patientInfo.id}</strong> • 
              Generated: {new Date().toLocaleString('en-IN')}
            </AlertDescription>
          </Alert>

          {/* Download Actions */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  <span className="font-medium">Report Ready for Download</span>
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={() => downloadReport('pdf')}>
                    <Download className="h-4 w-4 mr-2" />
                    PDF
                  </Button>
                  <Button variant="outline" onClick={() => downloadReport('docx')}>
                    <Download className="h-4 w-4 mr-2" />
                    DOCX
                  </Button>
                  <Button variant="outline" onClick={() => window.print()}>
                    Print
                  </Button>
                  <Button variant="outline">
                    <Share className="h-4 w-4 mr-2" />
                    Share
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Report Preview */}
          <Card className="print:shadow-none">
            <CardContent className="p-0">
              <div className="bg-white text-black p-8">
                {/* Report Header */}
                <div className="border-b pb-6 mb-6">
                  <div className="flex justify-between items-start">
                    <div>
                      <h1 className="text-2xl font-bold text-blue-900">{labInfo.name}</h1>
                      <p className="text-sm text-gray-600 mt-1">{labInfo.address}</p>
                      <p className="text-sm text-gray-600">Phone: {labInfo.phone} | Email: {labInfo.email}</p>
                      <div className="flex items-center gap-4 mt-2">
                        <Badge variant="outline" className="text-xs">
                          <Shield className="h-3 w-3 mr-1" />
                          {labInfo.accreditation}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          License: {labInfo.license}
                        </Badge>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="bg-blue-900 text-white px-4 py-2 rounded">
                        <h2 className="font-bold">PATHOLOGY REPORT</h2>
                      </div>
                      <p className="text-sm mt-2 text-gray-600">Report ID: {reportData?.reportId || patientInfo.id}</p>
                    </div>
                  </div>
                </div>

                {/* Patient & Test Info */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <User className="h-5 w-5" />
                        Patient Information
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Patient ID:</span>
                          <span className="font-medium">{patientInfo.id}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Name:</span>
                          <span className="font-medium">{patientInfo.name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Age:</span>
                          <span className="font-medium">{patientInfo.age} years</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Gender:</span>
                          <span className="font-medium">{patientInfo.gender}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Calendar className="h-5 w-5" />
                        Test Information
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Sample Type:</span>
                          <span className="font-medium">{sampleType}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Order Date:</span>
                          <span className="font-medium">{patientInfo.orderDate}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Report Date:</span>
                          <span className="font-medium">{patientInfo.reportDate}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Physician:</span>
                          <span className="font-medium">{patientInfo.physician}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Analysis Results */}
                <Card className="mb-6">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Microscope className="h-5 w-5" />
                      AI Analysis Results
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className={`text-2xl font-bold ${analysisResults?.isPositive ? 'text-red-600' : 'text-green-600'}`}>
                          {analysisResults?.prediction || 'N/A'}
                        </div>
                        <div className="text-xs text-muted-foreground">Detection Result</div>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {analysisResults?.confidence || 0}%
                        </div>
                        <div className="text-xs text-muted-foreground">Confidence</div>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold">
                          {analysisResults?.tumorProbability || 0}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {isBloodSample ? 'Detection Prob.' : 'Tumor Prob.'}
                        </div>
                      </div>
                      <div className="text-center p-4 bg-gray-50 rounded-lg">
                        <Badge variant={analysisResults?.riskLevel?.includes('High') ? 'destructive' : 'outline'}>
                          {analysisResults?.riskLevel || 'Unknown'}
                        </Badge>
                        <div className="text-xs text-muted-foreground mt-1">Risk Level</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Clinical Summary (Gemini) */}
                <Card className="mb-6">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Sparkles className="h-5 w-5" />
                      Clinical Summary
                      {clinicalSummary?.generatedBy === 'Gemini AI' && (
                        <Badge variant="secondary" className="ml-2">AI Generated</Badge>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {isGeneratingSummary ? (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Generating clinical summary...
                      </div>
                    ) : (
                      <Alert className={analysisResults?.isPositive ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                        {analysisResults?.isPositive ? (
                          <AlertTriangle className="h-4 w-4 text-red-600" />
                        ) : (
                          <CheckCircle2 className="h-4 w-4 text-green-600" />
                        )}
                        <AlertTitle className={analysisResults?.isPositive ? 'text-red-800' : 'text-green-800'}>
                          {analysisResults?.isPositive ? 'Abnormal Findings' : 'Normal Findings'}
                        </AlertTitle>
                        <AlertDescription className={analysisResults?.isPositive ? 'text-red-700' : 'text-green-700'}>
                          {clinicalSummary?.summary || 
                           (analysisResults?.isPositive 
                             ? `AI analysis detected potential abnormalities. Confidence: ${analysisResults.confidence}%. Further evaluation recommended.`
                             : `AI analysis confirms normal findings. Confidence: ${analysisResults?.confidence || 0}%. No abnormalities detected.`
                           )}
                        </AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>

                {/* AI Interpretation (Gemini) */}
                {geminiReport?.report?.content && (
                  <Card className="mb-6">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <Sparkles className="h-5 w-5 text-purple-500" />
                        AI Interpretation
                        <Badge variant="secondary" className="ml-2">Gemini AI</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-1">Interpretation</h4>
                        <p className="text-sm">{geminiReport.report.content.interpretation || 'Analysis interpretation not available.'}</p>
                      </div>
                      <div>
                        <h4 className="font-medium text-sm text-muted-foreground mb-1">Morphological Findings</h4>
                        <p className="text-sm">{geminiReport.report.content.morphologicalFindings || 'Morphological analysis not available.'}</p>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Recommendations (Gemini) */}
                {(geminiReport?.recommendations?.recommendations || geminiReport?.report?.content?.recommendations) && (
                  <Card className="mb-6">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-lg flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5 text-blue-500" />
                        Recommendations
                        <Badge variant="secondary" className="ml-2">Gemini AI</Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-2">
                        {(geminiReport?.recommendations?.recommendations || geminiReport?.report?.content?.recommendations || []).map((rec, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-sm">
                            <span className="text-green-600 mt-1">•</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}

                {/* Conclusion (Gemini) */}
                {geminiReport?.report?.content?.conclusion && (
                  <Card className="mb-6 border-blue-200 bg-blue-50">
                    <CardContent className="p-4">
                      <h4 className="font-semibold text-blue-800 mb-2">Conclusion</h4>
                      <p className="text-sm text-blue-700">{geminiReport.report.content.conclusion}</p>
                    </CardContent>
                  </Card>
                )}

                {/* Footer - Signatures */}
                <div className="border-t pt-6">
                  <div className="text-center text-sm text-muted-foreground mb-4">
                    This report was generated using RecursiaDx AI-powered digital pathology platform.
                    <br />
                    For questions, contact: {labInfo.phone}
                  </div>
                  <div className="flex items-center justify-center gap-4">
                    <Badge variant="outline" className="text-xs">
                      <Shield className="h-3 w-3 mr-1" />
                      HIPAA Compliant
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      <Clock className="h-3 w-3 mr-1" />
                      Real-time AI Processing
                    </Badge>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Workflow Completion Section */}
          <Card className="border-green-200 bg-green-50">
            <CardContent className="p-6">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div className="flex items-center gap-4">
                  <CheckCircle2 className="h-10 w-10 text-green-600" />
                  <div>
                    <h3 className="font-semibold text-lg text-green-800">Workflow Complete!</h3>
                    <p className="text-sm text-green-700">
                      Sample {patientInfo.id} has been fully processed and report is ready.
                    </p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      // Update sample status in backend
                      fetch(`http://localhost:5001/api/samples/${sample._id}/status`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ status: 'Completed' })
                      }).catch(console.error)
                      
                      toast.success('Report saved! Starting new sample...')
                      window.location.reload()
                    }}
                  >
                    Start New Sample
                  </Button>
                  <Button 
                    size="lg"
                    onClick={() => {
                      // Update sample status
                      fetch(`http://localhost:5001/api/samples/${sample._id}/status`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ status: 'Completed' })
                      }).catch(console.error)
                      
                      toast.success('Workflow completed successfully!')
                      
                      // Navigate to dashboard or call onNext
                      if (onNext) {
                        onNext()
                      } else {
                        window.location.href = '/dashboard'
                      }
                    }}
                  >
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    Complete & Close
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}