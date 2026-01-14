import React, { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'sonner'
import { 
  Thermometer, 
  Palette, 
  Download, 
  Upload,
  Zap,
  BarChart3,
  RefreshCw,
  Image as ImageIcon
} from 'lucide-react'

export function HeatmapViewer({ sample, className = "" }) {
  const [heatmapData, setHeatmapData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [heatmapType, setHeatmapType] = useState('tumor_probability')
  const [colormap, setColormap] = useState('hot')
  const [selectedImage, setSelectedImage] = useState(null)
  const fileInputRef = useRef(null)

  // Available heatmap types
  const heatmapTypes = [
    { value: 'tumor_probability', label: 'Tumor Probability', icon: '🎯' },
    { value: 'confidence', label: 'Confidence Level', icon: '📊' },
    { value: 'risk_score', label: 'Risk Score', icon: '⚠️' }
  ]

  // Available colormaps
  const colormaps = [
    { value: 'hot', label: 'Hot (Medical)', description: 'Classic medical imaging' },
    { value: 'viridis', label: 'Viridis', description: 'Perceptually uniform' },
    { value: 'plasma', label: 'Plasma', description: 'High contrast' },
    { value: 'inferno', label: 'Inferno', description: 'Dark background' },
    { value: 'jet', label: 'Jet', description: 'Rainbow colors' },
    { value: 'coolwarm', label: 'Cool-Warm', description: 'Blue to red' }
  ]

  // Generate heatmap
  const generateHeatmap = async (imageFile = null, imagePath = null) => {
    setLoading(true)
    
    try {
      const formData = new FormData()
      
      if (imageFile) {
        formData.append('image', imageFile)
        console.log('📁 Using uploaded file for heatmap')
      } else if (imagePath) {
        formData.append('imagePath', imagePath)
        console.log('🔗 Using existing image path:', imagePath)
      } else {
        toast.error('No image selected for heatmap generation')
        setLoading(false)
        return
      }
      
      formData.append('heatmapType', heatmapType)
      formData.append('colormap', colormap)
      
      console.log('🎨 Generating heatmap...')
      console.log(`   Type: ${heatmapType}`)
      console.log(`   Colormap: ${colormap}`)
      
      const response = await fetch('/api/samples/generate-heatmap', {
        method: 'POST',
        body: formData
      })
      
      const result = await response.json()
      
      if (result.success) {
        setHeatmapData(result.data)
        toast.success('Heatmap generated successfully!')
        console.log('✅ Heatmap data received:', result.data)
      } else {
        console.error('❌ Heatmap generation failed:', result.error)
        toast.error(`Heatmap generation failed: ${result.error}`)
      }
      
    } catch (error) {
      console.error('❌ Error generating heatmap:', error)
      toast.error('Failed to generate heatmap. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedImage(file)
      console.log('📷 Image selected:', file.name)
      toast.success(`Image selected: ${file.name}`)
    }
  }

  // Use sample image
  const useSampleImage = (imageData) => {
    if (imageData && imageData.filename) {
      const imagePath = `/uploads/${imageData.filename}`
      generateHeatmap(null, imagePath)
    }
  }

  // Download heatmap
  const downloadHeatmap = () => {
    if (heatmapData && heatmapData.heatmap.image_base64) {
      const link = document.createElement('a')
      link.href = heatmapData.heatmap.image_base64
      link.download = `heatmap_${heatmapType}_${colormap}_${Date.now()}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      toast.success('Heatmap downloaded!')
    }
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Thermometer className="h-5 w-5 text-orange-500" />
            AI Heatmap Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Heatmap Type Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Analysis Type</label>
              <Select value={heatmapType} onValueChange={setHeatmapType}>
                <SelectTrigger>
                  <SelectValue placeholder="Select analysis type" />
                </SelectTrigger>
                <SelectContent>
                  {heatmapTypes.map((type) => (
                    <SelectItem key={type.value} value={type.value}>
                      <div className="flex items-center gap-2">
                        <span>{type.icon}</span>
                        <span>{type.label}</span>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Colormap Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Color Scheme</label>
              <Select value={colormap} onValueChange={setColormap}>
                <SelectTrigger>
                  <SelectValue placeholder="Select colormap" />
                </SelectTrigger>
                <SelectContent>
                  {colormaps.map((cmap) => (
                    <SelectItem key={cmap.value} value={cmap.value}>
                      <div className="space-y-1">
                        <div className="font-medium">{cmap.label}</div>
                        <div className="text-xs text-muted-foreground">{cmap.description}</div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Generate Button */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Generate</label>
              <Button
                onClick={() => {
                  if (selectedImage) {
                    generateHeatmap(selectedImage)
                  } else {
                    toast.error('Please select an image first')
                  }
                }}
                disabled={loading || !selectedImage}
                className="w-full"
              >
                {loading ? (
                  <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                ) : (
                  <Zap className="h-4 w-4 mr-2" />
                )}
                {loading ? 'Generating...' : 'Generate Heatmap'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Image Selection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Upload New Image */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload Image
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="hidden"
              />
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                className="w-full"
              >
                <ImageIcon className="h-4 w-4 mr-2" />
                Choose Image File
              </Button>
              {selectedImage && (
                <div className="text-sm text-green-600">
                  ✅ Selected: {selectedImage.name}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Use Sample Images */}
        {sample && sample.images && sample.images.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Sample Images
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {sample.images.slice(0, 3).map((image, index) => (
                  <Button
                    key={index}
                    variant="outline"
                    onClick={() => useSampleImage(image)}
                    className="w-full justify-start"
                    disabled={loading}
                  >
                    <ImageIcon className="h-4 w-4 mr-2" />
                    Image {index + 1} ({image.filename})
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Heatmap Display */}
      {(loading || heatmapData) && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <Palette className="h-5 w-5 text-purple-500" />
                Generated Heatmap
              </CardTitle>
              {heatmapData && (
                <Button onClick={downloadHeatmap} size="sm" variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-4">
                <Skeleton className="h-64 w-full" />
                <div className="text-center text-sm text-muted-foreground">
                  Generating heatmap... This may take a few moments.
                </div>
              </div>
            ) : heatmapData ? (
              <div className="space-y-4">
                {/* Heatmap Image */}
                <div className="flex justify-center">
                  <img
                    src={heatmapData.heatmap.image_base64}
                    alt={`${heatmapData.heatmap.type} heatmap`}
                    className="max-w-full h-auto border rounded-lg shadow-sm"
                  />
                </div>

                {/* Heatmap Info */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-sm text-muted-foreground">Type</div>
                    <Badge variant="secondary">
                      {heatmapData.heatmap.type.replace('_', ' ')}
                    </Badge>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-muted-foreground">Colormap</div>
                    <Badge variant="secondary">
                      {heatmapData.heatmap.colormap}
                    </Badge>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-muted-foreground">Processing</div>
                    <div className="text-sm font-medium">
                      {(heatmapData.processing_time / 1000).toFixed(1)}s
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-muted-foreground">Status</div>
                    <Badge variant="default" className="bg-green-500">
                      ✅ Complete
                    </Badge>
                  </div>
                </div>

                {/* Analytics */}
                {heatmapData.heatmap.analytics && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Analytics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                        {Object.entries(heatmapData.heatmap.analytics).map(([key, value]) => (
                          <div key={key} className="text-center">
                            <div className="text-muted-foreground capitalize">
                              {key.replace('_', ' ')}
                            </div>
                            <div className="font-medium">
                              {typeof value === 'number' ? value.toFixed(3) : value}
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* Instructions */}
      {!heatmapData && !loading && (
        <Alert>
          <Thermometer className="h-4 w-4" />
          <AlertDescription>
            <strong>How to generate heatmaps:</strong>
            <ol className="list-decimal list-inside mt-2 space-y-1 text-sm">
              <li>Select your preferred analysis type (tumor probability, confidence, or risk score)</li>
              <li>Choose a color scheme that suits your needs</li>
              <li>Upload a new image or use one from your current sample</li>
              <li>Click "Generate Heatmap" to create the visualization</li>
            </ol>
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}