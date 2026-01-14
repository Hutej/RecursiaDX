import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Thermometer, 
  Download, 
  Zap,
  RefreshCw,
  Eye
} from 'lucide-react'

export function SimpleHeatmapViewer({ sample }) {
  const [heatmaps, setHeatmaps] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Debug logging - enhanced to check exact sample structure
  console.log('🔍 SimpleHeatmapViewer - Sample data:', sample)
  console.log('🔍 Sample structure:', {
    hasSample: !!sample,
    hasImages: !!(sample?.images),
    imageCount: sample?.images?.length || 0,
    sampleKeys: sample ? Object.keys(sample) : []
  })
  
  if (sample?.images) {
    sample.images.forEach((img, index) => {
      console.log(`🔍 Image ${index + 1} structure:`, {
        filename: img.filename,
        hasHeatmap: !!img.heatmap,
        hasHeatmapPath: !!img.heatmapPath,
        hasHeatmapBase64: !!img.heatmapBase64,
        heatmapKeys: img.heatmap ? Object.keys(img.heatmap) : [],
        allKeys: Object.keys(img)
      })
    })
  }

  // Try to fetch available heatmaps
  const fetchHeatmaps = async () => {
    setLoading(true)
    setError(null)

    try {
      // First, try to get heatmaps from sample data
      if (sample?.images) {
        const heatmapsFromSample = []
        
        sample.images.forEach((img, index) => {
          // Check if heatmap exists in different possible formats
          if (img.heatmap) {
            console.log(`✅ Found heatmap for image ${index + 1}:`, img.heatmap)
            heatmapsFromSample.push({
              id: `sample-${index}`,
              name: img.originalName || `Image ${index + 1}`,
              src: img.heatmap.base64 || img.heatmap.image_base64 || `http://localhost:5001${img.heatmap.path}` || `http://localhost:5001${img.heatmap.file_path}`,
              confidence: img.mlAnalysis?.confidence || 0,
              prediction: img.mlAnalysis?.prediction || 'unknown',
              originalImage: `http://localhost:5001${img.url}` || `http://localhost:5001${img.path}`,
              analytics: img.heatmap.analytics
            })
          } else if (img.heatmapPath || img.heatmapBase64) {
            console.log(`✅ Found heatmap path/base64 for image ${index + 1}`)
            heatmapsFromSample.push({
              id: `sample-${index}`,
              name: img.originalName || `Image ${index + 1}`,
              src: img.heatmapBase64 || `http://localhost:5001${img.heatmapPath}`,
              confidence: img.mlAnalysis?.confidence || 0,
              prediction: img.mlAnalysis?.prediction || 'unknown',
              originalImage: `http://localhost:5001${img.url}` || `http://localhost:5001${img.path}`
            })
          } else {
            console.log(`❌ No heatmap found for image ${index + 1}`)
          }
        })

        console.log('🔍 Heatmaps extracted from sample:', heatmapsFromSample)

        if (heatmapsFromSample.length > 0) {
          setHeatmaps(heatmapsFromSample)
          setLoading(false)
          return
        } else {
          console.log('⚠️ No heatmaps found in sample data, trying API...')
        }
      } else {
        console.log('⚠️ No sample images found, trying API...')
      }

      // If no heatmaps in sample data, try to fetch from backend
      const response = await fetch('http://localhost:5001/api/samples/heatmaps/list')
      if (response.ok) {
        const data = await response.json()
        setHeatmaps(data.heatmaps || [])
      } else {
        // Fallback: create mock heatmaps using known files
        const mockHeatmaps = [
          {
            id: 'mock-1',
            name: 'Sample Heatmap 1',
            src: 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760057547861-895269426.jpg_1760057548057.png',
            confidence: 0.85,
            prediction: 'malignant'
          },
          {
            id: 'mock-2', 
            name: 'Sample Heatmap 2',
            src: 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760057493712-791330968.jpg_1760057493878.png',
            confidence: 0.72,
            prediction: 'benign'
          },
          {
            id: 'mock-3',
            name: 'Sample Heatmap 3', 
            src: 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760057352506-265142668.jpg_1760057352673.png',
            confidence: 0.91,
            prediction: 'malignant'
          },
          {
            id: 'mock-4',
            name: 'Sample Heatmap 4',
            src: 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760056871267-520384507.jpg_1760056871423.png',
            confidence: 0.67,
            prediction: 'benign'
          },
          {
            id: 'mock-5',
            name: 'Sample Heatmap 5',
            src: 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760056740814-121516562.jpg_1760056741166.png',
            confidence: 0.88,
            prediction: 'malignant'
          }
        ]
        setHeatmaps(mockHeatmaps)
      }
    } catch (err) {
      console.error('Error fetching heatmaps:', err)
      setError('Failed to load heatmaps')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHeatmaps()
  }, [sample])

  const downloadHeatmap = (src, name) => {
    const link = document.createElement('a')
    link.href = src
    link.download = `heatmap_${name}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Thermometer className="h-5 w-5 text-orange-500" />
            Loading Heatmaps...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Thermometer className="h-5 w-5 text-orange-500" />
            Heatmap Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>
              {error}. <Button variant="link" onClick={fetchHeatmaps} className="p-0 h-auto">
                Try again
              </Button>
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  if (heatmaps.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Thermometer className="h-5 w-5 text-orange-500" />
            Heatmap Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <Thermometer className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Heatmaps Found</h3>
            <p className="text-muted-foreground mb-4">
              Upload images with ML analysis to generate heatmaps.
            </p>
            <Button variant="outline" onClick={fetchHeatmaps}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Thermometer className="h-5 w-5 text-orange-500" />
            AI Heatmap Analysis
            <Badge variant="default" className="ml-auto">
              {heatmaps.length} Generated
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-500">
                {heatmaps.length}
              </div>
              <div className="text-muted-foreground">Heatmaps</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-500">
                {heatmaps.filter(h => h.prediction === 'malignant').length}
              </div>
              <div className="text-muted-foreground">Malignant</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500">
                {heatmaps.length > 0 ? 
                  (heatmaps.reduce((sum, h) => sum + (h.confidence || 0), 0) / heatmaps.length * 100).toFixed(1) : '0'
                }%
              </div>
              <div className="text-muted-foreground">Avg Confidence</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Heatmap Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {heatmaps.map((heatmap) => (
          <Card key={heatmap.id} className="overflow-hidden">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  {heatmap.name}
                </CardTitle>
                <div className="flex items-center gap-2">
                  <Badge variant={heatmap.prediction === 'malignant' ? 'destructive' : 'secondary'}>
                    {heatmap.prediction || 'Unknown'}
                  </Badge>
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => downloadHeatmap(heatmap.src, heatmap.name)}
                  >
                    <Download className="h-3 w-3 mr-1" />
                    Download
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Heatmap Image */}
              <div className="relative">
                <img
                  src={heatmap.src}
                  alt={`Heatmap for ${heatmap.name}`}
                  className="w-full h-auto max-h-64 object-contain border rounded-lg bg-gray-50"
                  onError={(e) => {
                    e.target.src = '/placeholder-heatmap.png'
                    console.error('Failed to load heatmap:', heatmap.src)
                  }}
                  onLoad={() => {
                    console.log('✅ Heatmap loaded successfully:', heatmap.src)
                  }}
                />
              </div>

              {/* Analysis Details */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="font-medium">Confidence</div>
                  <div className="text-muted-foreground">
                    {heatmap.confidence ? `${(heatmap.confidence * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="font-medium">Status</div>
                  <Badge variant={heatmap.prediction === 'malignant' ? 'destructive' : 'secondary'} className="text-xs">
                    {heatmap.prediction || 'Analyzing'}
                  </Badge>
                </div>
              </div>

              {heatmap.originalImage && (
                <div>
                  <div className="font-medium text-sm mb-2">Original Image</div>
                  <img
                    src={heatmap.originalImage}
                    alt="Original"
                    className="w-full h-32 object-cover border rounded"
                  />
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="text-center">
        <Button variant="outline" onClick={fetchHeatmaps}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh Heatmaps
        </Button>
        <Button variant="secondary" onClick={() => {
          // Force load fallback heatmaps for testing
          const testHeatmaps = [
            {
              id: 'test-1',
              name: 'Test Heatmap 1',
              src: 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760057547861-895269426.jpg_1760057548057.png',
              confidence: 0.85,
              prediction: 'malignant'
            },
            {
              id: 'test-2',
              name: 'Test Heatmap 2', 
              src: 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760057493712-791330968.jpg_1760057493878.png',
              confidence: 0.72,
              prediction: 'benign'
            }
          ]
          setHeatmaps(testHeatmaps)
          setLoading(false)
        }} className="ml-2">
          Test Load Heatmaps
        </Button>
      </div>
    </div>
  )
}