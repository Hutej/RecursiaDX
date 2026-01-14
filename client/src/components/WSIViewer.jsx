import React, { useState, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { ImageCard } from '@/components/ui/image-card'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from '@/components/ui/carousel'
import { Drawer, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle, DrawerTrigger } from '@/components/ui/drawer'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'sonner'
import { 
  ZoomIn, 
  ZoomOut, 
  Move, 
  RotateCw, 
  Maximize2, 
  Minimize2,
  Download, 
  Info,
  Eye,
  Grid3X3,
  Layers,
  Target,
  Upload,
  Plus,
  X,
  Undo2,
  Redo2
} from 'lucide-react'

export function WSIViewer({ onNext, sample, onSampleUpdated }) {
  const [selectedImage, setSelectedImage] = useState(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isLoading, setIsLoading] = useState(false)
  const [showGrid, setShowGrid] = useState(false)
  const [annotations, setAnnotations] = useState([])
  const [uploadedImages, setUploadedImages] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [activeTool, setActiveTool] = useState('pan') // 'pan' or 'annotate'
  const [annotationColor, setAnnotationColor] = useState('red') // 'red', 'black', 'yellow'
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [undoStack, setUndoStack] = useState([]) // Stack for redo
  const [isFullscreen, setIsFullscreen] = useState(false) // Fullscreen toggle
  const canvasRef = useRef(null)
  const viewerRef = useRef(null)
  const fileInputRef = useRef(null)

  // Debug: Log sample data
  console.log('🔍 WSIViewer received sample:', sample)
  console.log('🔍 Sample images:', sample?.images)
  
  // Alert for debugging in browser
  if (sample) {
    console.log('✅ WSI Viewer has sample data')
    if (sample.images && sample.images.length > 0) {
      console.log('✅ WSI Viewer has images:', sample.images.length)
      sample.images.forEach((img, index) => {
        console.log(`📸 Image ${index + 1}:`, {
          url: img.url,
          filename: img.filename,
          hasML: !!img.mlAnalysis
        })
      })
    } else {
      console.log('❌ WSI Viewer: No images in sample')
    }
  } else {
    console.log('❌ WSI Viewer: No sample data received')
  }

  // Handle file upload
  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files)
    if (files.length === 0) return

    setIsUploading(true)
    
    try {
      const newImages = []
      
      for (const file of files) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp']
        if (!validTypes.includes(file.type)) {
          toast.error(`${file.name}: Invalid file type. Please upload images only.`)
          continue
        }

        // Validate file size (max 50MB)
        if (file.size > 50 * 1024 * 1024) {
          toast.error(`${file.name}: File size exceeds 50MB limit.`)
          continue
        }

        // Create a preview URL
        const previewUrl = URL.createObjectURL(file)
        
        const newImage = {
          id: `upload-${Date.now()}-${Math.random()}`,
          name: file.name,
          type: 'uploaded',
          resolution: 'N/A',
          size: `${file.size} bytes`,
          staining: 'Unknown',
          thumbnail: previewUrl,
          fullImage: previewUrl,
          analysis: 'pending',
          file: file,
          uploadedAt: new Date().toISOString()
        }
        
        newImages.push(newImage)
      }

      setUploadedImages(prev => [...prev, ...newImages])
      toast.success(`Successfully added ${newImages.length} image(s)`)
      
      // Automatically select the first uploaded image
      if (newImages.length > 0 && !selectedImage) {
        handleImageSelect(newImages[0])
      }
      
    } catch (error) {
      console.error('Upload error:', error)
      toast.error('Failed to add images')
  // Combine sample images with uploaded images
  const allImages = [...wsiImages, ...uploadedImages]

    } finally {
      setIsUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const handleRemoveImage = (imageId) => {
    setUploadedImages(prev => prev.filter(img => img.id !== imageId))
    if (selectedImage?.id === imageId) {
      setSelectedImage(null)
    }
    toast.info('Image removed')
  }

  // Use real sample images or mock data
  const wsiImages = sample?.images?.length > 0 ? sample.images.map(img => {
    console.log('🔍 Processing image:', img)
    const imageData = {
      id: img._id || img.filename,
      name: img.originalName || img.filename,
      type: sample.sampleType?.toLowerCase() || 'unknown',
      resolution: img.magnification || 'Unknown',
      size: `${Math.floor(Math.random() * 2048)}x${Math.floor(Math.random() * 1536)}`, // Mock size for now
      staining: img.staining || 'H&E',
      thumbnail: `http://localhost:5001${img.url}`,
      fullImage: `http://localhost:5001${img.url}`,
      analysis: img.mlAnalysis ? 'completed' : 'pending',
      mlAnalysis: img.mlAnalysis,
      uploadedAt: img.uploadedAt
    }
    console.log('🔍 Mapped image data:', imageData)
    return imageData
  }) : []

  // Combine sample images with uploaded images
  const allImages = [...wsiImages, ...uploadedImages]

  const handleImageSelect = (image) => {
    setIsLoading(true)
    setSelectedImage(image)
    setAnnotations([]) // Clear annotations when switching images
    setZoomLevel(1) // Reset zoom level
    setPosition({ x: 0, y: 0 }) // Reset position
    
    // Simulate image loading
    setTimeout(() => {
      setIsLoading(false)
      toast.success(`Loaded ${image.name}`)
    }, 1500)
  }

  const handleZoom = (direction) => {
    if (direction === 'in') {
      setZoomLevel(prev => Math.min(prev + 0.25, 5))
    } else {
      setZoomLevel(prev => Math.max(prev - 0.25, 0.25))
    }
  }

  const handleAnnotation = (e) => {
    if (activeTool !== 'annotate') return
    
    // Get the viewer container dimensions
    const rect = viewerRef.current?.getBoundingClientRect()
    if (!rect) return
    
    // Calculate click position relative to the viewer container
    const clickX = e.clientX - rect.left
    const clickY = e.clientY - rect.top
    
    // Convert viewport coordinates to image coordinates
    // Account for the transform (scale and translate)
    const imageX = (clickX / zoomLevel) - position.x
    const imageY = (clickY / zoomLevel) - position.y
    
    const newAnnotation = {
      id: Date.now(),
      x: imageX,
      y: imageY,
      color: annotationColor,
      type: 'marker',
      note: `Annotation at ${Math.round(imageX)}, ${Math.round(imageY)}`
    }
    setAnnotations(prev => [...prev, newAnnotation])
    setUndoStack([]) // Clear redo stack when new annotation added
    toast.info(`${annotationColor.charAt(0).toUpperCase() + annotationColor.slice(1)} annotation added`)
  }

  // Undo last annotation
  const handleUndo = () => {
    if (annotations.length === 0) {
      toast.error('Nothing to undo')
      return
    }
    const lastAnnotation = annotations[annotations.length - 1]
    setAnnotations(prev => prev.slice(0, -1))
    setUndoStack(prev => [...prev, lastAnnotation])
    toast.info('Annotation undone')
  }

  // Redo last undone annotation
  const handleRedo = () => {
    if (undoStack.length === 0) {
      toast.error('Nothing to redo')
      return
    }
    const lastUndone = undoStack[undoStack.length - 1]
    setUndoStack(prev => prev.slice(0, -1))
    setAnnotations(prev => [...prev, lastUndone])
    toast.info('Annotation restored')
  }

  const handleMouseDown = (e) => {
    if (activeTool === 'pan') {
      setIsDragging(true)
      setDragStart({
        x: e.clientX - position.x,
        y: e.clientY - position.y
      })
    }
  }

  const handleMouseMove = (e) => {
    if (isDragging && activeTool === 'pan') {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      })
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const proceedToAnalysis = async () => {
    if (!selectedImage) {
      toast.error("Please select an image to analyze")
      return
    }
    
    // Check if we have uploaded images that need analysis
    const imagesToAnalyze = uploadedImages.filter(img => img.file)
    
    if (imagesToAnalyze.length === 0) {
      // No new uploads, just proceed with existing sample data
      toast.success("Proceeding to AI Analysis...")
      setTimeout(() => onNext(), 1500)
      return
    }
    
    // Analyze uploaded images with ML
    try {
      setIsLoading(true)
      toast.info(`Analyzing ${imagesToAnalyze.length} image(s) with ML model...`)
      
      const formData = new FormData()
      
      // Add all uploaded image files
      imagesToAnalyze.forEach((img, index) => {
        formData.append('images', img.file)
      })
      
      // Add patient info if available from sample
      if (sample?.patientInfo) {
        formData.append('patientInfo', JSON.stringify(sample.patientInfo))
      } else {
        formData.append('patientInfo', JSON.stringify({
          name: 'Unknown Patient',
          age: 0,
          gender: 'Unknown'
        }))
      }
      
      // Call backend ML analysis endpoint
      const response = await fetch('http://localhost:5001/api/samples/upload-with-analysis', {
        method: 'POST',
        body: formData
      })
      
      if (!response.ok) {
        let errorMessage = 'Analysis failed'
        try {
          const errorData = await response.json()
          errorMessage = errorData.error || errorData.message || errorMessage
        } catch (parseError) {
          // If response is not JSON, use status text
          errorMessage = `Server error: ${response.statusText || response.status}`
        }
        throw new Error(errorMessage)
      }
      
      const result = await response.json()
      console.log('✅ ML Analysis complete:', result)
      
      // Update the sample with analysis results
      if (result.sample) {
        let updatedSample
        if (sample) {
          // Merge with existing sample
          updatedSample = {
            ...sample,
            images: [...(sample.images || []), ...(result.sample.images || [])]
          }
        } else {
          // Use the new sample
          updatedSample = result.sample
        }
        
        console.log('🔬 Updated sample with ML results:', updatedSample)
        
        // Pass updated sample back to parent
        if (onSampleUpdated) {
          onSampleUpdated(updatedSample)
        }
      }
      
      toast.success(`Analysis complete! ${result.sample?.images?.length || 0} images analyzed.`)
      setTimeout(() => onNext(), 1500)
      
    } catch (error) {
      console.error('Analysis error:', error)
      toast.error(`Analysis failed: ${error.message}`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="grid gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">WSI Viewer & Analysis</h1>
          <p className="text-muted-foreground">
            Step 2: Whole Slide Image viewing and preparation
          </p>
        </div>
        <Badge variant="outline" className="text-sm">
          <Eye className="h-4 w-4 mr-2" />
          Image Analysis
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Image Gallery */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg">Available Images</CardTitle>
                  <CardDescription>
                    Select an image to view and analyze
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {/* Upload Button */}
                <div className="border-2 border-dashed rounded-lg p-4 text-center hover:border-primary/50 transition-colors">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/jpg,image/png,image/tiff,image/bmp"
                    multiple
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                  >
                    {isUploading ? (
                      <>
                        <Skeleton className="h-4 w-4 mr-2" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Plus className="h-4 w-4 mr-2" />
                        Add New Image
                      </>
                    )}
                  </Button>
                  <p className="text-xs text-muted-foreground mt-2">
                    JPG, PNG, TIFF, BMP up to 50MB
                  </p>
                </div>

                {allImages.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Upload className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No images available</p>
                    <p className="text-xs">Upload an image to get started</p>
                  </div>
                ) : (
                  allImages.map((image) => (
                    <div
                      key={image.id}
                      className={`relative border rounded-lg p-3 cursor-pointer transition-all ${
                        selectedImage?.id === image.id 
                          ? 'border-primary bg-primary/5' 
                          : 'border-muted hover:border-primary/50'
                      }`}
                      onClick={() => handleImageSelect(image)}
                    >
                      {/* Remove button for uploaded images */}
                      {image.type === 'uploaded' && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="absolute top-1 right-1 z-10 h-6 w-6 p-0 hover:bg-destructive hover:text-destructive-foreground"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleRemoveImage(image.id)
                          }}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      )}
                      
                      <div className="aspect-video bg-muted rounded mb-2 flex items-center justify-center">
                        <ImageCard
                          src={image.thumbnail}
                          alt={image.name}
                          className="w-full h-full object-cover rounded"
                        />
                      </div>
                      
                      <div className="space-y-1">
                        <p className="text-sm font-medium truncate">{image.name}</p>
                        <div className="flex items-center gap-1 flex-wrap">
                          <Badge variant="secondary" className="text-xs">
                            {image.type}
                          </Badge>
                          {image.resolution && image.resolution !== 'N/A' && (
                            <Badge variant="outline" className="text-xs">
                              {image.resolution}
                            </Badge>
                          )}
                          {image.analysis && image.analysis !== 'pending' && (
                            <Badge 
                              variant={image.analysis === 'completed' ? 'default' : 'secondary'}
                              className="text-xs"
                            >
                              {image.analysis}
                            </Badge>
                          )}
                        </div>
                        {image.staining && image.staining !== 'Unknown' && (
                          <p className="text-xs text-muted-foreground">
                            {image.staining}
                          </p>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Viewer */}
        <div className="lg:col-span-3">
          <Card className="h-full">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>
                    {selectedImage ? selectedImage.name : 'Select an image to view'}
                  </CardTitle>
                  {selectedImage && (
                    <CardDescription>
                      Resolution: {selectedImage.resolution} • Staining: {selectedImage.staining}
                    </CardDescription>
                  )}
                </div>
                
                <div className="flex items-center gap-2">
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => setShowGrid(!showGrid)}
                        >
                          <Grid3X3 className="h-4 w-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Toggle Grid</TooltipContent>
                    </Tooltip>
                  </TooltipProvider>

                  <Drawer>
                    <DrawerTrigger asChild>
                      <Button variant="outline" size="sm">
                        <Info className="h-4 w-4" />
                      </Button>
                    </DrawerTrigger>
                    <DrawerContent>
                      <DrawerHeader>
                        <DrawerTitle>Image Information</DrawerTitle>
                        <DrawerDescription>
                          Detailed information about the selected image
                        </DrawerDescription>
                      </DrawerHeader>
                      <div className="p-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <p className="text-sm font-medium">File Size</p>
                          <p className="text-sm text-muted-foreground">45.2 MB</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Color Space</p>
                          <p className="text-sm text-muted-foreground">RGB</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Compression</p>
                          <p className="text-sm text-muted-foreground">JPEG2000</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Levels</p>
                          <p className="text-sm text-muted-foreground">6 pyramid levels</p>
                        </div>
                      </div>
                    </DrawerContent>
                  </Drawer>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-0">
              {selectedImage ? (
                <div className="relative">
                  {/* Toolbar */}
                  <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-background/95 backdrop-blur-sm rounded-lg p-2 border">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="outline" size="sm" onClick={() => handleZoom('in')}>
                            <ZoomIn className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Zoom In</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    <span className="text-sm font-mono px-2">
                      {(zoomLevel * 100).toFixed(0)}%
                    </span>

                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button variant="outline" size="sm" onClick={() => handleZoom('out')}>
                            <ZoomOut className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Zoom Out</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    <div className="h-4 w-px bg-border" />

                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button 
                            variant={activeTool === 'pan' ? 'default' : 'outline'} 
                            size="sm"
                            onClick={() => setActiveTool('pan')}
                          >
                            <Move className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Pan Tool (Drag to move)</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button 
                            variant={activeTool === 'annotate' ? 'default' : 'outline'} 
                            size="sm"
                            onClick={() => setActiveTool('annotate')}
                          >
                            <Target className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Annotation Tool (Click to mark)</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    {/* Color Picker for Annotations */}
                    {activeTool === 'annotate' && (
                      <>
                        <div className="h-4 w-px bg-border" />
                        <div className="flex items-center gap-1">
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button 
                                  variant={annotationColor === 'red' ? 'default' : 'outline'} 
                                  size="sm"
                                  className="w-8 h-8 p-0"
                                  onClick={() => setAnnotationColor('red')}
                                >
                                  <div className="w-4 h-4 rounded-full bg-red-500 border-2 border-white" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>Red Marker</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>

                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button 
                                  variant={annotationColor === 'black' ? 'default' : 'outline'} 
                                  size="sm"
                                  className="w-8 h-8 p-0"
                                  onClick={() => setAnnotationColor('black')}
                                >
                                  <div className="w-4 h-4 rounded-full bg-black border-2 border-white" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>Black Marker</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>

                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button 
                                  variant={annotationColor === 'yellow' ? 'default' : 'outline'} 
                                  size="sm"
                                  className="w-8 h-8 p-0"
                                  onClick={() => setAnnotationColor('yellow')}
                                >
                                  <div className="w-4 h-4 rounded-full bg-yellow-400 border-2 border-white" />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent>Yellow Marker</TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </div>
                      </>
                    )}

                    {/* Undo/Redo Buttons */}
                    <div className="h-4 w-px bg-border" />
                    
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={handleUndo}
                            disabled={annotations.length === 0}
                          >
                            <Undo2 className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Undo Annotation</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={handleRedo}
                            disabled={undoStack.length === 0}
                          >
                            <Redo2 className="h-4 w-4" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>Redo Annotation</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>

                    {/* Fullscreen Toggle */}
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button 
                            variant="outline" 
                            size="sm"
                            onClick={() => setIsFullscreen(!isFullscreen)}
                          >
                            {isFullscreen ? (
                              <Minimize2 className="h-4 w-4" />
                            ) : (
                              <Maximize2 className="h-4 w-4" />
                            )}
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>{isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}</TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>

                  {/* Image Viewer Area */}
                  <div 
                    ref={viewerRef}
                    className={`relative ${isFullscreen ? 'h-[80vh]' : 'h-96'} bg-muted/50 overflow-hidden ${activeTool === 'pan' ? 'cursor-move' : 'cursor-crosshair'} transition-all duration-300`}
                    onClick={(e) => {
                      if (activeTool === 'annotate') {
                        handleAnnotation(e)
                      }
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                  >
                    {isLoading ? (
                      <div className="flex items-center justify-center h-full">
                        <div className="space-y-4 w-full max-w-md p-4">
                          <Skeleton className="h-8 w-3/4" />
                          <Skeleton className="h-64 w-full" />
                          <div className="flex gap-2">
                            <Skeleton className="h-4 w-20" />
                            <Skeleton className="h-4 w-16" />
                            <Skeleton className="h-4 w-24" />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div 
                        className="w-full h-full bg-white flex items-center justify-center"
                        style={{
                          transform: `scale(${zoomLevel}) translate(${position.x}px, ${position.y}px)`,
                          transformOrigin: 'center'
                        }}
                      >
                        <ImageCard
                          src={selectedImage.fullImage}
                          alt={selectedImage.name}
                          className="max-w-full max-h-full object-contain"
                        />
                        
                        {/* Grid Overlay */}
                        {showGrid && (
                          <div className="absolute inset-0 pointer-events-none">
                            <svg className="w-full h-full">
                              <defs>
                                <pattern
                                  id="grid"
                                  width="50"
                                  height="50"
                                  patternUnits="userSpaceOnUse"
                                >
                                  <path
                                    d="M 50 0 L 0 0 0 50"
                                    fill="none"
                                    stroke="rgba(0,0,0,0.2)"
                                    strokeWidth="1"
                                  />
                                </pattern>
                              </defs>
                              <rect width="100%" height="100%" fill="url(#grid)" />
                            </svg>
                          </div>
                        )}

                        {/* Annotations - positioned relative to image, transforms with it */}
                        {annotations.map((annotation) => {
                          const colorClass = annotation.color === 'black' 
                            ? 'bg-black' 
                            : annotation.color === 'yellow' 
                              ? 'bg-yellow-400' 
                              : 'bg-red-500'
                          return (
                            <div
                              key={annotation.id}
                              className={`absolute w-4 h-4 ${colorClass} rounded-full border-2 border-white shadow-lg pointer-events-none z-20`}
                              style={{
                                left: `${annotation.x}px`,
                                top: `${annotation.y}px`,
                                transform: 'translate(-50%, -50%)'
                              }}
                            />
                          )
                        })}

                      </div>
                    )}
                  </div>

                  {/* Image Navigation */}
                  <div className="p-4">
                    <Carousel className="w-full">
                      <CarouselContent>
                        {wsiImages.map((image) => (
                          <CarouselItem key={image.id} className="basis-1/4">
                            <div 
                              className={`aspect-video border rounded cursor-pointer ${
                                selectedImage?.id === image.id ? 'border-primary' : 'border-muted'
                              }`}
                              onClick={() => handleImageSelect(image)}
                            >
                              <ImageCard
                                src={image.thumbnail}
                                alt={image.name}
                                className="w-full h-full object-cover rounded"
                              />
                            </div>
                          </CarouselItem>
                        ))}
                      </CarouselContent>
                      <CarouselPrevious />
                      <CarouselNext />
                    </Carousel>
                  </div>
                </div>
              ) : (
                <div className="h-96 flex items-center justify-center text-muted-foreground">
                  <div className="text-center">
                    <Layers className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Select an image from the gallery to begin viewing</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between">
        <Button variant="outline">
          <Download className="h-4 w-4 mr-2" />
          Export Annotations
        </Button>
        
        <Button onClick={proceedToAnalysis} disabled={!selectedImage}>
          Proceed to AI Analysis
        </Button>
      </div>
    </div>
  )
}