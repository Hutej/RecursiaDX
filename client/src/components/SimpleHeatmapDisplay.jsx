import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export function SimpleHeatmapDisplay() {
  const [heatmaps, setHeatmaps] = useState([])

  useEffect(() => {
    // Load some known heatmaps directly
    const knownHeatmaps = [
      'auto_heatmap_images-1760057547861-895269426.jpg_1760057548057.png',
      'auto_heatmap_images-1760057493712-791330968.jpg_1760057493878.png',
      'auto_heatmap_images-1760057352506-265142668.jpg_1760057352673.png',
      'auto_heatmap_images-1760057283325-179501232.jpg_1760057283768.png',
      'auto_heatmap_images-1760057058886-613946401.jpg_1760057059008.png'
    ]

    const heatmapData = knownHeatmaps.map((filename, index) => ({
      id: index + 1,
      name: `Heatmap ${index + 1}`,
      src: `http://localhost:5001/uploads/heatmaps/${filename}`,
      filename: filename
    }))

    setHeatmaps(heatmapData)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>🔥 Generated Heatmaps</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {heatmaps.map((heatmap) => (
            <div key={heatmap.id} className="border rounded-lg p-4">
              <div className="text-sm font-medium mb-2">{heatmap.name}</div>
              <img
                src={heatmap.src}
                alt={heatmap.name}
                className="w-full h-48 object-cover rounded border"
                onError={(e) => {
                  e.target.style.display = 'none'
                  console.log('❌ Failed to load:', heatmap.src)
                }}
                onLoad={() => {
                  console.log('✅ Loaded heatmap:', heatmap.src)
                }}
              />
              <div className="text-xs text-gray-500 mt-2">
                {heatmap.filename}
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-4 text-center">
          <Button 
            variant="outline" 
            onClick={() => window.location.reload()}
          >
            Refresh Heatmaps
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}