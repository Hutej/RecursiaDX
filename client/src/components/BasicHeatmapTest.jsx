import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function BasicHeatmapTest() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>🔥 Test Heatmap Display</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="text-center">
            <h3 className="text-lg font-semibold mb-4">Generated Heatmap</h3>
            
            {/* Direct image test */}
            <img
              src="http://localhost:3000/uploads/heatmaps/auto_heatmap_images-1735569690481.jpg_1735569690503.png"
              alt="Test Heatmap"
              className="w-full max-w-md mx-auto h-auto border rounded-lg"
              onLoad={() => console.log('✅ Heatmap loaded successfully!')}
              onError={(e) => {
                console.log('❌ Failed to load heatmap')
                e.target.style.display = 'none'
              }}
            />
            
            <p className="text-sm text-gray-500 mt-2">
              Direct heatmap from: /uploads/heatmaps/
            </p>
            
            {/* Alternative test with different heatmap */}
            <div className="mt-6">
              <h4 className="font-medium mb-2">Alternative Heatmap</h4>
              <img
                src="http://localhost:3000/uploads/heatmaps/auto_heatmap_images-1735569814265.jpg_1735569814284.png"
                alt="Alternative Heatmap"
                className="w-full max-w-md mx-auto h-auto border rounded-lg"
                onLoad={() => console.log('✅ Alternative heatmap loaded!')}
                onError={(e) => {
                  console.log('❌ Failed to load alternative heatmap')
                  e.target.style.display = 'none'
                }}
              />
            </div>

            {/* Fallback message */}
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <p className="text-sm">
                ℹ️ If images don't load, check that the backend server is running on port 3000
                and the heatmap files exist in the uploads/heatmaps folder.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}