// Quick test of the heatmap API endpoint
fetch('http://localhost:5001/api/samples/heatmaps/list')
  .then(response => response.json())
  .then(data => {
    console.log('🔍 Heatmap API response:', data)
    if (data.heatmaps && data.heatmaps.length > 0) {
      console.log('✅ Found heatmaps via API:', data.heatmaps.length)
      data.heatmaps.forEach((heatmap, index) => {
        console.log(`🎨 Heatmap ${index + 1}:`, heatmap.name, heatmap.src)
      })
    } else {
      console.log('❌ No heatmaps found via API')
    }
  })
  .catch(error => {
    console.error('❌ API Error:', error)
  })

// Also test direct file access
const testImage = 'http://localhost:5001/uploads/heatmaps/auto_heatmap_images-1760057547861-895269426.jpg_1760057548057.png'
const img = new Image()
img.onload = () => console.log('✅ Direct image access works:', testImage)
img.onerror = () => console.log('❌ Direct image access failed:', testImage)
img.src = testImage