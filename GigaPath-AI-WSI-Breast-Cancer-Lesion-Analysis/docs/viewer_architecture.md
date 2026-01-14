# Progressive WSI Heatmap Viewer Architecture

> [!WARNING]
> **Architecture specification only. No viewer code implemented.**
> Requires user approval of frontend technology stack.

---

## Overview

Proposed architecture for an interactive, Google Maps-style WSI heatmap viewer enabling:
- Progressive loading of large WSI files
- Zoom/pan navigation
- Attention heatmap overlay toggle
- Memory-efficient tile streaming

---

## Technology Recommendations

### Option 1: OpenSeadragon (Recommended)

**Pros**:
- Purpose-built for deep zoom images
- DZI (Deep Zoom Images) support
- Smooth panning/zooming
- Lightweight  

**Cons**:
- Requires tile pyramid generation

**Example**:
```javascript
var viewer = OpenSeadragon({
    id: "wsi-viewer",
    tileSources: "/tiles/slide_001.dzi",
    overlays: [{
        id: 'heatmap-overlay',
        px: 0, py: 0,
        width: 1.0,
        className: 'heatmap'
    }]
});
```

---

### Option 2: Leaflet

**Pros**:
- Popular mapping library
- Rich plugin ecosystem
- Simple API

**Cons**:
- Designed for geographic maps

---

### Option 3: Custom WebGL (Three.js)

**Pros**:
- Maximum control
- GPU-accelerated

**Cons**:
- Complex implementation
- Longer development time

---

## Architecture Components

### 1. Tile Generation

Convert WSI to tile pyramid:

```python
# pyvips example
slide = pyvips.Image.new_from_file("slide.svs")
slide.dzsave("output_dir/slide", tile_size=256, overlap=1)
```

**Output Structure**:
```
tiles/
└── slide_001/
    ├── slide_001.dzi  # Metadata
    └── slide_001_files/
        ├── 0/  # Zoom level 0 (lowest res)
        ├── 1/
        └── 12/  # Zoom level 12 (highest res)
```

---

### 2. Heatmap Overlay Generation

Map attention weights to tile coordinates:

```python
def generate_heatmap_tiles(attention_weights, coordinates, slide_dims):
    # Create heatmap image
    heatmap = create_attention_heatmap(attention_weights, coordinates)
    
    # Generate overlay tiles
    overlay = pyvips.Image.new_from_array(heatmap)
    overlay.dzsave("heatmap_overlay/slide")
```

---

### 3. Streaming Endpoint

Serve tiles on-demand:

```python
@app.get("/tiles/{slide_id}/{level}/{col}_{row}.{format}")
async def get_tile(slide_id: str, level: int, col: int, row: int, format: str):
    tile_path = f"tiles/{slide_id}_files/{level}/{col}_{row}.{format}"
    return FileResponse(tile_path)
```

---

## Memory-Efficient Design

### Challenge
- Full WSI: 100,000 × 80,000 pixels (8GB uncompressed)
- Cannot load entire slide in browser

### Solution: Tiled Pyramids

1. **Pre-generate** tile pyramids offline
2. **Stream** only visible tiles to browser
3. **Cache** recently viewed tiles
4. **Unload** off-screen tiles

---

## Viewer Features

### Core Functionality

- ✅ **Zoom**: Mouse wheel or +/- buttons
- ✅ **Pan**: Click-drag navigation
- ✅ **Full screen**: Maximize viewer
- ✅ **Home**: Reset to initial view

### Heatmap Controls

- ✅ **Toggle overlay**: Show/hide attention
- ✅ **Opacity slider**: Adjust transparency (0-100%)
- ✅ **Colormap selector**: Jet, hot, viridis
- ✅ **Legend**: Color scale explanation

### Annotations (Optional)

- 📌 Click to place markers
- 📏 Measure distances
- 📝 Add text notes

---

## Implementation Example (OpenSeadragon)

```html
<!DOCTYPE html>
<html>
<head>
    <script src="openseadragon.min.js"></script>
</head>
<body>
    <div id="wsi-viewer" style="width: 100%; height: 100vh;"></div>

    <div class="controls">
        <label>
            <input type="checkbox" id="toggle-heatmap" checked>
            Show Heatmap
        </label>
        <input type="range" id="opacity" min="0" max="100" value="50">
    </div>

    <script>
        var viewer = OpenSeadragon({
            id: "wsi-viewer",
            prefixUrl: "images/",
            tileSources: "/api/v1/tiles/slide_001.dzi",
            showNavigationControl: true,
            navigatorPosition: "BOTTOM_RIGHT"
        });

        // Add heatmap overlay
        viewer.addOverlay({
            element: document.getElementById('heatmap-canvas'),
            location: new OpenSeadragon.Rect(0, 0, 1, 1)
        });

        // Toggle heatmap
        document.getElementById('toggle-heatmap').addEventListener('change', function(e) {
            var overlay = viewer.currentOverlays[0];
            overlay.element.style.display = e.target.checked ? 'block' : 'none';
        });

        // Adjust opacity
        document.getElementById('opacity').addEventListener('input', function(e) {
            var overlay = viewer.currentOverlays[0];
            overlay.element.style.opacity = e.target.value / 100;
        });
    </script>
</body>
</html>
```

---

## Offline Viewer Support

### Cached Viewer

Bundle viewer assets locally:

```
viewer/
├── index.html
├── openseadragon.min.js
├── css/
└── images/
```

**No CDN required** — fully offline compatible.

---

## Performance Optimization

1. **Lazy Loading**: Load tiles only when visible
2. **Image Caching**: Browser caches tiles automatically
3. **Progressive JPEG**: Faster initial load
4. **WebWorkers**: Decode images off main thread

---

## Security Considerations

1. **De-identified Only**: No PHI in viewer
2. **Auth Required**: Protect tile endpoints
3. **CORS**: Restrict tile access
4. **Rate Limiting**: Prevent tile scraping

---

## Implementation Checklist

> [!WARNING]
> **Do NOT implement without user approval**

- [ ] Choose viewer framework
- [ ] Generate WSI tile pyramids
- [ ] Create heatmap overlay generator
- [ ] Implement tile streaming endpoint
- [ ] Build viewer HTML/JS interface
- [ ] Add overlay toggle controls
- [ ] Test with large WSIs (>50,000px)
- [ ] Optimize for mobile devices (optional)
- [ ] Bundle for offline deployment

---

**Last Updated**: 2026-01-12

**Status**: **Architecture Only - Not Implemented**
