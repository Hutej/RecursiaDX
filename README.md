# RecursiaDx

AI-powered digital pathology platform for automated tumor detection, malaria detection, and platelet counting with AI-generated clinical reports.

## Overview

RecursiaDx is a comprehensive medical image analysis platform that integrates state-of-the-art machine learning models for:
- **Tissue Analysis**: Tumor detection in histopathology slides using GigaPath-AttentionMIL
- **Malaria Detection**: Blood smear analysis using InceptionV3
- **Platelet Counting**: Automated platelet detection using YOLOv11
- **AI Report Generation**: Gemini AI-powered clinical summary generation

## Key Features

✅ **Multi-Modal Analysis**
- Tissue tumor detection (GigaPath-based AttentionML)
- Malaria parasite detection (Transfer Learning)
- Platelet counting (YOLO object detection)

✅ **AI-Powered Workflows**
- 5-step clinical workflow (Upload → Analysis → Dashboard → Review → Report)
- Real-time ML inference with interactive visualizations
- Gemini AI-generated clinical summaries and recommendations

✅ **Professional Reporting**
- Dynamic report generation with AI interpretation
- Morphological findings analysis
- Clinical recommendations
- HIPAA-compliant data handling

✅ **Interactive UI**
- Dark/Light theme support
- Sample type adaptation (Blood vs. Tissue)
- Real-time status tracking
- Demo mode for testing

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | React 18 + Vite + Tailwind CSS |
| **Backend** | Node.js 18+ + Express + MongoDB |
| **ML Models** | PyTorch + GigaPath + InceptionV3 + YOLOv11 |
| **AI Integration** | Google Gemini 2.5 Flash |
| **Database** | MongoDB Atlas |

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- MongoDB (local or Atlas)
- Google Gemini API key (optional, for AI reports)

### Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/AyushX1602/Recursia-Dx-ML-.git
   cd RecursiaDx
   ```

2. **Backend setup**
   ```bash
   cd backend
   npm install
   cp .env.example .env  # Configure MongoDB URI and Gemini API key
   ```

3. **Frontend setup**
   ```bash
   cd client
   npm install
   ```

4. **ML setup**
   ```bash
   cd ml/api
   pip install -r requirements.txt
   # Download model files (see ml/README.md)
   ```

### Running the Application

**Option 1: Use startup script (Windows)**
```bash
.\start_all.bat
```

**Option 2: Manual start**
```bash
# Terminal 1 - Backend
cd backend
node server.js

# Terminal 2 - Frontend
cd client
npm run dev

# Terminal 3 - Tissue ML
cd ml/api
python gigapath_api.py

# Terminal 4 - Blood ML
cd ml/api
python app.py
```

Access the application at `http://localhost:5173`

## Environment Variables

### Backend (.env)
```env
MONGODB_URI=mongodb://localhost:27017/recursiadx
PORT=5001
GEMINI_API_KEY=your_gemini_api_key_here  # Optional
ML_SERVICE_URL=http://localhost:5000
GIGAPATH_SERVICE_URL=http://localhost:5002
```

### ML (.env)
```env
GIGAPATH_MODEL_PATH=path/to/gigapath_model.pth
MALARIA_MODEL_PATH=path/to/InceptionV3_Malaria_PyTorch.pth
PLATELET_MODEL_PATH=path/to/yolo11n.pt
```

## Workflow Steps

1. **Sample Upload** - Upload tissue/blood images
2. **Analysis** - ML models process the images
3. **Dashboard** - View results and visualizations
4. **Technician Review** - Approve or request re-analysis
5. **Report Generation** - Generate AI-powered clinical reports

## Project Structure

```
RecursiaDx/
├── backend/          # Node.js API server
│   ├── routes/      # API endpoints
│   ├── models/      # MongoDB schemas
│   ├── services/    # Gemini integration
│   └── server.js    # Entry point
├── client/          # React frontend
│   └── src/
│       ├── components/   # UI components
│       └── lib/         # Utilities
├── ml/              # ML services
│   └── api/
│       ├── app.py            # Malaria/Platelet API
│       └── gigapath_api.py   # Tissue analysis API
└── test/            # Test images

```

## Model Information

| Model | Task | Architecture | Accuracy |
|-------|------|--------------|----------|
| GigaPath-AttentionMIL | Tissue Tumor Detection | Vision Transformer | ~85% |
| InceptionV3 | Malaria Detection | Transfer Learning | ~95% |
| YOLOv11n | Platelet Counting | Object Detection | ~90% |

## API Endpoints

### Backend API (Port 5001)
- `POST /api/samples/upload` - Upload sample images
- `POST /api/samples/demo-analysis` - Demo mode analysis
- `POST /api/reports/generate/:id` - Generate report
- `POST /api/reports/generate-full/:id` - Generate with Gemini

### ML APIs
- **Tissue**: `http://localhost:5002/analyze` (GigaPath)
- **Blood**: `http://localhost:5000/analyze` (Malaria + Platelet)

## Gemini Integration

The platform uses Google Gemini 2.5 Flash for:
- Clinical summary generation
- Result interpretation
- Morphological findings description
- Clinical recommendations
- Diagnostic conclusions

**Without Gemini API key**: System falls back to rule-based summaries.

## Documentation

- [Startup Guide](STARTUP_GUIDE.md)
- [Integration Report](INTEGRATION_REPORT.md)
- [ML Setup](ml/README.md)
- [Backend API](backend/README.md)

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This project is for educational and research purposes.

## Acknowledgments

- GigaPath model by Microsoft Research
- Gemini AI by Google
- Open-source ML communities

---

**Status**: ✅ Production Ready | 🔄 Active Development

For issues or questions, please open a GitHub issue.
