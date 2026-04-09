# Music Popularity Predictor

A full-stack application for predicting music track popularity using machine learning models and audio analysis.

## Project Structure

```
├── backend/                    # FastAPI backend (Railway deployment root)
│   ├── api/                    # API endpoints and core logic
│   │   ├── app.py             # Main FastAPI application
│   │   ├── model.py           # ML model inference
│   │   ├── feature_extraction.py # Audio feature extraction
│   │   └── models/            # API models/schemas
│   ├── helpers/               # Helper utilities (moved from root)
│   │   ├── preprocessor.py    # Data preprocessing
│   │   ├── trainer.py         # Model training utilities
│   │   └── fetcher.py         # Data fetching utilities
│   ├── models/                # Trained ML models
│   ├── requirements.txt       # Python dependencies
│   └── __init__.py
├── frontend/                   # React frontend (Netlify deployment)
│   ├── src/
│   ├── public/
│   └── package.json
├── notebooks/                  # Jupyter notebooks for analysis
├── temp/                       # Temporary data files
├── context/                    # Project documentation
└── package.json               # Root package.json for scripts
```

## Deployment

### Backend (Railway)

- **Deployment Root**: `backend/` folder
- **Main File**: `api/app.py`
- **Requirements**: `requirements.txt`
- **Environment Variables**:
  - `SPOTIFY_CLIENT_ID`
  - `SPOTIFY_CLIENT_SECRET`
  - `BEATOVEN_API_KEY`

### Frontend (Netlify)

- **Build Command**: `npm run build`
- **Publish Directory**: `build/`
- **Environment Variables**:
  - `REACT_APP_BACKEND_URL`: Railway backend URL

## Development Setup

1. **Backend**:

   ```bash
   cd backend
   pip install -r requirements.txt
   python -m api.app
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

## API Endpoints

- `POST /predict`: Predict track popularity
- `POST /extract-features`: Extract audio features
- `GET /health`: Health check

## Features

- Audio file upload and analysis
- Spotify track search integration
- Machine learning popularity prediction
- Optimization wizard for track improvement
- Multi-language support (i18n)
