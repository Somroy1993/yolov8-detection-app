[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-url)

# VisionScan — AI Object Detection & Analysis

A Streamlit app that uses YOLOv8 nano model for real-time object detection. Detect objects in images, get annotated results, analyze object distribution, and export as CSV.

## Features

- 🎯 **YOLOv8n Detection**: Fast, accurate object detection on CPU
- 📸 **Batch Processing**: Upload multiple images at once
- 🔍 **Confidence Threshold**: Adjustable detection confidence
- 📊 **Detailed Results**: Bounding boxes, confidence scores, object counts
- 📈 **Analysis Charts**: Plotly visualization of object distribution
- 📥 **CSV Export**: Download detection results
- 🛡️ **Email Gate**: NeonDB integration for lead tracking
- 🚀 **REST API**: Ship as FastAPI endpoint

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nano+-yellowgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NEON_DATABASE_URL` | Yes | PostgreSQL connection string |
| `SMTP_HOST` | Yes | SMTP server (default: smtp.gmail.com) |
| `SMTP_PORT` | Yes | SMTP port (default: 587) |
| `SMTP_USER` | Yes | SMTP username/email |
| `SMTP_PASS` | Yes | SMTP password or app password |
| `NOTIFY_EMAIL` | Yes | Email to send lead notifications |

## Local Setup

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd yolov8-detection-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Run Streamlit app**
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch `main`, main file path `app.py`.
4. Under **Advanced settings → Secrets**, paste the contents of your local `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example` for the schema). At minimum:
   ```toml
   NEON_DATABASE_URL = "postgresql://user:pass@host/db?sslmode=require"
   SMTP_HOST = "smtp.gmail.com"
   SMTP_PORT = "587"
   SMTP_USER = "you@gmail.com"
   SMTP_PASS = "your-16-char-app-password"
   NOTIFY_EMAIL = "you@gmail.com"
   ```
5. Click **Deploy**. First build takes ~3 min; subsequent builds are faster.

**No AI API keys required** — YOLOv8n weights auto-download on first run and are cached.

## Running as FastAPI

```bash
uvicorn src.api:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## API Endpoints

### POST `/api/detect`
Detect objects in image.

**Request:**
- File upload (multipart/form-data)
- `confidence`: Confidence threshold (0.1-0.9, default: 0.4)
- `class_filter`: Comma-separated class names (optional)

**Response:**
```json
{
  "detections": [{...}],
  "annotated_image": "base64_encoded_image",
  "total_objects": 5
}
```

## Usage Limits

- Max images per session: 5
- Max file size: 8 MB
- Supported formats: JPG, PNG, WebP, BMP

## Hire Me

Looking for custom AI solutions? Visit my portfolio at **[somroy1993.github.io](https://somroy1993.github.io)**

---

Built by Somnath Roy · This is a free demo with usage limits.
