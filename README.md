# Wildlife Detection Capstone Project

A custom SpeciesNet server that extends Google's wildlife detection model to automatically save annotated images with bounding boxes drawn around detected animals and humans.

## Features

- 🦌 **Wildlife Detection** - Detects and classifies animals using Google's SpeciesNet model
- 🖼️ **Automatic Annotation** - Draws bounding boxes on detected objects and saves annotated images
- 🚀 **Fast API Server** - HTTP server powered by LitServe for efficient inference
- 🐳 **Containerized** - Ready-to-deploy Docker container

## Quick Start

### Pull from GitHub Container Registry

```bash
docker pull ghcr.io/fabcontigiani/wildlife-detection-capstone-project:main
```

### Or Build Locally

```bash
docker build -t speciesnet-server .
```

### Run the Server

```bash
docker run -p 8000:8000 -v /path/to/images:/images speciesnet-server
```

The server will:
1. Download the SpeciesNet model on first run (~200MB)
2. Start listening on port 8000

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"filepath": "/images/photo.jpg"}]}'
```

### Response Format

```json
{
  "predictions": [{
    "filepath": "/images/photo.jpg",
    "prediction": "mammalia;cetartiodactyla;cervidae;odocoileus;virginianus;white-tailed deer",
    "prediction_score": 0.95,
    "detections": [...],
    "classifications": {...},
    "annotated_filepath": "/images/photo_annotated.jpg"
  }]
}
```

The `annotated_filepath` points to the saved image with bounding boxes drawn.

## Configuration

Command-line flags available:

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8000 | Server port |
| `--timeout` | 30 | Request timeout (seconds) |
| `--geofence` | True | Enable geographic filtering |
| `--save_annotated` | True | Save annotated images |

## Development

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager

### Local Setup

```bash
uv sync
uv run python main.py
```

## License

Apache License 2.0 - See source file headers for details.
