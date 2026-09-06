# Image Analysis Tool with Gemini API

A Python tool for batch analyzing indoor space images using Google's Gemini 2.5 Flash model. Designed for extracting spatial information, location identifiers, and fixed reference objects from university engineering faculty images.

## Overview

This tool processes a folder of images and extracts structured spatial information, including:
- Room numbers, codes, and signage text
- Space types and relative positions (hallways, intersections, etc.)
- Fixed reference objects (doors, walls, signs, columns)

The output is saved as a JSON file with both structured data and a brief Persian description.

## Key Features

- **Batch Processing**: Analyze all images in a folder with resume capability
- **Rate Limiting**: Built-in RPM (requests per minute) management to stay within free tier limits
- **Resume Support**: Continue from where you left off if processing is interrupted
- **JSON Output**: Structured extraction with location identifiers and reference objects
- **API Key Rotation**: Load multiple API keys from a file

## Requirements

```bash
pip install google-generativeai
```

## Configuration

1. **API Keys**: Create a file `API_keys.txt` with one Gemini API key per line
2. **Image Folder**: Place your images in the target folder (supports jpg, jpeg, png, webp, bmp, tiff)
3. **Output File**: JSON file where results will be saved

## Usage

```python
# Configure paths and run
API_KEYs = load_api_keys()  # Loads from API_keys.txt
FOLDER_PATH = "images_test"  # Your image folder
OUTPUT_FILE = "image_test_descriptions.json"

descriptions = analyze_image_folder_with_resume(FOLDER_PATH, api_key, OUTPUT_FILE)
```

## Output Format

```json
{
  "image_name.jpg": {
    "position_analysis": {
      "location_identifiers": ["A310", "Room 205"],
      "reference_objects": [
        {
          "object_name": "Main Door",
          "object_type": "door",
          "relative_position": "front",
          "suitability_for_distance_estimation": "yes"
        }
      ]
    },
    "detailed_description": "Brief Persian description (one sentence max)",
    "file_path": "/path/to/image.jpg",
    "file_size": 123456,
    "processed_at": "2026-09-06 14:30:22"
  }
}
```

## Error Handling

- Automatic JSON extraction from markdown code blocks
- Graceful fallback for malformed responses
- Per-image error handling with progress tracking
- Rate limiter auto-adjusts to avoid hitting API limits

## Limitations

- Requires valid Gemini API keys with rate limits (free tier: ~5 RPM)
- Designed specifically for indoor university engineering faculty spaces
- Images should depict indoor environments with visible spatial markers

## Run

```bash
python script_name.py
```