# API Documentation

## Base URL
```
http://localhost:5000/api/v1
```

## Endpoints

### POST /classify
Classify a waste item from an image.

**Request:**
```json
{
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "category": "plastic",
  "type": "recyclable",
  "bin_color": "yellow",
  "confidence": 0.951,
  "inference_time": 0.045,
  "all_predictions": {
    "plastic": 0.951,
    "glass": 0.023,
    "metal": 0.012,
    ...
  }
}
```

### GET /stats
Get classification statistics.

**Response:**
```json
{
  "total_classifications": 1234,
  "accuracy_rate": 0.951,
  "recyclables_detected": 892,
  "waste_diverted": "342 kg"
}
```

### GET /categories
Get all supported waste categories.

**Response:**
```json
{
  "categories": [
    {
      "id": 0,
      "name": "cardboard",
      "type": "recyclable",
      "bin_color": "blue",
      "description": "Cardboard boxes, paper packaging"
    },
    ...
  ]
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request - Invalid image data |
| 413 | Image too large (max 10MB) |
| 500 | Internal server error |

## Rate Limiting
- 100 requests per minute per IP
- 1000 requests per day per API key

## Authentication
For production use, add API key to headers:
```
X-API-Key: your_api_key_here
```
