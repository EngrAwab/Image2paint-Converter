# 🎨 Image-2-Paint Converter – HTTP API

> Version 1.0 – 2025-07-03  
> Base URL (default): **http://localhost:5000**

---

## Index of Endpoints

| # | Method | Path | Purpose |
|---|:------:|------|---------|
| 1 | **POST** | `/api/oil_paint` | Stylise an image with an oil-paint filter |
| 2 | **POST** | `/api/foogle_art` | Generate **Fogle-man style art** (random shapes) |
| 3 | **POST** | `/api/recipes` | Suggest acrylic / oil colour mixes for a target RGB |
| 4 | **POST** | `/api/merge_colors` | Blend multiple RGBs with weights |
| 5 | **POST** | `/api/shape_art` | Place random geometric shapes onto an image |
| 6 | **POST** | `/api/shape_detect` | Detect & annotate shapes inside an image |
| 7 | **POST** | `/api/geometrize` | Primitive-based geometrisation (PNG / SVG / JSON) |
| 8 | **POST** | `/api/generate` | *New* unified geometrizer (PNG / SVG / JSON) |

---

## 1 · `/api/oil_paint`

### Description
Applies a fast oil-paint effect to an uploaded image.

### Request

| Part | Type | Notes |
|------|------|-------|
| `oil_image` | **file** | PNG / JPG |
| `intensity` | form-field *float* | Default `10.0` – higher ⇒ thicker paint |

### Response `200 OK`

``` json
{
  "result_image": "data:image/png;base64,iVBORw0K..."
}
``` 

### cURL quick-test

``` bash
curl -F oil_image=@cover.jpg    \
     -F intensity=5             \
     http://localhost:5000/api/oil_paint
``` 

The body contains a **data-URL**; strip the header and `base64 –d` to save.

---

## 2 · `/api/foogle_art`

### Description
Creates abstract “Foogle-man” art by overlaying random shapes.

### Request parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `image` | **file** | – | Source photo |
| `shape_type` | enum<br>`Triangles | Rectangles | Circles` | `"Circles"` | What shape to scatter |
| `min_size` / `max_size` | int | `5` / `30` | Pixel diameter / edge |
| `num_shapes` | int | `100` | How many shapes |

### Success response

``` json
{
  "shapes": [
    {
      "type": "Circle",
      "center": [123, 456],
      "radius": 17,
      "color": [234, 120, 80]
    }, ...
  ]
}
``` 

---

## 3 · `/api/recipes`

### Purpose  
Finds **paint-mix recipes** to approximate a target RGB.

### JSON body

``` json
{
  "target":    [200, 100, 150],      // RGB 0-255
  "db_choice": "Artisan - Winsor & Newton",
  "step":      5.0                   // granularity %
}
``` 

### Response

List ordered by lowest ∆E / Euclidean error.

``` json
{
  "recipes": [
    {
      "mix": [198, 102, 148],
      "error": 2.8,
      "recipe": [
        { "name": "Permanent Rose", "perc": 70.0 },
        { "name": "Pthalo Blue",    "perc": 30.0 }
      ]
    }, ...
  ]
}
``` 

---

## 4 · `/api/merge_colors`

### Description  
Combines arbitrary colours with weights → returns weighted average.

### JSON body

``` json
{
  "colors": [
    { "rgb": [255, 0, 0], "weight": 0.7 },
    { "rgb": [0, 0, 255], "weight": 0.3 }
  ]
}
``` 

### Response

``` json
{
  "mixed_rgb": [179, 0, 76]
}
``` 

---

## 5 · `/api/shape_art`

### Goal  
Randomly stamp shapes onto the image, then return **Base-64 PNG + metadata**.

| Field | Type | Default |
|-------|------|---------|
| `image` | file | – |
| `shape_type` | `"Circles" | "Triangles" | "Squares"` | `"Circles"` |
| `num_shapes` | int | `20` |
| `min_size` / `max_size` | int | `10` / `40` |

Response identical to `/api/foogle_art` plus Base-64 PNG in `"image"` key.

---

## 6 · `/api/shape_detect`

### Behaviour
Runs classical CV + clustering to find shapes, returns annotated PNG.

| Param | Meaning |
|-------|---------|
| `encoded_image` | **file** upload |
| `shape_type` | Detect a single kind (“Circle”, “Triangle”, “Square”). |
| `min_size` / `max_size` | Filter results by approximate pixels. |

Response

``` json
{
  "annotated_image": "data:image/png;base64,iVBOR...",
  "grouped_colors": [
    { "color": "#ffcc00", "count": 12 },
    { "color": "#0077ff", "count": 5  }
  ]
}
``` 

---

## 7 · `/api/geometrize`  *(legacy)*

### Summary  
Uses primitive‐CLI in *single-shot* mode; returns **data-URL PNG**.

JSON payload

``` json
{
  "image":      "data:image/png;base64,...",
  "shapeType":  "ellipse",
  "alpha":      0.5,
  "trials":     100,
  "iterations": 10
}
``` 

Response `200`

``` json
{ "result": "data:image/png;base64,iVBOR..." }
``` 

---

## 8 · `/api/generate`  ***← recommended***

### Why new?  
Re-implements primitive but supports **three response modes**: `png`, `svg`, `json`.

### Multipart/form fields

| Key | Type | Req? | Notes |
|-----|------|------|-------|
| `image` **OR** `image_base64` | file / str | ✓ | Choose one |
| `output_format` | `png | svg | json` | ✓ | What you want back |
| `shape_types` | repeatable list | – | `triangle`, `rectangle`, `ellipse`,… |
| `opacity` | 0-255 | 128 | Alpha of shapes |
| `shape_count` | int >0 | 200 | Iterations for primitive |
| `background_color` | CSS | – | Force backdrop |
| `resize_width` / `resize_height` | int | – | Pre-resize before processing |

### Response variants

* **PNG** → `Content-Type: image/png` (binary).
* **SVG** → `image/svg+xml` (text).
* **JSON** →  
  ``` json
  {
    "shapes": [
      { "type": "ellipse", "color": "#001400", "opacity": 180,
        "center": [0,0], "radius": [1,1] }
    ],
    "background_color": "#ffffff",
    "canvas_size": [1024, 1024]
  }
  ``` 

### Example – get JSON metadata

``` python
from geometrize_client import geometrize

meta = geometrize(
    "http://localhost:5000",
    image_path="flower.jpg",
    output_format="json",
    shape_types=["triangle", "ellipse"],
    shape_count=10,
    opacity=180
)
print(meta["shapes"][0])
``` 

---

## 9 · Client Helper (`geometrize_client.py`)

``` python
import os, base64, requests
from typing import List, Optional, Union

class GeometrizeClientError(RuntimeError):
    pass

def geometrize(
    server_url: str,
    *,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    output_format: str = "png",
    shape_types: Optional[List[str]] = None,
    opacity: int = 128,
    shape_count: int = 200,
    background_color: Optional[str] = None,
    resize_width: Optional[int] = None,
    resize_height: Optional[int] = None,
    timeout: int = 90,
) -> Union[bytes, str, dict]:
    # ... full source identical to example script ...
``` 

---

## Appendix A · Error model

All endpoints return **JSON** on error:

``` json
{ "detail": "Human-readable message" }
``` 

Typical HTTP status codes used:

| Code | Meaning |
|------|---------|
| `400` | Validation error (missing fields, wrong type, etc.) |
| `413` | Image too large (if enforced) |
| `500` | Primitive execution, PIL error, or uncaught exception |

---

> **Maintainer:** Dr Awab – Feel free to extend, fork, or PR! 🚀
``` 

