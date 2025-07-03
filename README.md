# 🎨 Image-2-Paint Converter API  
*Developer Handbook – Version 1.0 (2025-07-03)*

**Base URL**&nbsp;`http://localhost:5000`

---

## 📑 Endpoint Catalogue

| # | Verb | Path | Summary |
|---|:---:|------|---------|
| 1 | POST | `/api/oil_paint`   | Apply an oil-paint filter |
| 2 | POST | `/api/foogle_art` | Create Fogle-man random-shape art |
| 3 | POST | `/api/recipes`    | Suggest paint-mix recipes for a target colour |
| 4 | POST | `/api/merge_colors` | Blend arbitrary RGBs with weights |
| 5 | POST | `/api/shape_art`  | Stamp random shapes onto an image |
| 6 | POST | `/api/shape_detect` | Detect & annotate shapes present in an image |
| 7 | POST | `/api/generate`   | **Unified geometrizer** → PNG / SVG / JSON |

*(Legacy `/api/geometrize` intentionally omitted.)*

---

## 1 · `/api/oil_paint`

### Request

| Part | Type | Default | Notes |
|------|------|---------|-------|
| `oil_image` | **file** | – | PNG / JPG |
| `intensity` | float | 10.0 | Higher ⇒ thicker brush strokes |

### Response `200`

``` json
{
  "result_image": "data:image/png;base64,iVBORw0KGgoAAA..."
}
``` 

### Quick cURL

``` bash
curl -F oil_image=@cover.jpg -F intensity=5 \
     http://localhost:5000/api/oil_paint
``` 

---

## 2 · `/api/foogle_art`

| Field | Type / Default | Description |
|-------|----------------|-------------|
| `image`        | **file** (required) | Source photo |
| `shape_type`   | `"Circles"` | `Triangles`, `Rectangles`, or `Circles` |
| `min_size`     | `5` | Pixel radius / edge |
| `max_size`     | `30` | Pixel radius / edge |
| `num_shapes`   | `100` | How many to draw |

### Output

``` json
{
  "shapes": [
    {
      "type": "Circle",
      "center": [123, 456],
      "radius": 17,
      "color": [234, 120,  80]
    },
    ...
  ]
}
``` 

---

## 3 · `/api/recipes`

### Request body

``` json
{
  "target":    [200, 100, 150],
  "db_choice": "Artisan - Winsor & Newton",
  "step":      5.0
}
``` 

### Response

``` json
{
  "recipes": [
    {
      "mix":   [198, 102, 148],
      "error": 2.8,
      "recipe": [
        { "name": "Permanent Rose", "perc": 70.0 },
        { "name": "Pthalo Blue",    "perc": 30.0 }
      ]
    }
  ]
}
``` 

---

## 4 · `/api/merge_colors`

### Example call

``` json
POST /api/merge_colors
{
  "colors": [
    { "rgb": [255, 0, 0], "weight": 0.7 },
    { "rgb": [  0, 0,255], "weight": 0.3 }
  ]
}
``` 

### Returns

``` json
{ "mixed_rgb": [179, 0, 76] }
``` 

---

## 5 · `/api/shape_art`

Randomly stamps shapes, returns Base-64 PNG plus metadata.

| Param | Default | Comment |
|-------|---------|---------|
| `image`             | (file) | Required |
| `shape_type`        | `"Circles"` | `"Triangles"`, `"Squares"` |
| `num_shapes`        | 20 | Quantity |
| `min_size` / `max_size` | 10 / 40 | Pixel bounds |

JSON format mirrors **§2** but includes `"image": "data:image/png;base64,..."`.

---

## 6 · `/api/shape_detect`

| Field | Purpose |
|-------|---------|
| `encoded_image` | **file** upload |
| `shape_type` | `"Circle"`, `"Triangle"`, `"Square"` |
| `min_size` / `max_size` | Pixel filters |

``` json
{
  "annotated_image": "data:image/png;base64,...",
  "grouped_colors": [
    { "color": "#ffcc00", "count": 12 },
    { "color": "#0077ff", "count":  5 }
  ]
}
``` 

---

## 7 · `/api/generate` – Unified Geometrizer (Recommended)

### Form fields

| Key | Req | Notes |
|-----|-----|-------|
| `image` **or** `image_base64` | ✓ | File or data-URL |
| `output_format` | ✓ | `png` · `svg` · `json` |
| `shape_types[]` | – | Repeat for each, e.g. `triangle`, `ellipse` |
| `opacity` | 128 | 0-255 |
| `shape_count` | 200 | # of mutations |
| `background_color` | – | `#rrggbb` or `rgb(r,g,b)` |
| `resize_width` / `resize_height` | – | Pre-resize |

### Response matrix

| Requested | Content-Type | Return type |
|-----------|--------------|-------------|
| `png`  | `image/png` | **bytes** |
| `svg`  | `image/svg+xml` | **string** |
| `json` | `application/json` | **dict** |

JSON structure:

``` json
{
  "shapes": [
    {
      "type": "ellipse",
      "color": "#001400",
      "opacity": 180,
      "center": [0, 0],
      "radius": [1, 1]
    }
  ],
  "background_color": "#ffffff",
  "canvas_size": [1024, 1024]
}
``` 

---

## 🐍 Full Python Client Helper

``` python
"""
geometrize_client.py
Minimal wrapper around /api/generate
"""
from __future__ import annotations
import os
import base64
import requests
from typing import Optional, List, Union, Dict, Tuple

class GeometrizeClientError(RuntimeError):
    """Non-2xx HTTP from the server."""

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
) -> Union[bytes, str, Dict]:
    # --- parameter sanity --------------------------------------------
    if bool(image_path) == bool(image_base64):
        raise ValueError("Provide exactly one of image_path or image_base64")

    url = f"{server_url.rstrip('/')}/api/generate"

    # --- assemble form fields ----------------------------------------
    data: Dict[str, Union[str, List[str]]] = {
        "output_format": output_format.lower(),
        "opacity": str(opacity),
        "shape_count": str(shape_count),
    }
    if shape_types:
        data["shape_types"] = shape_types
    if background_color:
        data["background_color"] = background_color
    if resize_width:
        data["resize_width"] = str(resize_width)
    if resize_height:
        data["resize_height"] = str(resize_height)

    # --- file vs base-64 ---------------------------------------------
    files: Dict[str, Tuple[str, bytes, str]] = {}
    if image_path:
        mime = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        files["image"] = (os.path.basename(image_path), open(image_path, "rb"), mime)
    else:
        if not image_base64.startswith("data:image"):
            image_base64 = "data:image/png;base64," + image_base64
        data["image_base64"] = image_base64

    # --- send ---------------------------------------------------------
    resp = requests.post(url, data=data, files=files or None, timeout=timeout)
    if not resp.ok:
        raise GeometrizeClientError(f"{resp.status_code} {resp.reason}: {resp.text}")

    fmt = output_format.lower()
    if fmt == "png":
        return resp.content
    if fmt == "svg":
        return resp.text
    if fmt == "json":
        return resp.json()
    raise ValueError(f"Unsupported output_format '{output_format}'")

def save_b64_image(b64str: str, filename: str) -> None:
    """Decode a (data-URL or raw) Base-64 string and write a file."""
    if b64str.startswith("data:image"):
        b64str = b64str.split(",", 1)[1]
    with open(filename, "wb") as f:
        f.write(base64.b64decode(b64str))
``` 

---

## 🔧 Example Workflow

``` python
from geometrize_client import geometrize

# 1 ) Fetch JSON metadata
meta = geometrize(
    "http://localhost:5000",
    image_path="flower.jpg",
    output_format="json",
    shape_types=["triangle", "ellipse"],
    shape_count=10,
    opacity=180,
)
print("Canvas:", meta["canvas_size"], "shapes:", len(meta["shapes"]))

# 2 ) Render PNG
png = geometrize(
    "http://localhost:5000",
    image_path="flower.jpg",
    output_format="png",
    shape_types=["triangle", "ellipse"],
    shape_count=10,
    opacity=180,
)
with open("flower_geom.png", "wb") as fh:
    fh.write(png)
print("✓ Saved flower_geom.png")
``` 

---

## 🚨 Error Schema

Every error uses this JSON envelope:

``` json
{ "detail": "Explanation message" }
``` 

Common statuses: `400` (validation) • `413` (file too large) • `500` (server fault).

---

*Maintainer – Dr Awab · Pull requests, issues, and feedback are welcome!* 🚀
``` 

