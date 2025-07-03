# # import time
# # import requests
# # import base64

# # BASE_URL = "http://localhost:5000"
# # session = requests.Session()

# # def save_b64_image(b64str, filename):
# #     with open(filename, "wb") as f:
# #         f.write(base64.b64decode(b64str))

# # def test_oil_paint(image_path, intensity=10):
# #     url = f"{BASE_URL}/api/oil_paint"
# #     with open(image_path, "rb") as imgf:
# #         files = {"oil_image": imgf}
# #         data  = {"intensity": intensity}
# #         start = time.perf_counter()
# #         resp  = session.post(url, files=files, data=data)
# #         dt = (time.perf_counter() - start) * 1000
# #         print(f"[oil_paint] {dt:.1f}ms  status={resp.status_code}")
# #         if resp.ok:
# #             b64 = resp.json()["result_image"]
# #             save_b64_image(b64, "painted.png")
# #             print("Saved → painted.png")
# #         else:
# #             print("Error:", resp.json())
# # def test_foogle_art(image_path, shape_type="Circles",
# #                     min_size=5, max_size=30, num_shapes=100):
# #     """
# #     Calls /api/foogle_art and prints JSON details for each shape:
# #     type, coordinates, and RGB color.
# #     """
# #     url = f"{BASE_URL}/api/foogle_art"
# #     with open(image_path, "rb") as imgf:
# #         files = {"image": imgf}
# #         data = {
# #             "shape_type": shape_type,
# #             "min_size": str(min_size),
# #             "max_size": str(max_size),
# #             "num_shapes": str(num_shapes)
# #         }
# #         start = time.perf_counter()
# #         resp = session.post(url, files=files, data=data)
# #         elapsed = (time.perf_counter() - start) * 1000
# #         print(f"[foogle_art] {elapsed:.1f} ms  status={resp.status_code}\n")

# #         if resp.ok:
# #             print (resp.json())
# #             shapes = resp.json().get("shapes", [])
# #             print(f"Received {len(shapes)} shapes:\n")
# #             for idx, shape in enumerate(shapes, start=1):
# #                 print(f"Shape {idx}:")
# #                 print(f"  Type : {shape['type']}")
# #                 if shape["type"] == "Circle":
# #                     center = shape.get("center", [])
# #                     radius = shape.get("radius")
# #                     print(f"  Center: {center}")
# #                     print(f"  Radius: {radius}")
# #                 else:
# #                     points = shape.get("points", [])
# #                     print(f"  Points: {points}")
# #                 color = shape.get("color", [])
# #                 print(f"  Color : {color}\n")
# #         else:
# #             # print error payload if any
# #             try:
# #                 err = resp.json()
# #             except ValueError:
# #                 err = resp.text
# #             print("Error response:", err)
# # # if __name__ == "__main__":
# # #     # 1) Oil painting
# # #     # test_oil_paint(r"D:\Programmes\Freelance\Sunny_work\Image2paint Converter\cover.jpg", intensity=5.0)

# # #     # 2) Foogle-man art
# # #     test_foogle_art("D:\Programmes\Freelance\Sunny_work\Image2paint Converter\cover.jpg",
# # #                     shape_type="Triangles",
# # #                     min_size=10, max_size=40,
# # #                     num_shapes=150)

# # import requests

# # BASE_URL = "http://localhost:5000"

# # def get_recipes(target, db_choice, step=10.0):
# #     """
# #     Calls POST /api/recipes with:
# #       { "target": [R,G,B], "db_choice": str, "step": float }
# #     """
# #     url = f"{BASE_URL}/api/recipes"
# #     payload = {
# #         "target": target,
# #         "db_choice": db_choice,
# #         "step": step
# #     }
# #     resp = requests.post(url, json=payload)

# #     # If we didn't get a 200 OK, print the error message
# #     if resp.status_code != 200:
# #         print(f"[ERROR] Status Code: {resp.status_code}")
# #         try:
# #             print("Response JSON:", resp.json())
# #         except ValueError:
# #             print("Response Text:", resp.text)
# #         return

# #     data = resp.json()
# #     print("=== Recipe Generator Response ===")
# #     print(data)
# #     for idx, rec in enumerate(data.get("recipes", []), start=1):
# #         print(f"Recipe {idx}:")
# #         print("  mix:", rec["mix"])
# #         print("  error:", rec["error"])
# #         print("  components:")
# #         for comp in rec["recipe"]:
# #             print(f"    - {comp['name']}: {comp['perc']}%")
# #     print()


# # def merge_colors(colors):
# #     """
# #     Calls POST /api/merge_colors with:
# #       { "colors": [ {"rgb":[R,G,B],"weight":w}, ... ] }
# #     """
# #     url = f"{BASE_URL}/api/merge_colors"
# #     payload = {"colors": colors}
# #     resp = requests.post(url, json=payload)

# #     if resp.status_code != 200:
# #         print(f"[ERROR] Status Code: {resp.status_code}")
# #         try:
# #             print("Response JSON:", resp.json())
# #         except ValueError:
# #             print("Response Text:", resp.text)
# #         return

# #     data = resp.json()
# #     print("=== Colour Merger Response ===")
# #     print("mixed_rgb:", data.get("mixed_rgb"))
# #     print()


# # def call_shape_art_api(
# #     image_path: str,
# #     shape_type: str = "Circles",
# #     num_shapes: int = 20,
# #     min_size: int = 10,
# #     max_size: int = 40,
# # ):
# #     """
# #     1) Upload `image_path` to /api/shape_art with the given parameters.
# #     2) Decode and save the returned Base64‐PNG as generated_shape_art.png.
# #     3) Print out each shape's metadata.
# #     """
# #     url = f"{BASE_URL}/api/shape_art"
# #     with open(image_path, "rb") as f:
# #         files = {"image": f}
# #         data = {
# #             "shape_type": shape_type,
# #             "num_shapes": str(num_shapes),
# #             "min_size":   str(min_size),
# #             "max_size":   str(max_size),
# #         }
# #         t0 = time.perf_counter()
# #         resp = session.post(url, files=files, data=data)
# #         dt = (time.perf_counter() - t0) * 1000
# #         print(f"[shape_art_api] {dt:.1f} ms  status={resp.status_code}")

# #     resp.raise_for_status()
# #     result = resp.json()
# #     art_b64 = result["image"]
# #     shapes  = result.get("shapes", [])

# #     # Save the generated shape-art image
# #     with open("generated_shape_art.png", "wb") as out:
# #         out.write(base64.b64decode(art_b64))
# #     print("Saved → generated_shape_art.png")

# # # if __name__ == "__main__":
# # #     call_shape_art_api(
# # #         "D:/Programmes/Freelance/Sunny_work/Image2paint Converter/cover.jpg",
# # #         shape_type="Circles",
# # #         num_shapes=20,
# # #         min_size=10,
# # #         max_size=40
# # #     )
# # def call_shape_detect_api(image_path, shape_type="Triangle", min_size=3, max_size=10):
# #     """
# #     Calls the /api/shape_detect endpoint:
# #     - Uploads an image and detection params
# #     - Saves the returned annotated image as 'annotated.png'
# #     - Prints grouped color counts
# #     """
# #     url = f"{BASE_URL}/api/shape_detect"
# #     with open(image_path, "rb") as f:
# #         files = {"encoded_image": f}
# #         data = {
# #             "shape_type": shape_type,
# #             "min_size":   str(min_size),
# #             "max_size":   str(max_size)
# #         }
# #         start = time.perf_counter()
# #         resp = session.post(url, files=files, data=data)
# #         elapsed = (time.perf_counter() - start) * 1000
# #         print(f"[shape_detect_api] {elapsed:.1f} ms  status={resp.status_code}")

# #     if resp.ok:
# #         result = resp.json()
# #         print (result)
# #         # Save annotated image
# #         annotated_b64 = result["annotated_image"]
# #         with open("annotated.png", "wb") as out:
# #             out.write(base64.b64decode(annotated_b64))
# #         print("Saved annotated image to annotated.png")

# #         # Print grouped colors
# #         print("Grouped colors (color: count):")
# #         for grp in result["grouped_colors"]:
# #             print(f"  {grp['color']}: {grp['count']}")

# #     else:
# #         print("Error response:", resp.text)


# # if __name__ == "__main__":
# #     # 1) Try recipe generation
# #     target_rgb = [200, 100, 150]
# #     # Make sure this DB name exactly matches one in your color.txt!
# #     db = "Artisan - Winsor & Newton"
# #     # get_recipes(target_rgb, db, step=5.0)

# #     # 2) Try colour merging
# #     colors_to_merge = [
# #         {"rgb": [255, 0, 0], "weight": 0.7},
# #         {"rgb": [0, 0, 255], "weight": 0.3},
# #     ]
# #     # merge_colors(colors_to_merge)
# #     # test_foogle_art("D:\Programmes\Freelance\Sunny_work\Image2paint Converter\cover.jpg",
# #     #                 shape_type="Circles",
# #     #                 min_size=10, max_size=40,
# #     #                 num_shapes=15)
# #     # call_shape_art_api("D:\Programmes\Freelance\Sunny_work\Image2paint Converter\cover.jpg",
# #     #     shape_type="Triangle",
# #     #     num_shapes=200,
# #     #     min_size=10,
# #     #     max_size=40
# #     # )
# #     call_shape_detect_api(
# #         r"D:\Programmes\Freelance\Sunny_work\Image2paint Converter\generated_shape_art.png",
# #         shape_type="Triangle",
# #         min_size=2,
# #         max_size=200
# #     )




# import time
# import base64
# import requests

# BASE_URL = "http://localhost:5000"
# session  = requests.Session()

# def save_b64_image(b64str: str, filename: str):
#     """Decode a Base64 string and save it to `filename`."""
#     with open(filename, "wb") as f:
#         f.write(base64.b64decode(b64str))

# def call_geometrize_api(
#     image_path: str,
#     shape_type: str   = "triangle",
#     alpha: float      = 0.5,
#     trials: int       = 50,
#     iterations: int   = 10,
#     output_name: str  = "geometrized.png"
# ):
#     """
#     1) Reads `image_path`, encodes it as a data URL
#     2) POSTs JSON to /api/geometrize
#     3) Saves the returned Base64‐PNG as `output_name`
#     """
#     url = f"{BASE_URL}/api/geometrize"

#     # 1) Load & encode the image
#     with open(image_path, "rb") as f:
#         raw = f.read()
#     data_url = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

#     # 2) Build and send JSON payload
#     payload = {
#         "image":      data_url,
#         "shapeType":  shape_type,
#         "alpha":      alpha,
#         "trials":     trials,
#         "iterations": iterations
#     }
#     start = time.perf_counter()
#     resp  = session.post(url, json=payload)
#     elapsed_ms = (time.perf_counter() - start) * 1000
#     print(f"[geometrize_api] {elapsed_ms:.1f} ms  status={resp.status_code}")

#     if not resp.ok:
#         print("Error response:", resp.text)
#         return

#     # 3) Extract & save result
#     result_url = resp.json().get("result")
#     if not result_url:
#         print("No 'result' in response")
#         return

#     # the part after the comma is the Base64 data
#     b64data = result_url.split(",", 1)[1]
#     save_b64_image(b64data, output_name)
#     print(f"✔️ Saved → {output_name}")

# # Example usage:
# if __name__ == "__main__":
#     call_geometrize_api(
#         r"D:\Programmes\Freelance\Sunny_work\Image2paint Converter\orig_preview.png",
#         shape_type="ellipse",
#         alpha=0.3,
#         trials=100,
#         iterations=200,
#         output_name="orig_preview_geometrized.png"
#     )
import os
import base64
import requests
from typing import List, Optional, Union

class GeometrizeClientError(RuntimeError):
    """Raised when the API returns a non-2xx status."""

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
    """
    Call the Flask-powered /api/generate endpoint.

    Parameters
    ----------
    server_url       e.g. 'http://localhost:5000'
    image_path       path to a PNG/JPEG/GIF on disk  (mutually exclusive with image_base64)
    image_base64     raw base-64 string (with or without data: prefix)
    output_format    'png', 'svg', or 'json'
    shape_types      list like ['triangle', 'circle']  – empty list means 'auto'
    opacity          0-255
    shape_count      integer >0
    background_color '#rrggbb', 'rgb(r,g,b)', or 'r,g,b'
    resize_width     optional target width  in pixels
    resize_height    optional target height in pixels
    timeout          seconds for the HTTP request

    Returns
    -------
    bytes | str | dict
        • PNG  → bytes (ready to write to file)
        • SVG  → str
        • JSON → dict
    """
    if bool(image_path) == bool(image_base64):
        raise ValueError("Specify exactly one of image_path or image_base64")

    url = f"{server_url.rstrip('/')}/api/generate"

    data = {
        "output_format": output_format.lower(),
        "opacity":       str(opacity),
        "shape_count":   str(shape_count),
    }
    if shape_types:
        # Multiple values for the same key → pass a list
        data["shape_types"] = shape_types
    if background_color:
        data["background_color"] = background_color
    if resize_width:
        data["resize_width"] = str(resize_width)
    if resize_height:
        data["resize_height"] = str(resize_height)

    files = {}
    if image_path:
        mime = "image/" + ("jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "png")
        files["image"] = (os.path.basename(image_path), open(image_path, "rb"), mime)
    else:  # base-64
        # Accept both raw strings and data URLs
        if not image_base64.startswith("data:image"):
            image_base64 = "data:image/png;base64," + image_base64
        data["image_base64"] = image_base64

    resp = requests.post(url, data=data, files=files, timeout=timeout)
    if not resp.ok:
        raise GeometrizeClientError(
            f"API {url} responded {resp.status_code}: {resp.text}"
        )

    if output_format == "png":
        return resp.content                      # bytes
    if output_format == "svg":
        return resp.text                         # str
    return resp.json()                           # dict for 'json'

# ─────────────── example usage ──────────────────────────────────────
if __name__ == "__main__":
    png_bytes = geometrize(
        "http://localhost:5000",
        image_path=r"D:\Programmes\Freelance\Sunny_work\Image2paint Converter\flower.jpg",
        output_format="json",
        shape_types=["triangle", "ellipse"],
        shape_count=10,
        opacity=180,
    )
    print (png_bytes)
    # with open("cat_geom.png", "wb") as f:
    #     f.write(png_bytes)