import os
import base64
import itertools
import math
from io import BytesIO
# at the top of app.py
import colormixer as mixbox
import os, re, base64, shutil, stat, tempfile, subprocess, platform, requests
from typing import List
from flask import Flask, request, jsonify, Response
from PIL import Image
import cv2
import numpy as np
from PIL import Image
from flask import (                       # ← jsonify appended
    Flask, render_template, request,
    session, redirect, url_for, send_file, jsonify
)
from werkzeug.utils import secure_filename

# ─────────── Flask setup ─────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder="templates",   # your HTML lives here
    static_folder="templates",     # serve static assets from here too
    static_url_path=""             # mount them at the web-root (/css/…, /js/…, /assets/…)
)

app.secret_key = os.urandom(16)

# Ensure uploads folder exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# helpers/upload.py
import os, cv2, numpy as np
from werkzeug.utils import secure_filename
from flask import current_app

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_and_decode(file_storage, subdir: str = ""):
    """
    • Validates extension  
    • Saves the file into UPLOAD_FOLDER[/subdir]  
    • Returns (img_bgr, path, error)  – exactly one of img_bgr / error will be None
    """
    if not file_storage or file_storage.filename == "":
        return None, None, "No file provided."
    if not allowed_file(file_storage.filename):
        return None, None, "Unsupported file type."

    root = current_app.config["UPLOAD_FOLDER"]
    if subdir:
        root = os.path.join(root, subdir)
        os.makedirs(root, exist_ok=True)

    fname = secure_filename(file_storage.filename)
    path  = os.path.join(root, fname)
    file_storage.save(path)

    data = np.fromfile(path, dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, "Failed to read image."
    return img, path, None

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────── Colour-database helpers ─────────────────────────────────
def read_color_file(path: str = "color.txt") -> str:
    # Ensure the file exists before trying to read it
    if not os.path.exists(path):
        open(path, 'w').close() # Create the file if it doesn't exist
    with open(path, encoding="utf8") as f:
        return f.read()


def parse_color_db(txt: str):
    dbs, cur = {}, None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # If line does not start with a digit, it's a database name
        if not line[0].isdigit():
            cur = line
            dbs[cur] = []
        # Otherwise, it's a color entry
        else:
            tok = line.split()
            if len(tok) < 3:
                continue
            # The RGB value is always the second to last token
            parts = tok[-2].split(",")
            if len(parts) != 3:
                continue
            try:
                r, g, b = map(int, parts)
            except ValueError:
                continue
            # The name is everything between the first and second-to-last token
            name = " ".join(tok[1:-2])
            if cur is not None:
                dbs[cur].append((name, (r, g, b)))
    return dbs

# +++ START OF ADDED/MODIFIED CODE +++

def write_color_db(databases, path: str = "color.txt"):
    """Writes the databases dictionary back to the file in the correct format."""
    with open(path, "w", encoding="utf8") as f:
        db_items = list(databases.items())
        for i, (db_name, colors) in enumerate(db_items):
            f.write(f"{db_name}\n")
            # The original file has a number at the start and end of color lines.
            # We'll replicate that. The last number seems arbitrary, so we use 1000.
            for j, (color_name, rgb) in enumerate(colors):
                r, g, b = rgb
                f.write(f"{j+1} {color_name} {r},{g},{b} 1000\n")
            # Add a blank line between databases, but not after the last one
            if i < len(db_items) - 1:
                f.write("\n")

def hex_to_rgb(hex_str: str):
    """Converts a hex color string (e.g., '#RRGGBB') to an (r, g, b) tuple."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return None
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return None

@app.route("/upload_shared_image", methods=["POST"])
def upload_shared_image():
    file = request.files.get("image")
    if not file or file.filename == "":
        return {"ok": False, "msg": "No file selected"}, 400
    if not allowed_file(file.filename):
        return {"ok": False, "msg": "Unsupported file type"}, 400

    filename = secure_filename(file.filename)
    path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # cache for all pages
    session["shared_img_path"]  = path

    # tiny 32-px thumbnail
    img_bgr   = cv2.imread(path)
    thumb     = cv2.resize(img_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    buf = BytesIO();  Image.fromarray(thumb_rgb).save(buf, format="PNG")
    session["shared_img_thumb"] = base64.b64encode(buf.getvalue()).decode()

    return {"ok": True}, 200

def convert_db_list_to_dict(lst):
    return {n: list(rgb) for n, rgb in lst}


def mix_colors(recipe):
    total = sum(p for _, p in recipe)
    r = sum(rgb[0] * p for rgb, p in recipe) / total
    g = sum(rgb[1] * p for rgb, p in recipe) / total
    b = sum(rgb[2] * p for rgb, p in recipe) / total
    return (round(r), round(g), round(b))


def color_error(c1, c2):
    return math.dist(c1, c2)


def generate_recipes(target, base_colors, step=10.0):
    base = list(base_colors.items())
    candidates = []

    # single-colour quick matches
    for name, rgb in base:
        err = color_error(rgb, target)
        if err < 5:
            candidates.append(([(name, 100)], rgb, err))

    # triple-mix brute-force search
    for (n1, r1), (n2, r2), (n3, r3) in itertools.combinations(base, 3):
        p_range = np.arange(0, 101, step)
        for p1 in p_range:
            for p2 in np.arange(0, 101 - p1, step):
                p3 = 100 - p1 - p2
                recipe = [(n1, p1), (n2, p2), (n3, p3)]
                mixed = mix_colors([(r1, p1), (r2, p2), (r3, p3)])
                err = color_error(mixed, target)
                candidates.append((recipe, mixed, err))

    candidates.sort(key=lambda x: x[2])
    top, seen = [], set()
    for rec, mix, err in candidates:
        key = tuple(sorted((n, p) for n, p in rec if p > 0))
        if key not in seen:
            seen.add(key)
            top.append((rec, mix, err))
        if len(top) == 3:
            break
    return top


# ─────────── Color-grouping helper ───────────────────────────────────
def color_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))


def group_similar_colors(rgb_vals, threshold=1):
    grouped_colors = []
    counts = []

    for color in rgb_vals:
        found_group = False
        for i, group in enumerate(grouped_colors):
            if color_distance(color, group[0]) < threshold:
                grouped_colors[i].append(color)
                counts[i] += 1
                found_group = True
                break
        if not found_group:
            grouped_colors.append([color])
            counts.append(1)
    return [(group[0], count) for group, count in zip(grouped_colors, counts)]


# ─────────── Shape-encoding/decoding (EnDe) ─────────────────────────
from EnDe import encode, decode

import random
# ─────────── Oil-painting helper ─────────────────────────────────────
from painterfun import oil_main  # your oil painting function


# ─────────── Geometrize & Foogle Man Repo placeholders ──────────────
try:
    from shape_art_generator import main_page as foogle_man_page
except ImportError:
    foogle_man_page = None

try:
    from geometrize import geometrize_app
except ImportError:
    geometrize_app = None


# ─────────── Routes ───────────────────────────────────────────────────

@app.route("/")
def root():
    return redirect(url_for("image_generator_page"))


ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


@app.route("/image_generator", methods=["GET", "POST"])
def image_generator_page():
    """
    Part A: Generate shape-art on the shared image.
    Part B: Generate paint recipes from a sampled pixel.
    """
    error               = None
    result_image_data   = None
    selected_recipe_color = None
    recipe_results      = None

    # sticky form defaults
    shape_option = "Triangle"
    num_shapes   = 100
    min_size     = 10
    max_size     = 50

    # flag: is there a shared image in session?
    shared_path = session.get("shared_img_path")
    shared_exists = bool(shared_path and os.path.exists(shared_path))

    # ─────────────────────── PART A: Shape-Art ───────────────────────
    if request.method == "POST" and request.form.get("action") != "generate_recipe":
        if not shared_exists:
            error = "Please upload an image with the header button first."
        else:
            # load the cached image
            img_bgr = cv2.imread(shared_path)
            if img_bgr is None:
                error = "Failed to read the shared image."
            else:
                # pull in form parameters
                shape_option = request.form.get("shape_type", "Triangle")
                num_shapes   = int(request.form.get("num_shapes", 100))
                if shape_option == "Triangle":
                    min_size = int(request.form.get("min_triangle_size", 15))
                    max_size = int(request.form.get("max_triangle_size", 50))
                else:
                    min_size = int(request.form.get("min_size", 10))
                    max_size = int(request.form.get("max_size", 15))
                if min_size > max_size:
                    min_size, max_size = max_size, min_size

                try:
                    # run your shape-art generator
                    encoded_img, _ = encode(
                        img_bgr,
                        shape_option,
                        output_path="",
                        num_shapes=num_shapes,
                        min_size=min_size,
                        max_size=max_size,
                    )
                    # convert to PNG/base64
                    rgb_img = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
                    buf = BytesIO()
                    Image.fromarray(rgb_img).save(buf, format="PNG")
                    result_image_data = base64.b64encode(buf.getvalue()).decode("ascii")

                    # stash for download
                    tmp_name = f"shape_art_{os.getpid()}.png"
                    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
                    with open(tmp_path, "wb") as f:
                        f.write(buf.getvalue())
                    session["shape_art_path"] = tmp_path

                except Exception as e:
                    error = f"Error generating shape art: {e}"

    # ───────────────────── PART B: Recipe Generation ─────────────────────
    if request.method == "POST" and request.form.get("action") == "generate_recipe":
        sel = request.form.get("selected_color", "")
        if sel:
            r, g, b = map(int, sel.split(","))
            selected_recipe_color = (r, g, b)
            step      = float(request.form.get("step", 10.0))
            db_choice = request.form.get("db_choice")

            # load color DB
            full_txt = read_color_file("color.txt")
            all_dbs  = parse_color_db(full_txt)
            raw_list = all_dbs.get(db_choice, [])
            base_dict= {name: list(rgb) for name, rgb in raw_list}

            recipe_results = generate_recipes(selected_recipe_color, base_dict, step=step)

    # ───────────────────────────── render ──────────────────────────────
    return render_template(
        "image_generator.html",
        error                   = error,
        result_image_data       = result_image_data,
        shape_option            = shape_option,
        num_shapes              = num_shapes,
        min_size                = min_size,
        max_size                = max_size,
        shared_img_exists       = shared_exists,
        selected_recipe_color   = selected_recipe_color,
        recipe_results          = recipe_results,
        db_list                 = list(parse_color_db(read_color_file("color.txt")).keys()),
        active_page             = "image_generator",
    )

@app.route("/download_shape_art")
def download_shape_art():
    path = session.get("shape_art_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("image_generator_page"))

@app.route("/shape_detector", methods=["GET", "POST"])
def shape_detector_page():
    error                = None
    decoded_image_data   = None
    grouped_colors       = session.get("grouped_colors", [])
    selected_recipe_color= None
    recipe_results       = None

    if request.method == "POST" and request.form.get("action") != "generate_recipe":
        file = request.files.get("encoded_image")
        img_bgr = None
        path    = None

        if file and file.filename:
            if not allowed_file(file.filename):
                error = "Unsupported file type."
            else:
                filename = secure_filename(file.filename)
                path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                session["shared_img_path"] = path
                file_bytes = np.fromfile(path, dtype=np.uint8)
                img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    error = "Failed to read uploaded image."
        else:
            path = session.get("shared_img_path")
            if path and os.path.exists(path):
                img_bgr = cv2.imread(path)
            else:
                error = "Please upload an encoded PNG/JPG first."

        if img_bgr is not None and error is None:
            shape_opt = request.form.get("shape_detect", "Triangle")
            min_size  = int(request.form.get("min_size", 3))
            max_size  = int(request.form.get("max_size", 10))

            _, annotated_img, rgb_vals = decode(
                img_bgr, shape_opt,
                boundaries=[], min_size=min_size, max_size=max_size
            )

            ann_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            buf     = BytesIO()
            Image.fromarray(ann_rgb).save(buf, format="PNG")
            decoded_image_data = base64.b64encode(buf.getvalue()).decode()

            rgb_py   = [list(map(int, col)) for col in rgb_vals]
            grouped  = sorted(group_similar_colors(rgb_py, threshold=10),
                              key=lambda x: x[1], reverse=True)
            grouped_colors              = grouped
            session["grouped_colors"]   = grouped

            tmp_name = f"shape_analysis_{os.getpid()}.png"
            tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
            with open(tmp_path, "wb") as f_out:
                f_out.write(buf.getvalue())
            session["analysis_path"] = tmp_path

    if request.form.get("action") == "generate_recipe":
        sel = request.form.get("selected_color")
        if sel:
            r, g, b = map(int, sel.split(","))
            selected_recipe_color = (r, g, b)
            step      = float(request.form.get("step", 10.0))
            db_choice = request.form.get("db_choice")

            full_txt = read_color_file("color.txt")
            all_dbs  = parse_color_db(full_txt)
            base_dict= {name: list(rgb) for name, rgb in all_dbs.get(db_choice, [])}

            recipe_results = generate_recipes(selected_recipe_color, base_dict, step=step)

    return render_template(
        "shape_detector.html",
        error                 = error,
        decoded_image_data    = decoded_image_data,
        grouped_colors        = grouped_colors,
        selected_recipe_color = selected_recipe_color,
        recipe_results        = recipe_results,
        db_list               = list(parse_color_db(read_color_file("color.txt")).keys()),
        active_page           = "shape_detector",
        shared_img_exists     = bool(session.get("shared_img_path")),
    )


@app.route("/download_analysis")
def download_analysis():
    path = session.get("analysis_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("shape_detector_page"))
@app.route("/download_oil")
def download_oil():
    path = session.get("oil_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("oil_painting_page"))
@app.route("/oil_painting", methods=["GET", "POST"])
def oil_painting_page():
    error                 = None
    original_image_data   = None
    result_image_data     = None
    intensity             = int(request.form.get("intensity", 10))
    selected_recipe_color = None
    recipe_results        = None
    db_list               = list(parse_color_db(read_color_file("color.txt")).keys())

    if request.method == "POST" and request.form.get("action") != "generate_recipe":
        img_bgr = None
        file = request.files.get("oil_image")
        if file and file.filename:
            img_bgr, path, err = save_and_decode(file, subdir="oil_painting")
            if err:
                error = err
            else:
                session["shared_img_path"] = path
        if img_bgr is None and error is None:
            shared_path = session.get("shared_img_path")
            if shared_path and os.path.exists(shared_path):
                img_bgr = cv2.imread(shared_path)
            else:
                error = "ERROR"

        if img_bgr is not None and error is None:
            buf = BytesIO()
            orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(orig_rgb).save(buf, format="PNG")
            original_image_data = base64.b64encode(buf.getvalue()).decode()

            try:
                painted = oil_main(img_bgr, intensity)
                painted = (painted * 255).astype(np.uint8)
                rgb_img = cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)

                buf = BytesIO()
                Image.fromarray(rgb_img).save(buf, format="PNG")
                result_image_data = base64.b64encode(buf.getvalue()).decode()

                tmp_name = f"oil_painting_{os.getpid()}.png"
                tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
                with open(tmp_path, "wb") as f:
                    f.write(buf.getvalue())
                session["oil_path"] = tmp_path

            except Exception as e:
                error = f"Error generating oil painting: {e}"

    if request.method == "POST" and request.form.get("action") == "generate_recipe":
        sel = request.form.get("selected_color", "")
        try:
            r, g, b = map(int, sel.split(","))
            selected_recipe_color = (r, g, b)
        except Exception:
            error = "Invalid RGB — click the oil-painted image to pick a colour."
        else:
            try:
                step = float(request.form.get("step", 10.0))
            except ValueError:
                step = 10.0
            db_choice = request.form.get("db_choice", db_list[0])
            if db_choice not in db_list:
                error = f"Unknown colour DB “{db_choice}”."
            else:
                full_txt   = read_color_file("color.txt")
                all_dbs    = parse_color_db(full_txt)
                base_dict  = {name: tuple(rgb) for name, rgb in all_dbs[db_choice]}
                recipe_results = generate_recipes(
                    selected_recipe_color,
                    base_dict,
                    step=step
                )

    return render_template(
        "oil_painting.html",
        error                 = error,
        original_image_data   = original_image_data,
        result_image_data     = result_image_data,
        intensity             = intensity,
        shared_img_exists     = bool(session.get("shared_img_path")),
        db_list               = db_list,
        selected_recipe_color = selected_recipe_color,
        recipe_results        = recipe_results,
        active_page           = "oil_painting",
    )


@app.route("/colour_merger", methods=["GET", "POST"])
def colour_merger_page():
    if "colors" not in session:
        session["colors"] = [
            {"rgb": [255, 0, 0], "weight": 0.5},
            {"rgb": [0, 255, 0], "weight": 0.5}
        ]

    if request.method == "POST":
        new_colors = []
        idx = 0
        while True:
            rgb_str = request.form.get(f"rgb-{idx}")
            weight_str = request.form.get(f"weight-{idx}")
            if rgb_str is None or weight_str is None:
                break
            try:
                r, g, b = map(int, rgb_str.split(","))
                w = float(weight_str)
                new_colors.append({"rgb": [r, g, b], "weight": w})
            except ValueError:
                pass
            idx += 1
        if new_colors:
            session["colors"] = new_colors

    colors = session["colors"]

    def get_mixed_rgb(colors_list):
        z_mix = [0] * mixbox.LATENT_SIZE
        total = sum(c["weight"] for c in colors_list)
        for i in range(len(z_mix)):
            z_mix[i] = sum(
                c["weight"] * mixbox.rgb_to_latent(c["rgb"])[i]
                for c in colors_list
            ) / total
        return mixbox.latent_to_rgb(z_mix)

    mixed_rgb = get_mixed_rgb(colors)

    return render_template(
        "colour_merger.html",
        colors=colors,
        mixed_rgb=mixed_rgb,
        active_page="colour_merger"
    )
@app.route("/recipe_generator", methods=["GET", "POST"])
def recipe_generator_page():
    error = None
    recipes = None
    selected_color = (255, 0, 0)

    full_txt = read_color_file("color.txt")
    databases = parse_color_db(full_txt)
    db_keys = list(databases.keys())

    if request.method == "POST":
        hex_color = request.form.get("hex_color")
        if hex_color:
            hv = hex_color.lstrip('#')
            selected_color = (
                int(hv[0:2], 16),
                int(hv[2:4], 16),
                int(hv[4:6], 16),
            )
        else:
            try:
                r = int(request.form.get("r", 0))
                g = int(request.form.get("g", 0))
                b = int(request.form.get("b", 0))
                selected_color = (r, g, b)
            except:
                pass

        step = float(request.form.get("step", 10.0))
        db_choice = request.form.get("db_choice", db_keys[0])

        base_list = databases.get(db_choice, [])
        base_dict = {name: list(rgba) for name, rgba in base_list}

        recipes = generate_recipes(selected_color, base_dict, step=step)

    return render_template(
        "recipe_generator.html",
        databases=db_keys,
        selected_color=selected_color,
        recipes=recipes,
        active_page="recipe_generator"
    )

@app.route("/colors_db", methods=["GET", "POST"])
def colors_db_page():
    """
    Browse/Add/Remove colors from color.txt.
    """
    full_txt = read_color_file("color.txt")
    databases = parse_color_db(full_txt)

    subpage = "databases"
    message = None
    selected_db_name = None

    if request.method == "POST":
        action = request.form.get("action")

        # Determine the subpage to show based on the action from the top buttons or forms
        if action == "browse": subpage = "databases"
        elif action == "add": subpage = "add"
        elif action == "remove_colors": subpage = "remove_colors"
        elif action == "create_db": subpage = "custom"
        elif action == "remove_db": subpage = "remove_database"

        # --- Handle DB Creation ---
        if action == "create_db" and "new_db_name" in request.form:
            new_db_name = request.form.get("new_db_name", "").strip()
            if not new_db_name:
                message = ("error", "Database name cannot be empty.")
            elif new_db_name in databases:
                message = ("error", f"Database '{new_db_name}' already exists.")
            else:
                databases[new_db_name] = []
                write_color_db(databases)
                message = ("success", f"Database '{new_db_name}' created successfully.")
                subpage = "databases"

        # --- Handle Adding a Color ---
        elif action == "add" and "color_name" in request.form:
            db_name = request.form.get("db_name")
            color_name = request.form.get("color_name", "").strip()
            hex_value = request.form.get("hex_value", "").strip()
            rgb = hex_to_rgb(hex_value)

            if not db_name or db_name not in databases:
                message = ("error", "Please select a valid database.")
            elif not color_name:
                message = ("error", "Color name cannot be empty.")
            elif not rgb:
                message = ("error", "Invalid hex value. Please use #RRGGBB format.")
            elif any(c[0].lower() == color_name.lower() for c in databases.get(db_name, [])):
                message = ("error", f"Color '{color_name}' already exists in '{db_name}'.")
            else:
                databases[db_name].append((color_name, rgb))
                write_color_db(databases)
                message = ("success", f"Color '{color_name}' added to '{db_name}'.")

        # --- Handle Removing Colors ---
        elif action == "remove_colors":
            db_name = request.form.get("db_name")
            colors_to_remove = request.form.getlist("colors")
            
            selected_db_name = db_name if db_name else (list(databases.keys())[0] if databases else None)

            # This block only runs if the "Remove Selected" button was clicked (not just a dropdown change)
            if colors_to_remove and db_name in databases:
                initial_count = len(databases[db_name])
                databases[db_name] = [c for c in databases[db_name] if c[0] not in colors_to_remove]
                write_color_db(databases)
                removed_count = initial_count - len(databases[db_name])
                if removed_count > 0:
                    message = ("success", f"Removed {removed_count} color(s) from '{db_name}'.")

        # --- Handle DB Deletion ---
        elif action == "remove_db" and "db_name" in request.form:
            db_to_remove = request.form.get("db_name")
            if db_to_remove in databases:
                del databases[db_to_remove]
                write_color_db(databases)
                message = ("success", f"Database '{db_to_remove}' has been deleted.")
                subpage = "databases"
            else:
                message = ("error", "Database not found.")
                subpage = "remove_database"

    # This ensures the 'remove_colors' page shows the first DB by default on GET request
    if subpage == "remove_colors" and not selected_db_name:
        if databases:
            selected_db_name = list(databases.keys())[0]

    return render_template(
        "colors_db.html",
        databases=databases,
        subpage=subpage,
        message=message,
        active_page="colors_db",
        selected_db_name=selected_db_name
    )

# +++ END OF ADDED/MODIFIED CODE +++

def resize_for_processing(image, max_dim=800):
    """Resize image for speed, return (resized, scale)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / w, max_dim / h)
    if scale < 1.0:
        resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized, scale
    return image, 1.0

def pixelate_image(image, block_size=5):
    """Pixelate by downscaling & upscaling."""
    h, w = image.shape[:2]
    small = cv2.resize(image,
                       (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def draw_random_circles(image, min_radius, max_radius, num_circles):
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(num_circles):
        r = random.randint(min_radius, max_radius)
        x = random.randint(r, w - r)
        y = random.randint(r, h - r)
        color = out[y, x].tolist()
        cv2.circle(out, (x, y), r, color, -1)
    return out

def draw_random_rectangles(image, min_size, max_size, num_rects):
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(num_rects):
        rw = random.randint(min_size, max_size)
        rh = random.randint(min_size, max_size)
        x  = random.randint(0, w - rw)
        y  = random.randint(0, h - rh)
        angle = random.randint(0, 360)
        color = out[y, x].tolist()
        rect = np.array([[x, y],
                         [x+rw, y],
                         [x+rw, y+rh],
                         [x, y+rh]], dtype=np.float32)
        M = cv2.getRotationMatrix2D((x+rw/2, y+rh/2), angle, 1.0)
        pts = cv2.transform(np.array([rect]), M)[0].astype(int)
        cv2.fillPoly(out, [pts], color)
    return out

def draw_random_triangles(image, min_size, max_size, num_triangles):
    out = image.copy()
    h, w = out.shape[:2]
    for _ in range(num_triangles):
        side = random.randint(min_size, max_size)
        tri_h = int(side * np.sqrt(3) / 2)
        x = random.randint(0, w - side)
        y = random.randint(tri_h, h)
        color = out[y - tri_h//2, x + side//2].tolist()
        tri = np.array([(x, y),
                        (x+side, y),
                        (x+side//2, y-tri_h)],
                       dtype=np.int32)
        angle = random.randint(0, 360)
        M = cv2.getRotationMatrix2D((x+side//2, y-tri_h/3), angle, 1.0)
        pts = cv2.transform(np.array([tri]), M)[0].astype(int)
        cv2.fillPoly(out, [pts], color)
    return out

def get_db_list():
    full_txt = read_color_file("color.txt")
    all_dbs  = parse_color_db(full_txt)
    return list(all_dbs.keys())


@app.route("/foogle_man_repo", methods=["GET", "POST"])
def foogle_man_repo_page():
    original_b64   = None
    generated_b64  = None
    download_url   = None
    num_shapes     = 0
    error                  = None
    recipe_results         = None
    selected_recipe_color  = None
    db_list                = get_db_list()

    if request.method == "POST":
        if request.form.get("action") == "generate_recipe":
            sel = request.form.get("selected_color", "").strip()
            try:
                r, g, b = [int(x) for x in sel.split(",")]
                selected_recipe_color = (r, g, b)
            except Exception:
                error = "Invalid RGB — click the art to pick a colour."
            else:
                try:
                    step = float(request.form.get("step", 10.0))
                except ValueError:
                    step = 10.0

                db_choice = request.form.get("db_choice", db_list[0])
                if db_choice not in db_list:
                    error = f"Unknown colour DB “{db_choice}”."
                else:
                    full_txt   = read_color_file("color.txt")
                    all_dbs    = parse_color_db(full_txt)
                    base_dict  = {n: tuple(rgb) for n, rgb in all_dbs[db_choice]}

                    recipe_results = generate_recipes(
                        selected_recipe_color,
                        base_dict,
                        step=step
                    )
        else:
            img_bgr = None
            file    = request.files.get("image")

            if file and file.filename:
                img_bgr, path, err = save_and_decode(file, subdir="foogle_man")
                if err:
                    error = err
                else:
                    session["shared_img_path"] = path

            if img_bgr is None and not error:
                path = session.get("shared_img_path")
                if path and os.path.exists(path):
                    img_bgr = cv2.imread(path)
                else:
                    error = "Please upload an image first (on any page)."

            if img_bgr is not None and not error:
                shape_type = request.form.get("shape_type", "Circles")
                min_size   = int(request.form.get("min_size", 5))
                max_size   = int(request.form.get("max_size", 30))
                num_shapes = int(request.form.get("num_shapes", 100))
                block_size = (min_size + max_size) // 5

                proc, scale = resize_for_processing(img_bgr)
                pix         = pixelate_image(proc, block_size)

                if shape_type == "Circles":
                    art = draw_random_circles(pix, min_size, max_size, num_shapes)
                elif shape_type == "Rectangles":
                    art = draw_random_rectangles(pix, min_size, max_size, num_shapes)
                else:
                    art = draw_random_triangles(pix, min_size, max_size, num_shapes)

                if scale < 1.0:
                    art = cv2.resize(
                        art,
                        (img_bgr.shape[1], img_bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )

                buf = BytesIO()
                Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))\
                     .save(buf, format="PNG")
                original_b64 = base64.b64encode(buf.getvalue()).decode()

                buf = BytesIO()
                Image.fromarray(cv2.cvtColor(art, cv2.COLOR_BGR2RGB))\
                     .save(buf, format="PNG")
                generated_b64 = base64.b64encode(buf.getvalue()).decode()

                download_url = f"data:image/png;base64,{generated_b64}"

    return render_template(
        "foogle_man_repo.html",
        original_image    = original_b64,
        generated_image   = generated_b64,
        num_shapes        = num_shapes,
        download_url      = download_url,
        error                   = error,
        db_list                 = db_list,
        selected_recipe_color   = selected_recipe_color,
        recipe_results          = recipe_results,
        shared_img_exists = bool(session.get("shared_img_path")),
    )


@app.route("/paint_geometrize")
def paint_geometrize_page():
    error               = None
    recipe_results      = None
    selected_recipe_rgb = None
    db_list = get_db_list()

    if request.method == "POST" and request.form.get("action") == "generate_recipe":
        sel = request.form.get("selected_color", "")
        try:
            r, g, b = [int(x.strip()) for x in sel.split(",")]
            selected_recipe_rgb = (r, g, b)
        except ValueError:
            error = "Invalid RGB—please click on the image to pick a colour."
        else:
            try:
                step = float(request.form.get("step", 10.0))
            except ValueError:
                step = 10.0
            db_choice = request.form.get("db_choice", "")
            if db_choice not in db_list:
                error = f"Unknown colour DB '{db_choice}'."
            else:
                full_txt = read_color_file("color.txt")
                all_dbs   = parse_color_db(full_txt)
                base_dict = {name: tuple(rgb) for name, rgb in all_dbs[db_choice]}

                recipe_results = generate_recipes(selected_recipe_rgb, base_dict, step=step)

                for rec in recipe_results:
                    if isinstance(rec.get("mix"), (tuple, list)):
                        rec["mix"] = ", ".join(str(c) for c in rec["mix"])

    return render_template(
        "paint_geometrize.html",
        error=error,
        db_list=db_list,
        selected_recipe_rgb=selected_recipe_rgb,
        recipe_results=recipe_results,
        active_page="paint_geometrize"
    )

@app.route("/generate_recipe", methods=["POST"])
def ajax_generate_recipe():
    sel = request.form.get("selected_color", "")
    if not sel:
        return jsonify(ok=False, msg="No color selected"), 400
    try:
        target = tuple(int(x) for x in sel.split(","))
    except ValueError:
        return jsonify(ok=False, msg="Bad color string"), 400

    step      = float(request.form.get("step", 10.0))
    db_choice = request.form.get("db_choice")

    full_txt  = read_color_file("color.txt")
    all_dbs   = parse_color_db(full_txt)
    raw_list  = all_dbs.get(db_choice, [])
    base_dict = {name: list(rgb) for name, rgb in raw_list}

    recipes = generate_recipes(target, base_dict, step=step)

    payload = []
    for recipe, mixed, err in recipes:
        payload.append({
            "recipe": [{"name": n, "perc": p} for n, p in recipe if p > 0],
            "mix": list(mixed),
            "error": err
        })
    return jsonify(ok=True, recipes=payload)

@app.route("/api/sum", methods=["GET"])
def api_sum_get():
    a = request.args.get("a", type=float)
    b = request.args.get("b", type=float)
    if a is None or b is None:
        return jsonify(error="Please provide numeric 'a' and 'b' query parameters"), 400
    return jsonify(sum=a + b)


@app.route("/api/sum", methods=["POST"])
def api_sum_post():
    data = request.get_json(silent=True)
    if not data or "a" not in data or "b" not in data:
        return jsonify(error="Please send JSON with 'a' and 'b' fields"), 400
    try:
        a = float(data["a"])
        b = float(data["b"])
    except (TypeError, ValueError):
        return jsonify(error="'a' and 'b' must be numbers"), 400
    return jsonify(sum=a + b)



import traceback
@app.errorhandler(Exception)
def handle_all_exceptions(e):
    """
    Catch any uncaught exception and return it (plus a stack trace)
    as JSON so clients can see exactly what failed.
    """
    tb = traceback.format_exc()
    return jsonify(
        error=str(e),
        traceback=tb
    ), 50
@app.route("/api/recipes", methods=["POST"])
def api_generate_recipes():
    data = request.get_json(silent=True)
    if not data or "target" not in data or "db_choice" not in data:
        return jsonify(error="Please send JSON with 'target' and 'db_choice'"), 400

    # 1) Parse inputs
    try:
        target = tuple(int(x) for x in data["target"])
        step   = float(data.get("step", 10.0))
        dbc    = data["db_choice"]
    except (TypeError, ValueError):
        return jsonify(error="Bad 'target' format or 'step' not a number"), 400

    # 2) Load & lookup database
    full_txt = read_color_file("color.txt")
    all_dbs  = parse_color_db(full_txt)
    raw_list = all_dbs.get(dbc)
    if raw_list is None:
        return jsonify(error=f"Unknown db_choice '{dbc}'"), 400

    # base_dict maps name → [r,g,b]
    base_dict = {name: list(rgb) for name, rgb in raw_list}

    # 3) Generate recipes
    recipes = generate_recipes(target, base_dict, step=step)

    # 4) Build response with rgb for each component
    payload = []
    for rec_list, mixed_rgb, err in recipes:
        components = []
        for name, perc in rec_list:
            rgb = base_dict.get(name, [0,0,0])
            components.append({
                "name": name,
                "rgb": rgb,
                "perc": perc
            })
        payload.append({
            "recipe": components,
            "mix": mixed_rgb,
            "error": err
        })

    return jsonify(recipes=payload), 200

@app.route("/api/merge_colors", methods=["POST"])
def api_merge_colors():
    data = request.get_json(silent=True)
    if not data or "colors" not in data:
        return jsonify(error="Please send JSON with 'colors' list"), 400

    try:
        colors = data["colors"]
        # reuse your colour-merger logic
        # build latent mix
        total_w = sum(c["weight"] for c in colors)
        z_mix = [0] * mixbox.LATENT_SIZE
        for i in range(len(z_mix)):
            z_mix[i] = sum(
                c["weight"] * mixbox.rgb_to_latent(c["rgb"])[i]
                for c in colors
            ) / total_w
        mixed_rgb = mixbox.latent_to_rgb(z_mix)
    except Exception as e:
        return jsonify(error=f"Invalid payload: {e}"), 400

    return jsonify(mixed_rgb=list(mixed_rgb)), 200
@app.route("/api/oil_paint", methods=["POST"])
def api_oil_paint():
    try:
        # 1) Check file
        file = request.files.get("oil_image")
        if not file:
            return jsonify(error="No 'oil_image' file provided"), 400

        # 2) Parse & coerce intensity to int
        try:
            intensity = int(float(request.form.get("intensity", 10.0)))
        except ValueError:
            return jsonify(error="'intensity' must be a number"), 400

        # 3) Decode upload (validates extension, reads into BGR array)
        img_bgr, path, err = save_and_decode(file, subdir="oil_painting")
        if err:
            return jsonify(error=err), 400

        # 4) Run your oil-paint filter
        painted = oil_main(img_bgr, intensity)
        painted = (painted * 255).astype(np.uint8)
        rgb_img = cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)

        # 5) Encode result as base64‐PNG
        buf = BytesIO()
        Image.fromarray(rgb_img).save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify(result_image=result_b64), 200

    except Exception as e:
        # This will be caught by handle_all_exceptions too, but here we add local traceback
        return jsonify(error=str(e), traceback=traceback.format_exc()), 500
@app.route("/api/foogle_art", methods=["POST"])
def api_foogle_art():
    """
    Expects multipart/form-data:
      - image: file
      - shape_type: "Circles"|"Rectangles"|"Triangles"
      - min_size, max_size, num_shapes: ints
    Returns JSON:
      {
        "shapes": [
          {
            "type": "Circle",
            "center": [x, y],
            "radius": r,
            "color": [R, G, B]
          },
          {
            "type": "Rectangle",
            "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
            "color": [R, G, B]
          },
          {
            "type": "Triangle",
            "points": [[x1,y1],[x2,y2],[x3,y3]],
            "color": [R, G, B]
          },
          …
        ]
      }
    """
    try:
        file = request.files.get("image")
        if not file:
            return jsonify(error="No 'image' file provided"), 400

        # decode upload
        img_bgr, path, err = save_and_decode(file, subdir="foogle_man")
        if err:
            return jsonify(error=err), 400

        # parameters
        shape_type = request.form.get("shape_type", "Circles")
        try:
            min_size   = int(request.form.get("min_size", 5))
            max_size   = int(request.form.get("max_size", 30))
            num_shapes = int(request.form.get("num_shapes", 100))
        except ValueError:
            return jsonify(error="min_size, max_size, num_shapes must be integers"), 400

        # prepare input for drawing
        proc, scale = resize_for_processing(img_bgr)
        block_size  = (min_size + max_size) // 5
        pix         = pixelate_image(proc, block_size)

        # we'll record each shape here
        shapes = []
        out = pix.copy()

        import random

        if shape_type == "Circles":
            for _ in range(num_shapes):
                r = random.randint(min_size, max_size)
                x = random.randint(r, out.shape[1] - r)
                y = random.randint(r, out.shape[0] - r)
                # BGR→RGB
                b, g, r_col = out[y, x].tolist()
                shapes.append({
                    "type": "Circle",
                    "center": [x, y],
                    "radius": r,
                    "color": [r_col, g, b]
                })
                cv2.circle(out, (x, y), r, (b, g, r_col), -1)

        elif shape_type == "Rectangles":
            for _ in range(num_shapes):
                rw = random.randint(min_size, max_size)
                rh = random.randint(min_size, max_size)
                x  = random.randint(0, out.shape[1] - rw)
                y  = random.randint(0, out.shape[0] - rh)
                pts = np.array([[x, y],
                                [x+rw, y],
                                [x+rw, y+rh],
                                [x, y+rh]], dtype=np.int32)
                b, g, r_col = out[y, x].tolist()
                shapes.append({
                    "type": "Rectangle",
                    "points": pts.tolist(),
                    "color": [r_col, g, b]
                })
                cv2.fillPoly(out, [pts], (b, g, r_col))

        else:  # Triangles
            for _ in range(num_shapes):
                side = random.randint(min_size, max_size)
                tri_h = int(side * math.sqrt(3) / 2)
                x = random.randint(0, out.shape[1] - side)
                y = random.randint(tri_h, out.shape[0])
                tri = np.array([
                    [x,       y],
                    [x + side, y],
                    [x + side//2, y - tri_h]
                ], dtype=np.int32)
                # pick color at one vertex
                b, g, r_col = out[y - tri_h//2, x + side//2].tolist()
                shapes.append({
                    "type": "Triangle",
                    "points": tri.tolist(),
                    "color": [r_col, g, b]
                })
                cv2.fillPoly(out, [tri], (b, g, r_col))

        # scale back if needed
        if scale < 1.0:
            out = cv2.resize(
                out,
                (img_bgr.shape[1], img_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        return jsonify(shapes=shapes), 200

    except Exception as e:
        import traceback
        return jsonify(error=str(e), traceback=traceback.format_exc()), 500





@app.route("/api/shape_art", methods=["POST"])
def api_shape_art():
    """
    API equivalent of image_generator_page Part A:
      - Accepts multipart/form-data:
          • image: file
          • shape_type: "Triangle"|"Circles"|"Rectangles"
          • num_shapes: int
          • min_size: int
          • max_size: int
      - Returns JSON { "image": "<Base64-PNG>" }
      - Side-effect: saves the same PNG to disk & sets session["shape_art_path"]
                     so your unchanged download_shape_art and shape_detector_page work.
    """
    # 1) Validate upload
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify(error="No image file provided"), 400

    # 2) Parse parameters
    shape_type = request.form.get("shape_type", "Triangle")
    try:
        num_shapes = int(request.form.get("num_shapes", 100))
        min_size   = int(request.form.get("min_size", 10))
        max_size   = int(request.form.get("max_size", 50))
    except ValueError:
        return jsonify(error="num_shapes, min_size and max_size must be integers"), 400

    # 3) Decode upload into BGR array
    img_bgr, _, err = save_and_decode(file, subdir="shape_art")
    if err:
        return jsonify(error=err), 400

    # 4) Generate shape art exactly as in image_generator_page
    try:
        encoded_img, _ = encode(
            img_bgr,
            shape_type,
            output_path="",
            num_shapes=num_shapes,
            min_size=min_size,
            max_size=max_size,
        )
    except Exception as e:
        return jsonify(error=f"Error generating shape art: {e}"), 500

    # 5) Convert to PNG → Base64
    rgb_img = cv2.cvtColor(encoded_img, cv2.COLOR_BGR2RGB)
    buf     = BytesIO()
    Image.fromarray(rgb_img).save(buf, format="PNG")
    b64     = base64.b64encode(buf.getvalue()).decode("ascii")

    # 6) Save the same PNG to disk & session["shape_art_path"]
    tmp_name = f"shape_art_{os.getpid()}.png"
    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
    with open(tmp_path, "wb") as out:
        out.write(buf.getvalue())
    session["shape_art_path"] = tmp_path

    # 7) Return JSON
    return jsonify(image=b64), 200
@app.route("/api/shape_detect", methods=["POST"])
def api_shape_detect():
    """
    POST supports two modes:
    1) multipart/form-data with file field 'encoded_image'
    2) application/json with keys:
         - image:      Base64‐PNG string
         - shape_type: "Triangle"|"Circles"|"Rectangles"
         - min_size:   int
         - max_size:   int
    Returns JSON:
      {
        "annotated_image": "<Base64‐PNG>",
        "grouped_colors": [
          { "color": [R,G,B], "count": N },
          …
        ]
      }
    """
    try:
        # ── 1) Read parameters ───────────────────────────────
        # Default values
        shape_opt = None
        min_size  = None
        max_size  = None

        # If JSON body:
        if request.is_json:
            data = request.get_json()
            b64 = data.get("image")
            shape_opt = data.get("shape_type", "Triangle")
            try:
                min_size = int(data.get("min_size", 3))
                max_size = int(data.get("max_size", 10))
            except (TypeError, ValueError):
                return jsonify(error="min_size and max_size must be integers"), 400

            if not b64:
                return jsonify(error="Missing Base64 'image'"), 400

            # Decode Base64 → bytes → numpy array → BGR
            img_bytes = base64.b64decode(b64)
            arr       = np.frombuffer(img_bytes, np.uint8)
            img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return jsonify(error="Could not decode Base64 image"), 400

        else:
            # multipart upload
            file = request.files.get("encoded_image")
            if not file or file.filename == "":
                return jsonify(error="No 'encoded_image' file provided"), 400
            shape_opt = request.form.get("shape_type", "Triangle")
            try:
                min_size = int(request.form.get("min_size", 3))
                max_size = int(request.form.get("max_size", 10))
            except ValueError:
                return jsonify(error="min_size and max_size must be integers"), 400

            # Use your helper to read & decode
            img_bgr, _, err = save_and_decode(file, subdir="shape_detector")
            if err:
                return jsonify(error=err), 400

        # ── 2) Run your EnDe.decode ────────────────────────────
        _, annotated_img, rgb_vals = decode(
            img_bgr,
            shape_opt,
            boundaries=[],
            min_size=min_size,
            max_size=max_size
        )

        # ── 3) Encode annotated preview as Base64‐PNG ───────────
        ann_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        buf     = BytesIO()
        Image.fromarray(ann_rgb).save(buf, format="PNG")
        annotated_b64 = base64.b64encode(buf.getvalue()).decode()

        # ── 4) Group similar colors ─────────────────────────────
        rgb_list = [list(map(int, c)) for c in rgb_vals]
        grouped  = sorted(
            group_similar_colors(rgb_list, threshold=10),
            key=lambda x: x[1],
            reverse=True
        )
        grouped_list = [{"color": col, "count": cnt} for col, cnt in grouped]

        return jsonify(
            annotated_image=annotated_b64,
            grouped_colors=grouped_list
        ), 200

    except Exception as e:
        import traceback
        return jsonify(
            error=str(e),
            traceback=traceback.format_exc()
        ), 500

# ─────────────────────── helpers reused from FastAPI version ───────────────
def _prepare_primitive_binary() -> str:
    """Return path to the `primitive` CLI, downloading the Windows build
    on-demand if we are on a Win32 host; otherwise assume it is in $PATH."""
    if platform.system().lower().startswith("win"):
        bin_path = os.path.join(os.path.dirname(__file__), "primitive.exe")
        if not os.path.exists(bin_path):
            print("Downloading primitive binary for Windows …")
            url = ("https://github.com/fogleman/primitive/releases/download/"
                   "v0.1.1/primitive_windows_amd64.exe")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with open(bin_path, "wb") as f:
                f.write(resp.content)
            # mark executable
            os.chmod(bin_path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        return bin_path
    return "primitive"                       # assume *nix package / PATH

PRIMITIVE_CMD = _prepare_primitive_binary()
HEX_RE = re.compile(r"^[0-9a-fA-F]{6}$")

def hex_from_color(color: str) -> str:
    """Accept '#rrggbb', 'rrggbb', 'rgb(r,g,b)', or 'r,g,b'. Return lowercase 6-hex."""
    if not color:
        return "ffffff"
    if color.startswith("#"):
        color = color[1:]
    if HEX_RE.fullmatch(color):
        return color.lower()
    if color.lower().startswith("rgb"):
        nums = re.findall(r"\d+", color)
        if len(nums) >= 3:
            r, g, b = map(int, nums[:3])
            return f"{r:02x}{g:02x}{b:02x}"
    if "," in color:
        try:
            r, g, b, *_ = map(int, color.split(","))
            return f"{r:02x}{g:02x}{b:02x}"
        except ValueError:
            pass
    raise ValueError("Invalid background_color format")
import xml.etree.ElementTree as ET
import re

# ─── Internal helpers ────────────────────────────────────────────
_HEX_RE = re.compile(r'^#?[0-9a-fA-F]{6}$')

def _norm_hex(col: str) -> str:
    """
    Normalise colour strings to '#rrggbb' (lower-case, leading #).
    Accepts '#rrggbb', 'rrggbb', 'rgb(r,g,b)', or 'r,g,b'.
    """
    if not col:
        return "#000000"

    if _HEX_RE.fullmatch(col):
        return f"#{col.lstrip('#').lower()}"

    if col.lower().startswith("rgb"):
        nums = list(map(int, re.findall(r'\d+', col)))
        if len(nums) >= 3:
            r, g, b = nums[:3]
            return f"#{r:02x}{g:02x}{b:02x}"

    if "," in col:
        try:
            r, g, b, *_ = map(int, col.split(","))
            return f"#{r:02x}{g:02x}{b:02x}"
        except ValueError:
            pass

    return "#000000"

# ─── Main converter ──────────────────────────────────────────────
def _svg_to_json(svg_txt: str) -> dict:
    """
    Convert Primitive-style SVG text into the geometrizer JSON schema.

    Returns a dict with keys:
        • shapes            – list[…]
        • background_color  – '#rrggbb'
        • canvas_size       – [width, height]
    """
    if not svg_txt.strip().startswith("<svg"):
        raise ValueError("Input does not appear to be valid SVG")

    root = ET.fromstring(svg_txt)
    width  = int(float(root.attrib.get("width", 0)))
    height = int(float(root.attrib.get("height", 0)))

    # Background colour: first <rect> that spans the canvas
    bg_hex = "#ffffff"
    for el in root:
        if el.tag.lower().endswith("rect"):
            bg_hex = _norm_hex(el.attrib.get("fill", "#ffffff"))
            break

    shapes = []
    for el in root.iter():
        tag = el.tag.lower().split('}')[-1]      # strip any SVG namespace
        if tag in {"svg", "rect"}:
            continue                             # skip root & background

        fill     = _norm_hex(el.attrib.get("fill", "#000000"))
        opacity  = int(float(el.attrib.get("fill-opacity", "1")) * 255)

        if tag == "polygon":
            pts = [[*map(float, p.split(','))]
                   for p in el.attrib["points"].split()]
            shape_type = "triangle" if len(pts) == 3 else "rectangle"
            shapes.append({
                "type":     shape_type,
                "color":    fill,
                "opacity":  opacity,
                "points":   pts,
            })

        elif tag == "ellipse":
            cx = float(el.attrib.get("cx", 0))
            cy = float(el.attrib.get("cy", 0))
            rx = float(el.attrib.get("rx", 0))
            ry = float(el.attrib.get("ry", 0))
            shape_type = "circle" if abs(rx - ry) < 1e-6 else "ellipse"
            shapes.append({
                "type":     shape_type,
                "color":    fill,
                "opacity":  opacity,
                "center":   [cx, cy],
                "radius":   rx if shape_type == "circle" else [rx, ry],
            })

        elif tag == "path":
            shapes.append({
                "type":     "quadratic_bezier",
                "color":    fill,
                "opacity":  opacity,
                "path":     el.attrib.get("d", ""),
            })

    return {
        "shapes":            shapes,
        "background_color":  bg_hex,
        "canvas_size":       [width, height],
    }

# ────────────────────────────── /api/generate ──────────────────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Parameters (multipart-form or application/x-www-form-urlencoded):

        image=<file>             – optional; mutually exclusive with image_base64
        image_base64=<string>    – optional; base-64-encoded PNG/JPEG/GIF
        output_format=png|svg|json   [required]
        shape_types=triangle|rectangle|ellipse|circle|rotated_rectangle|rotated_ellipse (repeatable)
        opacity=0-255                  (default 128)
        shape_count=int>0              (default 200)
        background_color=#rrggbb|rgb(r,g,b)|r,g,b  (default white)
        resize_width=int, resize_height=int (optional)
    """
    # ---------------- read request body ----------------
    img_file = request.files.get("image")
    img_b64  = request.form.get("image_base64")
    if img_file:
        contents = img_file.read()
        orig_ext = os.path.splitext(img_file.filename)[1] or ".png"
    elif img_b64:
        try:
            if img_b64.startswith("data:image"):
                img_b64 = img_b64.split(",", 1)[1]
            contents = base64.b64decode(img_b64)
        except Exception as exc:
            return jsonify(detail=f"Invalid base64 image: {exc}"), 400
        orig_ext = ".png"
    else:
        return jsonify(detail="Image file or base64 string is required."), 400

    output_format = (request.form.get("output_format") or "").lower()
    if output_format not in {"png", "svg", "json"}:
        return jsonify(detail="output_format must be 'png', 'svg', or 'json'"), 400

    shape_types: List[str] = request.form.getlist("shape_types")
    opacity       = int(request.form.get("opacity", 128))
    shape_count   = int(request.form.get("shape_count", 200))
    background    = request.form.get("background_color")
    resize_w      = request.form.get("resize_width",  type=int)
    resize_h      = request.form.get("resize_height", type=int)
    if not (0 <= opacity <= 255):
        return jsonify(detail="opacity must be 0-255"), 400
    if shape_count <= 0:
        return jsonify(detail="shape_count must be positive"), 400

    # ---------------- temp dir & optional resize ----------------
    tmp = tempfile.mkdtemp()
    try:
        inp_path = os.path.join(tmp, f"input{orig_ext}")
        with open(inp_path, "wb") as f:
            f.write(contents)

        if resize_w or resize_h:
            with Image.open(inp_path) as im:
                im = im.resize((resize_w or im.width, resize_h or im.height))
                im.save(inp_path)

        # ---------------- build primitive CLI args ----------------
        mapping = {
            "triangle": 1, "rectangle": 2, "ellipse": 3, "circle": 4,
            "rotated_rectangle": 5, "rotated_ellipse": 7
        }
        mode = 0
        if shape_types:
            try:
                modes = [mapping[st.lower()] for st in shape_types]
            except KeyError as bad:
                return jsonify(detail=f"Unsupported shape type: {bad}"), 400
            mode = modes[0] if len(modes) == 1 else 0

        bg_hex = hex_from_color(background) if background else "ffffff"
        out_path = os.path.join(tmp, "output." + ("png" if output_format == "png" else "svg"))

        cmd = [
            PRIMITIVE_CMD, "-i", inp_path, "-o", out_path,
            "-n", str(shape_count), "-m", str(mode), "-a", str(opacity), "-bg", bg_hex
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("primitive stderr:", result.stderr)
            return jsonify(detail=f"Primitive failed: {result.stderr}"), 500

        # ---------------- format-specific response ----------------
        if output_format == "png":
            return Response(open(out_path, "rb").read(), mimetype="image/png")

        if output_format == "svg":
            return Response(open(out_path, "r", encoding="utf-8").read(),
                            mimetype="image/svg+xml")
        if output_format == "json":
            svg_txt = open(out_path, "r", encoding="utf-8").read()
            return jsonify(_svg_to_json(svg_txt))
        # JSON: parse basic shapes from SVG
        import xml.etree.ElementTree as ET
        svg_data = open(out_path, "r", encoding="utf-8").read()
        root = ET.fromstring(svg_data)
        shapes = []
        for el in root:
            tag   = el.tag.lower().split("}", 1)[-1]  # strip xmlns
            fill  = el.attrib.get("fill", "#000000")
            alpha = float(el.attrib.get("fill-opacity", "1"))
            op    = int(alpha * 255)
            if tag == "polygon":
                pts = [[*map(float, p.split(","))] for p in el.attrib.get("points", "").split()]
                t   = "triangle" if len(pts) == 3 else "rectangle"
                shapes.append({"type": t, "color": fill, "opacity": op, "points": pts})
            elif tag == "ellipse":
                cx, cy = float(el.attrib["cx"]), float(el.attrib["cy"])
                rx, ry = float(el.attrib["rx"]), float(el.attrib["ry"])
                t   = "circle" if abs(rx - ry) < 1e-6 else "ellipse"
                shapes.append({"type": t, "color": fill, "opacity": op,
                               "center": [cx, cy], "radius": rx if t == "circle" else [rx, ry]})
            elif tag == "path":
                shapes.append({"type": "quadratic_bezier", "color": fill,
                               "opacity": op, "path": el.attrib.get("d", "")})

        with Image.open(inp_path) as im:
            canvas = [im.width, im.height]

        return jsonify(shapes=shapes, background_color=f"#{bg_hex}", canvas_size=canvas)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─────────── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
