import os
import base64
import itertools
import math
from io import BytesIO
# at the top of app.py
import colormixer as mixbox

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
    with open(path, encoding="utf8") as f:
        return f.read()


def parse_color_db(txt: str):
    dbs, cur = {}, None
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line[0].isdigit():
            cur = line
            dbs[cur] = []
        else:
            tok = line.split()
            if len(tok) < 3:
                continue
            parts = tok[-2].split(",")
            if len(parts) != 3:
                continue
            try:
                r, g, b = map(int, parts)
            except ValueError:
                continue
            name = " ".join(tok[1:-2])
            dbs[cur].append((name, (r, g, b)))
    return dbs

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

# ────────────────────────────────────────────────────────────────────
#  /shape_detector  – shared-image + instant-preview backend
# --------------------------------------------------------------------
#  • If the user uploads a new file → save it and remember its path
#    in session["shared_img_path"]  so every page can reuse it.
#  • If no file is chosen but we already have session["shared_img_path"]
#    → reopen that cached file and run the decode.
#  • If neither is available → show an error asking for an upload.
#  • Keeps your recipe-generation branch exactly as before.
# ────────────────────────────────────────────────────────────────────
@app.route("/shape_detector", methods=["GET", "POST"])
def shape_detector_page():
    error                = None
    decoded_image_data   = None
    grouped_colors       = session.get("grouped_colors", [])
    selected_recipe_color= None
    recipe_results       = None

    # ─────────────────────────── PART A: Decode ───────────────────────────
    if request.method == "POST" and request.form.get("action") != "generate_recipe":
        file = request.files.get("encoded_image")            # may be empty
        img_bgr = None
        path    = None

        # --- A1. NEW upload ------------------------------------------------
        if file and file.filename:
            if not allowed_file(file.filename):
                error = "Unsupported file type."
            else:
                filename = secure_filename(file.filename)
                path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                session["shared_img_path"] = path            # ❶ remember
                file_bytes = np.fromfile(path, dtype=np.uint8)
                img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    error = "Failed to read uploaded image."

        # --- A2. REUSE cached upload --------------------------------------
        else:
            path = session.get("shared_img_path")
            if path and os.path.exists(path):
                img_bgr = cv2.imread(path)
            else:
                error = "Please upload an encoded PNG/JPG first."

        # --- A3. Run decode if we have an image ---------------------------
        if img_bgr is not None and error is None:
            shape_opt = request.form.get("shape_detect", "Triangle")
            min_size  = int(request.form.get("min_size", 3))
            max_size  = int(request.form.get("max_size", 10))

            _, annotated_img, rgb_vals = decode(
                img_bgr, shape_opt,
                boundaries=[], min_size=min_size, max_size=max_size
            )

            # ▶ encode annotated preview
            ann_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            buf     = BytesIO()
            Image.fromarray(ann_rgb).save(buf, format="PNG")
            decoded_image_data = base64.b64encode(buf.getvalue()).decode()

            # ▶ group colours & stash in session so recipe-side can reuse
            rgb_py   = [list(map(int, col)) for col in rgb_vals]
            grouped  = sorted(group_similar_colors(rgb_py, threshold=10),
                              key=lambda x: x[1], reverse=True)
            grouped_colors              = grouped
            session["grouped_colors"]   = grouped

            # ▶ keep annotated PNG for “Download”
            tmp_name = f"shape_analysis_{os.getpid()}.png"
            tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
            with open(tmp_path, "wb") as f_out:
                f_out.write(buf.getvalue())
            session["analysis_path"] = tmp_path

    # ──────────────────────── PART B: Generate recipe ──────────────────────
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

    # ───────────────────────────── render ─────────────────────────────
    return render_template(
        "shape_detector.html",
        error                 = error,
        decoded_image_data    = decoded_image_data,
        grouped_colors        = grouped_colors,
        selected_recipe_color = selected_recipe_color,
        recipe_results        = recipe_results,
        db_list               = list(parse_color_db(read_color_file("color.txt")).keys()),
        active_page           = "shape_detector",
        shared_img_exists     = bool(session.get("shared_img_path")),   # ❷ flag for template
    )


@app.route("/download_analysis")
def download_analysis():
    path = session.get("analysis_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return redirect(url_for("shape_detector_page"))
@app.route("/download_oil")
def download_oil():
    """
    Send the last oil-painting PNG that was stored in session['oil_path'].
    Falls back to the Oil-Painting page if the file is missing.
    """
    path = session.get("oil_path")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    # nothing to download → bounce back to the page
    return redirect(url_for("oil_painting_page"))
@app.route("/oil_painting", methods=["GET", "POST"])
def oil_painting_page():
    """
    GET  → show the form (and maybe the last result preview)
    POST → either run oil-paint or generate paint recipes, depending on `action`
    """
    error                 = None
    original_image_data   = None
    result_image_data     = None
    intensity             = int(request.form.get("intensity", 10))

    # For the recipe generator
    selected_recipe_color = None
    recipe_results        = None
    # Load available colour DB names once
    db_list               = list(parse_color_db(read_color_file("color.txt")).keys())

    # ─── PART A: Oil‐paint generation ──────────────────────────────────
    # Triggered when action!="generate_recipe"
    if request.method == "POST" and request.form.get("action") != "generate_recipe":
        img_bgr = None
        # 1. New upload?
        file = request.files.get("oil_image")
        if file and file.filename:
            img_bgr, path, err = save_and_decode(file, subdir="oil_painting")
            if err:
                error = err
            else:
                session["shared_img_path"] = path
        # 2. Reuse shared image
        if img_bgr is None and error is None:
            shared_path = session.get("shared_img_path")
            if shared_path and os.path.exists(shared_path):
                img_bgr = cv2.imread(shared_path)
            else:
                error = "ERROR"

        # 3. Encode previews & run filter
        if img_bgr is not None and error is None:
            # stash original preview
            buf = BytesIO()
            orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            Image.fromarray(orig_rgb).save(buf, format="PNG")
            original_image_data = base64.b64encode(buf.getvalue()).decode()

            # run the oil filter
            try:
                painted = oil_main(img_bgr, intensity)
                painted = (painted * 255).astype(np.uint8)
                rgb_img = cv2.cvtColor(painted, cv2.COLOR_BGR2RGB)

                buf = BytesIO()
                Image.fromarray(rgb_img).save(buf, format="PNG")
                result_image_data = base64.b64encode(buf.getvalue()).decode()

                # store for download
                tmp_name = f"oil_painting_{os.getpid()}.png"
                tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], tmp_name)
                with open(tmp_path, "wb") as f:
                    f.write(buf.getvalue())
                session["oil_path"] = tmp_path

            except Exception as e:
                error = f"Error generating oil painting: {e}"

    # ─── PART B: Paint‐recipe generation ───────────────────────────────
    # Triggered when action=="generate_recipe"
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

    # ─── Render everything to the template ─────────────────────────────
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
        # build a zeroed latent of the correct size
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
    """
    Standalone paint-recipe generator.
    """
    error = None
    recipes = None
    selected_color = (255, 0, 0)

    full_txt = read_color_file("color.txt")
    databases = parse_color_db(full_txt)
    db_keys = list(databases.keys())

    if request.method == "POST":
        # grab the posted hex (instead of non‐existent 'target_hex')
        hex_color = request.form.get("hex_color")
        if hex_color:
            # strip leading ‘#’ then parse RRGGBB
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
    # 1) Read & parse the file
    full_txt = read_color_file("color.txt")
    databases = parse_color_db(full_txt)  # returns { db_name: [(color_name, (r,g,b)), …], … }

    # 2) Decide which sub‐section to show
    action = request.form.get("action", "browse")
    if action == "browse":
        subpage = "databases"
    elif action == "add":
        subpage = "add"
    elif action == "remove_colors":
        subpage = "remove_colors"
    elif action == "create_db":
        subpage = "custom"
    elif action == "remove_db":
        subpage = "remove_database"
    else:
        subpage = "databases"

    # (optional) you can pass a message tuple (type, text) if you need feedback
    message = None

    return render_template(
        "colors_db.html",
        databases=databases,    # a dict for your template to iterate over
        subpage=subpage,        # controls which form/block shows
        message=message,
        active_page="colors_db"
    )



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

def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return ext in {"png", "jpg", "jpeg", "webp"}

# ─────────── Route ────────────────────────────────────────────────────


def get_db_list():
    # return just the list of DB names
    full_txt = read_color_file("color.txt")
    all_dbs  = parse_color_db(full_txt)    # returns { name: [...], … }
    return list(all_dbs.keys())


@app.route("/foogle_man_repo", methods=["GET", "POST"])
def foogle_man_repo_page():
    # ✧ 1. Art‐generation tab
    original_b64   = None
    generated_b64  = None
    download_url   = None
    num_shapes     = 0

    # ✧ 2. Paint‐recipe tab
    error                  = None
    recipe_results         = None     # list of (recipe, mixedRGB, err)
    selected_recipe_color  = None     # rename here
    db_list                = get_db_list()

    if request.method == "POST":
        # ── A) Paint‐recipe branch ───────────────────────────
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

        # ── B) Image‐to‐art branch ────────────────────────────
        else:
            img_bgr = None
            file    = request.files.get("image")

            # B1: New upload?
            if file and file.filename:
                img_bgr, path, err = save_and_decode(file, subdir="foogle_man")
                if err:
                    error = err
                else:
                    session["shared_img_path"] = path

            # B2: Reuse shared
            if img_bgr is None and not error:
                path = session.get("shared_img_path")
                if path and os.path.exists(path):
                    img_bgr = cv2.imread(path)
                else:
                    error = "Please upload an image first (on any page)."

            # B3: Generate art
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

                # original preview
                buf = BytesIO()
                Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))\
                     .save(buf, format="PNG")
                original_b64 = base64.b64encode(buf.getvalue()).decode()

                # generated preview
                buf = BytesIO()
                Image.fromarray(cv2.cvtColor(art, cv2.COLOR_BGR2RGB))\
                     .save(buf, format="PNG")
                generated_b64 = base64.b64encode(buf.getvalue()).decode()

                download_url = f"data:image/png;base64,{generated_b64}"

    # ─── Render ───────────────────────────────────────────────
    return render_template(
        "foogle_man_repo.html",

        # art tab
        original_image    = original_b64,
        generated_image   = generated_b64,
        num_shapes        = num_shapes,
        download_url      = download_url,

        # recipe tab
        error                   = error,
        db_list                 = db_list,
        selected_recipe_color   = selected_recipe_color,  # pass to HTML
        recipe_results          = recipe_results,

        # shared‐image flag
        shared_img_exists = bool(session.get("shared_img_path")),
    )


@app.route("/paint_geometrize")
def paint_geometrize_page():
    """
    1. GET:  render empty form
    2. POST: parse selected_color, db_choice, step → generate_recipes → re-render with results
    """
    error               = None
    recipe_results      = None
    selected_recipe_rgb = None

    # load DB choices for dropdown on every render
    db_list = get_db_list()

    if request.method == "POST" and request.form.get("action") == "generate_recipe":
        sel = request.form.get("selected_color", "")
        try:
            r, g, b = [int(x.strip()) for x in sel.split(",")]
            selected_recipe_rgb = (r, g, b)
        except ValueError:
            error = "Invalid RGB—please click on the image to pick a colour."
        else:
            # parse step & db_choice
            try:
                step = float(request.form.get("step", 10.0))
            except ValueError:
                step = 10.0
            db_choice = request.form.get("db_choice", "")
            if db_choice not in db_list:
                error = f"Unknown colour DB '{db_choice}'."
            else:
                # load that database
                full_txt = read_color_file("color.txt")
                all_dbs   = parse_color_db(full_txt)
                base_dict = {name: tuple(rgb) for name, rgb in all_dbs[db_choice]}

                # compute recipes
                recipe_results = generate_recipes(selected_recipe_rgb, base_dict, step=step)

                # stringify any tuple mixes for JSON/template safety
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


# ═══════════ NEW AJAX ENDPOINT ═══════════
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
# ══════════════════════════════════════════

    
# ─────────── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get port from environment (Render sets this)
    app.run(host="0.0.0.0", port=port, debug=True)
    
