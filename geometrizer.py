import math
import random
import numpy as np
from PIL import Image, ImageDraw
from numba import njit

# -----------------------------------------------
# Utility Functions
# -----------------------------------------------

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

@njit
def image_difference_numba(arrA, arrB):
    diff = 0
    for i in range(arrA.shape[0]):
        for j in range(arrA.shape[1]):
            for k in range(4):  # RGBA channels
                d = int(arrA[i, j, k]) - int(arrB[i, j, k])
                diff += d * d
    return diff


def image_difference(imgA, imgB):
    """
    Compute the sum of squared differences (SSD) between two RGBA images.
    """
    arrA = np.array(imgA, dtype=np.uint8)
    arrB = np.array(imgB, dtype=np.uint8)
    return image_difference_numba(arrA, arrB)


def blend_image(base_img, shape_img):
    """
    Alpha-blend 'shape_img' on top of 'base_img' (both RGBA, same size).
    """
    return Image.alpha_composite(base_img, shape_img)

# -----------------------------------------------
# Shape Base Class and Implementations
# -----------------------------------------------

class BaseShape:
    def __init__(self):
        self.color = (255, 0, 0, 128)  # Default RGBA
    
    def randomize_color(self):
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(20, 200)
        )
    
    def copy(self):
        raise NotImplementedError
    
    def randomize(self, width, height):
        raise NotImplementedError
    
    def mutate(self, width, height, amount=1.0):
        raise NotImplementedError
    
    def rasterize(self, width, height):
        raise NotImplementedError


class TriangleShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.points = [(0, 0), (0, 0), (0, 0)]
    
    def copy(self):
        new_shape = TriangleShape()
        new_shape.color = self.color
        new_shape.points = list(self.points)
        return new_shape
    
    def randomize(self, width, height):
        self.randomize_color()
        self.points = [
            (random.randint(0, width-1), random.randint(0, height-1)),
            (random.randint(0, width-1), random.randint(0, height-1)),
            (random.randint(0, width-1), random.randint(0, height-1))
        ]
    
    def mutate(self, width, height, amount=1.0):
        if random.random() < 0.3:
            r, g, b, a = self.color
            r = clamp(r + random.randint(-15, 15), 0, 255)
            g = clamp(g + random.randint(-15, 15), 0, 255)
            b = clamp(b + random.randint(-15, 15), 0, 255)
            a = clamp(a + random.randint(-15, 15), 20, 255)
            self.color = (r, g, b, a)
        new_points = []
        for (x, y) in self.points:
            if random.random() < 0.5:
                x = clamp(x + int(random.randint(-5, 5) * amount), 0, width-1)
            if random.random() < 0.5:
                y = clamp(y + int(random.randint(-5, 5) * amount), 0, height-1)
            new_points.append((x, y))
        self.points = new_points
    
    def rasterize(self, width, height):
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, 'RGBA')
        draw.polygon(self.points, fill=self.color)
        return img


class RectangleShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.x1 = self.y1 = self.x2 = self.y2 = 0
    
    def copy(self):
        new_shape = RectangleShape()
        new_shape.color = self.color
        new_shape.x1, new_shape.y1, new_shape.x2, new_shape.y2 = (
            self.x1, self.y1, self.x2, self.y2
        )
        return new_shape
    
    def randomize(self, width, height):
        self.randomize_color()
        self.x1 = random.randint(0, width-1)
        self.y1 = random.randint(0, height-1)
        self.x2 = clamp(self.x1 + random.randint(-50, 50), 0, width-1)
        self.y2 = clamp(self.y1 + random.randint(-50, 50), 0, height-1)
    
    def mutate(self, width, height, amount=1.0):
        if random.random() < 0.3:
            r, g, b, a = self.color
            r = clamp(r + random.randint(-15, 15), 0, 255)
            g = clamp(g + random.randint(-15, 15), 0, 255)
            b = clamp(b + random.randint(-15, 15), 0, 255)
            a = clamp(a + random.randint(-15, 15), 20, 255)
            self.color = (r, g, b, a)
        if random.random() < 0.5:
            self.x1 = clamp(self.x1 + int(random.randint(-5, 5) * amount), 0, width-1)
        if random.random() < 0.5:
            self.y1 = clamp(self.y1 + int(random.randint(-5, 5) * amount), 0, height-1)
        if random.random() < 0.5:
            self.x2 = clamp(self.x2 + int(random.randint(-5, 5) * amount), 0, width-1)
        if random.random() < 0.5:
            self.y2 = clamp(self.y2 + int(random.randint(-5, 5) * amount), 0, height-1)
    
    def rasterize(self, width, height):
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, 'RGBA')
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])
        draw.rectangle([x1, y1, x2, y2], fill=self.color)
        return img


class EllipseShape(BaseShape):
    def __init__(self):
        super().__init__()
        self.x1 = self.y1 = self.x2 = self.y2 = 0
    
    def copy(self):
        new_shape = EllipseShape()
        new_shape.color = self.color
        new_shape.x1, new_shape.y1, new_shape.x2, new_shape.y2 = (
            self.x1, self.y1, self.x2, self.y2
        )
        return new_shape
    
    def randomize(self, width, height):
        self.randomize_color()
        self.x1 = random.randint(0, width-1)
        self.y1 = random.randint(0, height-1)
        self.x2 = clamp(self.x1 + random.randint(-50, 50), 0, width-1)
        self.y2 = clamp(self.y1 + random.randint(-50, 50), 0, height-1)
    
    def mutate(self, width, height, amount=1.0):
        if random.random() < 0.3:
            r, g, b, a = self.color
            r = clamp(r + random.randint(-15, 15), 0, 255)
            g = clamp(g + random.randint(-15, 15), 0, 255)
            b = clamp(b + random.randint(-15, 15), 0, 255)
            a = clamp(a + random.randint(-15, 15), 20, 255)
            self.color = (r, g, b, a)
        if random.random() < 0.5:
            self.x1 = clamp(self.x1 + int(random.randint(-5, 5) * amount), 0, width-1)
        if random.random() < 0.5:
            self.y1 = clamp(self.y1 + int(random.randint(-5, 5) * amount), 0, height-1)
        if random.random() < 0.5:
            self.x2 = clamp(self.x2 + int(random.randint(-5, 5) * amount), 0, width-1)
        if random.random() < 0.5:
            self.y2 = clamp(self.y2 + int(random.randint(-5, 5) * amount), 0, height-1)
    
    def rasterize(self, width, height):
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img, 'RGBA')
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])
        draw.ellipse([x1, y1, x2, y2], fill=self.color)
        return img


def create_shape(shape_type):
    if shape_type == 'triangle':
        return TriangleShape()
    elif shape_type == 'rectangle':
        return RectangleShape()
    elif shape_type == 'ellipse':
        return EllipseShape()
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

# -----------------------------------------------
# Annealing and Refinement
# -----------------------------------------------

def simulated_annealing_shape(base_img, target_img, shape,
                              iterations, start_temp, end_temp,
                              step_scale=1.0):
    width, height = target_img.size
    current_shape = shape.copy()
    shape_img = current_shape.rasterize(width, height)
    blended = blend_image(base_img, shape_img)
    current_diff = image_difference(target_img, blended)
    best_shape = current_shape.copy()
    best_diff = current_diff

    for i in range(iterations):
        T = start_temp * ((end_temp / start_temp) ** (i / iterations))
        new_shape = current_shape.copy()
        new_shape.mutate(width, height, amount=step_scale)
        shape_img = new_shape.rasterize(width, height)
        candidate = blend_image(base_img, shape_img)
        diff = image_difference(target_img, candidate)
        delta = diff - current_diff
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_shape = new_shape
            current_diff = diff
            if diff < best_diff:
                best_shape = new_shape.copy()
                best_diff = diff
    return best_shape, best_diff


def refine_shape(base_img, target_img, shape,
                 coarse_iter, fine_iter,
                 coarse_start_temp, coarse_end_temp,
                 fine_start_temp, fine_end_temp):
    best_shape, best_diff = simulated_annealing_shape(
        base_img, target_img, shape,
        iterations=coarse_iter,
        start_temp=coarse_start_temp,
        end_temp=coarse_end_temp,
        step_scale=1.0
    )
    return simulated_annealing_shape(
        base_img=blend_image(base_img, best_shape.rasterize(*target_img.size)),
        target_img=target_img,
        shape=best_shape,
        iterations=fine_iter,
        start_temp=fine_start_temp,
        end_temp=fine_end_temp,
        step_scale=0.5
    )

# -----------------------------------------------
# Public API: run_geometrize
# -----------------------------------------------

def run_geometrize(target_img, shape_type, shape_count, resize_factor,
                   coarse_iterations=1000, fine_iterations=500,
                   coarse_start_temp=100.0, coarse_end_temp=10.0,
                   fine_start_temp=10.0, fine_end_temp=1.0):
    # Ensure RGBA
    target_img = target_img.convert("RGBA")
    orig_w, orig_h = target_img.size
    if resize_factor != 1.0:
        new_w = int(orig_w * resize_factor)
        new_h = int(orig_h * resize_factor)
        target_img = target_img.resize((new_w, new_h), Image.LANCZOS)
    width, height = target_img.size

    # Start with white canvas
    current_img = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    for i in range(shape_count):
        shape = create_shape(shape_type)
        shape.randomize(width, height)
        best_shape, best_diff = refine_shape(
            base_img=current_img,
            target_img=target_img,
            shape=shape,
            coarse_iter=coarse_iterations,
            fine_iter=fine_iterations,
            coarse_start_temp=coarse_start_temp,
            coarse_end_temp=coarse_end_temp,
            fine_start_temp=fine_start_temp,
            fine_end_temp=fine_end_temp
        )

        # Blend if improved
        shape_img = best_shape.rasterize(width, height)
        current_img = blend_image(current_img, shape_img)

    return current_img
