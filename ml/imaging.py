"""Preprocesamiento de imagen compartido por el entrenamiento y el backend.

Vive en un solo sitio a proposito. Las imagenes de entrenamiento pasaron por
"Fit (reflect edges)" a 240x240, asi que una foto del telefono tiene que
recibir exactamente el mismo tratamiento; si el servidor recortara el centro
en vez de ajustar con bordes reflejados, estaria enviando al modelo imagenes
de un dominio distinto al que aprendio y la exactitud caeria sin aviso.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

TARGET_SIZE = 240          # lado del cuadro que produjo Roboflow
CROP_SIZE = 224            # recorte central que consume el modelo
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def fit_reflect(img: Image.Image, size: int = TARGET_SIZE) -> Image.Image:
    """Encaja la imagen en un cuadro `size`x`size` rellenando con reflejo.

    Conserva el objeto completo y su proporcion, a diferencia de un recorte
    central, que puede cortar justo el envase que se quiere clasificar.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    if (nw, nh) == (size, size):
        return img

    pad_l, pad_t = (size - nw) // 2, (size - nh) // 2
    arr = np.pad(
        np.asarray(img),
        ((pad_t, size - nh - pad_t), (pad_l, size - nw - pad_l), (0, 0)),
        mode="reflect",
    )
    return Image.fromarray(arr)


def center_crop(img: Image.Image, size: int = CROP_SIZE) -> Image.Image:
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def to_input_array(img: Image.Image, crop: int = CROP_SIZE,
                   fit: int = TARGET_SIZE) -> np.ndarray:
    """Imagen PIL -> tensor NCHW float32 normalizado, listo para el modelo."""
    img = center_crop(fit_reflect(img, fit), crop)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / \
        np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
