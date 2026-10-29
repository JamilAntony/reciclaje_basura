"""Taxonomia NTP 900.058 - Codigo de colores ampliado.

Fuente de verdad compartida por el entrenamiento, la evaluacion y el backend.

El modelo predice MATERIAL (clase fina). El material se traduce a COLOR de
contenedor con MATERIAL_TO_COLOR. Separar ambos niveles permite:
  * mensajes utiles en la app ("Carton -> tacho AZUL"),
  * cambiar la politica de color sin reentrenar,
  * medir exactitud a nivel material y a nivel color.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(frozen=True)
class ColorBin:
    key: str
    nombre: str          # nombre del color segun la norma
    residuos: str        # que va en ese contenedor
    hex: str             # color para pintar la UI
    texto_hex: str       # color de texto legible sobre `hex`


# --- Los 7 contenedores de la NTP 900.058 (instituciones / no municipales) ---
COLOR_BINS: Dict[str, ColorBin] = {
    "azul": ColorBin("azul", "Azul", "Papel y carton", "#1565C0", "#FFFFFF"),
    "amarillo": ColorBin("amarillo", "Amarillo", "Metales", "#F9A825", "#1A1A1A"),
    "blanco": ColorBin("blanco", "Blanco", "Plasticos", "#F5F5F5", "#1A1A1A"),
    "gris": ColorBin("gris", "Gris", "Vidrio", "#757575", "#FFFFFF"),
    "marron": ColorBin("marron", "Marron", "Organicos", "#6D4C41", "#FFFFFF"),
    "rojo": ColorBin("rojo", "Rojo", "Peligrosos", "#C62828", "#FFFFFF"),
    "negro": ColorBin("negro", "Negro", "Generales / No aprovechables", "#212121", "#FFFFFF"),
}

# Orden oficial en que se presentan los contenedores.
COLOR_ORDER: List[str] = ["azul", "amarillo", "blanco", "gris", "marron", "rojo", "negro"]

# --- Clases de material que aprende el modelo (orden = indice de salida) ---
MATERIAL_CLASSES: List[str] = [
    "carton",
    "metal",
    "no_aprovechable",
    "organico",
    "papel",
    "peligroso",
    "plastico",
    "vidrio",
]

MATERIAL_TO_COLOR: Dict[str, str] = {
    "papel": "azul",
    "carton": "azul",
    "metal": "amarillo",
    "plastico": "blanco",
    "vidrio": "gris",
    "organico": "marron",
    "peligroso": "rojo",
    "no_aprovechable": "negro",
}

MATERIAL_LABEL: Dict[str, str] = {
    "papel": "Papel",
    "carton": "Carton",
    "metal": "Metal",
    "plastico": "Plastico",
    "vidrio": "Vidrio",
    "organico": "Organico",
    "peligroso": "Residuo peligroso",
    "no_aprovechable": "No aprovechable",
}

# Consejo que muestra la app junto al color del tacho.
MATERIAL_TIP: Dict[str, str] = {
    "papel": "Manten el papel seco y sin restos de comida; el papel engrasado no se recicla.",
    "carton": "Aplana las cajas y retira cintas o grapas antes de depositarlas.",
    "metal": "Enjuaga latas y envases metalicos; aplastarlos ahorra espacio.",
    "plastico": "Vacia y enjuaga el envase; retira la tapa y deposita ambos.",
    "vidrio": "Deposita el envase sin tapa. El vidrio roto va protegido para evitar cortes.",
    "organico": "Restos de comida y poda. Sin bolsas plasticas ni liquidos.",
    "peligroso": "Pilas, baterias, aerosoles y medicinas requieren punto de acopio especial. No lo mezcles.",
    "no_aprovechable": "Residuos sin valor de reciclaje: papel higienico, colillas, ceramica, envoltorios mixtos.",
}

# --- Mapa desde las etiquetas de origen de los datasets hacia MATERIAL ---
# Las imagenes de Roboflow conservan la etiqueta original en el nombre de
# archivo (paper_100_jpg.rf.<hash>.jpg), lo que permite recuperar las clases
# finas que el export habia colapsado en carpetas genericas.
SOURCE_LABEL_TO_MATERIAL: Dict[str, str] = {
    "paper": "papel",
    "cardboard": "carton",
    "biological": "organico",
    "glass": "vidrio",
    "brown-glass": "vidrio",
    "green-glass": "vidrio",
    "white-glass": "vidrio",
    "metal": "metal",
    "plastic": "plastico",
    "trash": "no_aprovechable",
    "clothes": "no_aprovechable",
    "shoes": "no_aprovechable",
    "battery": "peligroso",
}


def material_index(material: str) -> int:
    return MATERIAL_CLASSES.index(material)


def color_for_material(material: str) -> ColorBin:
    return COLOR_BINS[MATERIAL_TO_COLOR[material]]


def taxonomy_payload() -> dict:
    """Taxonomia serializable, usada por el endpoint /api/classes."""
    return {
        "norma": "NTP 900.058 - codigo de colores ampliado",
        "materiales": [
            {
                "key": m,
                "label": MATERIAL_LABEL[m],
                "color": MATERIAL_TO_COLOR[m],
                "tip": MATERIAL_TIP[m],
            }
            for m in MATERIAL_CLASSES
        ],
        "colores": [asdict(COLOR_BINS[c]) for c in COLOR_ORDER],
    }
