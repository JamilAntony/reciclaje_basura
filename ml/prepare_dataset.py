"""Construye el dataset NTP 900.058 (8 materiales -> 7 colores).

Dos fuentes:

1. `Clasificacion de Basura.v4i.folder` (Roboflow, ya descargado). Su export
   colapso las clases finas en carpetas genericas y encima etiqueto el carton
   como "Tetrapack". Las etiquetas originales sobreviven en el nombre de
   archivo (`paper_100_jpg.rf.<hash>.jpg`), asi que se reconstruyen desde ahi.
   Eso recupera papel, carton y organicos --que la estructura de carpetas
   habia perdido-- y rescata la carpeta `Unlabeled`.

2. Clase `battery` del dataset de 12 clases (Hugging Face). El export de
   Roboflow no tiene residuos peligrosos, que es el contenedor ROJO de la
   norma. Las imagenes se normalizan con el mismo preprocesamiento de Roboflow
   (fit a 240x240 con bordes reflejados) para que el modelo no pueda usar el
   borde reflejado como atajo para reconocer esta clase.

Uso:
    python ml/prepare_dataset.py --battery-zip "<ruta.zip>"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imaging import fit_reflect
from taxonomy import MATERIAL_CLASSES, MATERIAL_TO_COLOR, SOURCE_LABEL_TO_MATERIAL

SPLITS = ("train", "valid", "test")
IMG_EXT = {".jpg", ".jpeg", ".png"}
# Reparto de la clase nueva, alineado con la proporcion del export de Roboflow.
BATTERY_SPLIT = {"train": 0.80, "valid": 0.12, "test": 0.08}


# Respaldo para los pocos archivos que son solo una marca de tiempo
# (`20240626_161207_jpg.rf...`): ahi la unica pista es la carpeta de Roboflow.
# `Basura` y `Unlabeled` mezclan materiales, asi que no se pueden resolver y
# se descartan en lugar de inventarles una etiqueta.
FOLDER_FALLBACK: dict[str, str | None] = {
    "metal": "metal",
    "plastico": "plastico",
    "tetrapack": "carton",   # el export etiqueto como Tetrapack lo que es carton
    "vidrio": "vidrio",
    "basura": None,
    "unlabeled": None,
}


def source_label(filename: str) -> str | None:
    """Etiqueta original a partir del nombre de archivo de Roboflow.

    Los nombres vienen en varias formas: `paper_100`, `glass_20240626_161832`,
    `trash_12_png`. Basta con el primer token alfabetico.
    """
    stem = filename.split(".rf.")[0].lower()
    m = re.match(r"^([a-z][a-z\-]*)", stem)
    return m.group(1) if m else None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink: mismo volumen, no duplica los 202 MB
    except OSError:
        shutil.copy2(src, dst)


def prepare_roboflow(root: Path, out: Path) -> tuple[Counter, int]:
    counts: Counter = Counter()
    unknown: Counter = Counter()

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise SystemExit(f"No existe {split_dir}")
        for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                label = source_label(f.name)
                material = SOURCE_LABEL_TO_MATERIAL.get(label or "")
                if material is None:
                    material = FOLDER_FALLBACK.get(folder.name.lower())
                if material is None:
                    unknown[f"{folder.name}/{label or '<sin-patron>'}"] += 1
                    continue
                dst_dir = out / split / material
                dst_dir.mkdir(parents=True, exist_ok=True)
                link_or_copy(f, dst_dir / f.name)
                counts[(split, material)] += 1

    if unknown:
        print(f"  avisos: {sum(unknown.values())} imagenes sin etiqueta reconocible "
              f"-> {dict(unknown.most_common(5))}")
    return counts, sum(unknown.values())


def prepare_battery(zip_path: Path, out: Path, seed: int = 42) -> Counter:
    """Extrae y normaliza la clase `battery` -> material `peligroso` (ROJO)."""
    counts: Counter = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            n for n in zf.namelist()
            if not n.endswith("/")
            and "/battery/" in n.replace("\\", "/").lower()
            and Path(n).suffix.lower() in IMG_EXT
        ]
        if not members:
            raise SystemExit(f"No hay imagenes de 'battery' dentro de {zip_path}")
        print(f"  {len(members)} imagenes de 'battery' en el zip")

        # Reparto determinista: ordenar por hash del nombre lo hace estable
        # aunque el zip cambie de orden interno.
        order = sorted(members, key=lambda n: hashlib.md5(n.encode()).hexdigest())
        random.Random(seed).shuffle(order)

        n = len(order)
        n_train = round(n * BATTERY_SPLIT["train"])
        n_valid = round(n * BATTERY_SPLIT["valid"])
        assignment = (
            [("train", m) for m in order[:n_train]]
            + [("valid", m) for m in order[n_train:n_train + n_valid]]
            + [("test", m) for m in order[n_train + n_valid:]]
        )

        for split, member in assignment:
            dst_dir = out / split / "peligroso"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{Path(member).stem}_fit240.jpg"
            if not dst.exists():
                with zf.open(member) as fh:
                    fit_reflect(Image.open(fh)).save(dst, "JPEG", quality=92)
            counts[(split, "peligroso")] += 1
    return counts


def report(counts: Counter, out: Path) -> None:
    materials = sorted({m for _, m in counts})
    width = max(len(m) for m in materials) + 2

    print("\n" + "=" * 70)
    print("DATASET NTP 900.058 -- imagenes por material y split")
    print("=" * 70)
    print(f"{'material':<{width}}{'color':<11}{'train':>8}{'valid':>8}{'test':>8}{'total':>9}")
    print("-" * 70)
    totals: dict[str, int] = defaultdict(int)
    for m in materials:
        row = [counts[(s, m)] for s in SPLITS]
        for s, v in zip(SPLITS, row):
            totals[s] += v
        print(f"{m:<{width}}{MATERIAL_TO_COLOR[m]:<11}"
              f"{row[0]:>8}{row[1]:>8}{row[2]:>8}{sum(row):>9}")
    print("-" * 70)
    print(f"{'TOTAL':<{width}}{'':<11}"
          f"{totals['train']:>8}{totals['valid']:>8}{totals['test']:>8}"
          f"{sum(totals.values()):>9}")

    faltan = [m for m in MATERIAL_CLASSES if m not in materials]
    if faltan:
        print(f"\nADVERTENCIA: materiales sin datos: {faltan}")
    else:
        print(f"\nLos {len(MATERIAL_CLASSES)} materiales y los 7 contenedores tienen datos.")
    print(f"Salida: {out}")


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roboflow", type=Path,
                    default=base / "Clasificacion de Basura.v4i.folder")
    ap.add_argument("--battery-zip", type=Path, default=None,
                    help="zip del dataset de 12 clases (aporta el contenedor ROJO)")
    ap.add_argument("--out", type=Path, default=base / "data" / "processed" / "ntp")
    ap.add_argument("--clean", action="store_true", help="borra la salida antes de construir")
    args = ap.parse_args()

    if args.clean and args.out.exists():
        print(f"Limpiando {args.out}")
        shutil.rmtree(args.out)

    print(f"[1/2] Roboflow: {args.roboflow}")
    counts, descartadas = prepare_roboflow(args.roboflow, args.out)
    print(f"  {sum(counts.values())} imagenes mapeadas, {descartadas} descartadas")

    if args.battery_zip:
        print(f"[2/2] Residuos peligrosos: {args.battery_zip}")
        counts.update(prepare_battery(args.battery_zip, args.out))
    else:
        print("[2/2] omitido (sin --battery-zip): el contenedor ROJO quedara sin datos")

    report(counts, args.out)


if __name__ == "__main__":
    main()
