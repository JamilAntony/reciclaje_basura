"""Clasifica imagenes desde la terminal, sin backend ni telefono.

Sirve para probar el modelo exportado con fotos propias antes de montar la app.

Uso:
    python ml/predict_cli.py foto.jpg
    python ml/predict_cli.py carpeta_con_fotos/ --tta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "backend"))

from inference import Clasificador, ImagenInvalida, ModelNotReady  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rutas", nargs="+", type=Path, help="imagenes o carpetas")
    ap.add_argument("--tta", action="store_true", help="promedia con el espejo horizontal")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="fuerza inferencia en CPU")
    args = ap.parse_args()

    try:
        clf = Clasificador(device="cpu" if args.cpu else None)
    except ModelNotReady as exc:
        raise SystemExit(str(exc))

    archivos: list[Path] = []
    for ruta in args.rutas:
        if ruta.is_dir():
            archivos += sorted(p for p in ruta.rglob("*") if p.suffix.lower() in IMG_EXT)
        elif ruta.is_file():
            archivos.append(ruta)
        else:
            print(f"aviso: no existe {ruta}", file=sys.stderr)

    if not archivos:
        raise SystemExit("No se encontraron imagenes.")

    colores = {c["key"]: c for c in clf.meta["taxonomy"]["colores"]}
    print(f"Modelo: {clf.meta['arch']} en {clf.device} | {len(archivos)} imagenes\n")

    for p in archivos:
        try:
            pred = clf.predecir(p.read_bytes(), tta=args.tta, topk=args.topk)
        except ImagenInvalida as exc:
            print(f"{p.name}: {exc}")
            continue

        bin_ = colores[pred.color]
        marca = "  (?)" if pred.incierto else ""
        print(f"{p.name}")
        print(f"   -> tacho {bin_['nombre'].upper():<9} ({bin_['residuos']})")
        print(f"      material {pred.material_label}  "
              f"confianza {pred.confianza:.1%}{marca}  {pred.ms:.0f} ms")
        if args.topk > 1:
            resto = ", ".join(f"{c['label']} {c['probabilidad']:.1%}"
                              for c in pred.top[1:])
            print(f"      alternativas: {resto}")
        print()


if __name__ == "__main__":
    main()
