"""Exporta el checkpoint entrenado a formatos de despliegue.

Genera:
  * `model_scripted.pt`  -- TorchScript. Es lo que carga el backend: no
    necesita el codigo de train.py, asi que el servidor no depende de este
    repo de entrenamiento.
  * `model.onnx`         -- opcional (--onnx). Util para servir con
    onnxruntime, en CPU o en el propio telefono.
  * `model_meta.json`    -- clases, tamano de entrada y normalizacion. El
    backend lo lee para no tener constantes duplicadas.

Uso:
    python ml/export_model.py
    python ml/export_model.py --onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import MATERIAL_TO_COLOR, taxonomy_payload
from train import build_model


def main() -> None:
    # El exportador de ONNX imprime emojis y la consola de Windows usa cp1252,
    # lo que reventaba la exportacion por un simple caracter de adorno.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")

    base = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=base / "artifacts" / "model_best.pt")
    ap.add_argument("--out-dir", type=Path, default=base / "artifacts")
    ap.add_argument("--onnx", action="store_true", help="exporta tambien a ONNX")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"No existe {args.checkpoint}. Corre primero ml/train.py")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    size = ckpt["img_size"]

    model = build_model(ckpt["arch"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, size, size)

    # trace + freeze: el grafo queda con los pesos dentro y sin ramas de
    # entrenamiento, que es lo que se quiere servir.
    #
    # Sin optimize_for_inference a proposito: deja el modulo en un estado que
    # torch.jit.load no puede volver a leer ("required keyword attribute
    # 'value' is undefined") y el archivo sale de 0.1 MB. freeze ya hace el
    # trabajo que importa aqui.
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
        scripted = torch.jit.freeze(scripted)
    ts_path = args.out_dir / "model_scripted.pt"
    scripted.save(str(ts_path))
    print(f"TorchScript: {ts_path} ({ts_path.stat().st_size / 1e6:.1f} MB)")

    # Comprobar que el grafo exportado da lo mismo que el modelo original.
    with torch.no_grad():
        ref = model(example)
        got = torch.jit.load(str(ts_path))(example)
    delta = (ref - got).abs().max().item()
    print(f"  diferencia maxima vs. el modelo original: {delta:.2e}")
    if delta > 1e-3:
        raise SystemExit("El modelo exportado no coincide con el original.")

    meta = {
        "arch": ckpt["arch"],
        "classes": classes,
        "img_size": size,
        "normalization": ckpt["normalization"],
        "material_to_color": MATERIAL_TO_COLOR,
        "trained_epoch": ckpt["epoch"],
        "val_metrics": ckpt.get("val_metrics", {}),
        "taxonomy": taxonomy_payload(),
    }
    meta_path = args.out_dir / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metadatos  : {meta_path}")

    if args.onnx:
        onnx_path = args.out_dir / "model.onnx"
        try:
            torch.onnx.export(
                model, example, str(onnx_path),
                input_names=["input"], output_names=["logits"],
                # Lote dinamico para poder enviar varias fotos en una peticion.
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=args.opset,
            )
            print(f"ONNX       : {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
        except ImportError as exc:
            # ONNX es opcional: el backend sirve con TorchScript. No tiene
            # sentido perder la exportacion principal por una dependencia extra.
            print(f"ONNX       : omitido ({exc}). Instala: pip install onnx onnxscript")

    print("\nListo. El backend carga model_scripted.pt + model_meta.json.")


if __name__ == "__main__":
    main()
