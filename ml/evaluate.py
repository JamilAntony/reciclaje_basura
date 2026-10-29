"""Evalua el modelo entrenado sobre el split de test.

Mide en dos niveles:
  * material (8 clases) -- lo que el modelo predice directamente;
  * color de contenedor (7 clases) -- lo que de verdad decide la app. Papel y
    carton van al mismo tacho azul, asi que confundirlos entre si no afecta al
    usuario y el nivel color lo refleja.

Uso:
    python ml/evaluate.py
    python ml/evaluate.py --split valid --tta
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import (COLOR_BINS, COLOR_ORDER, MATERIAL_LABEL, MATERIAL_TO_COLOR)
from train import build_model, build_transforms


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_all(model, loader, device, tta: bool) -> tuple[np.ndarray, np.ndarray]:
    probs_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x).float()
            if tta:
                # Promediar con el espejo horizontal: barato y suele dar unas
                # decimas de exactitud.
                logits = (logits + model(torch.flip(x, dims=[3])).float()) / 2
        probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(probs_all), np.concatenate(y_all)


def per_class_table(cm: np.ndarray, names: list[str], titulo: str) -> list[dict]:
    rows = []
    width = max(len(n) for n in names) + 2
    print(f"\n{titulo}")
    print("-" * (width + 42))
    print(f"{'clase':<{width}}{'prec':>8}{'recall':>8}{'F1':>8}{'soporte':>10}")
    print("-" * (width + 42))
    for i, name in enumerate(names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        sup = int(cm[i, :].sum())
        rows.append({"clase": name, "precision": prec, "recall": rec,
                     "f1": f1, "soporte": sup})
        print(f"{name:<{width}}{prec:>8.4f}{rec:>8.4f}{f1:>8.4f}{sup:>10}")
    print("-" * (width + 42))
    acc = np.trace(cm) / cm.sum()
    print(f"{'exactitud':<{width}}{'':>8}{'':>8}{acc:>8.4f}{int(cm.sum()):>10}")
    print(f"{'macro-F1':<{width}}{'':>8}{'':>8}"
          f"{np.mean([r['f1'] for r in rows]):>8.4f}")
    return rows


def plot_confusion(cm: np.ndarray, names: list[str], path: Path, titulo: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 1.0 * len(names) + 2.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("prediccion"); ax.set_ylabel("real"); ax.set_title(titulo)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"Matriz de confusion: {path}")


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=base / "artifacts" / "model_best.pt")
    ap.add_argument("--data-dir", type=Path, default=base / "data" / "processed" / "ntp")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tta", action="store_true", help="promedia con el espejo horizontal")
    ap.add_argument("--out-dir", type=Path, default=base / "artifacts")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"No existe {args.checkpoint}. Corre primero ml/train.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes: list[str] = ckpt["classes"]
    print(f"Modelo   : {ckpt['arch']} (epoca {ckpt['epoch']}, {args.checkpoint.name})")
    print(f"Split    : {args.split}   TTA: {'si' if args.tta else 'no'}")

    _, eval_tf = build_transforms(ckpt["img_size"])
    ds = datasets.ImageFolder(args.data_dir / args.split, transform=eval_tf)
    if ds.classes != classes:
        raise SystemExit(f"Clases en disco {ds.classes} != del modelo {classes}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    probs, y_true = predict_all(model, dl, device, args.tta)
    y_pred = probs.argmax(1)

    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Nivel color: se agrega sumando los materiales que comparten contenedor.
    color_idx = np.array([COLOR_ORDER.index(MATERIAL_TO_COLOR[c]) for c in classes])
    cmc = np.zeros((len(COLOR_ORDER), len(COLOR_ORDER)), dtype=np.int64)
    for t, p in zip(color_idx[y_true], color_idx[y_pred]):
        cmc[t, p] += 1

    print("\n" + "=" * 62)
    print(f"RESULTADOS -- split {args.split} ({len(ds)} imagenes)")
    print("=" * 62)
    mat_rows = per_class_table(cm, [MATERIAL_LABEL[c] for c in classes],
                               "Nivel MATERIAL (lo que predice el modelo)")
    col_names = [f"{COLOR_BINS[c].nombre} ({COLOR_BINS[c].residuos})" for c in COLOR_ORDER]
    col_rows = per_class_table(cmc, col_names,
                               "Nivel COLOR DE CONTENEDOR (lo que ve el usuario)")

    conf = probs.max(1)
    acierto = y_pred == y_true
    print(f"\nConfianza media  acertando: {conf[acierto].mean():.3f}  "
          f"fallando: {conf[~acierto].mean():.3f}")
    for umbral in (0.5, 0.7, 0.9):
        sel = conf >= umbral
        if sel.sum():
            print(f"  confianza >= {umbral:.1f}: cubre {sel.mean():6.1%} de las fotos, "
                  f"exactitud {acierto[sel].mean():.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, classes, args.out_dir / f"confusion_material_{args.split}.png",
                   f"Material -- {args.split}")
    plot_confusion(cmc, COLOR_ORDER, args.out_dir / f"confusion_color_{args.split}.png",
                   f"Contenedor NTP 900.058 -- {args.split}")

    metrics = {
        "split": args.split,
        "n_imagenes": len(ds),
        "tta": args.tta,
        "arch": ckpt["arch"],
        "epoca": ckpt["epoch"],
        "material": {
            "exactitud": float(np.trace(cm) / cm.sum()),
            "macro_f1": float(np.mean([r["f1"] for r in mat_rows])),
            "por_clase": mat_rows,
            "matriz_confusion": cm.tolist(),
            "clases": classes,
        },
        "color": {
            "exactitud": float(np.trace(cmc) / cmc.sum()),
            "macro_f1": float(np.mean([r["f1"] for r in col_rows])),
            "por_clase": col_rows,
            "matriz_confusion": cmc.tolist(),
            "clases": COLOR_ORDER,
        },
    }
    out = args.out_dir / f"metrics_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetricas: {out}")


if __name__ == "__main__":
    main()
