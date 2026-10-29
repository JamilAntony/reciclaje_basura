"""Entrena el clasificador de residuos NTP 900.058 en GPU.

Tarea: clasificacion de imagen unica (una foto -> un material -> un color de
contenedor). No se usa deteccion tipo YOLO porque el dataset viene en formato
"folder": no trae cajas delimitadoras, asi que no hay con que supervisar un
detector. Para el caso de uso --apuntar la camara a un residuo y saber en que
tacho va-- clasificar la imagen es lo correcto y rinde mas con los mismos
datos.

Modelo: EfficientNetV2-S preentrenado en ImageNet. Buen equilibrio entre
exactitud y costo, y entra con holgura en los 8 GB de la RTX 5060.

Uso:
    python ml/train.py                      # valores por defecto
    python ml/train.py --epochs 20 --arch convnext_tiny
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import MATERIAL_CLASSES, MATERIAL_TO_COLOR, COLOR_ORDER

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data_dir: Path
    out_dir: Path
    arch: str = "efficientnet_v2_s"
    img_size: int = 224          # las fuentes son 240x240; 224 deja margen de recorte
    epochs: int = 18
    batch_size: int = 48
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    workers: int = 6
    seed: int = 42
    patience: int = 6            # epocas sin mejorar antes de parar
    amp: bool = True
    compile_model: bool = False


ARCHS = {
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        # scale=(0.6, 1.0): el residuo suele ocupar casi todo el cuadro, asi que
        # recortes mas agresivos lo dejarian fuera.
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 240 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader, list[str]]:
    train_tf, eval_tf = build_transforms(cfg.img_size)
    train_ds = datasets.ImageFolder(cfg.data_dir / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(cfg.data_dir / "valid", transform=eval_tf)

    # El backend asume el orden de MATERIAL_CLASSES para mapear indice -> color.
    # ImageFolder ordena alfabeticamente; si algun dia dejan de coincidir, el
    # modelo exportado devolveria colores equivocados en silencio.
    if train_ds.classes != MATERIAL_CLASSES:
        raise SystemExit(
            "Las clases en disco no coinciden con la taxonomia.\n"
            f"  en disco : {train_ds.classes}\n"
            f"  esperado : {MATERIAL_CLASSES}\n"
            "Vuelve a correr ml/prepare_dataset.py"
        )
    if valid_ds.classes != MATERIAL_CLASSES:
        raise SystemExit("Las clases de valid no coinciden con las de train.")

    common = dict(num_workers=cfg.workers, pin_memory=True,
                  persistent_workers=cfg.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          drop_last=True, **common)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                          **common)
    return train_dl, valid_dl, train_ds.classes


def class_weights(train_dir: Path, classes: list[str]) -> torch.Tensor:
    """Pesos por clase ~ 1/sqrt(frecuencia).

    El dataset esta desbalanceado (vidrio 4286 vs peligroso 756). La raiz
    cuadrada compensa sin llegar a sobre-ponderar la clase chica, que es lo que
    pasa con 1/frecuencia pura.
    """
    counts = np.array([len(list((train_dir / c).iterdir())) for c in classes],
                      dtype=np.float64)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def build_model(arch: str, n_classes: int) -> nn.Module:
    if arch not in ARCHS:
        raise SystemExit(f"Arquitectura desconocida: {arch}. Opciones: {list(ARCHS)}")
    ctor, weights = ARCHS[arch]
    model = ctor(weights=weights)

    # Cada familia nombra su cabeza distinto; se reemplaza la capa final por
    # una del tamano de nuestra taxonomia.
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            idx = max(i for i, m in enumerate(model.classifier)
                      if isinstance(m, nn.Linear))
            in_f = model.classifier[idx].in_features
            model.classifier[idx] = nn.Linear(in_f, n_classes)
        else:
            model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    else:
        raise SystemExit(f"No se encontro la cabeza de clasificacion de {arch}")
    return model


# --------------------------------------------------------------------------- #
# Metricas
# --------------------------------------------------------------------------- #
COLOR_OF_INDEX = torch.tensor(
    [COLOR_ORDER.index(MATERIAL_TO_COLOR[m]) for m in MATERIAL_CLASSES],
    dtype=torch.long,
)


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, amp: bool) -> dict:
    model.eval()
    n_cls = len(MATERIAL_CLASSES)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    color_ok = total = 0
    loss_sum = 0.0
    color_of = COLOR_OF_INDEX.to(device)

    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
            loss = criterion(logits.float(), y)
        pred = logits.argmax(1)
        loss_sum += loss.item() * y.size(0)
        total += y.size(0)
        color_ok += (color_of[pred] == color_of[y]).sum().item()
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t, p] += 1

    acc = float(np.trace(cm) / max(total, 1))
    return {
        "loss": loss_sum / max(total, 1),
        "acc": acc,
        "macro_f1": macro_f1(cm),
        "color_acc": color_ok / max(total, 1),
        "cm": cm,
    }


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def train(cfg: Config) -> dict:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("AVISO: no se detecto GPU; el entrenamiento sera muy lento.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    amp = cfg.amp and device.type == "cuda"
    train_dl, valid_dl, classes = build_loaders(cfg)
    print(f"train: {len(train_dl.dataset)} imagenes | valid: {len(valid_dl.dataset)}")
    print(f"clases ({len(classes)}): {classes}")

    model = build_model(cfg.arch, len(classes)).to(device, memory_format=torch.channels_last)
    if cfg.compile_model:
        model = torch.compile(model)

    weights = class_weights(cfg.data_dir / "train", classes).to(device)
    print("pesos por clase: " +
          ", ".join(f"{c}={w:.2f}" for c, w in zip(classes, weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)

    # Sin decaimiento de peso en normalizaciones y sesgos: es la receta estandar
    # y evita castigar parametros que no son escalas de features.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )

    steps_per_epoch = len(train_dl)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)
    total_steps = cfg.epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.out_dir / "model_best.pt"
    history: list[dict] = []
    best_f1, best_epoch, epochs_sin_mejora = -1.0, -1, 0
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss = seen = correct = 0.0
        ep_start = time.time()

        for i, (x, y) in enumerate(train_dl, 1):
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            run_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
            if i % 50 == 0 or i == steps_per_epoch:
                print(f"\r  ep {epoch:02d}/{cfg.epochs}  "
                      f"{i:>4}/{steps_per_epoch}  "
                      f"loss {run_loss / seen:.4f}  acc {correct / seen:.4f}",
                      end="", flush=True)

        val = evaluate(model, valid_dl, criterion, device, amp)
        dur = time.time() - ep_start
        print(f"\r  ep {epoch:02d}/{cfg.epochs}  "
              f"train_loss {run_loss / seen:.4f}  train_acc {correct / seen:.4f}  |  "
              f"val_loss {val['loss']:.4f}  val_acc {val['acc']:.4f}  "
              f"macroF1 {val['macro_f1']:.4f}  color_acc {val['color_acc']:.4f}  "
              f"({dur:.0f}s)")

        history.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": run_loss / seen,
            "train_acc": correct / seen,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "val_color_acc": val["color_acc"],
            "seconds": round(dur, 1),
        })

        # Se selecciona por macro-F1, no por exactitud: con clases
        # desbalanceadas la exactitud premia acertar vidrio e ignorar peligroso.
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, epochs_sin_mejora = val["macro_f1"], epoch, 0
            torch.save({
                "state_dict": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
                "arch": cfg.arch,
                "classes": classes,
                "img_size": cfg.img_size,
                "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                "material_to_color": MATERIAL_TO_COLOR,
                "epoch": epoch,
                "val_metrics": {k: v for k, v in val.items() if k != "cm"},
                "config": {k: str(v) if isinstance(v, Path) else v
                           for k, v in asdict(cfg).items()},
            }, ckpt_path)
            print(f"      nuevo mejor macro-F1 -> {ckpt_path.name}")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= cfg.patience:
                print(f"      sin mejora en {cfg.patience} epocas; parada temprana")
                break

    total_min = (time.time() - t0) / 60
    print(f"\nTerminado en {total_min:.1f} min. "
          f"Mejor macro-F1 {best_f1:.4f} en la epoca {best_epoch}.")

    summary = {
        "best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "minutes": round(total_min, 1),
        "history": history,
        "classes": classes,
        "arch": cfg.arch,
    }
    (cfg.out_dir / "training_history.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    plot_history(history, cfg.out_dir / "training_curves.png")
    return summary


def plot_history(history: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ep = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], label="train")
    ax1.plot(ep, [h["val_loss"] for h in history], label="valid")
    ax1.set_title("Perdida"); ax1.set_xlabel("epoca"); ax1.legend(); ax1.grid(alpha=.3)

    ax2.plot(ep, [h["val_acc"] for h in history], label="exactitud material")
    ax2.plot(ep, [h["val_macro_f1"] for h in history], label="macro-F1")
    ax2.plot(ep, [h["val_color_acc"] for h in history], label="exactitud color")
    ax2.set_title("Validacion"); ax2.set_xlabel("epoca"); ax2.legend(); ax2.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Curvas: {path}")


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=base / "data" / "processed" / "ntp")
    ap.add_argument("--out-dir", type=Path, default=base / "artifacts")
    ap.add_argument("--arch", default="efficientnet_v2_s", choices=list(ARCHS))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--compile", action="store_true", dest="compile_model")
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"No existe {args.data_dir}. Corre primero ml/prepare_dataset.py")

    cfg = Config(
        data_dir=args.data_dir, out_dir=args.out_dir, arch=args.arch,
        img_size=args.img_size, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, workers=args.workers, patience=args.patience,
        amp=not args.no_amp, compile_model=args.compile_model,
    )
    train(cfg)


if __name__ == "__main__":
    main()
