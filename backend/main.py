"""API de clasificacion de residuos segun el codigo de colores NTP 900.058.

Endpoints:
    GET  /api/health     estado del servicio y del modelo
    GET  /api/classes    taxonomia completa (8 materiales, 7 contenedores)
    POST /api/predict    una foto -> material, contenedor y confianza

Arranque:
    python backend/main.py                 # escucha en 0.0.0.0:8000
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "backend"))

from inference import (Clasificador, ImagenInvalida,      # noqa: E402
                       ModelNotReady, DEFAULT_UMBRAL)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reciclaje")

# Las fotos del S24 Ultra pueden pasar de 20 MB; el limite corta subidas
# absurdas antes de gastar memoria decodificandolas.
MAX_BYTES = int(os.getenv("RECICLAJE_MAX_MB", "12")) * 1024 * 1024
UMBRAL = float(os.getenv("RECICLAJE_UMBRAL", str(DEFAULT_UMBRAL)))

estado: dict[str, Any] = {"clasificador": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El modelo se carga una vez al arrancar, no por peticion. Si no esta
    # exportado todavia, el servidor igual levanta y /api/health lo explica,
    # que es mas util que un arranque fallido sin mensaje.
    try:
        estado["clasificador"] = Clasificador(umbral=UMBRAL)
        log.info("Modelo cargado: %s", estado["clasificador"].info)
    except ModelNotReady as exc:
        estado["error"] = str(exc)
        log.error("Modelo no disponible: %s", exc)
    yield


app = FastAPI(
    title="Clasificador de residuos NTP 900.058",
    description="Clasifica una foto de un residuo y devuelve el color de "
                "contenedor del codigo de colores ampliado.",
    version="1.0.0",
    lifespan=lifespan,
)

# La app de Expo corre en el telefono y pega a la IP de la laptop, asi que el
# origen no es predecible: se permite cualquiera. Es una API local de lectura,
# sin credenciales ni datos propios.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def clasificador() -> Clasificador:
    clf = estado["clasificador"]
    if clf is None:
        raise HTTPException(503, detail=estado["error"] or "Modelo no cargado")
    return clf


@app.get("/api/health")
def health() -> dict[str, Any]:
    clf = estado["clasificador"]
    return {
        "estado": "ok" if clf else "sin_modelo",
        "detalle": estado["error"],
        "modelo": clf.info if clf else None,
        "max_mb": MAX_BYTES // (1024 * 1024),
    }


@app.get("/api/classes")
def classes() -> dict[str, Any]:
    clf = clasificador()
    return clf.meta["taxonomy"]


def _clasificar_bytes(raw: bytes, tta: bool, topk: int, via: str) -> dict[str, Any]:
    """Logica comun a las dos formas de enviar la foto."""
    clf = clasificador()

    if not raw:
        raise HTTPException(400, detail="La imagen llego vacia.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(
            413,
            detail=f"La imagen pesa {len(raw) / (1024 * 1024):.1f} MB y el limite es "
                   f"{MAX_BYTES // (1024 * 1024)} MB. "
                   "Reduce la resolucion antes de subirla.",
        )

    try:
        pred = clf.predecir(raw, tta=tta, topk=topk)
    except ImagenInvalida as exc:
        raise HTTPException(400, detail=str(exc)) from exc

    colores = {c["key"]: c for c in clf.meta["taxonomy"]["colores"]}
    bin_ = colores[pred.color]

    log.info("predict(%s) material=%s conf=%.3f incierto=%s %.0fms",
             via, pred.material, pred.confianza, pred.incierto, pred.ms)

    return {
        "material": pred.material,
        "material_label": pred.material_label,
        "confianza": pred.confianza,
        "incierto": pred.incierto,
        "contenedor": {
            "color": bin_["key"],
            "nombre": bin_["nombre"],
            "residuos": bin_["residuos"],
            "hex": bin_["hex"],
            "texto_hex": bin_["texto_hex"],
        },
        "consejo": clf.tip(pred.material),
        "top": pred.top,
        "ms": pred.ms,
        "mensaje": ("No estoy seguro; vuelve a tomar la foto mas cerca y con "
                    "buena luz." if pred.incierto else None),
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(..., description="foto del residuo (jpg/png)"),
    tta: bool = Query(False, description="promedia con el espejo horizontal"),
    topk: int = Query(3, ge=1, le=8),
) -> dict[str, Any]:
    """Subida multipart. Es la forma comoda desde curl, la pagina /docs o
    cualquier cliente HTTP normal."""
    return _clasificar_bytes(await file.read(), tta, topk, "multipart")


class ImagenBase64(BaseModel):
    imagen_base64: str


@app.post("/api/predict-base64")
async def predict_base64(
    cuerpo: ImagenBase64,
    tta: bool = Query(False, description="promedia con el espejo horizontal"),
    topk: int = Query(3, ge=1, le=8),
) -> dict[str, Any]:
    """Misma prediccion, con la imagen en base64 dentro de un JSON.

    Existe por la app de React Native: desde la version 0.88, su stack de red
    rechaza el objeto {uri, name, type} que se usaba para armar un multipart
    ("Unsupported FormDataPart implementation") y exige un Blob real, que no
    hay forma directa de obtener desde un file:// del telefono. Mandar la
    imagen en base64 evita el multipart por completo y funciona igual en
    cualquier version. Cuesta un 33 % mas de bytes, despreciable para fotos ya
    reducidas a ~60 KB.
    """
    datos = cuerpo.imagen_base64
    # Acepta tambien el formato data:image/jpeg;base64,<...> de los navegadores.
    if datos.startswith("data:"):
        _, _, datos = datos.partition(",")

    try:
        raw = base64.b64decode(datos, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, detail=f"base64 invalido: {exc}") from exc

    return _clasificar_bytes(raw, tta, topk, "base64")


@app.get("/")
def raiz() -> JSONResponse:
    return JSONResponse({
        "servicio": "Clasificador de residuos NTP 900.058",
        "docs": "/docs",
        "endpoints": ["/api/health", "/api/classes", "/api/predict"],
    })


def direcciones_ipv4() -> list[tuple[str, str]]:
    """Todas las IPv4 del equipo, con una etiqueta de a que red pertenecen.

    Esta maquina tiene varias interfaces (Wi-Fi, hotspot de Windows, adaptadores
    de VirtualBox/WSL). Anunciar solo una lleva a escribir en el telefono una
    direccion que esta en otra red y que nunca va a responder, asi que se
    listan todas y se marca cual es la probable.
    """
    import socket

    etiquetas = {
        "192.168.137.": "hotspot de Windows",
        "192.168.56.": "VirtualBox (no sirve para el telefono)",
        "192.168.146.": "adaptador virtual (no sirve para el telefono)",
        "172.": "WSL / Docker (no sirve para el telefono)",
        "169.254.": "sin conexion",
    }

    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except OSError:
        pass

    salida = []
    for ip in sorted(ips):
        nota = next((n for p, n in etiquetas.items() if ip.startswith(p)), "red local")
        salida.append((ip, nota))
    return salida


def main() -> None:
    import socket
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    # Cual es la interfaz de salida por defecto: suele ser la del router y es
    # la mejor apuesta cuando hay varias.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        principal = s.getsockname()[0]
        s.close()
    except OSError:
        principal = ""

    print("=" * 66)
    print("  API lista. Direcciones por las que el telefono puede entrar:")
    for ip, nota in direcciones_ipv4():
        marca = " <-- la mas probable" if ip == principal else ""
        print(f"    http://{ip}:{port}   ({nota}){marca}")
    print()
    print("  Usa la que este en la MISMA red que el telefono.")
    print("  Si el telefono esta en el hotspot de la laptop, es la 192.168.137.1")
    if principal:
        print(f"  Documentacion interactiva: http://{principal}:{port}/docs")
    print("=" * 66)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
