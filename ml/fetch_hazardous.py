"""Descarga solo la clase `battery` del dataset de 12 clases de Hugging Face.

El export de Roboflow no trae residuos peligrosos, que son el contenedor ROJO
de la NTP 900.058. El dataset de 12 clases si los tiene, pero su zip pesa
251 MB y solo necesitamos una carpeta (~1 % del total).

En vez de bajar el zip completo, se lee de forma remota: el CDN de Hugging
Face acepta peticiones por rango (HTTP 206), asi que se descarga el directorio
central del zip, se localizan los miembros de `battery` y se pide una sola vez
el tramo de bytes que los cubre. Son ~20 MB en lugar de 251 MB.

Uso:
    python ml/fetch_hazardous.py                 # -> data/external/battery.zip
"""

from __future__ import annotations

import argparse
import io
import zlib
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_URL = (
    "https://huggingface.co/datasets/UdaraChamidu/"
    "Garbage-Classification-with-12-classes/resolve/main/garbage_classification.zip"
)

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CDH_SIG = b"PK\x01\x02"


def http_get(url: str, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "reciclaje-ntp/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        if start is not None and r.status != 206:
            raise SystemExit("El servidor no respeta las peticiones por rango; "
                             "descarga el zip completo y usa --battery-zip.")
        return r.read()


def total_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "reciclaje-ntp/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        n = r.headers.get("Content-Length")
    if not n:
        raise SystemExit("El servidor no informa Content-Length.")
    return int(n)


def central_directory(url: str, size: int) -> tuple[bytes, int]:
    """Devuelve (bytes del directorio central, offset absoluto)."""
    tail_len = min(size, 65_536 + 64)
    tail = http_get(url, size - tail_len, size - 1)

    i = tail.rfind(EOCD_SIG)
    if i < 0:
        raise SystemExit("No se encontro el End Of Central Directory del zip.")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])

    # Zip64: los campos de 32 bits saturan y el valor real esta en otro bloque.
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        j = tail.rfind(EOCD64_LOCATOR_SIG)
        if j < 0:
            raise SystemExit("Zip64 sin localizador EOCD64.")
        (eocd64_off,) = struct.unpack("<Q", tail[j + 8:j + 16])
        blk = http_get(url, eocd64_off, eocd64_off + 55)
        if blk[:4] != EOCD64_SIG:
            raise SystemExit("Bloque EOCD64 invalido.")
        cd_size, cd_off = struct.unpack("<QQ", blk[40:56])

    return http_get(url, cd_off, cd_off + cd_size - 1), cd_off


def parse_central_directory(cd: bytes) -> list[dict]:
    entries: list[dict] = []
    p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == CDH_SIG:
        (comp_size, uncomp_size) = struct.unpack("<II", cd[p + 20:p + 28])
        n_len, e_len, c_len = struct.unpack("<HHH", cd[p + 28:p + 34])
        (local_off,) = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + n_len].decode("utf-8", "replace")
        extra = cd[p + 46 + n_len:p + 46 + n_len + e_len]

        # Zip64: los valores reales viven en el campo extra 0x0001.
        if 0xFFFFFFFF in (comp_size, uncomp_size, local_off):
            q = 0
            while q + 4 <= len(extra):
                tag, ln = struct.unpack("<HH", extra[q:q + 4])
                if tag == 0x0001:
                    vals = list(struct.unpack(f"<{ln // 8}Q", extra[q + 4:q + 4 + ln]))
                    if uncomp_size == 0xFFFFFFFF and vals:
                        uncomp_size = vals.pop(0)
                    if comp_size == 0xFFFFFFFF and vals:
                        comp_size = vals.pop(0)
                    if local_off == 0xFFFFFFFF and vals:
                        local_off = vals.pop(0)
                    break
                q += 4 + ln

        entries.append({"name": name, "local_off": local_off,
                        "comp_size": comp_size, "uncomp_size": uncomp_size})
        p += 46 + n_len + e_len + c_len
    return entries


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--folder", default="battery",
                    help="carpeta a extraer dentro del zip remoto")
    ap.add_argument("--out", type=Path, default=base / "data" / "external" / "battery.zip")
    args = ap.parse_args()

    print(f"Fuente : {args.url}")
    size = total_size(args.url)
    print(f"Tamano : {size / 1e6:.1f} MB (no se descarga completo)")

    cd, _ = central_directory(args.url, size)
    entries = parse_central_directory(cd)
    print(f"Miembros en el zip: {len(entries)}")

    needle = f"/{args.folder.lower()}/"
    wanted = [
        e for e in entries
        if needle in ("/" + e["name"].replace("\\", "/").lower())
        and not e["name"].endswith("/")
    ]
    if not wanted:
        raise SystemExit(f"No hay miembros bajo '{args.folder}'. "
                         f"Ejemplos: {[e['name'] for e in entries[:5]]}")

    lo = min(e["local_off"] for e in wanted)
    # El encabezado local mide 30 bytes + nombre + extra; se holgura 4 KB.
    hi = max(e["local_off"] + e["comp_size"] for e in wanted) + 4096
    hi = min(hi, size - 1)
    print(f"'{args.folder}': {len(wanted)} archivos, tramo de "
          f"{(hi - lo) / 1e6:.1f} MB -> descargando")

    blob = http_get(args.url, lo, hi)

    # Se reescriben los miembros en un zip local pequeno, que es lo que
    # consume prepare_dataset.py.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zout:
        for e in wanted:
            rel = e["local_off"] - lo
            if blob[rel:rel + 4] != b"PK\x03\x04":
                print(f"  aviso: encabezado local invalido en {e['name']}", file=sys.stderr)
                continue
            method, = struct.unpack("<H", blob[rel + 8:rel + 10])
            n_len, e_len = struct.unpack("<HH", blob[rel + 26:rel + 30])
            data_start = rel + 30 + n_len + e_len
            chunk = blob[data_start:data_start + e["comp_size"]]
            try:
                if method == 0:                      # almacenado sin comprimir
                    raw = chunk
                elif method == 8:                    # deflate crudo, sin cabecera zlib
                    raw = zlib.decompressobj(-zlib.MAX_WBITS).decompress(chunk)
                else:
                    raise ValueError(f"metodo de compresion {method} no soportado")
            except Exception as exc:                 # noqa: BLE001
                print(f"  aviso: no se pudo inflar {e['name']}: {exc}", file=sys.stderr)
                continue
            if len(raw) != e["uncomp_size"]:
                print(f"  aviso: tamano inesperado en {e['name']}", file=sys.stderr)
                continue
            zout.writestr(e["name"], raw)
            written += 1

    print(f"Listo: {written} imagenes en {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    print("Siguiente paso:")
    print(f'  python ml/prepare_dataset.py --clean --battery-zip "{args.out}"')


if __name__ == "__main__":
    main()
