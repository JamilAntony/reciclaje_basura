# Clasificador de residuos por código de colores — NTP 900.058

Sistema completo que fotografía un residuo y dice en qué contenedor va, según el
**código de colores ampliado** de la NTP 900.058 para instituciones, empresas y
residuos no municipales.

| Color | Residuos | Material que detecta el modelo |
|---|---|---|
| 🔵 Azul | Papel y cartón | `papel`, `carton` |
| 🟡 Amarillo | Metales | `metal` |
| ⚪ Blanco | Plásticos | `plastico` |
| ⚫ Gris | Vidrio | `vidrio` |
| 🟤 Marrón | Orgánicos | `organico` |
| 🔴 Rojo | Peligrosos | `peligroso` |
| ⬛ Negro | Generales / no aprovechables | `no_aprovechable` |

Tres piezas:

- **`ml/`** — preparación de datos, entrenamiento en GPU, evaluación y exportación.
- **`backend/`** — API FastAPI que recibe una foto y devuelve material + contenedor.
- **`mobile/`** — app Expo (React Native) que usa la cámara del teléfono.

---

## Qué hay y qué no en este repositorio

Viaja el **código completo** (entrenamiento, backend y app) más las métricas y
gráficas de los resultados. **No viajan** el dataset ni los modelos
entrenados: suman más de 400 MB y se regeneran enteros con los comandos de la
sección *Puesta en marcha*.

| No incluido | Cómo obtenerlo |
|---|---|
| `Clasificacion de Basura.v4i.folder/` (202 MB) | Descarga desde Roboflow — ver *Créditos de los datos* |
| `data/external/battery.zip` (5,8 MB) | `python ml/fetch_hazardous.py` |
| `data/processed/` | `python ml/prepare_dataset.py --clean --battery-zip data/external/battery.zip` |
| `artifacts/*.pt`, `*.onnx` (~240 MB) | `python ml/train.py` y después `python ml/export_model.py` |

Entrenar de cero toma unos 15 minutos en una GPU como la RTX 5060.

---

## 1. Decisiones de diseño

### Por qué clasificación y no YOLO

El dataset viene en **formato carpeta** (`folder`), es decir, una etiqueta por
imagen y **ninguna caja delimitadora**. Un detector tipo YOLO necesita cajas para
supervisarse; sin ellas habría que anotar a mano ~11 500 imágenes.

Además, el caso de uso es *apuntar la cámara a un residuo y saber en qué tacho
va*: una imagen, una decisión. Clasificar la imagen completa resuelve eso
directamente y aprovecha mejor los mismos datos. La detección sólo haría falta
para contar varios residuos en una misma escena.

Se usa **EfficientNetV2-S** preentrenada en ImageNet: buena exactitud por
parámetro y entra con holgura en los 8 GB de la RTX 5060.

### Dos niveles: material y color

El modelo predice **material** (8 clases) y una tabla traduce material → color de
contenedor. Separarlo tiene tres ventajas:

1. La app puede decir *"Cartón → tacho azul"*, más útil que sólo "azul".
2. La política de colores se puede cambiar **sin reentrenar** (editando
   `ml/taxonomy.py`).
3. Se mide la exactitud en los dos niveles. Confundir *papel* con *cartón* no
   afecta al usuario porque ambos van al mismo tacho azul, y la métrica de color
   lo refleja.

---

## 2. El dataset: dos problemas y cómo se resolvieron

Punto de partida: `Clasificacion de Basura.v4i.folder`, export de Roboflow con
19 570 imágenes de 240×240 en 6 carpetas —`Basura`, `Metal`, `Plastico`,
`Tetrapack`, `Unlabeled`, `Vidrio`.

### Problema 1 — las carpetas perdían información (y una estaba mal)

Al inspeccionar los nombres de archivo apareció que el export **había colapsado
clases finas** en carpetas genéricas:

```
train/Basura    ->  paper 2076 | biological 1404 | trash 742
train/Tetrapack ->  cardboard 2556          <-- no es tetrapak: es cartón
train/Unlabeled ->  trash 570 | paper 268   <-- sí tenían etiqueta
```

Es decir: tomar las carpetas al pie de la letra habría tirado **papel**,
**cartón** y **orgánicos** a un solo cajón de "basura", habría etiquetado todo el
cartón como tetrapak, y habría descartado 838 imágenes útiles en `Unlabeled`.

Las etiquetas originales sobreviven en el nombre (`paper_100_jpg.rf.<hash>.jpg`),
así que `ml/prepare_dataset.py` las reconstruye desde ahí. Se recuperaron
**19 568 de 19 570 imágenes** (las 2 restantes son fotos con nombre de sólo marca
de tiempo en una carpeta ambigua, y se descartan en lugar de inventarles clase).

### Problema 2 — faltaba el contenedor rojo

El export no tiene **ningún residuo peligroso**, que es uno de los 7
contenedores que pide la norma. Se complementó con la clase `battery` del dataset
de 12 clases de Hugging Face (945 imágenes de pilas y baterías).

Ese zip pesa 251 MB y sólo se necesitaba una carpeta.
`ml/fetch_hazardous.py` lee el **ZIP de forma remota**: descarga el directorio
central, localiza los miembros de `battery` y pide por rango HTTP sólo ese tramo
de bytes → **5,8 MB en lugar de 251 MB**.

Las imágenes nuevas pasan por el **mismo preprocesamiento que Roboflow** (ajuste
a 240×240 con bordes reflejados). Esto no es cosmético: si sólo la clase
`peligroso` llegara sin ese borde reflejado, el modelo podría aprender a
reconocerla por el borde en vez de por el objeto.

### Dataset resultante

| Material | Color | train | valid | test | total |
|---|---|---:|---:|---:|---:|
| carton | azul | 2556 | 365 | 182 | 3103 |
| metal | amarillo | 1418 | 209 | 102 | 1729 |
| no_aprovechable | negro | 1320 | 207 | 79 | 1606 |
| organico | marrón | 1404 | 189 | 106 | 1699 |
| papel | azul | 2348 | 329 | 177 | 2854 |
| peligroso | rojo | 756 | 113 | 76 | 945 |
| plastico | blanco | 2778 | 397 | 198 | 3373 |
| vidrio | gris | 4286 | 612 | 306 | 5204 |
| **TOTAL** | | **16 866** | **2421** | **1226** | **20 513** |

Se verificó que **ninguna imagen fuente aparece en dos splits** (11 514 fuentes
únicas; el resto son variantes de brillo que Roboflow generó y que quedan
siempre en el mismo split). Sin esa comprobación, las métricas estarían
infladas por fuga de datos.

El desbalance (vidrio 4286 vs. peligroso 756) se compensa con **pesos por clase
≈ 1/√frecuencia** en la función de pérdida, y el modelo se selecciona por
**macro-F1**, no por exactitud: con clases desbalanceadas la exactitud premia
acertar vidrio e ignorar las clases chicas.

---

## 3. Resultados

Entrenamiento: EfficientNetV2-S, 18 épocas, lote 64, **15,1 min** en la RTX 5060
(8 GB, CUDA 13.2, precisión mixta bf16). Mejor época: la 15, por macro-F1 en
validación.

Métricas sobre el **split de test** (1226 imágenes nunca vistas), con TTA de
espejo horizontal:

### Nivel material — 8 clases

| Material | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| Cartón | 0,977 | 0,945 | 0,961 | 182 |
| Metal | 0,935 | 0,980 | 0,957 | 102 |
| No aprovechable | 0,950 | 0,962 | 0,956 | 79 |
| Orgánico | 1,000 | 1,000 | **1,000** | 106 |
| Papel | 0,951 | 0,977 | 0,964 | 177 |
| Residuo peligroso | 0,950 | **1,000** | 0,974 | 76 |
| Plástico | 0,965 | 0,970 | 0,967 | 198 |
| Vidrio | 0,987 | 0,954 | 0,970 | 306 |

**Exactitud 96,82 % · macro-F1 0,9687**

### Nivel color de contenedor — 7 clases

| Contenedor | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| 🔵 Azul (papel y cartón) | 0,983 | 0,981 | 0,982 | 359 |
| 🟡 Amarillo (metales) | 0,935 | 0,980 | 0,957 | 102 |
| ⚪ Blanco (plásticos) | 0,965 | 0,970 | 0,967 | 198 |
| ⚫ Gris (vidrio) | 0,987 | 0,954 | 0,970 | 306 |
| 🟤 Marrón (orgánicos) | 1,000 | 1,000 | **1,000** | 106 |
| 🔴 Rojo (peligrosos) | 0,950 | **1,000** | 0,974 | 76 |
| ⬛ Negro (no aprovechables) | 0,950 | 0,962 | 0,956 | 79 |

**Exactitud 97,39 % · macro-F1 0,9724**

El nivel color sale por encima del nivel material, como era de esperar:
confundir papel con cartón no cambia el tacho, y esa confusión es justamente la
más frecuente del modelo.

Lo más importante para el uso real: **recall 1,000 en residuos peligrosos**. El
modelo no dejó pasar ninguna pila ni batería como reciclable, que es el error
más costoso que podía cometer.

### Calibración

| Umbral de confianza | Cobertura | Exactitud en lo cubierto |
|---|---:|---:|
| ≥ 0,5 | 99,4 % | 97,05 % |
| ≥ 0,7 | 97,7 % | 98,16 % |
| ≥ 0,9 | 42,9 % | 99,24 % |

Confianza media: 0,892 cuando acierta, 0,741 cuando falla. El umbral por
defecto del backend (0,55) descarta casi sólo errores. Subirlo a 0,7 lleva la
exactitud a 98,2 % a cambio de pedir repetir el 2 % de las fotos.

**Latencia de inferencia**: ~22 ms por foto en GPU con TTA (~50 ms la primera
petición). Gráficas en `artifacts/`: `training_curves.png`,
`confusion_material_test.png`, `confusion_color_test.png`.

---

## 4. Puesta en marcha

### Requisitos

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Comprobar la GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Paso 1 — construir el dataset

```bash
python ml/fetch_hazardous.py
python ml/prepare_dataset.py --clean --battery-zip data/external/battery.zip
```

### Paso 2 — entrenar

```bash
python ml/train.py --epochs 18 --batch-size 64
```

Produce en `artifacts/`: `model_best.pt`, `training_history.json`,
`training_curves.png`.

### Paso 3 — evaluar

```bash
python ml/evaluate.py --split test --tta
```

Produce `metrics_test.json` y las matrices de confusión a nivel material y a
nivel color.

### Paso 4 — exportar

```bash
python ml/export_model.py --onnx
```

Produce `model_scripted.pt` (TorchScript, lo que carga el backend),
`model_meta.json` y opcionalmente `model.onnx`. La exportación **verifica** que
el grafo exportado da la misma salida que el modelo original (diferencia máxima
observada: 2,7·10⁻⁷).

`--onnx` necesita `pip install onnx onnxscript`; si faltan, se omite ese paso sin
perder la exportación principal. El ONNX se escribe en dos archivos —
`model.onnx` y `model.onnx.data`— y **los dos** hacen falta para cargarlo.

### Paso 5 — levantar el backend

```powershell
.\backend\run.ps1
```

Imprime la dirección que hay que escribir en la app, por ejemplo
`http://192.168.1.37:8000`. La primera vez Windows pregunta por el firewall:
**permite el acceso en redes privadas**, si no el teléfono no alcanza al
servidor.

Probar sin teléfono:

```bash
python ml/predict_cli.py data/processed/ntp/test/vidrio --tta
curl -F "file=@foto.jpg" http://127.0.0.1:8000/api/predict
```

Documentación interactiva en `http://<ip>:8000/docs`.

### Paso 6 — app en el teléfono

```bash
cd mobile
npm install
npx expo start
```

Escanea el QR con **Expo Go** en el Galaxy S24 Ultra (la laptop y el teléfono
tienen que estar en la misma red Wi-Fi).

No hace falta configurar la dirección: la app la deduce de la IP desde la que
Expo le sirve el código. Si necesitas apuntar a otra máquina, **Ajustes** →
escribe la dirección → **Probar conexión** → **Guardar**, y desde ahí puedes
volver a la detección automática cuando quieras.

---

## 5. Operación: arrancar y parar a mano

Son **dos servicios independientes**, cada uno en su terminal. Ninguno se
lanza solo.

| Servicio | Qué hace | Puerto | ¿Cuándo hace falta? |
|---|---|---:|---|
| Backend (FastAPI) | Carga el modelo en la GPU y clasifica las fotos | 8000 | Siempre |
| Metro (Expo) | Sirve el código de la app al teléfono | 8081 | Sólo mientras uses Expo Go |

Arranca **primero el backend**: tarda unos segundos en cargar el modelo en la
GPU, y la app comprueba la conexión nada más abrir.

### Terminal 1 — backend

```powershell
cd D:\CURSOS_2026_2\MachineLearning\Reciclaje_basura
D:\CURSOS_2026_2\MachineLearning\.venv\Scripts\python.exe backend\main.py
```

O el atajo equivalente, que además avisa si falta exportar el modelo:

```powershell
.\backend\run.ps1
```

Está listo cuando imprime la lista de direcciones y `Application startup
complete`. Anota la marcada como *la más probable*: es la de tu Wi-Fi.

Comprobar desde la propia laptop (en PowerShell usa `curl.exe`, porque `curl`
a secas es un alias de `Invoke-WebRequest` y no acepta las mismas opciones):

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

Debe responder `"estado":"ok"` y `"device":"cuda"`.

### Terminal 2 — Metro (sólo para Expo Go)

```powershell
cd D:\CURSOS_2026_2\MachineLearning\Reciclaje_basura\mobile
npx expo start
```

Escanea el QR con Expo Go. Dentro de esa terminal: `r` recarga la app, `?`
lista los atajos.

### Parar

`Ctrl-C` en cada terminal. Para confirmar que no quedó nada:

```powershell
Get-NetTCPConnection -State Listen -LocalPort 8000,8081 -ErrorAction SilentlyContinue
```

Sin salida significa que ambos puertos están libres. Si algo quedó colgado:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*Reciclaje_basura*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

### Ajustes por variable de entorno

Se fijan en la terminal **antes** de arrancar el backend:

```powershell
$env:PORT = '8001'
```

| Variable | Por defecto | Para qué |
|---|---|---|
| `PORT` | `8000` | Puerto del backend |
| `RECICLAJE_UMBRAL` | `0.55` | Debajo de esta confianza la respuesta sale como *incierta* |
| `RECICLAJE_MAX_MB` | `12` | Tamaño máximo de foto aceptado |

### Si algo no responde

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `/api/health` dice `sin_modelo` | El modelo no está exportado | `python ml/export_model.py` |
| La app muestra *Sin conexión* | Backend apagado, o el teléfono está en otra red | Toca la píldora de estado en Inicio para reintentar |
| `address already in use` | Quedó un proceso anterior | El comando de `Stop-Process` de arriba |
| El teléfono no carga el QR | Teléfono y laptop en redes distintas | Misma Wi-Fi, y **apaga el hotspot de Windows**: con él activo el teléfono puede acabar en otra subred |
| Expo Go dice *incompatible* | Expo Go más antiguo que el SDK del proyecto | Actualizar Expo Go desde la Play Store |

La laptop tiene varias interfaces de red (Wi-Fi, hotspot de Windows,
VirtualBox, WSL). Por eso el backend lista **todas** sus direcciones al
arrancar y marca cuál es la probable: escribir la de VirtualBox en el teléfono
es un error fácil de cometer y difícil de diagnosticar.

---

## 6. API

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/api/health` | Estado del servicio y del modelo cargado |
| GET | `/api/classes` | Taxonomía completa: 8 materiales, 7 contenedores |
| POST | `/api/predict` | Foto en `multipart/form-data` (campo `file`) |
| POST | `/api/predict-base64` | Foto en JSON: `{"imagen_base64": "..."}` |

Las dos rutas de predicción hacen lo mismo y aceptan los parámetros `tta`
(promedia con el espejo horizontal) y `topk`. Existen las dos por un motivo
concreto: **la app usa base64, no multipart**.

Desde React Native 0.88, su stack de red rechaza el objeto
`{uri, name, type}` con el que se venía construyendo un `FormData` en móvil y
falla con `Unsupported FormDataPart implementation`. Exige un `Blob` real, y no
hay forma directa de obtener uno a partir de un `file://` del teléfono. Mandar
la imagen en base64 elimina el multipart del camino y funciona en cualquier
versión; cuesta un 33 % más de bytes, despreciable para fotos ya reducidas a
~60 KB. `multipart` se mantiene porque es lo cómodo desde curl y desde /docs.

Respuesta:

```json
{
  "material": "vidrio",
  "material_label": "Vidrio",
  "confianza": 0.9871,
  "incierto": false,
  "contenedor": {
    "color": "gris",
    "nombre": "Gris",
    "residuos": "Vidrio",
    "hex": "#757575",
    "texto_hex": "#FFFFFF"
  },
  "consejo": "Deposita el envase sin tapa...",
  "top": [{ "material": "vidrio", "probabilidad": 0.9871 }],
  "ms": 23.4,
  "mensaje": null
}
```

Cuando la confianza queda por debajo del umbral (0,55 por defecto) la respuesta
trae `incierto: true` y un mensaje para repetir la foto, en vez de mandar al
usuario a un tacho equivocado con una predicción dudosa. Se ajusta con la
variable de entorno `RECICLAJE_UMBRAL`.

---

## 7. Detalles que importan

**El preprocesamiento vive en un solo sitio.** `ml/imaging.py` lo comparten el
preparador de datos y el backend. Las imágenes de entrenamiento pasaron por
"fit a 240×240 con bordes reflejados", así que una foto del teléfono recibe
exactamente el mismo tratamiento. Si el servidor recortara el centro, estaría
enviando al modelo imágenes de un dominio distinto al que aprendió y la
exactitud caería **sin ningún aviso**.

**La foto se reduce en el teléfono.** Una captura del S24 Ultra puede pasar de
20 MB y el modelo sólo consume 224×224. La app la reduce a 512 px de ancho antes
de subirla: mismo resultado, mucho menos tiempo de red.

**El orden de las clases está verificado.** El backend traduce índice → color
asumiendo el orden de `MATERIAL_CLASSES`. `train.py` falla de entrada si el orden
en disco no coincide, porque un desajuste silencioso devolvería colores
equivocados con toda la confianza del mundo.

**El modelo se carga una vez.** Al arrancar, no por petición. Si todavía no está
exportado, el servidor igual levanta y `/api/health` explica qué falta.

**El calentamiento son tres pasadas, no una.** Con una sola, la primera foto
real seguía tardando ~2 s mientras cuDNN elegía algoritmos; con tres baja a los
~25 ms del resto. También se calienta la rama del espejo horizontal, porque el
TTA la usa en cada petición.

**No se usa `torch.jit.optimize_for_inference`.** Deja el módulo en un estado
que `torch.jit.load` no puede volver a leer (`required keyword attribute 'value'
is undefined`) y el archivo sale de 0,1 MB en vez de 80 MB. `torch.jit.freeze`
hace el trabajo que importa aquí. En torch 2.14 `torch.jit.load` avisa de que
está obsoleto; sigue funcionando y es lo que mantiene el backend independiente
del código de entrenamiento.

---

## 8. Estructura

```
Reciclaje_basura/
├── ml/
│   ├── taxonomy.py          # NTP 900.058: materiales, colores y su relación
│   ├── imaging.py           # preprocesamiento compartido con el backend
│   ├── fetch_hazardous.py   # trae la clase `peligroso` por rango HTTP
│   ├── prepare_dataset.py   # reconstruye etiquetas y arma el dataset
│   ├── train.py             # entrenamiento en GPU
│   ├── evaluate.py          # métricas por material y por color
│   ├── export_model.py      # TorchScript + ONNX + metadatos
│   └── predict_cli.py       # pruebas desde la terminal
├── backend/
│   ├── inference.py         # carga del modelo y predicción
│   ├── main.py              # API FastAPI
│   └── run.ps1              # arranque con la IP de la LAN
├── mobile/                      # app Expo / React Native
│   ├── App.tsx                  # stack nativo y proveedores
│   └── src/
│       ├── api.ts               # cliente HTTP con timeout y errores tipados
│       ├── rutas.ts             # tipos de navegación
│       ├── theme.ts             # tokens de estilo y los 7 colores NTP
│       ├── useServidor.ts       # dirección del backend (autodetectada)
│       ├── ServidorContext.tsx  # dirección + estado de conexión
│       ├── historial.tsx        # escaneos recientes, persistidos
│       ├── componentes/
│       │   ├── Boton.tsx
│       │   └── PildoraEstado.tsx
│       └── pantallas/
│           ├── Inicio.tsx       # punto de entrada
│           ├── Camara.tsx       # captura y subida
│           ├── Resultado.tsx    # el tacho que corresponde
│           ├── Guia.tsx         # los 7 contenedores, sin conexión
│           └── Ajustes.tsx
├── artifacts/               # modelos, métricas y gráficas (se generan)
├── data/
│   ├── external/            # battery.zip descargado
│   └── processed/ntp/       # dataset final (enlaces duros, no copias)
└── Clasificacion de Basura.v4i.folder/   # dataset original, intacto
```

El dataset procesado usa **enlaces duros** al original, así que no duplica los
202 MB en disco.

---

## 9. La app

Cinco pantallas sobre un **stack nativo** de React Navigation, que trae el
gesto de volver, el botón físico de Android y las transiciones de la
plataforma sin escribirlas a mano.

- **Inicio** — no arranca en la cámara. Primero muestra si el servidor
  responde, y recién entonces se entra a la cámara a propósito. Antes el
  usuario acababa encuadrando sin saber si había conexión y sólo se enteraba
  del problema después de disparar.
- **Cámara** — recuadro guía, flash conmutable y vibración al capturar. Los
  errores salen sobre la propia cámara, así que reintentar es un toque y no
  volver a navegar.
- **Resultado** — el color del contenedor ocupa la pantalla, porque es lo
  único que hace falta para actuar. Debajo: material, confianza, qué más va en
  ese tacho y las alternativas que consideró el modelo.
- **Guía** — los 7 contenedores con ejemplos. Los datos están en la app, no en
  el backend: es justo lo que se necesita consultar cuando uno está frente al
  tacho y la laptop está apagada.
- **Ajustes** — dirección del servidor, prueba de conexión, ficha del modelo y
  borrado del historial.

Los **escaneos recientes** se guardan en el teléfono y se ven en Inicio. Se
almacena sólo lo necesario para redibujar la tarjeta, no la respuesta entera:
la foto vive en la caché del sistema y Android la borra cuando necesita
espacio, así que la entrada tiene que seguir siendo legible sin su miniatura.

---

## 10. Créditos de los datos

- *Clasificación de Basura* v4 — Roboflow Universe, licencia CC BY 4.0.
  <https://universe.roboflow.com/clasificacin-basura/clasificacion-de-basura-x2fnq>
  Descárgalo en formato **folder** y descomprímelo en la raíz del proyecto,
  como `Clasificacion de Basura.v4i.folder/`.
- *Garbage Classification (12 classes)* — de ahí sale la clase `battery`, que
  cubre el contenedor rojo. No hace falta bajarlo a mano:
  `ml/fetch_hazardous.py` extrae sólo esa carpeta.
  <https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes>
