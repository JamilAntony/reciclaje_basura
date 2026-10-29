# Arranca la API de clasificacion de residuos.
#
#   .\backend\run.ps1
#
# Imprime la direccion que hay que escribir en la app del telefono.
# Si Windows pregunta por el firewall, permite el acceso en redes privadas:
# sin eso el telefono no alcanza al servidor.

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent $PSScriptRoot
$python = Join-Path (Split-Path -Parent $raiz) '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host "No se encontro el entorno virtual en $python" -ForegroundColor Red
    Write-Host "Usa tu propio interprete: python backend\main.py"
    exit 1
}

$modelo = Join-Path $raiz 'artifacts\model_scripted.pt'
if (-not (Test-Path $modelo)) {
    Write-Host "Falta el modelo exportado ($modelo)." -ForegroundColor Yellow
    Write-Host "Corre primero:"
    Write-Host "  python ml\train.py"
    Write-Host "  python ml\export_model.py"
    exit 1
}

& $python (Join-Path $raiz 'backend\main.py')
