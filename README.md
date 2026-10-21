# Evaluación de daños en fachadas — Backend

Backend de una aplicación web que clasifica daños en fachadas de edificaciones después de
desastres naturales. Recibe una o varias fotografías, las pasa por un modelo de visión
(`modelo.onnx`) y devuelve el análisis: tipo de construcción, nivel de daño, si la
edificación es rescatable y una recomendación. Opcionalmente, redacta una explicación con
Gemini (Google AI Studio); sin clave, usa una explicación de plantilla.

Está hecho con FastAPI y guarda el historial de inspecciones en SQLite. También sirve el
frontend en `http://localhost:8000/`.

## Estructura esperada

El backend sirve el frontend desde la carpeta `../frontend`, así que los dos repositorios
deben quedar uno al lado del otro:

```
proyecto/
├── backend/    ← este repositorio
└── frontend/   ← https://github.com/Mvg6/FrontendWeb
```

```bash
git clone https://github.com/Mvg6/BackendWeb.git backend
git clone https://github.com/Mvg6/FrontendWeb.git frontend
```

La carpeta del frontend tiene que llamarse exactamente `frontend`.

## Modelo (`modelo.onnx`)

`modelo.onnx` **no está en el repositorio**. Se entrenó con datos de CrisisNLP, cuya licencia
permite usarlos solo para investigación sin fines comerciales, por lo que el modelo no se
distribuye públicamente. Si lo necesitas, pídeselo al equipo del proyecto y colócalo en la
carpeta `backend/`.

Sin el modelo, la aplicación arranca igual, pero `/analizar` y `/analizar-edificio`
responden 503 (nunca se inventan resultados).

## Instalación

Probado con Python 3.12.

```bash
cd backend
python -m venv .venv
```

Activa el entorno virtual:

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux / macOS: `source .venv/bin/activate`

Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Configuración (`.env`)

Copia `.env.example` como `.env` en la carpeta `backend/` y completa los valores:

```bash
cp .env.example .env
```

(En Windows: `copy .env.example .env`.)

| Variable | Uso |
| --- | --- |
| `GEMINI_API_KEY` | Clave de Gemini. Vacía: se usa la explicación de plantilla. |
| `GEMINI_MODEL` | Opcional. Modelo Flash a usar; vacío: se elige uno automáticamente. |
| `EXPLICACION_IA` | `on`/`off`. Con `off` no se envía nada a Gemini. |
| `LIMITE_IA_SIN_SESION_POR_HORA` | Explicaciones con IA por hora y por IP para usuarios sin sesión. |
| `MODO_PUBLICO` | `on` al exponer la app por un túnel: desactiva `/docs` y `/redoc` y restringe CORS al mismo origen. |

El archivo `.env` nunca se sube al repositorio.

## Arranque

Desarrollo local:

```bash
uvicorn main:app --port 8000
```

Luego abre `http://localhost:8000/` en el navegador. La documentación de la API está en
`http://localhost:8000/docs` (solo con `MODO_PUBLICO=off`).

Detrás de un túnel (por ejemplo, Cloudflare Tunnel), con `MODO_PUBLICO=on` en `.env`:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --no-proxy-headers
```

`--no-proxy-headers` es necesario: detrás del túnel todas las conexiones llegan desde
`127.0.0.1`, y sin esta opción uvicorn confiaría en encabezados de proxy que cualquiera puede
enviar, lo que permitiría saltarse los límites por IP.

## Pruebas

Las pruebas automatizadas (carpeta `tests/`) no están en este repositorio; se piden al equipo.
Para ejecutarlas, coloca la carpeta `tests/` dentro de `backend/`, instala las dependencias de
desarrollo y ejecuta pytest desde `backend/` (la configuración está en `pytest.ini`):

```bash
pip install -r requirements-dev.txt
pytest
```

Las pruebas que usan fotos reales necesitan la variable de entorno `FOTOS_PERU_DATASET` con la
ruta a la carpeta `fotos_peru_dataset`; si no está definida, se saltan.
