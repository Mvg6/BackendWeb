"""Configuración común de las pruebas.

Aislamiento:
- Cada prueba usa una base SQLite y una carpeta de miniaturas propias en tmp_path;
  nunca backend/historial.db (una salvaguarda hace fallar la prueba si se intenta abrir).
- Gemini siempre simulado: el cliente real de Google está bloqueado (sin red ni clave).
- El modelo real (modelo.onnx) solo se usa en pruebas marcadas @pytest.mark.modelo.
  Las demás usan una sesión ONNX simulada que pasa por el preprocesamiento real.

Fotos reales opcionales: variable de entorno FOTOS_PERU_DATASET con la ruta a la
carpeta fotos_peru_dataset. Si no está definida o las fotos no existen, esas pruebas se saltan.
"""
import asyncio
import base64
import io
import os
import re
import sqlite3
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from google.genai import errors as errores_genai
from PIL import Image

import main

BACKEND = Path(main.BASE_DIR)
RUTA_MODELO_REAL = BACKEND / "modelo.onnx"
RUTA_BD_REAL = (BACKEND / "historial.db").resolve()
DATASET = Path(os.environ["FOTOS_PERU_DATASET"]) if os.environ.get("FOTOS_PERU_DATASET") else None
CONTRASENA = "clave-segura-123"

# Estado global de main.py que las pruebas cambian: se restaura al terminar cada una.
GLOBALES = [
    "RUTA_BD", "DIR_MINIATURAS", "RUTA_ENV", "RUTA_MODELO",
    "sesion_onnx", "entrada_onnx_nombre", "cabezas_no_entrenadas", "analizar_imagen",
    "cliente_gemini", "modelo_gemini", "modelos_alternativos_gemini", "motivo_sin_ia", "_clave_gemini",
    "reconfigurar_ia_pendiente", "_ultima_configuracion_ia", "_tarea_configuracion_ia", "genai", "_esperar",
    "limite_analisis_sin_sesion", "limite_registros", "limite_ia_sin_sesion",
    "TIEMPO_TOTAL_IA_SEGUNDOS", "TIMEOUT_IA_SEGUNDOS", "ESPERAS_REINTENTO_IA",
]


# --------------------------------------------------------------------------
# Imágenes sintéticas
# --------------------------------------------------------------------------
def jpeg(ancho=640, alto=480, color=(120, 90, 60), exif_gps=False):
    imagen = Image.new("RGB", (ancho, alto), color)
    salida = io.BytesIO()
    if exif_gps:
        exif = imagen.getexif()
        exif[0x010F] = "CamaraPrivada"  # Make
        gps = exif.get_ifd(0x8825)
        gps[1], gps[2], gps[3], gps[4] = "S", (7.0, 9.0, 51.0), "W", (78.0, 30.0, 0.0)  # Cajamarca aprox.
        imagen.save(salida, "JPEG", exif=exif)
    else:
        imagen.save(salida, "JPEG")
    return salida.getvalue()


def png(modo="RGB", ancho=320, alto=240):
    salida = io.BytesIO()
    Image.new("RGB", (ancho, alto), (30, 140, 200)).convert(modo).save(salida, "PNG")
    return salida.getvalue()


def truncar(datos, fraccion=0.4):
    return datos[: int(len(datos) * fraccion)]


BASURA = bytes(range(256)) * 80


def texto_pdf(datos):
    """Texto de un PDF de reportlab (flujos ASCII85 + Flate), sin paquetes compilados."""
    contenido = []
    for m in re.finditer(rb"stream\s*(.*?)\s*endstream", datos, re.S):
        try:
            contenido.append(zlib.decompress(base64.a85decode(m.group(1).strip(), adobe=True)))
        except Exception:
            pass
    texto = b"\n".join(contenido).decode("latin-1")
    cadenas = re.findall(r"\(((?:\\.|[^\\)])*)\)", texto)

    def desescapar(m):  # \ddd octal (los caracteres no ASCII, p. ej. \361 = ñ) o \( \) \\
        codigo = m.group(1)
        return chr(int(codigo, 8)) if codigo[0] in "01234567" else codigo

    return " ".join(re.sub(r"\\([0-7]{1,3}|.)", desescapar, c) for c in cadenas)


def foto_dataset(*partes):
    if DATASET is None:
        pytest.skip("define FOTOS_PERU_DATASET para usar fotos reales")
    ruta = DATASET.joinpath(*partes)
    if not ruta.exists():
        pytest.skip(f"no está la foto {ruta} (define FOTOS_PERU_DATASET)")
    return ruta.read_bytes()


# --------------------------------------------------------------------------
# Dobles de prueba
# --------------------------------------------------------------------------
class SesionOnnxSimulada:
    """Imita onnxruntime.InferenceSession. `danos` es una cola de vectores de
    probabilidad del nivel de daño (uno por foto); si se vacía, usa `dano`."""

    def __init__(self):
        self.dano = [0.10, 0.20, 0.70]
        self.danos = []
        self.construccion = [0.6, 0.1, 0.1, 0.1, 0.1]
        self.rescatable = [0.2, 0.5, 0.3]
        self.entradas = []
        self.falla = None

    def run(self, _salidas, alimentacion):
        (arreglo,) = alimentacion.values()
        self.entradas.append(arreglo)
        if self.falla:
            raise self.falla
        dano = self.danos.pop(0) if self.danos else self.dano
        return [np.array([self.construccion]), np.array([dano]), np.array([self.rescatable])]


def explicacion_ia(estructura=("Muros de albañilería con columnas de concreto.",), fotos=(),
                   edificio="no_se_distingue", edificio_seguro=None, discrepancia=None):
    """ExplicacionIA válida. `fotos`: [(numero, tipo, seguro), ...]."""
    return main.ExplicacionIA(
        estructura=list(estructura), estado=["Grietas diagonales."], viabilidad="Daño visible.",
        acciones=["Acordonar el perímetro."], discrepancia=discrepancia, imagen_valida=True,
        tipo_construccion_foto=[main.TipoConstruccionFotoIA(foto=n, tipo_construccion=t, tipo_construccion_seguro=s)
                                for n, t, s in fotos],
        tipo_construccion_edificio=edificio,
        tipo_construccion_edificio_seguro=(edificio != "no_se_distingue") if edificio_seguro is None else edificio_seguro,
    )


def error_api(codigo, retry_delay=None, mensaje=None):
    estados = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
               429: "RESOURCE_EXHAUSTED", 500: "INTERNAL", 503: "UNAVAILABLE"}
    cuerpo = {"error": {"code": codigo, "message": mensaje or f"falla {codigo} simulada", "status": estados[codigo]}}
    if retry_delay:
        cuerpo["error"]["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return (errores_genai.ClientError if codigo < 500 else errores_genai.ServerError)(codigo, cuerpo)


class GeminiSimulado:
    """Imita client.aio.models.generate_content. Cada llamada consume un paso del
    `guion`: una excepción, "mal_esquema", ("lento", segundos) o una ExplicacionIA.
    Sin guion, responde `respuesta`."""

    def __init__(self):
        self.guion = []
        self.respuesta = explicacion_ia()
        self.llamadas = []  # (modelo, textos enviados, imágenes enviadas)

    async def generate_content(self, model, contents, config):
        textos = " ".join(p.text for p in contents if p.text)
        imagenes = [p.inline_data.data for p in contents if p.inline_data]
        self.llamadas.append((model, textos, imagenes))
        paso = self.guion.pop(0) if self.guion else self.respuesta
        if isinstance(paso, BaseException):
            raise paso
        if paso == "mal_esquema":
            return SimpleNamespace(parsed=None)
        if isinstance(paso, tuple) and paso[0] == "lento":
            await asyncio.sleep(paso[1])
            paso = self.respuesta
        return SimpleNamespace(parsed=paso)

    def activar(self, alternativos=("gemini-alternativo",)):
        """Pone este simulador como Gemini disponible (llamar después de arrancar la app)."""
        main.cliente_gemini = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=self.generate_content)))
        main.modelo_gemini = "gemini-principal"
        main.modelos_alternativos_gemini = list(alternativos)
        main.motivo_sin_ia = None
        main.reconfigurar_ia_pendiente = False


class _ClienteGoogleBloqueado:
    def __init__(self, *_, **__):
        raise RuntimeError("en las pruebas no se usa el cliente real de Google (sin red ni clave)")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def entorno_aislado(tmp_path, monkeypatch):
    for nombre in GLOBALES:
        monkeypatch.setattr(main, nombre, getattr(main, nombre))
    env = tmp_path / ".env"
    env.write_text("EXPLICACION_IA=off\nLIMITE_IA_SIN_SESION_POR_HORA=3\n", encoding="utf-8")
    monkeypatch.setattr(main, "RUTA_BD", str(tmp_path / "historial_prueba.db"))
    monkeypatch.setattr(main, "DIR_MINIATURAS", str(tmp_path / "miniaturas"))
    monkeypatch.setattr(main, "RUTA_ENV", str(env))
    monkeypatch.setattr(main, "RUTA_MODELO", str(tmp_path / "sin_modelo.onnx"))
    monkeypatch.setattr(main, "genai", SimpleNamespace(Client=_ClienteGoogleBloqueado))
    monkeypatch.setattr(main, "limite_analisis_sin_sesion", main.LimitePorIP(main.LIMITE_ANALISIS_SIN_SESION_POR_HORA, 3600))
    monkeypatch.setattr(main, "limite_registros", main.LimitePorIP(main.LIMITE_REGISTROS_POR_HORA, 3600))
    monkeypatch.setattr(main, "limite_ia_sin_sesion", main.LimitePorIP(3, 3600))
    esperas = []

    async def esperar_sin_demora(segundos):
        esperas.append(segundos)

    monkeypatch.setattr(main, "_esperar", esperar_sin_demora)

    conectar_real = sqlite3.connect

    def conectar_vigilado(base, *args, **kwargs):
        ruta = str(base).split("?")[0].removeprefix("file:")
        if ruta not in (":memory:", "") and Path(ruta).resolve() == RUTA_BD_REAL:
            raise AssertionError("una prueba intentó abrir backend/historial.db real")
        return conectar_real(base, *args, **kwargs)

    monkeypatch.setattr(main.sqlite3, "connect", conectar_vigilado)
    yield SimpleNamespace(tmp=tmp_path, bd=main.RUTA_BD, minis=Path(main.DIR_MINIATURAS), esperas=esperas, env=env)


@pytest.fixture
def entorno(entorno_aislado):
    return entorno_aislado


@pytest.fixture
def onnx():
    return SesionOnnxSimulada()


@pytest.fixture
def gemini():
    return GeminiSimulado()


@pytest.fixture
def app(onnx):
    """TestClient con la app arrancada (BD temporal) y la sesión ONNX simulada,
    con las mismas cabezas sin entrenar que el modelo actual."""
    with TestClient(main.app) as cliente:
        main.sesion_onnx, main.entrada_onnx_nombre = onnx, "entrada"
        main.cabezas_no_entrenadas = ["tipo_construccion", "rescatable"]
        yield cliente


@pytest.fixture
def app_sin_modelo():
    with TestClient(main.app) as cliente:
        assert main.sesion_onnx is None
        yield cliente


@pytest.fixture
def app_modelo_real(monkeypatch):
    if not RUTA_MODELO_REAL.exists():
        pytest.skip("no está backend/modelo.onnx")
    monkeypatch.setattr(main, "RUTA_MODELO", str(RUTA_MODELO_REAL))
    with TestClient(main.app) as cliente:
        if main.sesion_onnx is None:
            pytest.skip("modelo.onnx no se pudo cargar")
        yield cliente


def entrar(cliente, usuario="usuario_a", contrasena=CONTRASENA):
    """Registra (si hace falta) e inicia sesión; devuelve los encabezados con el token."""
    cliente.post("/auth/registro", json={"usuario": usuario, "contrasena": contrasena, "acepta_privacidad": True})
    respuesta = cliente.post("/auth/login", json={"usuario": usuario, "contrasena": contrasena})
    assert respuesta.status_code == 200, respuesta.text
    return {"Authorization": f"Bearer {respuesta.json()['token']}"}


def analizar(cliente, datos=None, nombre="fachada.jpg", tipo="image/jpeg", headers=None, **campos):
    return cliente.post("/analizar", headers=headers or {}, data=campos or None,
                        files={"archivo": (nombre, jpeg() if datos is None else datos, tipo)})


def analizar_edificio(cliente, fotos, headers=None, **campos):
    """fotos: [(datos, vista), ...] o [(datos, vista, nombre), ...]."""
    archivos = [("archivos", (f[2] if len(f) > 2 else f"foto{i}.jpg", f[0], "image/jpeg")) for i, f in enumerate(fotos, 1)]
    return cliente.post("/analizar-edificio", headers=headers or {}, files=archivos,
                        data={"vistas": [f[1] for f in fotos], **campos})


def contar(tabla):
    con = sqlite3.connect(main.RUTA_BD)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()[0]
    finally:
        con.close()
