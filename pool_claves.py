"""Pool de claves de API para la explicación con IA (Gemini, Groq, OpenRouter).

Las claves se guardan cifradas con Fernet (tabla api_keys de historial.db). La clave
maestra es POOL_MASTER_KEY en backend/.env; si falta, se genera y se agrega al archivo.
Nunca se imprime ni se registra una clave completa: como mucho sus últimos 4 caracteres.
Lo usan main.py y la línea de comandos pool_keys.py (sin dependencias de FastAPI).
"""
import base64
import io
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as hora
from typing import Optional

import httpx
from cryptography.fernet import Fernet, InvalidToken
from dotenv import dotenv_values

PROVEEDORES = ("gemini", "groq", "openrouter")
# Prioridad al importar desde .env la primera vez (menor = se usa primero).
VARIABLES_ENV = (("gemini", "GEMINI_API_KEY", 1), ("groq", "GROQ_API_KEY", 2), ("openrouter", "OPENROUTER_API_KEY", 3))
VARIABLE_MAESTRA = "POOL_MASTER_KEY"
URL_GROQ = "https://api.groq.com/openai/v1"
# Modelos de Groq que aceptan imágenes, en orden de preferencia (el listado de la API
# no indica la modalidad, así que se filtra por nombre y se prueba con una imagen).
PATRONES_VISION_GROQ = (r"llama-4-scout", r"llama-4-maverick", r"vision", r"-vl\b", r"llava", r"gemma-3")
# Modelos que no son de chat (voz, transcripción, clasificadores de seguridad): no se prueban.
PATRON_NO_CHAT_GROQ = r"whisper|orpheus|tts|guard"


@dataclass
class ClaveIA:
    id: Optional[int]
    proveedor: str
    modelo: Optional[str]
    alias: Optional[str]
    prioridad: int
    usos_hoy: int
    clave: str = field(repr=False)  # nunca aparece en repr ni en logs
    # Rotación por modelo: modelo -> timestamp hasta el que está apartado (429, 503...).
    esperas_modelo: dict = field(default_factory=dict)

    @property
    def nombre(self):
        """Para los logs: proveedor, alias y los últimos 4 caracteres."""
        return f"{self.proveedor}:{self.alias or self.id} ({enmascarar(self.clave)})"


def enmascarar(clave):
    clave = clave or ""
    return f"****{clave[-4:]}" if len(clave) >= 8 else "****"


# --- .env y cifrado ------------------------------------------------------------------
def escribir_variable_env(ruta_env, nombre, valor):
    """Agrega o reemplaza NOMBRE=valor en .env sin tocar las demás líneas."""
    lineas = []
    if os.path.exists(ruta_env):
        with open(ruta_env, encoding="utf-8") as archivo:
            lineas = archivo.read().splitlines()
    patron = re.compile(rf"^\s*{re.escape(nombre)}\s*=")
    nuevas = [f"{nombre}={valor}" if patron.match(linea) else linea for linea in lineas]
    if nuevas == lineas:
        nuevas.append(f"{nombre}={valor}")
    with open(ruta_env, "w", encoding="utf-8") as archivo:
        archivo.write("\n".join(nuevas) + "\n")


def obtener_fernet(ruta_env, crear=True):
    """Fernet con POOL_MASTER_KEY; si falta y `crear`, la genera y la agrega a .env.
    Devuelve (fernet, creada)."""
    valor = (dotenv_values(ruta_env).get(VARIABLE_MAESTRA) or "").strip()
    creada = False
    if not valor:
        if not crear:
            raise ValueError(f"falta {VARIABLE_MAESTRA} en .env")
        valor = Fernet.generate_key().decode()
        escribir_variable_env(ruta_env, VARIABLE_MAESTRA, valor)
        creada = True
    try:
        return Fernet(valor.encode()), creada
    except (ValueError, TypeError):
        raise ValueError(f"{VARIABLE_MAESTRA} en .env no es una clave Fernet válida") from None


def cifrar(fernet, clave):
    return fernet.encrypt(clave.encode()).decode()


def descifrar(fernet, cifrada):
    """None si no se puede descifrar (p. ej. cambió POOL_MASTER_KEY)."""
    try:
        return fernet.decrypt(cifrada.encode()).decode()
    except (InvalidToken, ValueError):
        return None


# --- Tabla -------------------------------------------------------------------------
def crear_tabla(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proveedor TEXT NOT NULL CHECK (proveedor IN ('gemini', 'groq', 'openrouter')),
            modelo TEXT,
            clave_cifrada TEXT NOT NULL,
            alias TEXT,
            activa INTEGER NOT NULL DEFAULT 1,
            prioridad INTEGER NOT NULL DEFAULT 10,
            cooldown_hasta REAL NOT NULL DEFAULT 0,
            usos_hoy INTEGER NOT NULL DEFAULT 0,
            fecha_usos TEXT,
            errores_seguidos INTEGER NOT NULL DEFAULT 0,
            ultimo_error TEXT,
            ultimo_uso REAL
        )
    """)
    # JSON {modelo: cooldown_hasta}: el cupo de Gemini es por modelo, un 429 aparta solo ese modelo.
    if "cooldown_modelos" not in {fila[1] for fila in conn.execute("PRAGMA table_info(api_keys)")}:
        conn.execute("ALTER TABLE api_keys ADD COLUMN cooldown_modelos TEXT")


def _esperas_vigentes(texto, ahora):
    try:
        esperas = json.loads(texto) if texto else {}
    except ValueError:
        return {}
    return {m: hasta for m, hasta in esperas.items() if isinstance(hasta, (int, float)) and hasta > ahora}


def apartar_modelo(conn, id_, modelo, hasta, detalle):
    """Aparta un modelo de una clave hasta `hasta` (la clave sigue activa para sus otros modelos)."""
    (texto,) = conn.execute("SELECT cooldown_modelos FROM api_keys WHERE id = ?", (id_,)).fetchone() or (None,)
    esperas = _esperas_vigentes(texto, time.time())
    esperas[modelo] = max(esperas.get(modelo, 0), hasta)
    conn.execute("UPDATE api_keys SET cooldown_modelos = ?, ultimo_error = ?, errores_seguidos = errores_seguidos + 1 "
                 "WHERE id = ?", (json.dumps(esperas), f"{modelo}: {detalle or ''}"[:300], id_))


def contar(conn):
    return conn.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]


def agregar(conn, fernet, proveedor, clave, modelo=None, alias=None, prioridad=10):
    """Devuelve el id nuevo. ValueError si el proveedor no es válido o la clave ya está."""
    proveedor = proveedor.strip().lower()
    clave = (clave or "").strip()
    if proveedor not in PROVEEDORES:
        raise ValueError(f"proveedor inválido: {proveedor} (usa {', '.join(PROVEEDORES)})")
    if len(clave) < 8:
        raise ValueError("la clave es demasiado corta")
    for (cifrada,) in conn.execute("SELECT clave_cifrada FROM api_keys WHERE proveedor = ?", (proveedor,)):
        if descifrar(fernet, cifrada) == clave:
            raise ValueError(f"esa clave ya está en el pool ({enmascarar(clave)})")
    cur = conn.execute(
        "INSERT INTO api_keys (proveedor, modelo, clave_cifrada, alias, prioridad) VALUES (?, ?, ?, ?, ?)",
        (proveedor, modelo or None, cifrar(fernet, clave), alias or None, int(prioridad)),
    )
    return cur.lastrowid


def importar_desde_env(conn, fernet, config):
    """Solo si la tabla está vacía: GEMINI_API_KEY (prioridad 1), GROQ_API_KEY (2) y
    OPENROUTER_API_KEY (3). Devuelve los proveedores importados."""
    if contar(conn):
        return []
    importados = []
    for proveedor, variable, prioridad in VARIABLES_ENV:
        clave = (config.get(variable) or "").strip()
        if clave:
            agregar(conn, fernet, proveedor, clave, alias=f"{variable} (.env)", prioridad=prioridad)
            importados.append(proveedor)
    return importados


def _usos_de_hoy(usos, fecha_usos, ahora):
    return usos if fecha_usos == date.fromtimestamp(ahora).isoformat() else 0


def disponibles(conn, fernet, ahora=None):
    """Claves activas sin cooldown: menor prioridad primero y, a igual prioridad, la
    menos usada hoy. Las que no se pueden descifrar se omiten."""
    ahora = time.time() if ahora is None else ahora
    filas = conn.execute("""
        SELECT id, proveedor, modelo, alias, prioridad, usos_hoy, fecha_usos, clave_cifrada, cooldown_modelos
        FROM api_keys WHERE activa = 1 AND cooldown_hasta <= ?
    """, (ahora,)).fetchall()
    claves = []
    for id_, proveedor, modelo, alias, prioridad, usos, fecha_usos, cifrada, esperas in filas:
        clave = descifrar(fernet, cifrada)
        if clave is not None:
            claves.append(ClaveIA(id_, proveedor, modelo, alias, prioridad, _usos_de_hoy(usos, fecha_usos, ahora), clave,
                                  _esperas_vigentes(esperas, ahora)))
    claves.sort(key=lambda c: (c.prioridad, c.usos_hoy, c.id))
    return claves


def todas(conn, fernet, ahora=None):
    """Estado de todas las claves para la CLI, con la clave ya enmascarada."""
    ahora = time.time() if ahora is None else ahora
    filas = conn.execute("""
        SELECT id, proveedor, modelo, alias, activa, prioridad, cooldown_hasta, usos_hoy, fecha_usos,
               errores_seguidos, ultimo_error, ultimo_uso, clave_cifrada, cooldown_modelos
        FROM api_keys ORDER BY prioridad, id
    """).fetchall()
    resultado = []
    for (id_, proveedor, modelo, alias, activa, prioridad, cooldown, usos, fecha_usos,
         errores, ultimo_error, ultimo_uso, cifrada, esperas) in filas:
        clave = descifrar(fernet, cifrada)
        resultado.append({
            "id": id_, "proveedor": proveedor, "modelo": modelo, "alias": alias, "activa": bool(activa),
            "prioridad": prioridad, "clave": enmascarar(clave) if clave is not None else "(no se puede descifrar)",
            "cooldown_restante": max(0.0, cooldown - ahora), "cooldown_hasta": cooldown,
            "usos_hoy": _usos_de_hoy(usos, fecha_usos, ahora), "errores_seguidos": errores,
            "ultimo_error": ultimo_error, "ultimo_uso": ultimo_uso,
            "modelos_en_espera": {m: hasta - ahora for m, hasta in _esperas_vigentes(esperas, ahora).items()},
        })
    return resultado


def claves_descifradas(conn, fernet):
    """Todas las claves en claro (para ocultarlas en los logs). No se imprimen."""
    return [c for (cifrada,) in conn.execute("SELECT clave_cifrada FROM api_keys")
            if (c := descifrar(fernet, cifrada))]


def activar(conn, id_, activa=True):
    """True si existía la clave. Al activar, se borran sus esperas (de la clave y de sus modelos) y sus errores."""
    if activa:
        cur = conn.execute("UPDATE api_keys SET activa = 1, cooldown_hasta = 0, cooldown_modelos = NULL, "
                           "errores_seguidos = 0 WHERE id = ?", (id_,))
    else:
        cur = conn.execute("UPDATE api_keys SET activa = 0 WHERE id = ?", (id_,))
    return cur.rowcount > 0


def registrar_uso(conn, id_, ahora=None):
    ahora = time.time() if ahora is None else ahora
    hoy = date.fromtimestamp(ahora).isoformat()
    conn.execute("""
        UPDATE api_keys SET usos_hoy = CASE WHEN fecha_usos = ? THEN usos_hoy + 1 ELSE 1 END,
                            fecha_usos = ?, ultimo_uso = ?, errores_seguidos = 0
        WHERE id = ?
    """, (hoy, hoy, ahora, id_))


def registrar_error(conn, id_, detalle, cooldown_hasta=0.0, desactivar=False):
    """`detalle` ya debe venir sin la clave (se guarda en ultimo_error)."""
    conn.execute("""
        UPDATE api_keys SET errores_seguidos = errores_seguidos + 1, ultimo_error = ?,
                            cooldown_hasta = MAX(cooldown_hasta, ?),
                            activa = CASE WHEN ? THEN 0 ELSE activa END
        WHERE id = ?
    """, ((detalle or "")[:300], cooldown_hasta, int(desactivar), id_))


# --- Reinicio de cupos diarios -----------------------------------------------------------
def _domingo(anio, mes, numero):
    primero = date(anio, mes, 1)
    return primero + timedelta(days=(6 - primero.weekday()) % 7 + 7 * (numero - 1))


def _desfase_pacifico(instante_utc):
    """UTC-7 en horario de verano (2.º domingo de marzo a 1.er domingo de noviembre,
    a las 2:00 locales); UTC-8 el resto del año. Sin depender de tzdata."""
    anio = instante_utc.year
    inicio = datetime.combine(_domingo(anio, 3, 2), hora(10), tzinfo=timezone.utc)
    fin = datetime.combine(_domingo(anio, 11, 1), hora(9), tzinfo=timezone.utc)
    return timedelta(hours=-7) if inicio <= instante_utc < fin else timedelta(hours=-8)


def proxima_medianoche_pacifico(ahora=None):
    """Timestamp de la próxima medianoche en la hora del Pacífico (reinicio del cupo de Gemini)."""
    utc = datetime.fromtimestamp(time.time() if ahora is None else ahora, timezone.utc)
    manana = (utc + _desfase_pacifico(utc)).date() + timedelta(days=1)
    medianoche = datetime.combine(manana, hora(0), tzinfo=timezone.utc)
    return (medianoche - _desfase_pacifico(medianoche + timedelta(hours=8))).timestamp()


def proxima_medianoche_utc(ahora=None):
    utc = datetime.fromtimestamp(time.time() if ahora is None else ahora, timezone.utc)
    return datetime.combine(utc.date() + timedelta(days=1), hora(0), tzinfo=timezone.utc).timestamp()


def reinicio_cupo_diario(proveedor, ahora=None):
    """Gemini reinicia a la medianoche del Pacífico; Groq y OpenRouter, a la medianoche UTC."""
    return proxima_medianoche_pacifico(ahora) if proveedor == "gemini" else proxima_medianoche_utc(ahora)


# --- Groq: modelo con imágenes ---------------------------------------------------------
def modelos_groq(clave, cliente=None):
    cliente = cliente or httpx.Client(timeout=20)
    r = cliente.get(f"{URL_GROQ}/models", headers={"Authorization": f"Bearer {clave}"})
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if m.get("active", True)]


def candidatos_vision_groq(ids):
    """Primero los nombres que suelen aceptar imágenes; después los demás modelos de chat
    (sin voz, transcripción ni filtros de seguridad). Cada uno se prueba con una imagen."""
    orden = []
    for patron in PATRONES_VISION_GROQ:
        orden += sorted(i for i in ids if re.search(patron, i) and i not in orden)
    return orden + sorted(i for i in ids if i not in orden and not re.search(PATRON_NO_CHAT_GROQ, i))


def _imagen_prueba_b64():
    """JPEG gris de 64x64 px para probar si un modelo acepta imágenes."""
    from PIL import Image
    salida = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(salida, "JPEG")
    return base64.b64encode(salida.getvalue()).decode()


def acepta_imagenes_groq(clave, modelo, cliente=None):
    """Prueba mínima: una imagen pequeña y respuesta JSON. True si el modelo responde."""
    cliente = cliente or httpx.Client(timeout=30)
    r = cliente.post(f"{URL_GROQ}/chat/completions", headers={"Authorization": f"Bearer {clave}"}, json={
        "model": modelo, "temperature": 0, "max_tokens": 20,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": 'Responde solo este JSON: {"ok": true}'},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_imagen_prueba_b64()}"}},
        ]}],
    })
    return r.status_code == 200


def elegir_modelo_groq(clave, cliente=None):
    """(modelo elegido o None, candidatos probados). Consulta GET /models y prueba los
    que por nombre aceptan imágenes, en orden de preferencia."""
    candidatos = candidatos_vision_groq(modelos_groq(clave, cliente))
    for modelo in candidatos:
        try:
            if acepta_imagenes_groq(clave, modelo, cliente):
                return modelo, candidatos
        except httpx.TimeoutException:
            continue  # un modelo lento no impide probar los demás
    return None, candidatos


def conectar(ruta_bd):
    conn = sqlite3.connect(ruta_bd)
    crear_tabla(conn)
    return conn
