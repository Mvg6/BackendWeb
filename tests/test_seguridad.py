"""Seguridad: límites por IP, encabezados de IP confiables, escape en PDF, metadatos,
tamaño de archivo, rutas de miniaturas, logs y celdas de Excel."""
import io
import os
import sqlite3
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from PIL import Image

import main
from conftest import BASURA, analizar, entrar, explicacion_ia, jpeg, texto_pdf


# --- Límites por IP ---------------------------------------------------------------------
def test_limite_de_analisis_sin_sesion(app):
    codigos = [analizar(app).status_code for _ in range(main.LIMITE_ANALISIS_SIN_SESION_POR_HORA + 1)]
    assert codigos == [200] * main.LIMITE_ANALISIS_SIN_SESION_POR_HORA + [429]
    r = analizar(app)
    assert r.status_code == 429 and "Retry-After" in r.headers and "inicia sesión" in r.json()["detail"]


def test_con_sesion_no_hay_limite_por_ip(app):
    h = entrar(app)
    assert all(analizar(app, headers=h).status_code == 200 for _ in range(main.LIMITE_ANALISIS_SIN_SESION_POR_HORA + 3))


def test_limite_de_registros(app):
    codigos = [app.post("/auth/registro", json={"usuario": f"cuenta{i}", "contrasena": "clave-segura-123",
                                                "acepta_privacidad": True}).status_code for i in range(6)]
    assert codigos == [201] * 5 + [429]


def test_limite_por_ip_ventana_deslizante():
    limite = main.LimitePorIP(2, 3600)
    assert [limite.consumir("1.1.1.1") for _ in range(2)] == [0, 0]
    assert limite.consumir("1.1.1.1") > 0 and limite.consumir("2.2.2.2") == 0
    assert main.LimitePorIP(0, 60).consumir("x") == 60, "límite 0 = siempre bloqueado, sin error"


# --- IP del cliente: encabezados de proxy ------------------------------------------------
def peticion(host, **encabezados):
    return SimpleNamespace(client=SimpleNamespace(host=host) if host else None,
                           headers={k.replace("_", "-"): v for k, v in encabezados.items()})


@pytest.mark.parametrize("req, ip", [
    (peticion("127.0.0.1", cf_connecting_ip="203.0.113.7"), "203.0.113.7"),
    (peticion("127.0.0.1", cf_connecting_ip="2001:db8::1"), "2001:db8::1"),
    (peticion("127.0.0.1", cf_connecting_ip="no-es-ip"), "127.0.0.1"),
    (peticion("127.0.0.1"), "127.0.0.1"),
    (peticion("10.0.0.5", cf_connecting_ip="203.0.113.7"), "10.0.0.5"),
    (peticion("127.0.0.1", x_forwarded_for="203.0.113.9"), "127.0.0.1"),
    (peticion("10.0.0.5", x_forwarded_for="203.0.113.9"), "10.0.0.5"),
    (peticion(None), "desconocida"),
])
def test_ip_confiable(req, ip):
    """CF-Connecting-IP solo cuenta si la conexión viene del túnel (127.0.0.1).
    X-Forwarded-For nunca lo usa la app (uvicorn debe correr con --no-proxy-headers)."""
    assert main._ip(req) == ip


def test_cambiar_encabezados_no_evade_el_limite_desde_afuera(app):
    afuera = TestClient(main.app, client=("10.0.0.5", 5000))  # sin `with`: comparte la app ya arrancada
    for i in range(main.LIMITE_ANALISIS_SIN_SESION_POR_HORA):
        assert analizar(afuera, headers={"CF-Connecting-IP": f"198.51.100.{i}", "X-Forwarded-For": f"192.0.2.{i}"}).status_code == 200
    assert analizar(afuera, headers={"CF-Connecting-IP": "198.51.100.200"}).status_code == 429


def test_detras_del_tunel_cada_visitante_tiene_su_limite(app):
    tunel = TestClient(main.app, client=("127.0.0.1", 5000))
    for _ in range(main.LIMITE_ANALISIS_SIN_SESION_POR_HORA):
        analizar(tunel, headers={"CF-Connecting-IP": "203.0.113.1"})
    assert analizar(tunel, headers={"CF-Connecting-IP": "203.0.113.1"}).status_code == 429
    assert analizar(tunel, headers={"CF-Connecting-IP": "203.0.113.2"}).status_code == 200


# --- Escape en el PDF ----------------------------------------------------------------------
def test_pdf_escapa_marcado_en_textos_de_usuario_y_de_gemini(app, gemini):
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["<b>negrita</b> & <script>alert(1)</script>"],
                                      discrepancia="Texto con <i>etiquetas</i> & símbolos")
    h = entrar(app)
    analizar(app, nombre="<img src=x onerror=alert(1)>.jpg", headers=h)
    r = app.get("/historial/exportar-pdf", headers=h)
    assert r.status_code == 200 and r.content.startswith(b"%PDF")
    # reportlab escribe cada entidad (&lt; &gt;) como un fragmento aparte: se compara sin espacios.
    compacto = "".join(texto_pdf(r.content).split())
    assert "<script>alert(1)</script>" in compacto and "<b>negrita</b>&" in compacto, "el marcado se muestra literal"
    assert "<imgsrc=xonerror=alert(1)>.jpg" in compacto and "<i>etiquetas</i>&símbolos" in compacto


# --- Metadatos y miniaturas ----------------------------------------------------------------
def test_miniatura_sin_exif_tamano_maximo_y_nombre_aleatorio(app, entorno):
    original = jpeg(3000, 2000, exif_gps=True)
    assert b"Exif" in original and Image.open(io.BytesIO(original)).getexif().get_ifd(0x8825)
    analizar(app, original, "casa_de_Juan_Perez.jpg", headers=entrar(app))
    (nombre,) = os.listdir(entorno.minis)
    datos = (entorno.minis / nombre).read_bytes()
    imagen = Image.open(io.BytesIO(datos))
    assert main.PATRON_MINIATURA.fullmatch(nombre) and "Juan" not in nombre
    assert imagen.format == "JPEG" and max(imagen.size) == main.LADO_MAXIMO_MINIATURA
    assert b"Exif" not in datos and b"CamaraPrivada" not in datos and not imagen.getexif()


def test_jpeg_sin_metadatos():
    datos = main.jpeg_sin_metadatos(jpeg(1500, 1000, exif_gps=True), 1024, 85)
    imagen = Image.open(io.BytesIO(datos))
    assert b"Exif" not in datos and "exif" not in imagen.info and max(imagen.size) == 1024


def test_nombres_de_miniatura_manipulados_no_salen_de_la_carpeta(app, entorno):
    afuera = entorno.tmp / "no_borrar.jpg"
    afuera.write_bytes(b"x")
    main.borrar_miniaturas(["../no_borrar.jpg", "..\\no_borrar.jpg", None])
    assert afuera.exists()
    h = entrar(app)
    i = analizar(app, headers=h).json()["inspeccion_id"]
    con = sqlite3.connect(main.RUTA_BD)
    with con:
        con.execute("UPDATE analisis SET miniatura = '../historial_prueba.db' WHERE inspeccion_id = ?", (i,))
    con.close()
    assert app.get(f"/inspecciones/{i}/fotos/1", headers=h).status_code == 404


def test_archivo_mayor_a_10_mb_responde_413(app):
    grande = b"\xff\xd8" + os.urandom(main.TAMANO_MAXIMO_ARCHIVO_BYTES)
    r = analizar(app, grande)
    assert r.status_code == 413


def test_el_log_no_incluye_el_nombre_del_archivo(app, caplog):
    nombre = "casa_de_Juan_Perez_DNI12345678.jpg"
    r = analizar(app, BASURA, nombre)
    assert r.status_code == 422 and nombre in r.json()["detail"]
    assert "No se pudo analizar" in caplog.text and "Juan" not in caplog.text and "DNI" not in caplog.text


# --- Excel: inyección de fórmulas -------------------------------------------------------------
@pytest.mark.parametrize("nombre", ['=HYPERLINK("http://ejemplo.invalid","clic").jpg', "=1+1.jpg",
                                    "+1+1.jpg", "-1+1.jpg", "@SUMA(1).jpg"])
def test_excel_no_convierte_textos_de_usuario_en_formulas(app, nombre):
    """Un nombre de archivo que empieza como fórmula se exporta como texto literal (CWE-1236)."""
    h = entrar(app)
    analizar(app, nombre=nombre, headers=h)
    hoja = load_workbook(io.BytesIO(app.get("/historial/exportar-excel", headers=h).content))["Fotos"]
    celdas = [c for fila in hoja.iter_rows(min_row=4) for c in fila]
    assert not [c.coordinate for c in celdas if c.data_type == "f"], "hay celdas convertidas en fórmula"
    archivo = hoja.cell(row=4, column=3)  # "Archivo"; el cliente de pruebas codifica las comillas como %22
    assert unquote(archivo.value) == nombre and archivo.data_type == "s", "texto literal, sin apóstrofo"


@pytest.mark.parametrize("prefijo", main.INICIOS_DE_FORMULA, ids=["igual", "mas", "menos", "arroba", "tab", "cr"])
def test_llenar_hoja_guarda_como_texto_todo_inicio_de_formula(prefijo):
    """Los seis inicios de OWASP (incluidos tabulación y retorno de carro, que no
    pueden llegar en un nombre de archivo) quedan como texto; los números no se tocan."""
    from openpyxl import Workbook
    libro = Workbook()
    texto = prefijo + 'HYPERLINK("http://ejemplo.invalid")'
    main._llenar_hoja(libro.active, "t", ["a", "b", "c"], [[texto, -5, "texto normal"]])
    salida = io.BytesIO()
    libro.save(salida)
    a, b, c = load_workbook(io.BytesIO(salida.getvalue())).active[4]
    # Al leer, el XML normaliza el retorno de carro a salto de línea: sigue siendo texto.
    assert a.value.replace("\n", "\r") == texto.replace("\n", "\r") and a.data_type == "s"
    assert (b.value, b.data_type) == (-5, "n") and (c.value, c.data_type) == ("texto normal", "s")
