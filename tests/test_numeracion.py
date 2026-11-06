"""Número de ficha por usuario (numero_usuario) y guardar el análisis hecho sin sesión al
ingresar: migración, numeración independiente, concurrencia, reportes, rutas con el id
interno, caché de IA al guardar y que sin sesión no se guarda nada (PRIV_2)."""
import io
import sqlite3
import threading
from pathlib import Path

import pytest
from openpyxl import load_workbook

import main
from conftest import analizar, analizar_edificio, contar, entrar, jpeg, texto_pdf


def foto(i):
    return jpeg(color=(20 + 11 * i, 70, 90))


def numeros(cliente, h):
    return sorted(i["numero_usuario"] for i in cliente.get("/historial", headers=h).json())


# --- Migración ---------------------------------------------------------------------------
def test_migracion_numera_por_usuario_en_orden_de_id_sin_perder_datos(entorno):
    con = sqlite3.connect(main.RUTA_BD)
    con.execute("CREATE TABLE inspecciones (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, nivel_dano_general TEXT, "
                "rescatable_general TEXT, tipo_construccion_general TEXT, recomendacion TEXT, usuario_id INTEGER)")
    # id:        1    2    3    4    5     6
    duenos = [7, 9, 7, 7, 9, None]
    con.executemany("INSERT INTO inspecciones (fecha, nivel_dano_general, recomendacion, usuario_id) VALUES (?, ?, ?, ?)",
                    [(f"2026-10-0{i}", "Leve", f"r{i}", u) for i, u in enumerate(duenos, 1)])
    con.commit()
    con.close()
    for _ in range(2):  # la segunda vez no cambia nada
        main.inicializar_bd()
    con = sqlite3.connect(main.RUTA_BD)
    filas = con.execute("SELECT id, usuario_id, numero_usuario, recomendacion FROM inspecciones ORDER BY id").fetchall()
    con.close()
    assert filas == [(1, 7, 1, "r1"), (2, 9, 1, "r2"), (3, 7, 2, "r3"), (4, 7, 3, "r4"), (5, 9, 2, "r5"),
                     (6, None, None, "r6")]


def test_el_indice_unico_impide_numeros_repetidos(entorno):
    main.inicializar_bd()
    con = sqlite3.connect(main.RUTA_BD)
    con.execute("INSERT INTO inspecciones (usuario_id, numero_usuario) VALUES (1, 1)")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO inspecciones (usuario_id, numero_usuario) VALUES (1, 1)")
    con.execute("INSERT INTO inspecciones (usuario_id, numero_usuario) VALUES (2, 1)")  # otro usuario: sí
    con.close()


# --- Numeración independiente --------------------------------------------------------------
def test_cada_usuario_numera_desde_1(app):
    a, b = entrar(app, "usuario_a"), entrar(app, "usuario_b")
    respuestas = [analizar(app, foto(i), headers=a).json() for i in range(2)]
    respuestas.append(analizar_edificio(app, [(foto(5), "exterior"), (foto(6), "interior")], a).json())
    rb = analizar(app, foto(9), headers=b).json()
    assert [r["numero_usuario"] for r in respuestas] == [1, 2, 3]
    assert (rb["numero_usuario"], rb["inspeccion_id"]) == (1, 4), "B empieza en 1 aunque su id interno sea 4"
    assert numeros(app, a) == [1, 2, 3] and numeros(app, b) == [1]
    assert analizar(app, foto(10), headers=b).json()["numero_usuario"] == 2


def test_sin_sesion_no_hay_numero(app):
    j = analizar(app, foto(1)).json()
    assert (j["guardado"], j["inspeccion_id"], j["numero_usuario"]) == (False, None, None)


def test_concurrencia_basica_sin_numeros_repetidos(app):
    entrar(app)
    usuario_id = 1
    veredicto = {"nivel_dano": "Leve", "rescatable": "Sí", "rescatable_fuente": "reglas", "tipo_construccion": None,
                 "tipo_construccion_fuente": None, "recomendacion": "r"}
    resultados, errores = [], []

    def guardar():
        try:
            resultados.append(main.guardar_inspeccion(veredicto, [], None, [], usuario_id, []))
        except Exception as e:  # noqa: BLE001
            errores.append(e)

    hilos = [threading.Thread(target=guardar) for _ in range(20)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    assert not errores
    assert sorted(n for _, n in resultados) == list(range(1, 21)), "1..20, sin repetidos ni huecos"


# --- Se muestra el número por usuario; el id queda interno ---------------------------------------
@pytest.fixture
def b_con_id_distinto(app):
    """A con 3 fichas (ids 1-3); B con 2 (ids 4-5, números 1-2)."""
    a, b = entrar(app, "usuario_a"), entrar(app, "usuario_b")
    for i in range(3):
        analizar(app, foto(i), headers=a)
    for i in range(2):
        analizar(app, foto(10 + i), headers=b)
    return b


def test_historial_trae_el_numero_y_el_id_interno(app, b_con_id_distinto):
    historial = app.get("/historial", headers=b_con_id_distinto).json()
    assert sorted((i["numero_usuario"], i["inspeccion_id"]) for i in historial) == [(1, 4), (2, 5)]


def test_excel_muestra_ficha_por_usuario(app, b_con_id_distinto):
    libro = load_workbook(io.BytesIO(app.get("/historial/exportar-excel", headers=b_con_id_distinto).content))
    inspecciones, fotos = libro["Inspecciones"], libro["Fotos"]
    assert inspecciones["A3"].value == "Ficha N.º" and fotos["A3"].value == "Ficha N.º"
    assert sorted(f[0] for f in inspecciones.iter_rows(min_row=4, values_only=True)) == [1, 2]
    assert sorted(f[0] for f in fotos.iter_rows(min_row=4, values_only=True)) == [1, 2]


def test_pdf_muestra_ficha_por_usuario(app, b_con_id_distinto):
    texto = texto_pdf(app.get("/historial/exportar-pdf", headers=b_con_id_distinto).content)
    assert "Ficha N.º 1" in texto and "Ficha N.º 2" in texto
    assert "Ficha N.º 4" not in texto and "Inspección #" not in texto


def test_las_fotos_se_piden_con_el_id_interno_y_el_acceso_no_cambia(app, b_con_id_distinto):
    assert app.get("/inspecciones/4/fotos/1", headers=b_con_id_distinto).status_code == 200
    assert app.get("/inspecciones/1/fotos/1", headers=b_con_id_distinto).status_code == 404, "es de A"


def test_la_ficha_del_frontend_usa_el_numero_por_usuario():
    index = (Path(main.BASE_DIR).parent / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "Ficha N.º ${Number(r.numero_usuario)}" in index and "${Number(i.numero_usuario)}" in index
    assert "Ficha N.º ${Number(r.inspeccion_id)}" not in index
    assert "¿Guardar el análisis anterior en tu historial?" in index


# --- Guardar el análisis anterior al ingresar ------------------------------------------------
def test_guardar_al_ingresar_usa_la_cache_de_ia_sin_llamar_a_ningun_proveedor(app, gemini):
    gemini.activar()
    sin_sesion = analizar(app, foto(1), acepta_envio_ia="true", vista="interior").json()
    assert sin_sesion["guardado"] is False and len(gemini.llamadas) == 1
    assert contar("inspecciones") == 0, "PRIV_2: sin sesión no se guarda"
    h = entrar(app)  # crea la cuenta e inicia sesión
    guardado = analizar(app, foto(1), acepta_envio_ia="true", vista="interior", headers=h).json()
    assert len(gemini.llamadas) == 1, "la explicación sale de la caché"
    assert guardado["guardado"] is True and guardado["numero_usuario"] == 1
    assert guardado["explicacion"] == sin_sesion["explicacion"] and guardado["vista"] == "interior"
    (ficha,) = app.get("/historial", headers=h).json()
    assert ficha["explicacion"] == sin_sesion["explicacion"] and ficha["fotos"][0]["vista"] == "interior"


def test_guardar_un_edificio_al_ingresar_tambien_sale_de_la_cache(app, gemini):
    gemini.activar()
    fotos = [(foto(1), "exterior"), (foto(2), "interior"), (foto(3), "exterior")]
    sin_sesion = analizar_edificio(app, fotos, acepta_envio_ia="true").json()
    h = entrar(app)
    guardado = analizar_edificio(app, fotos, h).json()
    assert len(gemini.llamadas) == 1 and guardado["explicacion"] == sin_sesion["explicacion"]
    assert [f["vista"] for f in guardado["fotos"]] == ["exterior", "interior", "exterior"]


def test_con_otra_vista_no_es_la_misma_entrada(app, gemini):
    gemini.activar()
    analizar(app, foto(1), acepta_envio_ia="true")
    analizar(app, foto(1), headers=entrar(app), vista="interior")
    assert len(gemini.llamadas) == 2


def test_priv2_sin_sesion_nunca_se_guarda(app, gemini):
    gemini.activar()
    analizar(app, foto(1), acepta_envio_ia="true")
    analizar(app, foto(2))
    analizar_edificio(app, [(foto(3), "exterior"), (foto(4), "interior")], acepta_envio_ia="true")
    assert contar("inspecciones") == 0 and contar("analisis") == 0
    assert not list(Path(main.DIR_MINIATURAS).glob("*.jpg"))
