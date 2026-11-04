"""Reportes: el Excel y el PDF se generan y contienen sus rótulos."""
import io

import pytest
from openpyxl import load_workbook

import main
from conftest import analizar, analizar_edificio, entrar, explicacion_ia, jpeg, texto_pdf


@pytest.fixture
def historial_con_datos(app, gemini, onnx):
    """Usuario con: una inspección de edificio con Gemini (exterior + interior) y una
    foto suelta con la explicación de respaldo."""
    h = entrar(app, "usuario_a")
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["Muros de ladrillo con columnas de concreto."],
                                      fotos=[(1, "Ladrillo/Albañilería", True), (2, "Ladrillo/Albañilería", True)],
                                      edificio="Ladrillo/Albañilería",
                                      discrepancia="La foto 2 parece de otro inmueble.")
    onnx.danos = [[0.05, 0.37, 0.58], [0.10, 0.20, 0.70]]
    analizar_edificio(app, [(jpeg(), "exterior", "fachada.jpg"), (jpeg(), "interior", "sala.jpg")], h)
    main.motivo_sin_ia = "desactivado"
    onnx.dano = [0.8, 0.1, 0.1]
    analizar(app, nombre="suelta.jpg", headers=h)
    return h


def test_excel_se_genera_con_sus_hojas_y_rotulos(app, historial_con_datos):
    r = app.get("/historial/exportar-excel", headers=historial_con_datos)
    assert r.status_code == 200 and "spreadsheetml" in r.headers["content-type"]
    assert "historial.xlsx" in r.headers["content-disposition"]
    libro = load_workbook(io.BytesIO(r.content))
    assert libro.sheetnames == ["Inspecciones", "Fotos"]
    inspecciones, fotos = libro["Inspecciones"], libro["Fotos"]
    for hoja in (inspecciones, fotos):
        assert str(hoja["A1"].value).startswith("Historial de usuario_a, generado el ")
        assert hoja.freeze_panes == "A4"
    assert [c.value for c in inspecciones[3]] == main.ENCABEZADOS_INSPECCIONES
    assert [c.value for c in fotos[3]] == ["Inspección"] + main.ENCABEZADOS_FOTOS


def test_excel_marca_el_origen_de_cada_valor(app, historial_con_datos):
    libro = load_workbook(io.BytesIO(app.get("/historial/exportar-excel", headers=historial_con_datos).content))
    columnas = {titulo: i for i, titulo in enumerate(main.ENCABEZADOS_INSPECCIONES)}
    filas = list(libro["Inspecciones"].iter_rows(min_row=4, values_only=True))
    edificio = next(f for f in filas if f[columnas["Fotos interiores"]] == 1)
    suelta = next(f for f in filas if f[columnas["Fotos interiores"]] == 0)
    assert edificio[columnas["Rescatable"]] == "No" + main.SUFIJO_RESCATABLE_REGLAS
    assert edificio[columnas["Construcción"]] == "Ladrillo/Albañilería" + main.SUFIJO_TIPO_GEMINI
    explicacion = edificio[columnas["Explicación IA"]]
    for rotulo in ("Fuente: Gemini (gemini-principal)", "Estructura:", "Estado:", "Viabilidad:", "Acciones:",
                   "Discrepancia con la clasificación:"):
        assert rotulo in explicacion
    assert suelta[columnas["Explicación IA"]].startswith("No disponible (desactivado)")
    assert suelta[columnas["Rescatable"]] == "Sí" + main.SUFIJO_RESCATABLE_REGLAS
    fotos = {f[2]: f for f in libro["Fotos"].iter_rows(min_row=4, values_only=True)}
    assert fotos["sala.jpg"][3] == "Interior" and fotos["sala.jpg"][6] == main.NO_EVALUADO_INTERIOR
    assert fotos["sala.jpg"][5] == "Ladrillo/Albañilería" + main.SUFIJO_TIPO_GEMINI, "el tipo de Gemini también en interiores"
    assert fotos["fachada.jpg"][5] == "Ladrillo/Albañilería" + main.SUFIJO_TIPO_GEMINI


def test_pdf_se_genera_con_titulo_y_secciones(app, historial_con_datos):
    r = app.get("/historial/exportar-pdf", headers=historial_con_datos)
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    assert "historial.pdf" in r.headers["content-disposition"]
    texto = texto_pdf(r.content)
    for rotulo in ("Historial de usuario_a, generado el", "2 inspecciones, ordenadas por prioridad",
                   "Inspección #1", "Inspección #2", "Daño general:", "Construcción:", "Rescatable:", "Recomendación:",
                   "Explicación IA", "Estructura", "Viabilidad:", "Acciones", "Discrepancia con la clasificación:",
                   "(estimación preliminar por reglas)", "(detectado por IA externa, referencial)",
                   "No evaluado (interior)", "Explicación IA no disponible (desactivado)",
                   "Foto", "Archivo", "Vista", "Confianza (%)"):
        assert rotulo in texto, rotulo


def test_excel_y_pdf_marcan_el_tipo_de_gemini_en_interiores(app, gemini):
    """Solo interiores: el tipo por foto y del edificio llevan el sufijo; rescatable de cada foto sigue sin evaluar."""
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["Muros de adobe."], fotos=[(1, "Adobe", True)], edificio="Adobe")
    h = entrar(app)
    analizar_edificio(app, [(jpeg(), "interior", "cocina.jpg")], h)
    libro = load_workbook(io.BytesIO(app.get("/historial/exportar-excel", headers=h).content))
    (fila,) = libro["Inspecciones"].iter_rows(min_row=4, values_only=True)
    columnas = {titulo: i for i, titulo in enumerate(main.ENCABEZADOS_INSPECCIONES)}
    assert fila[columnas["Construcción"]] == "Adobe" + main.SUFIJO_TIPO_GEMINI
    assert fila[columnas["Nota"]] == main.NOTA_SIN_EXTERIORES_TIPO_IA
    (foto,) = libro["Fotos"].iter_rows(min_row=4, values_only=True)
    assert foto[5] == "Adobe" + main.SUFIJO_TIPO_GEMINI and foto[6] == main.NO_EVALUADO_INTERIOR
    texto = " ".join(texto_pdf(app.get("/historial/exportar-pdf", headers=h).content).split())
    assert texto.count("Adobe (detectado por IA externa, referencial)") >= 2, "veredicto y foto"
    assert "No evaluado (interior)" in texto and "fotos interiores (referencial)" in texto


def test_pdf_muestra_la_advertencia_de_imagen_no_valida(app, gemini):
    gemini.activar()
    respuesta = explicacion_ia()
    respuesta.imagen_valida = False
    gemini.respuesta = respuesta
    h = entrar(app)
    analizar(app, headers=h)
    assert main.AVISO_IMAGEN_NO_VALIDA in texto_pdf(app.get("/historial/exportar-pdf", headers=h).content)


def test_reportes_vacios_se_generan(app):
    h = entrar(app)
    libro = load_workbook(io.BytesIO(app.get("/historial/exportar-excel", headers=h).content))
    assert libro["Inspecciones"].max_row == 3, "solo título y encabezados"
    texto = texto_pdf(app.get("/historial/exportar-pdf", headers=h).content)
    assert "0 inspecciones" in texto


def test_texto_explicacion():
    assert main.texto_explicacion(None) == ""
    assert main.texto_explicacion(main.explicacion_respaldo("r", "cuota")).startswith("No disponible (cuota)")
    gemini = {"fuente": "gemini", "modelo": "m", "imagen_valida": False, "estructura": ["a"], "estado": ["b"],
              "viabilidad": "c", "acciones": ["d"], "discrepancia": "e"}
    lineas = main.texto_explicacion(gemini).split("\n")
    assert lineas[:2] == ["Fuente: Gemini (m)", main.AVISO_IMAGEN_NO_VALIDA]
    assert "• a" in lineas and "Viabilidad: c" in lineas and lineas[-1] == "Discrepancia con la clasificación: e"
