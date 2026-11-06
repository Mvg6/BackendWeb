"""D-08: si el texto de la IA menciona un nivel de daño distinto al del ONNX, se marca
discrepancia; el nivel de daño (y con él el color de la ficha) no cambia."""
import pytest

import main
from conftest import analizar, analizar_edificio, entrar, explicacion_ia, jpeg

MODERADO = [0.10, 0.80, 0.10]
SEVERO = [0.05, 0.15, 0.80]
LEVE = [0.80, 0.15, 0.05]


def ia(estado=("Grietas diagonales.",), estructura=("Muros de albañilería.",), viabilidad="Daño visible.",
       acciones=("Acordonar el perímetro.",), discrepancia=None):
    e = explicacion_ia(estructura=estructura, discrepancia=discrepancia)
    e.estado, e.viabilidad, e.acciones = list(estado), viabilidad, list(acciones)
    return e


def test_moderado_con_colapso_en_el_texto_marca_discrepancia_sin_cambiar_el_nivel(app, gemini, onnx):
    onnx.dano = MODERADO
    gemini.activar()
    gemini.respuesta = ia(estado=["Colapso parcial del muro lateral."], viabilidad="El daño es severo.")
    j = analizar(app, headers=entrar(app)).json()
    assert (j["nivel_dano"], j["confianza"]) == ("Moderado", 80.0), "el nivel del ONNX no cambia"
    e = j["explicacion"]
    assert e["niveles_mencionados_distintos"] == ["colapso", "severo"]
    assert e["discrepancia"].startswith("Revisión automática: la explicación menciona «colapso», «severo»")
    assert "(Moderado)" in e["discrepancia"] and "Prevalece la clasificación del modelo" in e["discrepancia"]


def test_mismo_nivel_no_marca_nada(app, gemini, onnx):
    onnx.dano = SEVERO
    gemini.activar()
    gemini.respuesta = ia(estado=["Colapso total de la fachada."], viabilidad="Daño severo.")
    e = analizar(app, headers=entrar(app)).json()["explicacion"]
    assert e["discrepancia"] is None and e["niveles_mencionados_distintos"] == []


def test_un_nivel_menor_tambien_es_discrepancia(app, gemini, onnx):
    onnx.dano = SEVERO
    gemini.activar()
    gemini.respuesta = ia(estado=["Fisuras leves en el tarrajeo."])
    e = analizar(app, headers=entrar(app)).json()["explicacion"]
    assert e["niveles_mencionados_distintos"] == ["leves"] and "(Severo/Colapso)" in e["discrepancia"]


@pytest.mark.parametrize("texto", [
    "No se observa colapso de la estructura.",
    "Sin daño severo visible en las columnas.",
    "Se descarta colapso inminente.",
    "No hay ningún elemento colapsado.",
])
def test_una_negacion_no_cuenta_como_mencion(app, gemini, onnx, texto):
    onnx.dano = MODERADO
    gemini.activar()
    gemini.respuesta = ia(estado=[texto])
    assert analizar(app, headers=entrar(app)).json()["explicacion"]["discrepancia"] is None


def test_se_agrega_a_la_discrepancia_que_ya_dio_la_ia(app, gemini, onnx):
    onnx.dano = MODERADO
    gemini.activar()
    gemini.respuesta = ia(acciones=["Evacuar por colapso del techo."], discrepancia="La foto parece de otro inmueble.")
    d = analizar(app, headers=entrar(app)).json()["explicacion"]["discrepancia"]
    assert d.startswith("La foto parece de otro inmueble. Revisión automática:")


def test_edificio_compara_cada_frase_con_las_fotos_que_nombra(app, gemini, onnx):
    onnx.danos = [SEVERO, LEVE]  # foto 1 Severo, foto 2 Leve; veredicto Severo/Colapso
    gemini.activar()
    gemini.respuesta = ia(estado=["En la foto 2 las grietas son leves.", "La foto 1 muestra colapso del muro."])
    e = analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(color=(1, 2, 3)), "exterior")], entrar(app)).json()
    assert e["veredicto"]["nivel_dano"] == "Severo/Colapso" and e["explicacion"]["discrepancia"] is None
    onnx.danos = [SEVERO, LEVE]
    gemini.respuesta = ia(estado=["En la foto 1 el daño es leve."])
    e = analizar_edificio(app, [(jpeg(color=(4, 5, 6)), "exterior"), (jpeg(color=(7, 8, 9)), "exterior")],
                          entrar(app)).json()["explicacion"]
    assert e["niveles_mencionados_distintos"] == ["leve"], "la foto 1 es Severo/Colapso"


def test_frase_sin_foto_acepta_el_nivel_de_cualquier_foto_del_edificio(app, gemini, onnx):
    onnx.danos = [SEVERO, LEVE]
    gemini.activar()
    gemini.respuesta = ia(estado=["Hay zonas con daño leve y otras colapsadas."], viabilidad="Daño moderado en general.")
    e = analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(color=(1, 2, 3)), "exterior")], entrar(app)).json()
    assert e["explicacion"]["niveles_mencionados_distintos"] == ["moderado"]


def test_tambien_con_groq_y_queda_en_el_historial_y_el_excel(app, gemini, onnx, monkeypatch):
    import io
    import json

    import httpx
    from openpyxl import load_workbook

    import pool_claves
    onnx.dano = MODERADO
    main.motivo_sin_ia = None  # IA activa (sin Gemini configurado): responde la clave Groq
    main.config_ia["GROQ_MODEL"] = "groq-simulado"
    conn = pool_claves.conectar(main.RUTA_BD)
    with conn:
        pool_claves.agregar(conn, main._fernet_pool, "groq", "gsk_ClaveSimuladaD08xyz", prioridad=1)
    conn.close()
    contenido = json.dumps(ia(estado=["Se ven muros colapsados."]).model_dump())

    async def enviar(url, clave, cuerpo, timeout):
        return httpx.Response(200, json={"choices": [{"message": {"content": contenido}}]})

    monkeypatch.setattr(main, "_enviar_compatible", enviar)
    h = entrar(app)
    assert analizar(app, headers=h).json()["explicacion"]["niveles_mencionados_distintos"] == ["colapsados"]
    (guardada,) = app.get("/historial", headers=h).json()
    assert guardada["veredicto"]["nivel_dano"] == "Moderado" and "Revisión automática" in guardada["explicacion"]["discrepancia"]
    hoja = load_workbook(io.BytesIO(app.get("/historial/exportar-excel", headers=h).content))["Inspecciones"]
    assert any("Revisión automática" in str(c.value) for fila in hoja.iter_rows(min_row=4) for c in fila)


def test_la_plantilla_no_se_revisa(app, onnx):
    onnx.dano = MODERADO
    e = analizar(app, headers=entrar(app)).json()["explicacion"]
    assert e["fuente"] == "plantilla" and "niveles_mencionados_distintos" not in e and e["discrepancia"] is None


@pytest.mark.parametrize("texto, nivel, esperado", [
    ("La vivienda COLAPSÓ parcialmente.", "Moderado", ["colapso"]),
    ("Daño SEVERA en la esquina.", "Leve", ["severa"]),
    ("Grietas moderadas en el muro.", "Moderado", []),
    ("Fisuras moderadas y leves.", "Severo/Colapso", ["moderadas", "leves"]),
    ("Riesgo de colapso de la cornisa.", "Leve", ["colapso"]),
])
def test_niveles_mencionados_distintos(texto, nivel, esperado):
    e = {"estructura": [], "estado": [texto], "viabilidad": "", "acciones": []}
    assert main.niveles_mencionados_distintos(e, [{"nivel_dano": nivel}], nivel) == esperado
