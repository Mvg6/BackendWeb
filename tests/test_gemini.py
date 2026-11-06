"""Gemini (siempre simulado): reintentos, tipo seguro y regla de duda, consentimiento,
límite sin sesión, reconfiguración y contenido enviado."""
import asyncio
import io
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from conftest import analizar, analizar_edificio, entrar, error_api, explicacion_ia, jpeg

FOTOS = [{"bytes": jpeg(), "vista": "exterior", "nivel_dano": "Leve", "confianza": 80.0, "tipo_construccion": None,
          "rescatable": "Sí", "rescatable_fuente": "reglas", "nombre_archivo": "casa_de_Juan_Perez.jpg"}]
VEREDICTO = {**FOTOS[0], "recomendacion": "Recomendación por reglas."}


def explicar(gemini, *guion, alternativos=("gemini-alternativo",)):
    gemini.activar(alternativos)
    gemini.guion[:] = list(guion)
    return asyncio.run(main.obtener_explicacion(FOTOS, VEREDICTO))


def modelos(gemini):
    return [m for m, _, _ in gemini.llamadas]


# --- Reintentos ---------------------------------------------------------------------
def test_503_y_luego_responde(gemini, entorno):
    e = explicar(gemini, error_api(503), explicacion_ia())
    assert e["fuente"] == "gemini" and e["modelo"] == "gemini-principal" and entorno.esperas == [2]


def test_dos_fallas_y_el_tercer_intento_usa_el_alternativo(gemini, entorno):
    e = explicar(gemini, error_api(503), error_api(500), explicacion_ia())
    assert modelos(gemini) == ["gemini-principal", "gemini-principal", "gemini-alternativo"]
    assert e["modelo"] == "gemini-alternativo" and entorno.esperas == [2, 5]


@pytest.mark.parametrize("falla", [TimeoutError(), httpx.ReadTimeout("lento")])
def test_tras_un_timeout_el_siguiente_intento_usa_el_alternativo(gemini, falla):
    e = explicar(gemini, falla, explicacion_ia())
    assert modelos(gemini) == ["gemini-principal", "gemini-alternativo"] and e["fuente"] == "gemini"


def test_sin_alternativo_se_repite_el_principal(gemini):
    explicar(gemini, TimeoutError(), error_api(503), explicacion_ia(), alternativos=())
    assert modelos(gemini) == ["gemini-principal"] * 3


def test_429_con_retry_delay_largo_da_cuota_sin_reintentar(gemini, entorno, caplog):
    e = explicar(gemini, error_api(429, retry_delay="37s"))
    assert (e["fuente"], e["motivo_respaldo"]) == ("plantilla", "cuota")
    assert len(gemini.llamadas) == 1 and entorno.esperas == []
    assert "Google pide esperar 37 s" in caplog.text


def test_429_con_retry_delay_corto_reintenta_esperando_lo_sugerido(gemini, entorno):
    assert explicar(gemini, error_api(429, retry_delay="3.5s"), explicacion_ia())["fuente"] == "gemini"
    assert entorno.esperas == [3.5]


@pytest.mark.parametrize("codigo", [400, 401, 403, 404])
def test_errores_permanentes_no_se_reintentan(gemini, entorno, codigo):
    e = explicar(gemini, error_api(codigo))
    assert e["motivo_respaldo"] == "error" and len(gemini.llamadas) == 1 and entorno.esperas == []


def test_tres_fallas_seguidas_dan_plantilla_error(gemini, entorno):
    e = explicar(gemini, error_api(503), TimeoutError(), "mal_esquema")
    assert (e["fuente"], e["motivo_respaldo"]) == ("plantilla", "error")
    assert len(gemini.llamadas) == 3 and entorno.esperas == [2, 5]


def test_tres_429_seguidos_dan_cuota(gemini):
    assert explicar(gemini, error_api(429), error_api(429), error_api(429))["motivo_respaldo"] == "cuota"


def test_respuesta_fuera_de_esquema_se_reintenta(gemini):
    assert explicar(gemini, "mal_esquema", explicacion_ia())["fuente"] == "gemini"


def test_tope_total_de_tiempo(gemini, monkeypatch):
    monkeypatch.setattr(main, "TIEMPO_TOTAL_IA_SEGUNDOS", 1.0)
    monkeypatch.setattr(main, "TIMEOUT_IA_SEGUNDOS", 0.4)
    monkeypatch.setattr(main, "ESPERAS_REINTENTO_IA", (0.1, 0.1))
    monkeypatch.setattr(main, "_esperar", asyncio.sleep)
    inicio = time.perf_counter()
    e = explicar(gemini, ("lento", 3), ("lento", 3), ("lento", 3))
    assert e["motivo_respaldo"] == "error" and time.perf_counter() - inicio < 1.6


def test_la_clave_no_sale_en_el_log(gemini, caplog):
    main._clave_gemini = "AIzaClaveFalsaDePrueba123456"
    explicar(gemini, error_api(503, mensaje="key=AIzaClaveFalsaDePrueba123456"),
             error_api(401, mensaje="API key AIzaClaveFalsaDePrueba123456 not valid"))
    assert "AIzaClaveFalsaDePrueba123456" not in caplog.text and "***" in caplog.text


@pytest.mark.parametrize("falla, reintentable, motivo", [
    (error_api(503), True, "error"), (error_api(500), True, "error"), (error_api(429), True, "cuota"),
    (error_api(429, retry_delay="60s"), False, "cuota"), (error_api(401), False, "error"),
    (TimeoutError(), True, "error"), (main.RespuestaFueraDeEsquema(), True, "error"), (ValueError("x"), False, "error"),
])
def test_clasificar_falla(falla, reintentable, motivo):
    assert main._clasificar_falla_ia(falla)[:2] == (reintentable, motivo)


def test_sin_ia_disponible_devuelve_la_plantilla_con_su_motivo():
    main.motivo_sin_ia = "sin_clave"
    e = asyncio.run(main.obtener_explicacion(FOTOS, VEREDICTO))
    assert e == main.explicacion_respaldo("Recomendación por reglas.", "sin_clave")
    assert e["viabilidad"] == "Recomendación por reglas." and e["tipo_construccion_edificio_seguro"] is None


# --- Tipo de construcción: "seguro" y regla de duda --------------------------------------
@pytest.mark.parametrize("texto, tipo, dudoso", [
    ("bloques de piedra o adobe labrado", "Adobe", True),
    ("muros de adobe con grietas o fisuras", "Adobe", False),
    ("parece ser concreto armado", "Concreto armado", True),
    ("dos o tres pisos de ladrillo", "Ladrillo/Albañilería", False),
    ("concreto armado con tabiques de ladrillo", "Concreto armado", False),
    ("muros de ladrillo u adobe", "Adobe", True),
    ("pórticos de concreto armado o albañilería", "Concreto armado", True),
    ("Muros de PIEDRA O ADOBE", "Adobe", True),
    ("quincha o bahareque", "Madera", False),
    ("podría tratarse de tapial", "Adobe", True),
    ("parece uno dos tres concreto", "Concreto armado", True),
    ("parece uno dos tres cuatro concreto", "Concreto armado", False),
    ("muros de concreto; la cubierta posiblemente de madera", "Concreto armado", False),
])
def test_regla_de_duda(texto, tipo, dudoso):
    assert (main.fragmento_tipo_dudoso([texto], tipo) is not None) is dudoso


def test_regla_de_duda_por_foto():
    estructura = ["En la foto 2 se ven bloques de piedra o adobe.", "Foto 1: muros de ladrillo."]
    assert main.fragmento_tipo_dudoso(estructura, "Ladrillo/Albañilería", foto=1) is None
    assert main.fragmento_tipo_dudoso(estructura, "Adobe", foto=2) is not None
    assert main.fragmento_tipo_dudoso(estructura, "Ladrillo/Albañilería") is not None, "el edificio mira todas las frases"


def test_tipo_no_seguro_o_dudoso_cuenta_como_no_se_distingue():
    e = explicacion_ia(estructura=["Muros de ladrillo."], fotos=[(1, "Adobe", False), (2, "Ladrillo/Albañilería", True)],
                       edificio="Adobe", edificio_seguro=False).model_dump()
    assert main._tipos_por_foto(e) == {1: main.NO_SE_DISTINGUE, 2: "Ladrillo/Albañilería"}
    assert main._tipo_edificio_de_ia(e) == main.NO_SE_DISTINGUE
    assert main._tipo_de_ia(main.NO_SE_DISTINGUE) is None and main._tipo_de_ia("Adobe") == "Adobe"


def test_endpoint_tipo_de_gemini_con_regla_de_duda(app, gemini, caplog):
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["Muros portantes de bloques de piedra o adobe labrado."],
                                      fotos=[(1, "Adobe", True)], edificio="Adobe")
    j = analizar(app, headers=entrar(app)).json()
    assert (j["tipo_construccion"], j["tipo_construccion_fuente"]) == (None, "gemini")
    assert j["explicacion"]["tipo_construccion_edificio"] == "Adobe", "la explicación no se modifica"
    assert "«Muros portantes de bloques de piedra o adobe labrado»" in caplog.text


def _tipos(fotos):
    return [(f["tipo_construccion"], f["tipo_construccion_fuente"], f["rescatable"], f["rescatable_fuente"])
            for f in fotos]


def test_edificio_con_exteriores_e_interiores_tipo_por_foto_en_todas(app, gemini):
    """Interiores con tipo de Gemini como las exteriores; su rescatable sigue en null."""
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["Muros de ladrillo con columnas de concreto."],
                                      fotos=[(1, "Ladrillo/Albañilería", True), (2, "no_se_distingue", False),
                                             (3, "Ladrillo/Albañilería", True), (4, "Concreto armado", False)],
                                      edificio="Ladrillo/Albañilería")
    h = entrar(app)
    j = analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(), "exterior"), (jpeg(), "interior"),
                                (jpeg(), "interior"), (jpeg(), "interior")], h).json()
    esperado = [("Ladrillo/Albañilería", "gemini", "No", "reglas"), (None, "gemini", "No", "reglas"),
                ("Ladrillo/Albañilería", "gemini", None, None),
                (None, "gemini", None, None),   # seguro=false -> no se distingue
                (None, "gemini", None, None)]   # sin entrada de Gemini -> no se distingue
    assert _tipos(j["fotos"]) == esperado
    v = j["veredicto"]
    assert (v["tipo_construccion"], v["tipo_construccion_fuente"], v["nota"]) == ("Ladrillo/Albañilería", "gemini", None)
    (guardada,) = app.get("/historial", headers=h).json()
    assert _tipos(guardada["fotos"]) == esperado and guardada["veredicto"]["tipo_construccion"] == "Ladrillo/Albañilería"


def test_regla_de_duda_tambien_en_fotos_interiores(app, gemini, caplog):
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["Fachada de ladrillo (foto 1).", "En la foto 2 los muros parecen de adobe."],
                                      fotos=[(1, "Ladrillo/Albañilería", True), (2, "Adobe", True)],
                                      edificio="Ladrillo/Albañilería")
    j = analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(), "interior")], entrar(app)).json()
    assert _tipos(j["fotos"]) == [("Ladrillo/Albañilería", "gemini", "No", "reglas"), (None, "gemini", None, None)]
    assert "(foto 2)" in caplog.text


def test_solo_interiores_el_tipo_del_edificio_sale_de_ellas(app, gemini):
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["Muros de adobe con grietas en las esquinas."],
                                      fotos=[(1, "Adobe", True), (2, "Adobe", True)], edificio="Adobe")
    h = entrar(app)
    j = analizar_edificio(app, [(jpeg(), "interior"), (jpeg(), "interior")], h).json()
    assert _tipos(j["fotos"]) == [("Adobe", "gemini", None, None)] * 2
    v = j["veredicto"]
    assert (v["tipo_construccion"], v["tipo_construccion_fuente"]) == ("Adobe", "gemini")
    assert (v["rescatable"], v["rescatable_fuente"]) == ("Requiere inspección adicional", "reglas"), "no cambia"
    assert v["recomendacion"] == main.recomendacion_sin_exteriores(v["nivel_dano"]), "no cambia"
    assert v["nota"] == main.NOTA_SIN_EXTERIORES_TIPO_IA
    (guardada,) = app.get("/historial", headers=h).json()
    assert guardada["veredicto"] == v and _tipos(guardada["fotos"]) == _tipos(j["fotos"])


def test_solo_interiores_sin_gemini_el_tipo_queda_sin_evaluar(app):
    v = analizar_edificio(app, [(jpeg(), "interior")], entrar(app)).json()["veredicto"]
    assert (v["tipo_construccion"], v["tipo_construccion_fuente"], v["nota"]) == (None, None, main.NOTA_SIN_EXTERIORES)


def test_con_cnn_entrenada_para_el_tipo_se_usa_la_cnn(app, gemini):
    main.cabezas_no_entrenadas = ["rescatable"]
    gemini.activar()
    gemini.respuesta = explicacion_ia(fotos=[(1, "Adobe", True), (2, "Adobe", True)], edificio="Adobe")
    j = analizar(app, headers=entrar(app)).json()
    assert (j["tipo_construccion"], j["tipo_construccion_fuente"]) == ("Concreto armado", "modelo")
    fotos = analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(), "interior")], entrar(app, "otro")).json()["fotos"]
    assert _tipos(fotos) == [("Concreto armado", "modelo", "No", "reglas"), (None, None, None, None)], \
        "con la CNN entrenada para el tipo, Gemini no completa ninguna foto"


# --- Consentimiento y límite sin sesión ---------------------------------------------------
@pytest.mark.parametrize("valor", [None, "false", "on", "1", "si"])
def test_sin_sesion_sin_consentimiento_no_se_llama_a_gemini(app, gemini, valor):
    gemini.activar()
    campos = {} if valor is None else {"acepta_envio_ia": valor}
    j = analizar(app, **campos).json()
    assert j["explicacion"]["motivo_respaldo"] == "sin_sesion" and gemini.llamadas == []
    assert (j["tipo_construccion"], j["tipo_construccion_fuente"]) == (None, None)


def test_sin_sesion_con_consentimiento_usa_gemini_y_no_guarda(app, gemini):
    gemini.activar()
    j = analizar(app, acepta_envio_ia="true").json()
    assert j["explicacion"]["fuente"] == "gemini" and j["guardado"] is False and len(gemini.llamadas) == 1


def test_limite_de_explicaciones_sin_sesion(app, gemini):
    gemini.activar()
    # Fotos distintas: una foto repetida sale de la caché sin gastar el límite (ver test_pool_ia.py).
    resultados = [analizar(app, jpeg(color=(40 * i, 90, 60)), acepta_envio_ia="true").json()["explicacion"]
                  for i in range(4)]
    assert [r["fuente"] for r in resultados[:3]] == ["gemini"] * 3
    assert resultados[3]["motivo_respaldo"] == "limite_ia" and len(gemini.llamadas) == 3


def test_un_solo_consumo_del_limite_aunque_haya_reintentos(app, gemini):
    gemini.activar()
    gemini.guion[:] = [error_api(503), error_api(503), explicacion_ia()]
    assert analizar(app, acepta_envio_ia="true").json()["explicacion"]["fuente"] == "gemini"
    assert sum(len(u) for u in main.limite_ia_sin_sesion._usos.values()) == 1


def test_con_sesion_no_aplica_el_limite_sin_sesion(app, gemini):
    gemini.activar()
    h = entrar(app)
    assert all(analizar(app, headers=h).json()["explicacion"]["fuente"] == "gemini" for _ in range(5))


# --- Reconfiguración tras una falla al arrancar ---------------------------------------------
class ClienteGoogleSimulado:
    """Imita genai.Client para configurar_gemini (listado y sondeo)."""
    escenario = {"listado": None, "sondeo": {}}
    listados = 0
    sondeados = []

    def __init__(self, **_):
        self.models = SimpleNamespace(list=self._listar, generate_content=self._sondear)
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=None))

    def _listar(self):
        ClienteGoogleSimulado.listados += 1
        if self.escenario["listado"]:
            raise self.escenario["listado"]
        return [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
                for n in ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-pro")]

    def _sondear(self, model, contents, config):
        ClienteGoogleSimulado.sondeados.append(model)
        if model in self.escenario["sondeo"]:
            raise self.escenario["sondeo"][model]


@pytest.fixture
def google_simulado(monkeypatch, entorno):
    entorno.env.write_text("GEMINI_API_KEY=AIzaClaveDePrueba\nEXPLICACION_IA=on\n", encoding="utf-8")
    ClienteGoogleSimulado.escenario = {"listado": None, "sondeo": {}}
    ClienteGoogleSimulado.listados = 0
    ClienteGoogleSimulado.sondeados = []
    monkeypatch.setattr(main, "genai", SimpleNamespace(Client=ClienteGoogleSimulado))
    return ClienteGoogleSimulado


def test_configuracion_elige_flash_y_alternativa(google_simulado):
    """Sin GEMINI_MODELOS_EXCLUIDOS en .env, gemini-3.8-flash queda excluido: ni se prueba ni es alternativa."""
    google_simulado.escenario["sondeo"] = {"gemini-3.5-flash": error_api(503)}
    main.configurar_gemini()
    assert main.motivo_sin_ia is None and main.modelo_gemini == "gemini-3.5-flash-lite"
    assert main.modelos_alternativos_gemini == ["gemini-3.5-flash"], "la falla transitoria se conserva como alternativa"
    assert google_simulado.sondeados == ["gemini-3.5-flash", "gemini-3.5-flash-lite"]


@pytest.mark.parametrize("linea, principal, alternativos", [
    ("", "gemini-3.8-flash", ["gemini-3.5-flash", "gemini-3.5-flash-lite"]),                # variable vacía: ninguno excluido
    (" gemini-3.5-flash , GEMINI-3.8-FLASH,models/x ", "gemini-3.5-flash-lite", []),         # espacios, mayúsculas, prefijo
    ("gemini-3.5-flash-lite", "gemini-3.8-flash", ["gemini-3.5-flash"]),                     # lista propia: 3.8 vuelve a usarse
])
def test_modelos_excluidos_configurables_en_env(google_simulado, entorno, linea, principal, alternativos):
    entorno.env.write_text(f"GEMINI_API_KEY=AIzaClaveDePrueba\nEXPLICACION_IA=on\nGEMINI_MODELOS_EXCLUIDOS={linea}\n",
                           encoding="utf-8")
    main.configurar_gemini()
    assert (main.modelo_gemini, main.modelos_alternativos_gemini) == (principal, alternativos)


def test_gemini_model_excluido_no_se_usa(google_simulado, entorno, caplog):
    entorno.env.write_text("GEMINI_API_KEY=AIzaClaveDePrueba\nEXPLICACION_IA=on\nGEMINI_MODEL=gemini-3.8-flash\n",
                           encoding="utf-8")
    main.configurar_gemini()
    assert main.modelo_gemini == "gemini-3.5-flash" and "gemini-3.8-flash" not in main.modelos_alternativos_gemini
    assert "gemini-3.8-flash" not in google_simulado.sondeados and "está en GEMINI_MODELOS_EXCLUIDOS" in caplog.text


def test_modelos_excluidos():
    assert main.modelos_excluidos({}) == {"gemini-3.8-flash"}, "por defecto"
    assert main.modelos_excluidos({"GEMINI_MODELOS_EXCLUIDOS": ""}) == set()
    assert main.modelos_excluidos({"GEMINI_MODELOS_EXCLUIDOS": "a, ,b,"}) == {"a", "b"}


@pytest.mark.parametrize("falla", [error_api(503), error_api(429), TimeoutError(), httpx.ConnectError("sin red"), ConnectionResetError()])
def test_falla_transitoria_al_arrancar_queda_pendiente(google_simulado, falla):
    google_simulado.escenario["listado"] = falla
    main.configurar_gemini()
    assert (main.motivo_sin_ia, main.reconfigurar_ia_pendiente) == ("error", True)


@pytest.mark.parametrize("falla", [error_api(401), error_api(403), error_api(400)])
def test_falla_permanente_al_arrancar_desactiva_hasta_reiniciar(google_simulado, falla, caplog):
    google_simulado.escenario["listado"] = falla
    main.configurar_gemini()
    assert (main.motivo_sin_ia, main.reconfigurar_ia_pendiente) == ("error", False)
    assert "desactivada hasta reiniciar" in caplog.text


def test_se_reconfigura_en_la_siguiente_peticion_pasado_un_minuto(google_simulado, onnx):
    google_simulado.escenario["listado"] = error_api(503)
    with TestClient(main.app) as cliente:
        main.sesion_onnx, main.entrada_onnx_nombre, main.cabezas_no_entrenadas = onnx, "entrada", ["tipo_construccion", "rescatable"]
        estado = cliente.get("/estado").json()["explicacion_ia"]
        assert estado == {"lista": False, "modelo": None, "motivo": "error", "reintento_automatico": True}
        assert "AIzaClaveDePrueba" not in cliente.get("/estado").text
        h = entrar(cliente)
        listados = google_simulado.listados
        assert analizar(cliente, headers=h).json()["explicacion"]["motivo_respaldo"] == "error"
        assert google_simulado.listados == listados, "dentro del minuto no se reintenta"
        google_simulado.escenario["listado"] = None
        main._ultima_configuracion_ia = float("-inf")
        simulado = explicacion_ia()

        async def responder(model, contents, config):
            return SimpleNamespace(parsed=simulado)

        def cliente_que_responde(**kw):
            c = ClienteGoogleSimulado(**kw)
            c.aio = SimpleNamespace(models=SimpleNamespace(generate_content=responder))
            return c

        main.genai = SimpleNamespace(Client=cliente_que_responde)
        assert analizar(cliente, headers=h).json()["explicacion"]["fuente"] == "gemini"
        assert cliente.get("/estado").json()["explicacion_ia"]["lista"] is True


def test_sin_sesion_reconfiguracion_fallida_no_gasta_cuota(google_simulado, onnx):
    google_simulado.escenario["listado"] = error_api(503)
    with TestClient(main.app) as cliente:
        main.sesion_onnx, main.entrada_onnx_nombre = onnx, "entrada"
        main._ultima_configuracion_ia = float("-inf")
        j = analizar(cliente, acepta_envio_ia="true").json()
        assert j["explicacion"]["motivo_respaldo"] == "error"
        assert not any(main.limite_ia_sin_sesion._usos.values())


def test_candidatos_gemini():
    disponibles = {"gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-4.0-flash-preview",
                   "gemini-3.5-flash-tts", "gemini-3.5-pro", "gemini-flash-latest"}
    assert main.candidatos_gemini(disponibles, "") == ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"]
    assert main.candidatos_gemini(disponibles, "gemini-2.5-flash")[0] == "gemini-2.5-flash"
    assert main.candidatos_gemini(disponibles, "gemini-9-flash")[0] == "gemini-3.5-flash"
    assert main.candidatos_gemini(disponibles, "", {"gemini-3.5-flash"}) == ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
    assert main.candidatos_gemini(disponibles, "gemini-2.5-flash", {"gemini-2.5-flash"}) == ["gemini-3.5-flash",
                                                                                              "gemini-3.5-flash-lite"]


# --- Contenido enviado a Gemini -----------------------------------------------------------
def test_contenido_sin_nombres_de_archivo_sin_exif_y_sin_rescatable_por_reglas(app, gemini):
    gemini.activar()
    analizar(app, jpeg(3000, 2000, exif_gps=True), "casa_de_Juan_Perez_DNI12345678.jpg", headers=entrar(app))
    (_, textos, imagenes), = gemini.llamadas
    assert "Juan" not in textos and "DNI" not in textos and ".jpg" not in textos
    assert "rescatable no evaluado por el modelo" in textos
    assert "tipo de construcción no evaluado (el modelo aún no está entrenado para esto)" in textos
    (imagen,) = imagenes
    abierta = Image.open(io.BytesIO(imagen))
    assert b"Exif" not in imagen and not abierta.getexif() and max(abierta.size) <= main.LADO_MAXIMO_IMAGEN_IA


def test_prompt_pide_el_tipo_tambien_en_interiores():
    texto = " ".join(main.INSTRUCCIONES_IA.split())
    assert 'una entrada por CADA foto, exterior o interior' in texto
    assert "muros, columnas, vigas y techos visibles" in texto
    assert "prima lo que se vea con más claridad" in texto and 'contradicción real, usa "no_se_distingue"' in texto
    assert "Si solo hay fotos interiores, determínalo con ellas" in texto
    assert "solo para las fotos exteriores" not in texto and "ninguna para las fotos interiores" not in texto
