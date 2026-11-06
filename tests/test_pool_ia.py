"""Pool de claves de IA (Gemini, Groq, OpenRouter), siempre con proveedores simulados:
cifrado y enmascarado, selección y rotación ante 429/503/401, cooldowns, todas caídas,
tope total, caché, que la IA no altere el nivel de daño, D-05 (interior en "Una foto")
y modo feria."""
import asyncio
import io
import json
import sqlite3
import time

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from openpyxl import load_workbook

import main
import pool_claves
import pool_keys
from conftest import analizar, analizar_edificio, entrar, error_api, explicacion_ia, jpeg

CLAVE_GEMINI = "AIzaClaveGeminiSimulada0001"
CLAVE_GEMINI_2 = "AIzaClaveGeminiSimulada0002"
CLAVE_GROQ = "gsk_ClaveGroqSimulada0003"
MODELO_GROQ = "groq-vision-simulado"


def foto(i):
    """Fotos distintas (la misma foto saldría de la caché)."""
    return jpeg(color=(10 + 7 * i, 90, 60))


def respuesta_compatible(contenido, codigo=200, encabezados=None):
    if codigo != 200:
        return httpx.Response(codigo, headers=encabezados or {}, json={"error": {"message": contenido}})
    texto = contenido if isinstance(contenido, str) else json.dumps(contenido, ensure_ascii=False)
    return httpx.Response(200, json={"choices": [{"message": {"content": texto}}]})


class CompatibleSimulado:
    """Imita POST /chat/completions de Groq u OpenRouter. Cada llamada consume un paso del
    `guion`: una excepción, un httpx.Response, ("lento", s) o un dict/str de respuesta."""

    def __init__(self):
        self.guion = []
        self.respuesta = explicacion_ia().model_dump()
        self.llamadas = []

    async def enviar(self, url, clave, cuerpo, timeout):
        self.llamadas.append({"url": url, "clave": clave, "cuerpo": cuerpo})
        paso = self.guion.pop(0) if self.guion else self.respuesta
        if isinstance(paso, tuple) and paso[0] == "lento":
            await asyncio.sleep(paso[1])
            paso = self.respuesta
        if isinstance(paso, BaseException):
            raise paso
        if isinstance(paso, httpx.Response):
            return paso
        return respuesta_compatible(paso)


class Pool:
    def __init__(self, cliente, gemini, compatible):
        self.cliente, self.gemini, self.compatible = cliente, gemini, compatible

    def agregar(self, proveedor, clave, prioridad, modelo=None):
        conn = pool_claves.conectar(main.RUTA_BD)
        with conn:
            id_ = pool_claves.agregar(conn, main._fernet_pool, proveedor, clave, modelo, None, prioridad)
        conn.close()
        if proveedor == "gemini" and main._id_clave_gemini is None:
            main._id_clave_gemini = id_
        return id_

    def fila(self, id_):
        conn = pool_claves.conectar(main.RUTA_BD)
        try:
            return next(c for c in pool_claves.todas(conn, main._fernet_pool) if c["id"] == id_)
        finally:
            conn.close()


@pytest.fixture
def compatible(monkeypatch):
    simulado = CompatibleSimulado()
    monkeypatch.setattr(main, "_enviar_compatible", simulado.enviar)
    return simulado


@pytest.fixture
def pool(app, gemini, compatible):
    """App arrancada con el pool vacío; Gemini y Groq simulados (falta agregar las claves)."""
    gemini.activar(alternativos=())
    main.config_ia["GROQ_MODEL"] = MODELO_GROQ
    return Pool(app, gemini, compatible)


@pytest.fixture
def gemini_y_groq(pool):
    """Gemini con prioridad 1 y Groq con prioridad 2."""
    return pool.agregar("gemini", CLAVE_GEMINI, 1), pool.agregar("groq", CLAVE_GROQ, 2)


# --- Cifrado y enmascarado ---------------------------------------------------------------------
def test_las_claves_se_guardan_cifradas(pool, entorno):
    id_ = pool.agregar("groq", CLAVE_GROQ, 2)
    con = sqlite3.connect(main.RUTA_BD)
    (cifrada,) = con.execute("SELECT clave_cifrada FROM api_keys WHERE id = ?", (id_,)).fetchone()
    volcado = "\n".join(con.iterdump())
    con.close()
    assert CLAVE_GROQ not in volcado and CLAVE_GROQ not in cifrada
    assert pool_claves.descifrar(main._fernet_pool, cifrada) == CLAVE_GROQ
    assert pool_claves.descifrar(Fernet(Fernet.generate_key()), cifrada) is None, "otra clave maestra no descifra"
    assert "POOL_MASTER_KEY=" in entorno.env.read_text(encoding="utf-8")


def test_clave_maestra_se_genera_sin_mostrarse(entorno, caplog, capsys):
    entorno.env.write_text("EXPLICACION_IA=off\n", encoding="utf-8")
    fernet, creada = pool_claves.obtener_fernet(str(entorno.env))
    valor = entorno.env.read_text(encoding="utf-8").split("POOL_MASTER_KEY=")[1].strip()
    assert creada and pool_claves.obtener_fernet(str(entorno.env))[1] is False, "la segunda vez ya existe"
    assert valor not in caplog.text and valor not in capsys.readouterr().out
    assert pool_claves.descifrar(fernet, pool_claves.cifrar(fernet, "x" * 10)) == "x" * 10


def test_clave_maestra_invalida_no_rompe_el_arranque(entorno, onnx, caplog):
    entorno.env.write_text("EXPLICACION_IA=off\nPOOL_MASTER_KEY=no-es-una-clave\n", encoding="utf-8")
    with TestClient(main.app) as cliente:
        assert cliente.get("/estado").status_code == 200 and main._fernet_pool is None
    assert "Pool de claves IA no disponible" in caplog.text


@pytest.mark.parametrize("clave, mascara", [("AIzaSyEJEMPLO1234abcd", "****abcd"), ("corta", "****"), (None, "****")])
def test_enmascarar(clave, mascara):
    assert pool_claves.enmascarar(clave) == mascara


def test_repr_de_una_clave_no_la_muestra():
    clave = pool_claves.ClaveIA(1, "groq", None, "alias", 2, 0, CLAVE_GROQ)
    assert CLAVE_GROQ not in repr(clave) and CLAVE_GROQ not in clave.nombre and "0003" in clave.nombre


def test_ninguna_clave_completa_en_logs_ni_en_estado(pool, gemini_y_groq, caplog):
    pool.gemini.guion[:] = [error_api(503, mensaje=f"falla con key={CLAVE_GEMINI}")]
    analizar(pool.cliente, headers=entrar(pool.cliente))
    estado = pool.cliente.get("/estado").text
    g, _ = gemini_y_groq
    for clave in (CLAVE_GEMINI, CLAVE_GROQ):
        assert clave not in caplog.text and clave not in estado
    assert CLAVE_GEMINI not in (pool.fila(g)["ultimo_error"] or "") and "***" in pool.fila(g)["ultimo_error"]
    assert "****0001" in caplog.text, "los logs identifican la clave por sus últimos 4 caracteres"


# --- Migración desde .env ---------------------------------------------------------------------
def test_migracion_importa_las_claves_de_env_solo_con_la_tabla_vacia(entorno, onnx, caplog):
    entorno.env.write_text(f"EXPLICACION_IA=off\nGEMINI_API_KEY={CLAVE_GEMINI}\nGROQ_API_KEY={CLAVE_GROQ}\n"
                           f"GROQ_MODEL={MODELO_GROQ}\n", encoding="utf-8")
    for _ in range(2):  # el segundo arranque no vuelve a importar
        with TestClient(main.app):
            pass
    con = pool_claves.conectar(main.RUTA_BD)
    filas = con.execute("SELECT proveedor, prioridad FROM api_keys ORDER BY prioridad").fetchall()
    con.close()
    assert filas == [("gemini", 1), ("groq", 2)]
    assert "2 clave(s) cargada(s), 2 activa(s) (gemini: 1, groq: 1)" in caplog.text
    assert CLAVE_GEMINI not in caplog.text and CLAVE_GROQ not in caplog.text


def test_groq_sin_modelo_se_elige_uno_con_imagenes_y_se_guarda(entorno, onnx, monkeypatch, caplog):
    entorno.env.write_text(f"EXPLICACION_IA=off\nGROQ_API_KEY={CLAVE_GROQ}\n", encoding="utf-8")
    pedidas = []
    monkeypatch.setattr(pool_claves, "elegir_modelo_groq",
                        lambda clave: (pedidas.append(clave), ("meta-llama/llama-4-scout-17b-16e-instruct", []))[1])
    with TestClient(main.app):
        pass
    assert pedidas == [CLAVE_GROQ]
    assert "GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct" in entorno.env.read_text(encoding="utf-8")
    assert main.config_ia["GROQ_MODEL"] == "meta-llama/llama-4-scout-17b-16e-instruct"


def test_elegir_modelo_groq_consulta_models_y_prueba_con_imagen():
    probados = []

    def manejar(peticion):
        if peticion.url.path.endswith("/models"):
            assert peticion.headers["authorization"] == f"Bearer {CLAVE_GROQ}"
            return httpx.Response(200, json={"data": [{"id": i} for i in (
                "llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct", "whisper-large-v3")]})
        cuerpo = json.loads(peticion.content)
        probados.append(cuerpo["model"])
        assert any(p["type"] == "image_url" for p in cuerpo["messages"][0]["content"])
        return httpx.Response(400 if "scout" in cuerpo["model"] else 200, json={})

    cliente = httpx.Client(transport=httpx.MockTransport(manejar))
    modelo, candidatos = pool_claves.elegir_modelo_groq(CLAVE_GROQ, cliente)
    assert candidatos == ["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct",
                          "llama-3.3-70b-versatile"], "primero los de visión por nombre; nunca whisper"
    assert probados == candidatos[:2] and modelo == "meta-llama/llama-4-maverick-17b-128e-instruct"


# --- Selección ------------------------------------------------------------------------------
def test_seleccion_por_prioridad_y_luego_la_menos_usada_hoy():
    conn = sqlite3.connect(":memory:")
    pool_claves.crear_tabla(conn)
    fernet = Fernet(Fernet.generate_key())
    a = pool_claves.agregar(conn, fernet, "gemini", "clave-a-12345678", prioridad=2)
    b = pool_claves.agregar(conn, fernet, "gemini", "clave-b-12345678", prioridad=2)
    c = pool_claves.agregar(conn, fernet, "groq", "clave-c-12345678", prioridad=1)
    d = pool_claves.agregar(conn, fernet, "groq", "clave-d-12345678", prioridad=0)
    ahora = time.time()
    pool_claves.registrar_uso(conn, a, ahora)
    pool_claves.registrar_error(conn, c, "503", cooldown_hasta=ahora + 60)
    pool_claves.activar(conn, d, False)
    assert [k.id for k in pool_claves.disponibles(conn, fernet, ahora)] == [b, a], "c en espera, d desactivada"
    assert [k.id for k in pool_claves.disponibles(conn, fernet, ahora + 61)] == [c, b, a], "terminó la espera de c"
    manana = ahora + 86400
    assert pool_claves.disponibles(conn, fernet, manana)[1].usos_hoy == 0, "los usos se reinician cada día"


def test_agregar_rechaza_proveedor_invalido_y_clave_repetida():
    conn = sqlite3.connect(":memory:")
    pool_claves.crear_tabla(conn)
    fernet = Fernet(Fernet.generate_key())
    pool_claves.agregar(conn, fernet, "groq", CLAVE_GROQ)
    with pytest.raises(ValueError, match="ya está en el pool"):
        pool_claves.agregar(conn, fernet, "groq", f"  {CLAVE_GROQ} ")
    with pytest.raises(ValueError, match="proveedor inválido"):
        pool_claves.agregar(conn, fernet, "openai", CLAVE_GROQ)


# --- Rotación y cooldowns ------------------------------------------------------------------------
def test_503_deja_la_clave_en_espera_60_s_y_responde_la_siguiente(pool, gemini_y_groq):
    g, q = gemini_y_groq
    rotaciones = main.estadisticas_ia["rotaciones"]
    pool.gemini.guion[:] = [error_api(503)]
    h = entrar(pool.cliente)
    e = analizar(pool.cliente, foto(1), headers=h).json()["explicacion"]
    assert (e["fuente"], e["modelo"]) == ("groq", MODELO_GROQ)
    assert len(pool.gemini.llamadas) == 1 and len(pool.compatible.llamadas) == 1, "sin reintentar la clave que falló"
    assert 55 <= pool.fila(g)["cooldown_restante"] <= 60 and pool.fila(g)["errores_seguidos"] == 1
    assert pool.fila(q)["usos_hoy"] == 1 and main.estadisticas_ia["rotaciones"] == rotaciones + 1
    analizar(pool.cliente, foto(2), headers=h)
    assert len(pool.gemini.llamadas) == 1, "en espera: no se usa"


@pytest.mark.parametrize("falla", [TimeoutError(), httpx.ReadTimeout("lento"), error_api(500)])
def test_timeout_o_500_tambien_dejan_60_s(pool, gemini_y_groq, falla):
    g, _ = gemini_y_groq
    pool.gemini.guion[:] = [falla]
    assert analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]["fuente"] == "groq"
    assert 55 <= pool.fila(g)["cooldown_restante"] <= 60


def test_429_espera_lo_que_pide_el_proveedor(pool, gemini_y_groq):
    g, _ = gemini_y_groq
    pool.gemini.guion[:] = [error_api(429, retry_delay="37s")]
    assert analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]["fuente"] == "groq"
    assert 32 <= pool.fila(g)["cooldown_restante"] <= 37


def test_429_por_cupo_diario_espera_hasta_la_medianoche_del_pacifico(pool, gemini_y_groq):
    g, _ = gemini_y_groq
    pool.gemini.guion[:] = [error_api(429, mensaje="Quota exceeded for metric generate_content_free_tier_requests, "
                                                   "quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")]
    antes = time.time()
    assert analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]["fuente"] == "groq"
    assert abs(pool.fila(g)["cooldown_hasta"] - pool_claves.proxima_medianoche_pacifico(antes)) < 5


def test_groq_429_con_retry_after_pasa_a_gemini(pool):
    q = pool.agregar("groq", CLAVE_GROQ, 1)
    pool.agregar("gemini", CLAVE_GEMINI, 2)
    pool.compatible.guion[:] = [respuesta_compatible("Rate limit reached: requests per minute", 429, {"retry-after": "12"})]
    e = analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]
    assert e["fuente"] == "gemini" and 7 <= pool.fila(q)["cooldown_restante"] <= 12


def test_groq_429_por_cupo_diario_espera_hasta_la_medianoche_utc(pool):
    q = pool.agregar("groq", CLAVE_GROQ, 1)
    pool.agregar("gemini", CLAVE_GEMINI, 2)
    pool.compatible.guion[:] = [respuesta_compatible("Rate limit reached on requests per day (RPD)", 429)]
    antes = time.time()
    analizar(pool.cliente, headers=entrar(pool.cliente))
    assert abs(pool.fila(q)["cooldown_hasta"] - pool_claves.proxima_medianoche_utc(antes)) < 5


@pytest.mark.parametrize("codigo", [401, 403])
def test_401_o_403_desactivan_la_clave(pool, gemini_y_groq, codigo, caplog):
    g, _ = gemini_y_groq
    pool.gemini.guion[:] = [error_api(codigo)]
    h = entrar(pool.cliente)
    assert analizar(pool.cliente, foto(1), headers=h).json()["explicacion"]["fuente"] == "groq"
    assert pool.fila(g)["activa"] is False and "quedó desactivada" in caplog.text
    analizar(pool.cliente, foto(2), headers=h)
    assert len(pool.gemini.llamadas) == 1


def test_400_de_clave_invalida_de_gemini_tambien_la_desactiva(pool, gemini_y_groq):
    g, _ = gemini_y_groq
    pool.gemini.guion[:] = [error_api(400, mensaje="API key not valid. Please pass a valid API key.")]
    analizar(pool.cliente, headers=entrar(pool.cliente))
    assert pool.fila(g)["activa"] is False


def test_dos_claves_gemini_rotan_entre_si(pool, monkeypatch):
    g1 = pool.agregar("gemini", CLAVE_GEMINI, 1)
    g2 = pool.agregar("gemini", CLAVE_GEMINI_2, 1)
    otra = type(pool.gemini)()
    main._clientes_gemini[g2] = type(main.cliente_gemini)(aio=type(main.cliente_gemini.aio)(
        models=type(main.cliente_gemini.aio.models)(generate_content=otra.generate_content)))
    pool.gemini.guion[:] = [error_api(503)]
    e = analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]
    assert e["fuente"] == "gemini" and len(otra.llamadas) == 1 and pool.fila(g1)["cooldown_restante"] > 0


def test_a_igual_prioridad_se_reparte_entre_las_claves(pool):
    g = pool.agregar("gemini", CLAVE_GEMINI, 1)
    q = pool.agregar("groq", CLAVE_GROQ, 1)
    h = entrar(pool.cliente)
    for i in range(4):
        analizar(pool.cliente, foto(i), headers=h)
    assert pool.fila(g)["usos_hoy"] == 2 and pool.fila(q)["usos_hoy"] == 2


# --- Todas caídas y tope total -------------------------------------------------------------------
def test_todas_caidas_devuelve_el_onnx_sin_inventar_explicacion(pool, gemini_y_groq):
    g, q = gemini_y_groq
    pool.gemini.guion[:] = [error_api(503)]
    pool.compatible.guion[:] = [respuesta_compatible("sobrecargado", 503)] * 3
    r = analizar(pool.cliente, headers=entrar(pool.cliente))
    j = r.json()
    e = j["explicacion"]
    assert r.status_code == 200 and j["nivel_dano"] == "Severo/Colapso" and j["probabilidades"]
    assert (e["fuente"], e["modelo"], e["motivo_respaldo"]) == ("plantilla", None, "error")
    assert e["estructura"] == e["estado"] == e["acciones"] == [] and e["discrepancia"] is None
    assert "sin explicación" in e["mensaje"]
    assert len(pool.compatible.llamadas) == 3, "la última clave se reintenta como siempre"
    assert pool.fila(g)["cooldown_restante"] > 0 and pool.fila(q)["cooldown_restante"] > 0


def test_sin_claves_disponibles_no_llama_a_nadie(pool, gemini_y_groq):
    for id_ in gemini_y_groq:
        conn = pool_claves.conectar(main.RUTA_BD)
        with conn:
            pool_claves.activar(conn, id_, False)
        conn.close()
    e = analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]
    assert e["motivo_respaldo"] == "error" and not pool.gemini.llamadas and not pool.compatible.llamadas
    assert pool.cliente.get("/estado").json()["explicacion_ia"]["lista"] is False


def test_sin_sesion_sin_claves_disponibles_no_gasta_la_cuota(pool, gemini_y_groq):
    for id_ in gemini_y_groq:
        conn = pool_claves.conectar(main.RUTA_BD)
        with conn:
            pool_claves.registrar_error(conn, id_, "503", cooldown_hasta=time.time() + 60)
        conn.close()
    assert analizar(pool.cliente, acepta_envio_ia="true").json()["explicacion"]["motivo_respaldo"] == "error"
    assert not any(main.limite_ia_sin_sesion._usos.values())


def test_tope_total_de_60_s_entre_todas_las_claves(pool, gemini_y_groq, monkeypatch):
    monkeypatch.setattr(main, "TIEMPO_TOTAL_IA_SEGUNDOS", 1.0)
    monkeypatch.setattr(main, "TIMEOUT_IA_SEGUNDOS", 0.6)
    monkeypatch.setattr(main, "_esperar", asyncio.sleep)
    pool.gemini.guion[:] = [("lento", 3)]
    pool.compatible.guion[:] = [("lento", 3)] * 3
    inicio = time.perf_counter()
    e = analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]
    assert e["motivo_respaldo"] == "error" and time.perf_counter() - inicio < 1.8


# --- Consistencia: la IA solo explica -------------------------------------------------------------
def test_la_ia_no_altera_el_nivel_de_dano_ni_el_rescatable(pool):
    pool.agregar("groq", CLAVE_GROQ, 1)
    respuesta = {**explicacion_ia().model_dump(), "nivel_dano": "Leve", "rescatable": "Sí", "confianza": 99,
                 "tipo_construccion_edificio": "Bambú", "tipo_construccion_edificio_seguro": True,
                 "tipo_construccion_foto": [{"foto": 1, "tipo_construccion": "Bambú", "tipo_construccion_seguro": True}]}
    pool.compatible.respuesta = respuesta
    j = analizar(pool.cliente, headers=entrar(pool.cliente)).json()
    assert (j["nivel_dano"], j["confianza"]) == ("Severo/Colapso", 70.0)
    assert (j["rescatable"], j["rescatable_fuente"]) == ("No", "reglas")
    assert (j["tipo_construccion"], j["tipo_construccion_fuente"]) == (None, "groq"), "fuera de la lista -> no se distingue"
    assert "nivel_dano" not in j["explicacion"] and j["explicacion"]["tipo_construccion_edificio"] == "no_se_distingue"


@pytest.mark.parametrize("texto, valido", [
    ('```json\n{"x": 1}\n```', False),
    ("no es json", False),
    ("[1, 2]", False),
])
def test_respuesta_que_no_valida_se_rechaza(texto, valido):
    with pytest.raises(main.RespuestaFueraDeEsquema):
        main.validar_respuesta_ia(texto)


def test_validar_respuesta_acepta_cercas_y_normaliza_tipos():
    datos = {**explicacion_ia().model_dump(), "tipo_construccion_edificio": "Adobe", "tipo_construccion_edificio_seguro": "sí",
             "tipo_construccion_foto": [{"foto": 1, "tipo_construccion": "Adobe", "tipo_construccion_seguro": True},
                                        {"foto": "2", "tipo_construccion": "Adobe", "tipo_construccion_seguro": True}]}
    e = main.validar_respuesta_ia(f"```json\n{json.dumps(datos)}\n```")
    assert (e.tipo_construccion_edificio, e.tipo_construccion_edificio_seguro) == ("Adobe", False), "seguro no booleano"
    assert [f.foto for f in e.tipo_construccion_foto] == [1], "entrada sin número de foto válido"


def test_respuesta_fuera_de_esquema_de_groq_se_reintenta_si_es_la_ultima(pool):
    pool.agregar("groq", CLAVE_GROQ, 1)
    pool.compatible.guion[:] = ["esto no es json"]
    assert analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]["fuente"] == "groq"
    assert len(pool.compatible.llamadas) == 2


def test_mismo_prompt_temperatura_cero_y_json_en_todos_los_proveedores(pool):
    pool.agregar("groq", CLAVE_GROQ, 1)
    analizar(pool.cliente, headers=entrar(pool.cliente))
    (llamada,) = pool.compatible.llamadas
    cuerpo = llamada["cuerpo"]
    assert llamada["url"] == "https://api.groq.com/openai/v1/chat/completions" and llamada["clave"] == CLAVE_GROQ
    assert cuerpo["temperature"] == 0 and cuerpo["response_format"] == {"type": "json_object"}
    assert cuerpo["model"] == MODELO_GROQ and cuerpo["messages"][0]["content"].startswith(main.INSTRUCCIONES_IA)
    partes = cuerpo["messages"][1]["content"]
    assert any(p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg;base64,") for p in partes)
    assert main.CONFIG_IA.temperature == 0


def test_el_prompt_incluye_probabilidades_y_rescatable_y_prohibe_cambiarlos(app, gemini):
    gemini.activar()
    analizar(app, headers=entrar(app))
    (_, textos, _), = gemini.llamadas
    assert "probabilidades del modelo: Leve 0.100, Moderado 0.200, Severo/Colapso 0.700" in textos
    assert "Nivel de daño: Severo/Colapso" in textos and "Rescatable:" in textos and "rescatable" in textos
    assert "no puedes cambiar el nivel de daño ni el rescatable" in " ".join(main.INSTRUCCIONES_IA.split())


def test_groq_con_mas_de_5_fotos_no_se_usa(pool):
    pool.agregar("groq", CLAVE_GROQ, 1)
    pool.agregar("gemini", CLAVE_GEMINI, 2)
    j = analizar_edificio(pool.cliente, [(foto(i), "exterior") for i in range(6)], entrar(pool.cliente)).json()
    assert j["explicacion"]["fuente"] == "gemini" and not pool.compatible.llamadas


# --- Caché ------------------------------------------------------------------------------------
def test_misma_entrada_misma_explicacion_sin_gastar_cupo(app, gemini):
    gemini.activar()
    h = entrar(app)
    primera = analizar(app, foto(1), headers=h).json()["explicacion"]
    segunda = analizar(app, foto(1), headers=h).json()["explicacion"]
    assert primera == segunda and len(gemini.llamadas) == 1


def test_sin_sesion_la_cache_no_gasta_la_cuota(app, gemini):
    gemini.activar()
    for _ in range(3):
        assert analizar(app, foto(1), acepta_envio_ia="true").json()["explicacion"]["fuente"] == "gemini"
    assert len(gemini.llamadas) == 1 and sum(len(u) for u in main.limite_ia_sin_sesion._usos.values()) == 1


def test_la_cache_distingue_la_vista_y_las_fotos(app, gemini):
    gemini.activar()
    h = entrar(app)
    analizar(app, foto(1), headers=h)
    analizar(app, foto(1), headers=h, vista="interior")
    analizar(app, foto(2), headers=h)
    analizar_edificio(app, [(foto(1), "exterior"), (foto(2), "exterior")], h)
    assert len(gemini.llamadas) == 4


def test_la_plantilla_no_se_guarda_en_la_cache(app, gemini):
    gemini.activar(alternativos=())
    gemini.guion[:] = [error_api(400)]
    h = entrar(app)
    assert analizar(app, foto(1), headers=h).json()["explicacion"]["fuente"] == "plantilla"
    assert analizar(app, foto(1), headers=h).json()["explicacion"]["fuente"] == "gemini"


def test_sin_consentimiento_no_se_usa_ni_la_cache(app, gemini):
    gemini.activar()
    analizar(app, foto(1), headers=entrar(app))
    assert analizar(app, foto(1)).json()["explicacion"]["motivo_respaldo"] == "sin_sesion"


# --- Historial: proveedor y modelo -------------------------------------------------------------
def test_historial_guarda_proveedor_y_modelo(pool):
    pool.agregar("groq", CLAVE_GROQ, 1)
    h = entrar(pool.cliente)
    analizar(pool.cliente, foto(1), headers=h)
    main.motivo_sin_ia = "desactivado"
    analizar(pool.cliente, foto(2), headers=h)
    por_id = {i["inspeccion_id"]: i for i in pool.cliente.get("/historial", headers=h).json()}
    assert (por_id[1]["ia_proveedor"], por_id[1]["ia_modelo"]) == ("groq", MODELO_GROQ)
    assert (por_id[2]["ia_proveedor"], por_id[2]["ia_modelo"]) == (None, None)
    hoja = load_workbook(io.BytesIO(pool.cliente.get("/historial/exportar-excel", headers=h).content))["Inspecciones"]
    assert any(str(c.value).startswith(f"Fuente: Groq ({MODELO_GROQ})") for fila in hoja.iter_rows(min_row=4) for c in fila)


def test_migracion_de_proveedor_y_modelo_sin_perder_datos(entorno):
    con = sqlite3.connect(main.RUTA_BD)
    con.execute("CREATE TABLE inspecciones (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, nivel_dano_general TEXT, "
                "rescatable_general TEXT, tipo_construccion_general TEXT, recomendacion TEXT, explicacion TEXT)")
    con.executemany("INSERT INTO inspecciones (fecha, nivel_dano_general, recomendacion, explicacion) VALUES (?, ?, ?, ?)", [
        ("2026-10-01", "Leve", "r1", json.dumps({"fuente": "gemini", "modelo": "gemini-3.1-flash-lite"})),
        ("2026-10-02", "Moderado", "r2", json.dumps({"fuente": "plantilla", "modelo": None})),
        ("2026-10-03", "Severo/Colapso", "r3", None),
    ])
    con.commit()
    con.close()
    main.inicializar_bd()
    con = sqlite3.connect(main.RUTA_BD)
    filas = con.execute("SELECT nivel_dano_general, recomendacion, ia_proveedor, ia_modelo FROM inspecciones ORDER BY id").fetchall()
    con.close()
    assert filas == [("Leve", "r1", "gemini", "gemini-3.1-flash-lite"), ("Moderado", "r2", None, None),
                     ("Severo/Colapso", "r3", None, None)]


def test_estado_muestra_el_pool_sin_claves(pool, gemini_y_groq):
    estado = pool.cliente.get("/estado").json()
    assert estado["pool_ia"] == {"claves": 2, "activas": 2, "disponibles": 2, "modo_feria": False}
    assert estado["explicacion_ia"]["lista"] is True and estado["explicacion_ia"]["modelo"] == "gemini-principal"


# --- D-05: "Una foto" respeta la vista elegida -----------------------------------------------------
def test_d05_una_foto_interior_deja_rescatable_en_null(app):
    h = entrar(app)
    j = analizar(app, headers=h, vista="interior").json()
    assert j["vista"] == "interior" and j["nota"] == main.NOTA_SIN_EXTERIORES
    assert (j["rescatable"], j["rescatable_fuente"], j["tipo_construccion"]) == (None, None, None)
    assert j["nivel_dano"] == "Severo/Colapso" and j["recomendacion"] == main.recomendacion_sin_exteriores("Severo/Colapso")
    (guardada,) = app.get("/historial", headers=h).json()
    assert guardada["fotos"][0]["vista"] == "interior" and guardada["veredicto"]["fotos_exteriores"] == 0
    assert guardada["veredicto"]["rescatable"] is None
    fila = list(load_workbook(io.BytesIO(app.get("/historial/exportar-excel", headers=h).content))["Inspecciones"]
                .iter_rows(min_row=4, values_only=True))[0]
    assert fila[main.ENCABEZADOS_INSPECCIONES.index("Rescatable")] == main.NO_EVALUADO_SIN_EXTERIORES


def test_d05_sin_vista_sigue_siendo_exterior(app):
    j = analizar(app, headers=entrar(app)).json()
    assert j["vista"] == "exterior" and j["nota"] is None and (j["rescatable"], j["rescatable_fuente"]) == ("No", "reglas")


def test_d05_vista_invalida_400(app):
    assert analizar(app, vista="techo").status_code == 400


def test_d05_interior_con_ia_trae_el_tipo_pero_no_el_rescatable(app, gemini):
    gemini.activar()
    gemini.respuesta = explicacion_ia(estructura=["Muros de adobe."], fotos=[(1, "Adobe", True)], edificio="Adobe")
    j = analizar(app, headers=entrar(app), vista="interior").json()
    assert (j["tipo_construccion"], j["tipo_construccion_fuente"]) == ("Adobe", "gemini")
    assert (j["rescatable"], j["rescatable_fuente"]) == (None, None) and j["nota"] == main.NOTA_SIN_EXTERIORES_TIPO_IA
    (_, textos, _), = gemini.llamadas
    assert "Foto 1 (interior)" in textos


# --- Modo feria -------------------------------------------------------------------------------------
@pytest.fixture
def con_env(entorno, onnx):
    """Arranca la app con las líneas de .env indicadas."""
    def arrancar(*lineas):
        entorno.env.write_text("\n".join(("EXPLICACION_IA=off",) + lineas) + "\n", encoding="utf-8")
        cliente = TestClient(main.app)
        cliente.__enter__()
        main.sesion_onnx, main.entrada_onnx_nombre = onnx, "entrada"
        main.cabezas_no_entrenadas = ["tipo_construccion", "rescatable"]
        arrancados.append(cliente)
        return cliente
    arrancados = []
    yield arrancar
    for cliente in arrancados:
        cliente.__exit__(None, None, None)


def test_feria_off_todo_como_hoy(app):
    assert main.modo_feria is False and main.limite_rpm_ia is None
    assert (main.limite_analisis_sin_sesion.maximo, main.limite_ia_sin_sesion.maximo) == (10, 3)
    assert app.get("/estado").json()["pool_ia"]["modo_feria"] is False


def test_feria_on_sube_los_limites_sin_cuenta(con_env):
    cliente = con_env("MODO_FERIA=on")
    assert main.modo_feria is True and main.limite_rpm_ia.maximo == 20
    assert (main.limite_analisis_sin_sesion.maximo, main.limite_ia_sin_sesion.maximo) == (300, 300)
    assert all(analizar(cliente, foto(i)).status_code == 200 for i in range(main.LIMITE_ANALISIS_SIN_SESION_POR_HORA + 2))
    assert cliente.get("/estado").json()["pool_ia"]["modo_feria"] is True


def test_feria_valores_configurables(con_env):
    con_env("MODO_FERIA=on", "LIMITE_FERIA_HORA=50", "FERIA_IA_RPM=7")
    assert (main.limite_analisis_sin_sesion.maximo, main.limite_ia_sin_sesion.maximo, main.limite_rpm_ia.maximo) == (50, 50, 7)


def test_feria_por_minuto_pasa_a_la_siguiente_clave_y_luego_sin_explicacion(con_env, gemini, compatible):
    cliente = con_env("MODO_FERIA=on", "FERIA_IA_RPM=1")
    gemini.activar(alternativos=())
    main.config_ia["GROQ_MODEL"] = MODELO_GROQ
    pool = Pool(cliente, gemini, compatible)
    pool.agregar("gemini", CLAVE_GEMINI, 1)
    pool.agregar("groq", CLAVE_GROQ, 2)
    fuentes = [analizar(cliente, foto(i), acepta_envio_ia="true").json()["explicacion"] for i in range(3)]
    assert [f["fuente"] for f in fuentes] == ["gemini", "groq", "plantilla"]
    assert fuentes[2]["motivo_respaldo"] == "limite_feria" and "sin explicación" in fuentes[2]["mensaje"]
    assert analizar(cliente, foto(9)).json()["explicacion"]["motivo_respaldo"] == "sin_sesion", "sigue la casilla"


# --- CLI pool_keys.py ---------------------------------------------------------------------------
def test_cli_agregar_listar_activar_desactivar_y_estado(tmp_path, capsys):
    bd, env = str(tmp_path / "bd.db"), tmp_path / ".env"
    base = ["--bd", bd, "--env", str(env)]
    leer = lambda _texto: CLAVE_GROQ  # noqa: E731  (getpass simulado)
    assert pool_keys.main(base + ["agregar", "groq", "--alias", "grupo"], leer) == 0
    assert pool_keys.main(base + ["agregar", "groq"], leer) == 1, "repetida"
    assert pool_keys.main(base + ["listar"]) == 0
    assert pool_keys.main(base + ["desactivar", "1"]) == 0
    assert pool_keys.main(base + ["estado"]) == 0
    assert pool_keys.main(base + ["activar", "9"]) == 1
    salida = capsys.readouterr()
    texto = salida.out + salida.err
    assert CLAVE_GROQ not in texto and "****0003" in texto and "desactivada" in texto
    assert "POOL_MASTER_KEY=" in env.read_text(encoding="utf-8")
    valor = env.read_text(encoding="utf-8").split("POOL_MASTER_KEY=")[1].strip()
    assert valor not in texto, "la clave maestra tampoco se muestra"


def test_cli_estado_muestra_la_espera(tmp_path, capsys):
    bd, env = str(tmp_path / "bd.db"), str(tmp_path / ".env")
    pool_keys.main(["--bd", bd, "--env", env, "agregar", "gemini"], lambda _t: CLAVE_GEMINI)
    conn = pool_claves.conectar(bd)
    with conn:
        pool_claves.registrar_error(conn, 1, "HTTP 429 cuota", cooldown_hasta=time.time() + 600)
    conn.close()
    capsys.readouterr()
    pool_keys.main(["--bd", bd, "--env", env, "estado"])
    salida = capsys.readouterr().out
    assert "en espera" in salida and "10 min" in salida and "HTTP 429 cuota" in salida


# --- D-09: Groq sin razonamiento, salida acotada y 15 s por llamada ---------------------------------
def test_d09_groq_sin_razonamiento_y_con_salida_acotada(pool):
    pool.agregar("groq", CLAVE_GROQ, 1)
    analizar(pool.cliente, headers=entrar(pool.cliente))
    (llamada,) = pool.compatible.llamadas
    cuerpo = llamada["cuerpo"]
    assert (cuerpo["reasoning_effort"], cuerpo["reasoning_format"]) == ("none", "hidden")
    assert cuerpo["max_completion_tokens"] == 900 < 1000, "por debajo del límite de 1000 tokens de salida por minuto"
    assert main.TIMEOUT_POR_PROVEEDOR["groq"] == 15


def test_d09_openrouter_no_recibe_los_parametros_de_groq():
    cuerpo = main.cuerpo_compatible("openrouter", "modelo-x", [])
    assert not {"reasoning_effort", "reasoning_format", "max_completion_tokens"} & set(cuerpo)


def test_d09_groq_que_no_responde_pasa_a_la_siguiente_clave_sin_reintentar(pool, monkeypatch):
    monkeypatch.setitem(main.TIMEOUT_POR_PROVEEDOR, "groq", 0.3)
    q = pool.agregar("groq", CLAVE_GROQ, 1)
    pool.agregar("gemini", CLAVE_GEMINI, 2)
    pool.compatible.guion[:] = [("lento", 5)]
    inicio = time.perf_counter()
    e = analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]
    assert e["fuente"] == "gemini" and time.perf_counter() - inicio < 2
    assert len(pool.compatible.llamadas) == 1 and 55 <= pool.fila(q)["cooldown_restante"] <= 60
    assert "sin respuesta en 0.3 s" in pool.fila(q)["ultimo_error"]


def test_d09_groq_como_ultima_clave_tampoco_se_reintenta(pool, monkeypatch, onnx):
    monkeypatch.setitem(main.TIMEOUT_POR_PROVEEDOR, "groq", 0.3)
    onnx.dano = [0.10, 0.80, 0.10]
    pool.agregar("groq", CLAVE_GROQ, 1)
    pool.compatible.guion[:] = [("lento", 5)] * 3
    inicio = time.perf_counter()
    j = analizar(pool.cliente, headers=entrar(pool.cliente)).json()
    assert j["explicacion"]["motivo_respaldo"] == "error" and j["nivel_dano"] == "Moderado", "resultado del ONNX"
    assert len(pool.compatible.llamadas) == 1 and time.perf_counter() - inicio < 2


def test_d09_429_request_too_large_no_se_reintenta_y_espera_un_minuto(pool):
    q = pool.agregar("groq", CLAVE_GROQ, 1)
    pool.agregar("gemini", CLAVE_GEMINI, 2)
    pool.compatible.guion[:] = [respuesta_compatible(
        "Request too large for model `qwen` on output tokens per minute (OTPM): Limit 1000, Requested 1637.", 429)]
    assert analizar(pool.cliente, headers=entrar(pool.cliente)).json()["explicacion"]["fuente"] == "gemini"
    assert len(pool.compatible.llamadas) == 1 and 55 <= pool.fila(q)["cooldown_restante"] <= 60


def test_d09_clasificar_request_too_large():
    e = main.ErrorProveedorIA(429, "Request too large for model on tokens per minute (TPM)")
    reintentable, motivo, detalle, _ = main._clasificar_falla_ia(e)
    assert (reintentable, motivo) == (False, "cuota") and "límite de tokens por minuto" in detalle
    assert main._clasificar_falla_ia(main.ErrorProveedorIA(429, "Rate limit reached"))[0] is True


def test_d09_gemini_conserva_sus_reintentos_ante_un_timeout(gemini):
    gemini.activar()
    gemini.guion[:] = [TimeoutError(), explicacion_ia()]
    e = asyncio.run(main.obtener_explicacion(
        [{"bytes": jpeg(), "vista": "exterior", "nivel_dano": "Leve", "confianza": 80.0, "tipo_construccion": None,
          "rescatable": "Sí", "rescatable_fuente": "reglas"}],
        {"nivel_dano": "Leve", "tipo_construccion": None, "rescatable": "Sí", "rescatable_fuente": "reglas",
         "recomendacion": "r"}))
    assert e["fuente"] == "gemini" and len(gemini.llamadas) == 2
