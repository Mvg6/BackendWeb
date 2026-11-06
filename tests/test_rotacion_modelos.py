"""Rotación por modelo en las claves Gemini (GEMINI_MODELOS_ORDEN), siempre con Gemini
simulado: orden, 429 por modelo (no aparta la clave), cupo diario, 503, todos apartados,
clave inválida, modelos sin instrucción de sistema (Gemma) y el modelo real en la ficha."""
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main
import pool_claves
import pool_keys
from conftest import analizar, entrar, error_api, explicacion_ia, jpeg

ORDEN = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemma-4-26b-a4b-it"]
CLAVE = "AIzaClaveRotacionModelos01"
CLAVE_GROQ = "gsk_ClaveRotacionModelos02"
CUPO_DIARIO = "Quota exceeded, quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def foto(i):
    return jpeg(color=(15 + 9 * i, 80, 70))


class GeminiPorModelo:
    """Imita client.aio.models.generate_content con un guion por modelo: una excepción,
    "mal_esquema" o "sin_sistema" (400 si se envía system_instruction)."""

    def __init__(self):
        self.guiones = {}
        self.sin_sistema = set()
        self.llamadas = []  # (modelo, con instrucción de sistema)

    async def generate_content(self, model, contents, config):
        con_sistema = config.system_instruction is not None
        self.llamadas.append((model, con_sistema))
        if model in self.sin_sistema and con_sistema:
            raise error_api(400, mensaje=f"Developer instruction is not enabled for models/{model}")
        guion = self.guiones.get(model, [])
        paso = guion.pop(0) if guion else None
        if isinstance(paso, BaseException):
            raise paso
        if paso == "mal_esquema":
            return SimpleNamespace(parsed=None, text="no es json")
        respuesta = explicacion_ia()
        return SimpleNamespace(parsed=respuesta if con_sistema else None, text=respuesta.model_dump_json())

    def modelos(self):
        return [m for m, _ in self.llamadas]


@pytest.fixture
def rotacion(app, monkeypatch):
    """App con una clave Gemini en el pool y rotación por modelo en ORDEN."""
    simulado = GeminiPorModelo()
    main.cliente_gemini = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=simulado.generate_content)))
    main.modelo_gemini, main.modelos_alternativos_gemini = ORDEN[0], ORDEN[1:]
    main.modelos_gemini_orden = list(ORDEN)
    main.motivo_sin_ia, main.reconfigurar_ia_pendiente = None, False
    conn = pool_claves.conectar(main.RUTA_BD)
    with conn:
        id_ = pool_claves.agregar(conn, main._fernet_pool, "gemini", CLAVE, prioridad=1)
    conn.close()
    main._id_clave_gemini = id_
    return SimpleNamespace(cliente=app, gemini=simulado, id=id_)


def fila(id_):
    conn = pool_claves.conectar(main.RUTA_BD)
    try:
        return next(c for c in pool_claves.todas(conn, main._fernet_pool) if c["id"] == id_)
    finally:
        conn.close()


# --- Orden -----------------------------------------------------------------------------
def test_se_usa_el_primer_modelo_del_orden(rotacion):
    e = analizar(rotacion.cliente, headers=entrar(rotacion.cliente)).json()["explicacion"]
    assert (e["fuente"], e["modelo"]) == ("gemini", ORDEN[0]) and rotacion.gemini.modelos() == [ORDEN[0]]


def test_429_aparta_solo_ese_modelo_y_responde_el_siguiente(rotacion):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(429, retry_delay="40s")]
    h = entrar(rotacion.cliente)
    e = analizar(rotacion.cliente, foto(1), headers=h).json()["explicacion"]
    assert e["modelo"] == ORDEN[1] and rotacion.gemini.modelos() == ORDEN[:2]
    estado = fila(rotacion.id)
    assert estado["activa"] and estado["cooldown_restante"] == 0, "la clave no se aparta"
    assert list(estado["modelos_en_espera"]) == [ORDEN[0]] and 35 <= estado["modelos_en_espera"][ORDEN[0]] <= 40
    analizar(rotacion.cliente, foto(2), headers=h)
    assert rotacion.gemini.modelos() == [ORDEN[0], ORDEN[1], ORDEN[1]], "el modelo apartado no se vuelve a probar"


def test_429_por_cupo_diario_aparta_el_modelo_hasta_la_medianoche_del_pacifico(rotacion):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(429, mensaje=CUPO_DIARIO)]
    antes = time.time()
    assert analizar(rotacion.cliente, headers=entrar(rotacion.cliente)).json()["explicacion"]["modelo"] == ORDEN[1]
    conn = pool_claves.conectar(main.RUTA_BD)
    (texto,) = conn.execute("SELECT cooldown_modelos FROM api_keys WHERE id = ?", (rotacion.id,)).fetchone()
    conn.close()
    assert abs(json.loads(texto)[ORDEN[0]] - pool_claves.proxima_medianoche_pacifico(antes)) < 5


@pytest.mark.parametrize("falla", [error_api(503), error_api(500), TimeoutError()])
def test_503_500_o_timeout_apartan_el_modelo_60_s(rotacion, falla):
    rotacion.gemini.guiones[ORDEN[0]] = [falla]
    assert analizar(rotacion.cliente, headers=entrar(rotacion.cliente)).json()["explicacion"]["modelo"] == ORDEN[1]
    assert 55 <= fila(rotacion.id)["modelos_en_espera"][ORDEN[0]] <= 60


def test_respuesta_fuera_de_esquema_pasa_al_siguiente_sin_apartar(rotacion):
    rotacion.gemini.guiones[ORDEN[0]] = ["mal_esquema"]
    assert analizar(rotacion.cliente, headers=entrar(rotacion.cliente)).json()["explicacion"]["modelo"] == ORDEN[1]
    assert fila(rotacion.id)["modelos_en_espera"] == {}


def test_modelo_retirado_404_se_aparta_hasta_la_medianoche(rotacion):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(404)]
    analizar(rotacion.cliente, headers=entrar(rotacion.cliente))
    assert abs(time.time() + fila(rotacion.id)["modelos_en_espera"][ORDEN[0]]
               - pool_claves.proxima_medianoche_pacifico()) < 5


def test_con_todos_los_modelos_apartados_pasa_a_la_siguiente_clave(rotacion, monkeypatch):
    import httpx

    async def enviar(url, clave, cuerpo, timeout):
        return httpx.Response(200, json={"choices": [{"message": {"content": explicacion_ia().model_dump_json()}}]})

    monkeypatch.setattr(main, "_enviar_compatible", enviar)
    main.config_ia["GROQ_MODEL"] = "groq-simulado"
    conn = pool_claves.conectar(main.RUTA_BD)
    with conn:
        pool_claves.agregar(conn, main._fernet_pool, "groq", CLAVE_GROQ, prioridad=2)
    conn.close()
    for modelo in ORDEN:
        rotacion.gemini.guiones[modelo] = [error_api(429, retry_delay="300s")]
    h = entrar(rotacion.cliente)
    assert analizar(rotacion.cliente, foto(1), headers=h).json()["explicacion"]["fuente"] == "groq"
    assert rotacion.gemini.modelos() == ORDEN
    assert analizar(rotacion.cliente, foto(2), headers=h).json()["explicacion"]["fuente"] == "groq"
    assert rotacion.gemini.modelos() == ORDEN, "la clave Gemini sin modelos libres ni se intenta"
    assert rotacion.cliente.get("/estado").json()["explicacion_ia"]["modelo"] == "groq-simulado"


def test_todos_apartados_y_sin_otra_clave_queda_el_resultado_del_onnx(rotacion):
    for modelo in ORDEN:
        rotacion.gemini.guiones[modelo] = [error_api(429, retry_delay="300s")]
    j = analizar(rotacion.cliente, headers=entrar(rotacion.cliente)).json()
    assert j["explicacion"]["motivo_respaldo"] == "cuota" and j["nivel_dano"] == "Severo/Colapso"
    assert rotacion.cliente.get("/estado").json()["explicacion_ia"]["lista"] is False


@pytest.mark.parametrize("codigo", [401, 403])
def test_clave_invalida_desactiva_la_clave_entera(rotacion, codigo):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(codigo)]
    e = analizar(rotacion.cliente, headers=entrar(rotacion.cliente)).json()["explicacion"]
    assert e["fuente"] == "plantilla" and rotacion.gemini.modelos() == [ORDEN[0]], "no se prueban los otros modelos"
    assert fila(rotacion.id)["activa"] is False


def test_al_activar_la_clave_se_liberan_sus_modelos(rotacion):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(429, retry_delay="300s")]
    analizar(rotacion.cliente, headers=entrar(rotacion.cliente))
    conn = pool_claves.conectar(main.RUTA_BD)
    with conn:
        pool_claves.activar(conn, rotacion.id)
    conn.close()
    assert fila(rotacion.id)["modelos_en_espera"] == {}


def test_estado_indica_el_siguiente_modelo_disponible(rotacion):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(429, retry_delay="300s")]
    analizar(rotacion.cliente, headers=entrar(rotacion.cliente))
    assert rotacion.cliente.get("/estado").json()["explicacion_ia"]["modelo"] == ORDEN[1]


# --- Modelos sin instrucción de sistema (Gemma) -----------------------------------------------
def test_modelo_que_no_acepta_instruccion_de_sistema_recibe_el_prompt_en_el_mensaje(rotacion, caplog):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(429, retry_delay="300s")]
    rotacion.gemini.guiones[ORDEN[1]] = [error_api(429, retry_delay="300s")]
    rotacion.gemini.sin_sistema.add(ORDEN[2])
    h = entrar(rotacion.cliente)
    e = analizar(rotacion.cliente, foto(1), headers=h).json()["explicacion"]
    assert (e["fuente"], e["modelo"]) == ("gemini", "gemma-4-26b-a4b-it")
    assert rotacion.gemini.llamadas[-2:] == [(ORDEN[2], True), (ORDEN[2], False)]
    assert ORDEN[2] in main._modelos_sin_sistema and "no acepta instrucción de sistema" in caplog.text
    analizar(rotacion.cliente, foto(2), headers=h)
    assert rotacion.gemini.llamadas[-1] == (ORDEN[2], False), "se recuerda: va directo sin instrucción de sistema"


def test_sin_sistema_el_texto_se_valida_igual_y_si_no_es_json_falla(rotacion):
    main._modelos_sin_sistema = {ORDEN[0], ORDEN[1], ORDEN[2]}
    for modelo in ORDEN:
        rotacion.gemini.guiones[modelo] = ["mal_esquema"]
    e = analizar(rotacion.cliente, headers=entrar(rotacion.cliente)).json()["explicacion"]
    assert e["fuente"] == "plantilla" and len(rotacion.gemini.llamadas) == 3


def test_el_prompt_en_el_mensaje_es_el_unificado(rotacion):
    main._modelos_sin_sistema = set(ORDEN)
    capturado = {}

    async def generar(model, contents, config):
        capturado["texto"] = contents[0].text
        capturado["config"] = config
        return SimpleNamespace(parsed=None, text=explicacion_ia().model_dump_json())

    main.cliente_gemini.aio.models.generate_content = generar
    analizar(rotacion.cliente, headers=entrar(rotacion.cliente))
    assert capturado["texto"].startswith(main.INSTRUCCIONES_IA) and "esquema JSON" in capturado["texto"]
    assert capturado["config"].temperature == 0 and capturado["config"].system_instruction is None


# --- La ficha muestra el modelo real ----------------------------------------------------------
def test_la_ficha_muestra_el_modelo_real_usado(rotacion):
    rotacion.gemini.guiones[ORDEN[0]] = [error_api(503)]
    rotacion.gemini.guiones[ORDEN[1]] = [error_api(503)]
    h = entrar(rotacion.cliente)
    e = analizar(rotacion.cliente, headers=h).json()["explicacion"]
    assert e["modelo"] == "gemma-4-26b-a4b-it"
    (guardada,) = rotacion.cliente.get("/historial", headers=h).json()
    assert (guardada["ia_proveedor"], guardada["ia_modelo"]) == ("gemini", "gemma-4-26b-a4b-it")
    index = (Path(main.BASE_DIR).parent / "frontend" / "index.html").read_text(encoding="utf-8")
    assert re.search(r"Modelo usado: \$\{texto\(x\.modelo\)\}", index), "la ficha muestra x.modelo tal cual"


# --- Configuración ---------------------------------------------------------------------
class ClienteGoogle:
    disponibles = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                   "gemma-4-26b-a4b-it", "gemma-4-31b-it")
    sondeados = []

    def __init__(self, **_):
        self.models = SimpleNamespace(list=self._listar, generate_content=self._sondear)
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=None))

    def _listar(self):
        return [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"]) for n in self.disponibles]

    def _sondear(self, model, contents, config):
        ClienteGoogle.sondeados.append(model)


@pytest.fixture
def google(monkeypatch, entorno):
    ClienteGoogle.sondeados = []
    monkeypatch.setattr(main, "genai", SimpleNamespace(Client=ClienteGoogle))
    return entorno


def test_configuracion_con_orden_y_exclusion(google, caplog):
    google.env.write_text("GEMINI_API_KEY=AIzaClaveDePrueba\nEXPLICACION_IA=on\nGEMINI_MODELOS_ORDEN="
                          + ",".join(ORDEN + ["gemini-3.8-flash", "gemma-9-inexistente"]) + "\n", encoding="utf-8")
    main.configurar_gemini()
    assert main.modelos_gemini_orden == ORDEN and main.modelo_gemini == ORDEN[0]
    assert ClienteGoogle.sondeados == [ORDEN[0]], "solo se sondea hasta el primero que responde"
    assert "gemini-3.8-flash está en GEMINI_MODELOS_EXCLUIDOS" in caplog.text
    assert "gemma-9-inexistente no está disponible" in caplog.text


def test_sin_orden_queda_el_comportamiento_de_siempre(google):
    google.env.write_text("GEMINI_API_KEY=AIzaClaveDePrueba\nEXPLICACION_IA=on\n", encoding="utf-8")
    main.configurar_gemini()
    assert main.modelos_gemini_orden == [] and main.modelo_gemini == "gemini-3.5-flash"


def test_modelos_sin_sistema_configurables(google):
    google.env.write_text("GEMINI_API_KEY=AIzaClaveDePrueba\nEXPLICACION_IA=on\nGEMINI_MODELOS_ORDEN=gemma-4-31b-it\n"
                          "GEMINI_MODELOS_SIN_SISTEMA=gemma-4-31b-it\n", encoding="utf-8")
    main.configurar_gemini()
    assert main.modelos_gemini_orden == ["gemma-4-31b-it"] and main._modelos_sin_sistema == {"gemma-4-31b-it"}


def test_el_orden_lo_usa_la_app_al_arrancar(google, onnx):
    google.env.write_text(f"GEMINI_API_KEY={CLAVE}\nEXPLICACION_IA=on\nGEMINI_MODELOS_ORDEN={','.join(ORDEN)}\n",
                          encoding="utf-8")
    with TestClient(main.app) as cliente:
        assert main.modelos_gemini_orden == ORDEN
        assert cliente.get("/estado").json()["explicacion_ia"]["modelo"] == ORDEN[0]


# --- CLI ------------------------------------------------------------------------------------
def test_cli_estado_muestra_los_modelos_en_espera(tmp_path, capsys):
    bd, env = str(tmp_path / "bd.db"), str(tmp_path / ".env")
    pool_keys.main(["--bd", bd, "--env", env, "agregar", "gemini"], lambda _t: CLAVE)
    conn = pool_claves.conectar(bd)
    with conn:
        pool_claves.apartar_modelo(conn, 1, "gemini-3.1-flash-lite", time.time() + 3 * 3600, "HTTP 429 cupo diario")
    conn.close()
    capsys.readouterr()
    pool_keys.main(["--bd", bd, "--env", env, "estado"])
    salida = capsys.readouterr().out
    assert "disponible" in salida and "modelo gemini-3.1-flash-lite en espera 3.0 h" in salida
    assert "gemini-3.1-flash-lite: HTTP 429 cupo diario" in salida and CLAVE not in salida


def test_migracion_agrega_la_columna_sin_perder_claves(tmp_path):
    import sqlite3
    from cryptography.fernet import Fernet
    bd = str(tmp_path / "vieja.db")
    conn = sqlite3.connect(bd)
    conn.execute("CREATE TABLE api_keys (id INTEGER PRIMARY KEY AUTOINCREMENT, proveedor TEXT NOT NULL, modelo TEXT, "
                 "clave_cifrada TEXT NOT NULL, alias TEXT, activa INTEGER NOT NULL DEFAULT 1, prioridad INTEGER NOT NULL "
                 "DEFAULT 10, cooldown_hasta REAL NOT NULL DEFAULT 0, usos_hoy INTEGER NOT NULL DEFAULT 0, fecha_usos TEXT, "
                 "errores_seguidos INTEGER NOT NULL DEFAULT 0, ultimo_error TEXT, ultimo_uso REAL)")
    fernet = Fernet(Fernet.generate_key())
    conn.execute("INSERT INTO api_keys (proveedor, clave_cifrada) VALUES ('gemini', ?)", (pool_claves.cifrar(fernet, CLAVE),))
    conn.commit()
    conn.close()
    conn = pool_claves.conectar(bd)
    (clave,) = pool_claves.disponibles(conn, fernet)
    conn.close()
    assert clave.clave == CLAVE and clave.esperas_modelo == {}
