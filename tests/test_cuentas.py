"""Cuentas y sesiones: registro, login, bloqueo, vencimiento, logout y eliminación."""
import os
import sqlite3
import time

import pytest

import main
from conftest import CONTRASENA, analizar, analizar_edificio, contar, entrar, jpeg


def registrar(cliente, usuario="usuario_a", contrasena=CONTRASENA, **extra):
    datos = {"usuario": usuario, "contrasena": contrasena, "acepta_privacidad": True, **extra}
    return cliente.post("/auth/registro", json=datos)


def login(cliente, usuario="usuario_a", contrasena=CONTRASENA):
    return cliente.post("/auth/login", json={"usuario": usuario, "contrasena": contrasena})


def ejecutar(sql, *args):
    con = sqlite3.connect(main.RUTA_BD)
    with con:
        con.execute(sql, args)
    con.close()


# --- Registro ---------------------------------------------------------------------
def test_registro_correcto_guarda_la_fecha_de_aceptacion(app):
    r = registrar(app)
    assert r.status_code == 201 and r.json()["usuario"] == "usuario_a" and r.json()["privacidad_aceptada"]


def test_registro_con_usuario_repetido_sin_distinguir_mayusculas(app):
    registrar(app)
    assert registrar(app, "USUARIO_A").status_code == 409


@pytest.mark.parametrize("contrasena", ["corta", "x" * 7, "x" * 129])
def test_registro_rechaza_contrasena_fuera_de_rango(app, contrasena):
    assert registrar(app, contrasena=contrasena).status_code == 400


@pytest.mark.parametrize("usuario", ["ab", "con espacio", "muñoz", "x" * 31, "a/b", ""])
def test_registro_rechaza_usuario_invalido(app, usuario):
    assert registrar(app, usuario).status_code == 400


def test_registro_exige_aceptar_privacidad(app):
    assert app.post("/auth/registro", json={"usuario": "ana", "contrasena": CONTRASENA}).status_code == 400
    assert registrar(app, "ana", acepta_privacidad=False).status_code == 400
    assert registrar(app, "ana", acepta_privacidad="true").status_code == 422, "debe ser un booleano real"


# --- Login --------------------------------------------------------------------------
def test_login_devuelve_token_expiracion_y_usuario(app):
    registrar(app)
    j = login(app).json()
    assert set(j) == {"token", "expira", "usuario"} and j["usuario"] == {"usuario": "usuario_a"}
    assert app.get("/auth/yo", headers={"Authorization": f"Bearer {j['token']}"}).json()["usuario"] == "usuario_a"


def test_login_fallido_no_revela_si_el_usuario_existe(app):
    registrar(app)
    mala = login(app, contrasena="incorrecta-123")
    inexistente = login(app, "no_existe")
    assert mala.status_code == inexistente.status_code == 401 and mala.json() == inexistente.json()


def test_bloqueo_tras_5_intentos_fallidos(app):
    registrar(app)
    registrar(app, "usuario_b")
    codigos = [login(app, contrasena=f"mala-{i}-xxxx").status_code for i in range(main.MAXIMO_INTENTOS_FALLIDOS)]
    assert codigos == [401] * 4 + [429]
    bloqueado = login(app)
    assert bloqueado.status_code == 429 and "Retry-After" in bloqueado.headers, "bloqueada aun con la contraseña correcta"
    assert login(app, "usuario_b").status_code == 200, "el bloqueo es solo de esa cuenta"
    ejecutar("UPDATE usuarios SET bloqueado_hasta = ? WHERE usuario = 'usuario_a'", time.time() - 1)
    assert login(app).status_code == 200, "pasados los 5 minutos vuelve a entrar"


def test_login_correcto_reinicia_los_intentos(app):
    registrar(app)
    for _ in range(3):
        login(app, contrasena="mala-contrasena")
    assert login(app).status_code == 200
    assert [login(app, contrasena="mala-contrasena").status_code for _ in range(4)] == [401] * 4


# --- Sesiones -------------------------------------------------------------------------
def test_token_vencido_da_401_y_se_borra(app):
    h = entrar(app)
    token = h["Authorization"].split()[1]
    ejecutar("UPDATE sesiones SET expira = ? WHERE token_hash = ?", time.time() - 1, main._hash_token(token))
    r = app.get("/auth/yo", headers=h)
    assert r.status_code == 401 and "venció" in r.json()["detail"]
    assert contar("sesiones") == 0


@pytest.mark.parametrize("encabezado", ["Bearer token-inventado", "Bearer "])
def test_token_invalido_da_401(app, encabezado):
    assert app.get("/auth/yo", headers={"Authorization": encabezado}).status_code == 401


def test_token_invalido_en_analizar_da_401_y_no_se_trata_como_anonimo(app):
    assert analizar(app, headers={"Authorization": "Bearer inventado"}).status_code == 401


def test_logout_invalida_el_token(app):
    h = entrar(app)
    assert app.post("/auth/logout", headers=h).status_code == 200
    assert app.get("/auth/yo", headers=h).status_code == 401


def test_sin_sesion_auth_yo_y_logout_dan_401_con_www_authenticate(app):
    for r in (app.get("/auth/yo"), app.post("/auth/logout")):
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


# --- Eliminar cuenta --------------------------------------------------------------------
def test_eliminar_cuenta_con_contrasena_incorrecta_no_borra(app):
    h = entrar(app)
    r = app.request("DELETE", "/auth/cuenta", headers=h, json={"contrasena": "incorrecta-123"})
    assert r.status_code == 403 and contar("usuarios") == 1


def test_eliminar_cuenta_borra_sus_datos_y_miniaturas_pero_no_las_de_otros(app, entorno):
    ha, hb = entrar(app, "usuario_a"), entrar(app, "usuario_b")
    analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(), "interior")], ha)
    analizar(app, headers=ha)
    id_b = analizar(app, headers=hb).json()["inspeccion_id"]
    miniaturas_b = {n for (n,) in sqlite3.connect(main.RUTA_BD).execute(
        "SELECT miniatura FROM analisis WHERE inspeccion_id = ?", (id_b,))}
    assert len(os.listdir(entorno.minis)) == 4
    r = app.request("DELETE", "/auth/cuenta", headers=ha, json={"contrasena": CONTRASENA})
    assert r.status_code == 200 and r.json()["inspecciones_eliminadas"] == 2
    assert set(os.listdir(entorno.minis)) == miniaturas_b, "solo quedan las miniaturas de B"
    assert (contar("usuarios"), contar("inspecciones"), contar("analisis")) == (1, 1, 1)
    assert app.get("/auth/yo", headers=ha).status_code == 401, "su token deja de servir"
    assert app.get("/historial", headers=hb).json()[0]["inspeccion_id"] == id_b


# --- Almacenamiento seguro ---------------------------------------------------------------
def test_contrasenas_y_tokens_nunca_en_texto_plano(app, caplog):
    h = entrar(app, "usuario_a", "contrasena-muy-secreta-1")
    token = h["Authorization"].split()[1]
    contenido = open(main.RUTA_BD, "rb").read()
    assert b"contrasena-muy-secreta-1" not in contenido and token.encode() not in contenido
    assert sqlite3.connect(main.RUTA_BD).execute("SELECT hash_contrasena FROM usuarios").fetchone()[0].startswith("scrypt$")
    assert "contrasena-muy-secreta-1" not in caplog.text and token not in caplog.text


def test_hash_de_contrasena_con_sal_aleatoria():
    a, b = main.hashear_contrasena("misma-clave"), main.hashear_contrasena("misma-clave")
    assert a != b and main.verificar_contrasena("misma-clave", a) and not main.verificar_contrasena("otra", a)
