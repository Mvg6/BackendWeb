"""Historial privado: sesión obligatoria, aislamiento entre usuarios, orden y campos."""
import pytest

import main
from conftest import analizar, analizar_edificio, contar, entrar, jpeg

NIVEL_A_SALIDA = {"Leve": [0.8, 0.1, 0.1], "Moderado": [0.1, 0.8, 0.1], "Severo/Colapso": [0.1, 0.1, 0.8]}


@pytest.mark.parametrize("ruta", ["/historial", "/historial/exportar-excel", "/historial/exportar-pdf",
                                  "/inspecciones/1/fotos/1"])
def test_sin_sesion_401(app, ruta):
    r = app.get(ruta)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_sin_sesion_el_analisis_no_se_guarda(app):
    j = analizar(app).json()
    assert j["guardado"] is False and j["inspeccion_id"] is None and j["tiene_miniatura"] is False
    j = analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(), "interior")]).json()
    assert j["guardado"] is False and j["inspeccion_id"] is None
    assert contar("inspecciones") == 0


def test_con_sesion_se_guarda_a_nombre_del_usuario(app):
    h = entrar(app)
    j = analizar(app, headers=h).json()
    assert j["guardado"] is True and isinstance(j["inspeccion_id"], int) and j["tiene_miniatura"] is True
    assert [i["inspeccion_id"] for i in app.get("/historial", headers=h).json()] == [j["inspeccion_id"]]


def test_aislamiento_entre_usuarios(app):
    ha, hb = entrar(app, "usuario_a"), entrar(app, "usuario_b")
    id_a = analizar(app, headers=ha).json()["inspeccion_id"]
    id_b = analizar(app, headers=hb).json()["inspeccion_id"]
    assert [i["inspeccion_id"] for i in app.get("/historial", headers=ha).json()] == [id_a]
    assert [i["inspeccion_id"] for i in app.get("/historial", headers=hb).json()] == [id_b]
    assert app.get(f"/inspecciones/{id_a}/fotos/1", headers=hb).status_code == 404, "B no ve la foto de A pidiendo por id"
    assert app.get(f"/inspecciones/{id_b}/fotos/1", headers=ha).status_code == 404
    propia = app.get(f"/inspecciones/{id_a}/fotos/1", headers=ha)
    assert propia.status_code == 200 and propia.headers["content-type"] == "image/jpeg"
    assert "no-store" in propia.headers["cache-control"]


def test_indice_de_foto_inexistente_404(app):
    h = entrar(app)
    i = analizar(app, headers=h).json()["inspeccion_id"]
    for indice in (0, 2, 99):
        assert app.get(f"/inspecciones/{i}/fotos/{indice}", headers=h).status_code == 404


def test_orden_por_prioridad(app, onnx):
    h = entrar(app)
    ids = {}
    for nivel in ("Leve", "Severo/Colapso", "Moderado", "Leve", "Severo/Colapso"):
        onnx.dano = NIVEL_A_SALIDA[nivel]
        ids.setdefault(nivel, []).append(analizar(app, headers=h).json()["inspeccion_id"])
    historial = app.get("/historial", headers=h).json()
    assert [i["veredicto"]["nivel_dano"] for i in historial] == ["Severo/Colapso"] * 2 + ["Moderado"] + ["Leve"] * 2
    assert [i["inspeccion_id"] for i in historial] == ids["Severo/Colapso"] + ids["Moderado"] + ids["Leve"], \
        "dentro del mismo nivel, en orden de creación"


def test_campos_de_cada_foto_en_el_historial(app, onnx):
    h = entrar(app)
    onnx.danos = [[0.05, 0.37, 0.58], [0.1, 0.2, 0.7]]
    j = analizar_edificio(app, [(jpeg(), "exterior"), (jpeg(), "interior")], h).json()
    (inspeccion,) = app.get("/historial", headers=h).json()
    assert inspeccion["inspeccion_id"] == j["inspeccion_id"] and inspeccion["total_fotos"] == 2
    ext, inte = inspeccion["fotos"]
    assert ext["probabilidades"] == {"Leve": 0.05, "Moderado": 0.37, "Severo/Colapso": 0.58} and ext["baja_confianza"] is True
    assert inte["baja_confianza"] is False and inte["vista"] == "interior"
    assert (ext["rescatable"], ext["rescatable_fuente"]) == ("No", "reglas")
    assert (inte["rescatable"], inte["rescatable_fuente"], inte["tipo_construccion_fuente"]) == (None, None, None)
    assert ext["tiene_miniatura"] and inte["tiene_miniatura"]
    assert inspeccion["veredicto"]["rescatable_fuente"] == "reglas"
    assert [f[c] for f in inspeccion["fotos"] for c in ("probabilidades",)] == [f["probabilidades"] for f in j["fotos"]]


def test_inspecciones_anteriores_sin_dueno_no_se_muestran(app):
    import sqlite3
    con = sqlite3.connect(main.RUTA_BD)
    with con:
        con.execute("INSERT INTO inspecciones (fecha, nivel_dano_general) VALUES ('2026-01-01', 'Leve')")
    con.close()
    h = entrar(app)
    assert app.get("/historial", headers=h).json() == []
    assert app.get("/inspecciones/1/fotos/1", headers=h).status_code == 404


def test_estado_publico_sin_clave(app):
    estado = app.get("/estado")
    assert estado.status_code == 200 and estado.json()["modelo_disponible"] is True
    assert set(estado.json()["explicacion_ia"]) == {"lista", "modelo", "motivo", "reintento_automatico"}
