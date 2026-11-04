"""Inferencia: preprocesamiento, probabilidades, baja confianza, errores 422/503."""
from types import SimpleNamespace

import numpy as np
import pytest

import main
from conftest import BASURA, analizar, analizar_edificio, contar, entrar, foto_dataset, jpeg, png, truncar

MENSAJE_503 = {"detail": "El modelo de clasificación no está disponible"}


def mensaje_422(nombre, foto=None):
    texto = f"No se pudo leer la imagen {nombre}. Prueba con otra foto (JPG o PNG)."
    return {"detail": f"Foto {foto}: {texto}" if foto else texto}


# --- Preprocesamiento --------------------------------------------------------
@pytest.mark.parametrize("ancho, alto", [(640, 480), (100, 300), (224, 224), (2000, 50)])
def test_preprocesamiento_entrada_224x224_rgb_normalizada(onnx, ancho, alto):
    main.sesion_onnx, main.entrada_onnx_nombre, main.cabezas_no_entrenadas = onnx, "entrada", []
    main.analizar_imagen(jpeg(ancho, alto, color=(255, 128, 0)))
    entrada = onnx.entradas[-1]
    assert entrada.shape == (1, 224, 224, 3) and entrada.dtype == np.float32
    assert 0.0 <= entrada.min() and entrada.max() <= 1.0
    r, g, b = entrada[0, 112, 112]
    assert r > 0.95 and 0.45 < g < 0.56 and b < 0.05, "canales en orden RGB y divididos entre 255"


@pytest.mark.parametrize("modo", ["RGB", "RGBA", "L", "P"])
def test_png_en_cualquier_modo_se_convierte_a_rgb(onnx, modo):
    main.sesion_onnx, main.entrada_onnx_nombre, main.cabezas_no_entrenadas = onnx, "entrada", []
    main.analizar_imagen(png(modo))
    assert onnx.entradas[-1].shape == (1, 224, 224, 3)


# --- Probabilidades y baja confianza ----------------------------------------
@pytest.mark.parametrize("salida", [[0.05, 0.37, 0.58], [0.333, 0.333, 0.334], [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])
def test_probabilidades_suman_1_tres_decimales_y_nivel_es_el_maximo(onnx, salida):
    main.sesion_onnx, main.entrada_onnx_nombre, main.cabezas_no_entrenadas = onnx, "entrada", []
    onnx.dano = salida
    r = main.analizar_imagen(jpeg())
    p = r["probabilidades"]
    assert list(p) == main.CLASES_DANO
    assert all(round(v, 3) == v for v in p.values())
    assert abs(sum(p.values()) - 1) <= 0.003
    assert r["nivel_dano"] == max(p, key=p.get) == main.CLASES_DANO[int(np.argmax(salida))]
    assert r["confianza"] == round(max(salida) * 100, 1)


@pytest.mark.parametrize("salida, baja", [
    ([0.05, 0.37, 0.58], True),
    ([0.20, 0.20, 0.60], False),       # 0,60 exacto no es "menor a 0,60"
    ([0.20, 0.2004, 0.5996], False),   # se muestra 0.600 -> coherente con lo visible
    ([0.20, 0.2006, 0.5994], True),    # se muestra 0.599
    ([0.10, 0.20, 0.70], False),
])
def test_baja_confianza_en_el_borde_060(onnx, salida, baja):
    main.sesion_onnx, main.entrada_onnx_nombre, main.cabezas_no_entrenadas = onnx, "entrada", []
    onnx.dano = salida
    assert main.analizar_imagen(jpeg())["baja_confianza"] is baja


def test_baja_confianza_null_sin_probabilidades():
    assert main.es_baja_confianza(None) is None


def test_cabezas_sin_entrenar_se_devuelven_en_null(onnx):
    main.sesion_onnx, main.entrada_onnx_nombre = onnx, "entrada"
    main.cabezas_no_entrenadas = ["tipo_construccion", "rescatable"]
    r = main.analizar_imagen(jpeg())
    assert r["tipo_construccion"] is None and r["rescatable"] is None and r["nivel_dano"] == "Severo/Colapso"


def test_misma_foto_mismo_resultado_sin_azar(app):
    resultados = {analizar(app).json()["confianza"] for _ in range(3)}
    assert len(resultados) == 1
    assert not hasattr(main, "random"), "ya no hay modo simulado"


# --- Errores 422 / 400 -------------------------------------------------------
@pytest.mark.parametrize("nombre, datos, tipo", [
    ("basura.jpg", BASURA, "image/jpeg"),
    ("truncada.jpg", truncar(jpeg(800, 600)), "image/jpeg"),
    ("truncada.png", truncar(png(ancho=800, alto=600), 0.5), "image/png"),
    ("vacia.jpg", b"", "image/jpeg"),
    ("dibujo.svg", b'<svg xmlns="http://www.w3.org/2000/svg"/>', "image/svg+xml"),
], ids=["bytes_basura", "jpeg_truncado", "png_truncado", "archivo_vacio", "svg"])
def test_imagen_danada_responde_422(app, nombre, datos, tipo):
    r = analizar(app, datos, nombre, tipo)
    assert r.status_code == 422 and r.json() == mensaje_422(nombre)


def test_falla_de_inferencia_onnx_responde_422_sin_inventar(app, onnx, caplog):
    onnx.falla = RuntimeError("[ONNXRuntimeError] simulado")
    r = analizar(app, nombre="ok.jpg")
    assert r.status_code == 422 and r.json() == mensaje_422("ok.jpg")
    assert "inferencia ONNX" in caplog.text


def test_edificio_indica_la_foto_que_fallo_y_no_guarda_nada(app, gemini):
    gemini.activar()
    h = entrar(app)
    r = analizar_edificio(app, [(jpeg(), "exterior", "a.jpg"), (BASURA, "exterior", "mala.jpg"), (png(), "interior", "c.png")], h)
    assert r.status_code == 422 and r.json() == mensaje_422("mala.jpg", foto=2)
    assert contar("inspecciones") == 0 and contar("analisis") == 0
    assert gemini.llamadas == [], "no se procesa ninguna foto ni se llama a Gemini"


def test_archivo_que_no_es_imagen_responde_400(app):
    r = analizar(app, b"hola", "notas.txt", "text/plain")
    assert r.status_code == 400


def test_edificio_valida_vistas_y_cantidad(app):
    assert analizar_edificio(app, [(jpeg(), "techo")]).status_code == 400
    assert analizar_edificio(app, [(jpeg(), "exterior")] * (main.MAXIMO_FOTOS_POR_EDIFICIO + 1)).status_code == 400
    r = app.post("/analizar-edificio", files=[("archivos", ("a.jpg", jpeg(), "image/jpeg"))] * 2, data={"vistas": ["exterior"]})
    assert r.status_code == 400


# --- Sin modelo: 503 -----------------------------------------------------------
def test_sin_modelo_los_dos_endpoints_responden_503(app_sin_modelo):
    assert analizar(app_sin_modelo).json() == MENSAJE_503
    r = analizar_edificio(app_sin_modelo, [(jpeg(), "exterior")])
    assert r.status_code == 503 and r.json() == MENSAJE_503
    estado = app_sin_modelo.get("/estado").json()
    assert estado["modelo_disponible"] is False and estado["mensaje"] == MENSAJE_503["detail"]


def test_sin_modelo_no_gasta_el_limite_por_ip(app_sin_modelo):
    for _ in range(main.LIMITE_ANALISIS_SIN_SESION_POR_HORA + 2):
        assert analizar(app_sin_modelo).status_code == 503
    assert not any(main.limite_analisis_sin_sesion._usos.values())


def test_modelo_corrupto_no_carga(monkeypatch, entorno):
    corrupto = entorno.tmp / "corrupto.onnx"
    corrupto.write_bytes(BASURA)
    monkeypatch.setattr(main, "RUTA_MODELO", str(corrupto))
    assert main.cargar_sesion_onnx() is None


# --- Metadatos y salidas del modelo -------------------------------------------
@pytest.mark.parametrize("metadato, no_entrenadas", [
    (None, []),
    ("nivel_dano", ["tipo_construccion", "rescatable"]),
    ("nivel_dano,rescatable", ["tipo_construccion"]),
    ("nivel_dano, tipo_construccion ,rescatable", []),
    ("nivel_dano,cabeza_rara", ["tipo_construccion", "rescatable"]),
])
def test_leer_cabezas_no_entrenadas(metadato, no_entrenadas):
    mapa = {} if metadato is None else {"cabezas_entrenadas": metadato}
    sesion = SimpleNamespace(get_modelmeta=lambda: SimpleNamespace(custom_metadata_map=mapa))
    assert main.leer_cabezas_no_entrenadas(sesion) == no_entrenadas


def test_validar_orden_salidas():
    def sesion(*nombres):
        return SimpleNamespace(get_outputs=lambda: [SimpleNamespace(name=n) for n in nombres])

    main.validar_orden_salidas(sesion("salida_construccion", "salida_dano", "salida_rescatable"))
    with pytest.raises(ValueError):
        main.validar_orden_salidas(sesion("salida_dano", "salida_construccion", "salida_rescatable"))
    with pytest.raises(ValueError):
        main.validar_orden_salidas(sesion("salida_construccion", "salida_dano"))


# --- Modelo real (opcional) -----------------------------------------------------
@pytest.mark.modelo
def test_modelo_real_clasifica_una_foto_del_dataset(app_modelo_real):
    datos = foto_dataset("otros_desastres_pisco2007", "pisco2007_04.jpg")
    j = analizar(app_modelo_real, datos, "pisco2007_04.jpg").json()
    p = j["probabilidades"]
    assert j["usando_modelo_real"] is True and j["nivel_dano"] == max(p, key=p.get)
    assert abs(sum(p.values()) - 1) <= 0.003 and j["baja_confianza"] is (max(p.values()) < 0.60)


@pytest.mark.modelo
def test_modelo_real_es_determinista_y_lee_png(app_modelo_real):
    import io
    from PIL import Image
    datos = foto_dataset("otros_desastres_arequipa1868", "arequipa1868_01.jpg")
    salida = io.BytesIO()
    Image.open(io.BytesIO(datos)).save(salida, "PNG")
    jpg_1 = analizar(app_modelo_real, datos).json()
    jpg_2 = analizar(app_modelo_real, datos).json()
    png_r = analizar(app_modelo_real, salida.getvalue(), "a.png", "image/png").json()
    assert jpg_1["probabilidades"] == jpg_2["probabilidades"]
    assert png_r["nivel_dano"] == jpg_1["nivel_dano"]
