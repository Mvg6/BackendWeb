"""Motor de reglas: rescatable estimado, recomendaciones y veredicto consolidado."""
import pytest

import main

TEXTO_SEVERO_INGENIERO = ("Daño severo o colapso parcial/total. NO ingresar. Evacuar y acordonar la zona de "
                          "inmediato. Un ingeniero colegiado debe determinar si la edificación se puede "
                          "recuperar o debe demolerse.")


def foto(vista="exterior", nivel="Leve", tipo=None, rescatable=None, confianza=80.0, fuente_resc=None):
    if rescatable is None and vista == "exterior":
        rescatable, fuente_resc = main.RESCATABLE_POR_NIVEL[nivel], "reglas"
    return {"vista": vista, "nivel_dano": nivel, "tipo_construccion": tipo, "rescatable": rescatable,
            "rescatable_fuente": fuente_resc, "confianza": confianza}


@pytest.fixture
def cabezas_actuales():
    main.cabezas_no_entrenadas = ["tipo_construccion", "rescatable"]


@pytest.fixture
def cabezas_entrenadas():
    main.cabezas_no_entrenadas = []


# --- Rescatable por reglas ----------------------------------------------------
@pytest.mark.parametrize("nivel, rescatable", [("Leve", "Sí"), ("Moderado", "Requiere inspección adicional"),
                                                ("Severo/Colapso", "No")])
def test_rescatable_por_reglas_para_los_tres_niveles(cabezas_actuales, nivel, rescatable):
    r = main.completar_rescatable({"nivel_dano": nivel, "rescatable": None})
    assert (r["rescatable"], r["rescatable_fuente"]) == (rescatable, "reglas")


def test_con_cabeza_entrenada_se_usa_la_salida_del_modelo(cabezas_entrenadas):
    r = main.completar_rescatable({"nivel_dano": "Severo/Colapso", "rescatable": "Requiere inspección adicional"})
    assert (r["rescatable"], r["rescatable_fuente"]) == ("Requiere inspección adicional", "modelo")


def test_fuente_del_tipo_segun_la_cnn(cabezas_actuales):
    assert main.completar_resultado_cnn({"nivel_dano": "Leve", "rescatable": None})["tipo_construccion_fuente"] is None
    main.cabezas_no_entrenadas = []
    assert main.completar_resultado_cnn({"nivel_dano": "Leve", "rescatable": "Sí"})["tipo_construccion_fuente"] == "modelo"


# --- Recomendaciones ------------------------------------------------------------
@pytest.mark.parametrize("nivel, esperado", [
    ("Leve", "Daño leve. Se recomiendan reparaciones menores y la revisión de un profesional colegiado antes de "
             "confirmar el uso normal de la edificación." + main.AVISO_ADOBE_TAPIAL),
    ("Moderado", "Daño moderado. No habitar hasta contar con una inspección estructural detallada por un profesional colegiado."),
    ("Severo/Colapso", TEXTO_SEVERO_INGENIERO),
])
def test_recomendacion_para_los_tres_niveles_con_rescatable_por_reglas(nivel, esperado):
    assert main.generar_recomendacion(None, nivel, main.RESCATABLE_POR_NIVEL[nivel], "reglas") == esperado


def test_severo_con_no_del_modelo_tambien_deja_la_decision_al_ingeniero():
    assert main.generar_recomendacion(None, "Severo/Colapso", "No", "modelo") == TEXTO_SEVERO_INGENIERO


@pytest.mark.parametrize("tipo", ["Adobe", "Madera"])
def test_leve_con_adobe_o_madera_agrega_su_aviso_y_el_de_adobe_tapial(tipo):
    texto = main.generar_recomendacion(tipo, "Leve", "Sí", "reglas")
    assert f"construcción de {tipo.lower()}" in texto and texto.endswith(main.AVISO_ADOBE_TAPIAL)


TODAS = [(n, t, r, f) for n in main.CLASES_DANO for t in main.CLASES_CONSTRUCCION + [None]
         for r in main.CLASES_RESCATABLE for f in ("modelo", "reglas")]


def test_leve_siempre_lleva_el_aviso_de_adobe_y_tapial():
    for n, t, r, f in TODAS:
        if n == "Leve":
            assert main.AVISO_ADOBE_TAPIAL in main.generar_recomendacion(t, n, r, f)
    assert main.AVISO_ADOBE_TAPIAL in main.recomendacion_sin_exteriores("Leve")
    assert main.AVISO_ADOBE_TAPIAL not in main.recomendacion_sin_exteriores("Moderado")


def test_ningun_texto_afirma_habitable():
    textos = [main.generar_recomendacion(t, n, r, f) for n, t, r, f in TODAS]
    textos += [main.recomendacion_sin_exteriores(n) for n in main.CLASES_DANO]
    for texto in textos:
        sin_frase_permitida = texto.lower().replace("determinar si la edificación es habitable", "")
        assert "habitable" not in sin_frase_permitida, texto
        assert "estructura habitable" not in texto.lower()


def test_ningun_texto_recomienda_demolicion():
    for n, t, r, f in TODAS:
        texto = main.generar_recomendacion(t, n, r, f)
        assert "se recomienda demolición" not in texto
        assert "demol" not in texto.replace("debe demolerse", ""), texto


def test_rescatable_desconocido_se_trata_de_forma_conservadora():
    assert main.rescatable_para_reglas(None) == "Requiere inspección adicional"
    assert main.rescatable_para_reglas("Sí") == "Sí"


# --- Veredicto del edificio ------------------------------------------------------
def test_aplicar_vista_interior_anula_tipo_y_rescatable():
    r = main.aplicar_vista({"nivel_dano": "Leve", "tipo_construccion": "Adobe", "rescatable": "Sí",
                            "rescatable_fuente": "modelo", "tipo_construccion_fuente": "modelo"}, "interior")
    assert (r["tipo_construccion"], r["rescatable"], r["rescatable_fuente"], r["tipo_construccion_fuente"]) == (None,) * 4


def test_peor_caso_manda_aunque_venga_de_una_foto_interior(cabezas_actuales):
    v = main.consolidar_veredicto([foto(nivel="Leve"), foto(nivel="Leve"), foto("interior", "Severo/Colapso")])
    assert v["nivel_dano"] == "Severo/Colapso"
    assert (v["rescatable"], v["rescatable_fuente"]) == ("No", "reglas"), "rescatable con el nivel consolidado"
    assert v["recomendacion"] == TEXTO_SEVERO_INGENIERO
    assert (v["fotos_exteriores"], v["fotos_interiores"], v["nota"]) == (2, 1, None)


def test_solo_interiores(cabezas_actuales):
    v = main.consolidar_veredicto([foto("interior", "Moderado"), foto("interior", "Leve")])
    assert v["tipo_construccion"] is None and v["tipo_construccion_fuente"] is None
    assert (v["rescatable"], v["rescatable_fuente"]) == ("Requiere inspección adicional", "reglas")
    assert v["nota"] == main.NOTA_SIN_EXTERIORES
    assert v["recomendacion"] == main.recomendacion_sin_exteriores("Moderado")
    assert "No se evaluó la fachada" in v["recomendacion"]


def test_solo_interiores_severo_mantiene_evacuar(cabezas_actuales):
    v = main.consolidar_veredicto([foto("interior", "Severo/Colapso")])
    assert v["recomendacion"].startswith("Daño severo detectado. Evacuar preventivamente")


def test_con_cnn_entrenada_rescatable_es_el_mas_restrictivo_y_tipo_el_mas_repetido(cabezas_entrenadas):
    fotos = [foto(nivel="Moderado", tipo="Adobe", rescatable="Sí", fuente_resc="modelo"),
             foto(nivel="Leve", tipo="Adobe", rescatable="No", fuente_resc="modelo", confianza=60),
             foto(nivel="Leve", tipo="Madera", rescatable="Requiere inspección adicional", fuente_resc="modelo", confianza=99)]
    v = main.consolidar_veredicto(fotos)
    assert (v["rescatable"], v["rescatable_fuente"]) == ("No", "modelo")
    assert (v["tipo_construccion"], v["tipo_construccion_fuente"]) == ("Adobe", "modelo")


def test_empate_de_tipo_se_desempata_por_confianza(cabezas_entrenadas):
    fotos = [foto(tipo="Adobe", rescatable="Sí", fuente_resc="modelo", confianza=70),
             foto(tipo="Madera", rescatable="Sí", fuente_resc="modelo", confianza=95)]
    assert main.consolidar_veredicto(fotos)["tipo_construccion"] == "Madera"


def test_recomendacion_de_usa_los_campos_del_registro():
    registro = {"tipo_construccion": None, "nivel_dano": "Moderado", "rescatable": None, "rescatable_fuente": "reglas"}
    assert main.recomendacion_de(registro) == main.generar_recomendacion(None, "Moderado", "Requiere inspección adicional", "reglas")
