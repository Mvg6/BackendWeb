"""Genera los anexos del informe a partir de una corrida de pytest con cobertura.

Uso (desde backend/):
    python -m pytest --cov=main --cov-report=html:reporte_pruebas/htmlcov \
        --cov-report=json:reporte_pruebas/coverage.json --junitxml=reporte_pruebas/junit.xml
    python tests/generar_reporte.py

Escribe reporte_pruebas/tabla_pruebas.md y reporte_pruebas/cobertura_por_funcion.md.
"""
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

CARPETA = Path(__file__).resolve().parents[1] / "reporte_pruebas"

# Función de main.py que verifica cada prueba (agrupación para la tabla del informe).
FUNCION_PROBADA = {
    # test_inferencia
    "test_preprocesamiento_entrada_224x224_rgb_normalizada": "analizar_imagen (preprocesamiento 224x224)",
    "test_png_en_cualquier_modo_se_convierte_a_rgb": "analizar_imagen (preprocesamiento 224x224)",
    "test_probabilidades_suman_1_tres_decimales_y_nivel_es_el_maximo": "probabilidades_de_dano",
    "test_baja_confianza_en_el_borde_060": "es_baja_confianza",
    "test_baja_confianza_null_sin_probabilidades": "es_baja_confianza",
    "test_cabezas_sin_entrenar_se_devuelven_en_null": "analizar_imagen (cabezas sin entrenar)",
    "test_misma_foto_mismo_resultado_sin_azar": "analizar_imagen (sin modo simulado)",
    "test_imagen_danada_responde_422": "analizar_foto (422 imagen dañada)",
    "test_falla_de_inferencia_onnx_responde_422_sin_inventar": "analizar_foto (422 imagen dañada)",
    "test_edificio_indica_la_foto_que_fallo_y_no_guarda_nada": "analizar_foto (422 imagen dañada)",
    "test_archivo_que_no_es_imagen_responde_400": "analizar / analizar_edificio (validación)",
    "test_edificio_valida_vistas_y_cantidad": "analizar / analizar_edificio (validación)",
    "test_sin_modelo_los_dos_endpoints_responden_503": "analizar / analizar_edificio (503 sin modelo)",
    "test_sin_modelo_no_gasta_el_limite_por_ip": "analizar / analizar_edificio (503 sin modelo)",
    "test_modelo_corrupto_no_carga": "cargar_sesion_onnx",
    "test_leer_cabezas_no_entrenadas": "leer_cabezas_no_entrenadas",
    "test_validar_orden_salidas": "validar_orden_salidas",
    "test_modelo_real_clasifica_una_foto_del_dataset": "modelo.onnx real (@pytest.mark.modelo)",
    "test_modelo_real_es_determinista_y_lee_png": "modelo.onnx real (@pytest.mark.modelo)",
    # test_reglas
    "test_rescatable_por_reglas_para_los_tres_niveles": "completar_rescatable",
    "test_con_cabeza_entrenada_se_usa_la_salida_del_modelo": "completar_rescatable",
    "test_fuente_del_tipo_segun_la_cnn": "completar_resultado_cnn",
    "test_recomendacion_para_los_tres_niveles_con_rescatable_por_reglas": "generar_recomendacion",
    "test_severo_con_no_del_modelo_tambien_deja_la_decision_al_ingeniero": "generar_recomendacion",
    "test_leve_con_adobe_o_madera_agrega_su_aviso_y_el_de_adobe_tapial": "generar_recomendacion",
    "test_leve_siempre_lleva_el_aviso_de_adobe_y_tapial": "generar_recomendacion / recomendacion_sin_exteriores",
    "test_ningun_texto_afirma_habitable": "generar_recomendacion / recomendacion_sin_exteriores",
    "test_ningun_texto_recomienda_demolicion": "generar_recomendacion",
    "test_rescatable_desconocido_se_trata_de_forma_conservadora": "rescatable_para_reglas",
    "test_aplicar_vista_interior_anula_tipo_y_rescatable": "aplicar_vista",
    "test_peor_caso_manda_aunque_venga_de_una_foto_interior": "consolidar_veredicto",
    "test_solo_interiores": "consolidar_veredicto",
    "test_solo_interiores_severo_mantiene_evacuar": "consolidar_veredicto",
    "test_con_cnn_entrenada_rescatable_es_el_mas_restrictivo_y_tipo_el_mas_repetido": "consolidar_veredicto",
    "test_empate_de_tipo_se_desempata_por_confianza": "consolidar_veredicto",
    "test_recomendacion_de_usa_los_campos_del_registro": "recomendacion_de",
    # test_cuentas
    "test_registro_correcto_guarda_la_fecha_de_aceptacion": "registro",
    "test_registro_con_usuario_repetido_sin_distinguir_mayusculas": "registro",
    "test_registro_rechaza_contrasena_fuera_de_rango": "registro",
    "test_registro_rechaza_usuario_invalido": "registro",
    "test_registro_exige_aceptar_privacidad": "registro",
    "test_login_devuelve_token_expiracion_y_usuario": "login / yo",
    "test_login_fallido_no_revela_si_el_usuario_existe": "login / yo",
    "test_bloqueo_tras_5_intentos_fallidos": "_comprobar_contrasena_con_bloqueo",
    "test_login_correcto_reinicia_los_intentos": "_comprobar_contrasena_con_bloqueo",
    "test_token_vencido_da_401_y_se_borra": "usuario_opcional / usuario_requerido",
    "test_token_invalido_da_401": "usuario_opcional / usuario_requerido",
    "test_token_invalido_en_analizar_da_401_y_no_se_trata_como_anonimo": "usuario_opcional / usuario_requerido",
    "test_sin_sesion_auth_yo_y_logout_dan_401_con_www_authenticate": "usuario_opcional / usuario_requerido",
    "test_logout_invalida_el_token": "logout",
    "test_eliminar_cuenta_con_contrasena_incorrecta_no_borra": "eliminar_cuenta",
    "test_eliminar_cuenta_borra_sus_datos_y_miniaturas_pero_no_las_de_otros": "eliminar_cuenta",
    "test_contrasenas_y_tokens_nunca_en_texto_plano": "hashear_contrasena / _hash_token",
    "test_hash_de_contrasena_con_sal_aleatoria": "hashear_contrasena / _hash_token",
    # test_historial
    "test_sin_sesion_401": "historial / exportaciones / miniatura_de_foto (401)",
    "test_sin_sesion_el_analisis_no_se_guarda": "guardar_inspeccion_de_usuario",
    "test_con_sesion_se_guarda_a_nombre_del_usuario": "guardar_inspeccion_de_usuario",
    "test_aislamiento_entre_usuarios": "obtener_historial / miniatura_de_foto (aislamiento, 404)",
    "test_indice_de_foto_inexistente_404": "obtener_historial / miniatura_de_foto (aislamiento, 404)",
    "test_inspecciones_anteriores_sin_dueno_no_se_muestran": "obtener_historial / miniatura_de_foto (aislamiento, 404)",
    "test_orden_por_prioridad": "obtener_historial (orden por prioridad)",
    "test_campos_de_cada_foto_en_el_historial": "obtener_historial (campos)",
    "test_estado_publico_sin_clave": "estado",
    # test_gemini
    "test_503_y_luego_responde": "obtener_explicacion (reintentos)",
    "test_dos_fallas_y_el_tercer_intento_usa_el_alternativo": "obtener_explicacion (reintentos)",
    "test_tras_un_timeout_el_siguiente_intento_usa_el_alternativo": "obtener_explicacion (reintentos)",
    "test_sin_alternativo_se_repite_el_principal": "obtener_explicacion (reintentos)",
    "test_429_con_retry_delay_largo_da_cuota_sin_reintentar": "obtener_explicacion (reintentos)",
    "test_429_con_retry_delay_corto_reintenta_esperando_lo_sugerido": "obtener_explicacion (reintentos)",
    "test_errores_permanentes_no_se_reintentan": "obtener_explicacion (reintentos)",
    "test_tres_fallas_seguidas_dan_plantilla_error": "obtener_explicacion (reintentos)",
    "test_tres_429_seguidos_dan_cuota": "obtener_explicacion (reintentos)",
    "test_respuesta_fuera_de_esquema_se_reintenta": "obtener_explicacion (reintentos)",
    "test_tope_total_de_tiempo": "obtener_explicacion (reintentos)",
    "test_la_clave_no_sale_en_el_log": "obtener_explicacion (reintentos)",
    "test_sin_ia_disponible_devuelve_la_plantilla_con_su_motivo": "explicacion_respaldo",
    "test_clasificar_falla": "_clasificar_falla_ia",
    "test_regla_de_duda": "fragmento_tipo_dudoso",
    "test_regla_de_duda_por_foto": "fragmento_tipo_dudoso",
    "test_tipo_no_seguro_o_dudoso_cuenta_como_no_se_distingue": "_tipos_por_foto / _tipo_edificio_de_ia",
    "test_endpoint_tipo_de_gemini_con_regla_de_duda": "aplicar_tipo_de_gemini (endpoints)",
    "test_edificio_con_exteriores_e_interiores_tipo_por_foto_en_todas": "aplicar_tipo_de_gemini (exteriores e interiores)",
    "test_regla_de_duda_tambien_en_fotos_interiores": "aplicar_tipo_de_gemini (exteriores e interiores)",
    "test_solo_interiores_el_tipo_del_edificio_sale_de_ellas": "aplicar_tipo_de_gemini (solo interiores)",
    "test_solo_interiores_sin_gemini_el_tipo_queda_sin_evaluar": "aplicar_tipo_de_gemini (solo interiores)",
    "test_prompt_pide_el_tipo_tambien_en_interiores": "INSTRUCCIONES_IA (prompt)",
    "test_con_cnn_entrenada_para_el_tipo_se_usa_la_cnn": "aplicar_tipo_de_gemini (endpoints)",
    "test_sin_sesion_sin_consentimiento_no_se_llama_a_gemini": "explicacion_para (consentimiento y límite)",
    "test_sin_sesion_con_consentimiento_usa_gemini_y_no_guarda": "explicacion_para (consentimiento y límite)",
    "test_limite_de_explicaciones_sin_sesion": "explicacion_para (consentimiento y límite)",
    "test_un_solo_consumo_del_limite_aunque_haya_reintentos": "explicacion_para (consentimiento y límite)",
    "test_con_sesion_no_aplica_el_limite_sin_sesion": "explicacion_para (consentimiento y límite)",
    "test_configuracion_elige_flash_y_alternativa": "configurar_gemini",
    "test_falla_transitoria_al_arrancar_queda_pendiente": "configurar_gemini",
    "test_falla_permanente_al_arrancar_desactiva_hasta_reiniciar": "configurar_gemini",
    "test_se_reconfigura_en_la_siguiente_peticion_pasado_un_minuto": "asegurar_configuracion_ia",
    "test_sin_sesion_reconfiguracion_fallida_no_gasta_cuota": "asegurar_configuracion_ia",
    "test_candidatos_gemini": "candidatos_gemini",
    "test_modelos_excluidos_configurables_en_env": "modelos_excluidos (GEMINI_MODELOS_EXCLUIDOS)",
    "test_gemini_model_excluido_no_se_usa": "modelos_excluidos (GEMINI_MODELOS_EXCLUIDOS)",
    "test_modelos_excluidos": "modelos_excluidos (GEMINI_MODELOS_EXCLUIDOS)",
    "test_contenido_sin_nombres_de_archivo_sin_exif_y_sin_rescatable_por_reglas": "construir_contenido_ia",
    # test_seguridad
    "test_limite_de_analisis_sin_sesion": "aplicar_limite_sin_sesion / LimitePorIP",
    "test_con_sesion_no_hay_limite_por_ip": "aplicar_limite_sin_sesion / LimitePorIP",
    "test_limite_de_registros": "aplicar_limite_sin_sesion / LimitePorIP",
    "test_limite_por_ip_ventana_deslizante": "aplicar_limite_sin_sesion / LimitePorIP",
    "test_ip_confiable": "_ip (CF-Connecting-IP / X-Forwarded-For)",
    "test_cambiar_encabezados_no_evade_el_limite_desde_afuera": "_ip (CF-Connecting-IP / X-Forwarded-For)",
    "test_detras_del_tunel_cada_visitante_tiene_su_limite": "_ip (CF-Connecting-IP / X-Forwarded-For)",
    "test_pdf_escapa_marcado_en_textos_de_usuario_y_de_gemini": "exportar_historial_pdf (escape)",
    "test_miniatura_sin_exif_tamano_maximo_y_nombre_aleatorio": "crear_miniatura / jpeg_sin_metadatos (EXIF)",
    "test_jpeg_sin_metadatos": "crear_miniatura / jpeg_sin_metadatos (EXIF)",
    "test_nombres_de_miniatura_manipulados_no_salen_de_la_carpeta": "borrar_miniaturas / miniatura_de_foto (rutas)",
    "test_archivo_mayor_a_10_mb_responde_413": "_leer_archivo_limitado",
    "test_el_log_no_incluye_el_nombre_del_archivo": "analizar_foto (log sin datos personales)",
    "test_excel_no_convierte_textos_de_usuario_en_formulas": "exportar_historial_excel (inyección de fórmulas)",
    "test_llenar_hoja_guarda_como_texto_todo_inicio_de_formula": "_llenar_hoja (inyección de fórmulas)",
    # test_reportes
    "test_excel_se_genera_con_sus_hojas_y_rotulos": "exportar_historial_excel",
    "test_excel_marca_el_origen_de_cada_valor": "exportar_historial_excel",
    "test_excel_y_pdf_marcan_el_tipo_de_gemini_en_interiores": "exportar_historial_excel / exportar_historial_pdf",
    "test_pdf_se_genera_con_titulo_y_secciones": "exportar_historial_pdf",
    "test_pdf_muestra_la_advertencia_de_imagen_no_valida": "exportar_historial_pdf",
    "test_reportes_vacios_se_generan": "exportar_historial_excel / exportar_historial_pdf",
    "test_texto_explicacion": "texto_explicacion",
}

# Funciones principales de main.py para el cuadro de cobertura (en orden de lectura).
FUNCIONES_PRINCIPALES = [
    "cargar_sesion_onnx", "leer_cabezas_no_entrenadas", "inicializar_bd", "guardar_inspeccion", "obtener_historial",
    "generar_recomendacion", "recomendacion_sin_exteriores", "completar_rescatable", "completar_resultado_cnn",
    "aplicar_vista", "consolidar_veredicto", "fragmento_tipo_dudoso", "aplicar_tipo_de_gemini", "nota_sin_exteriores", "validar_orden_salidas",
    "analizar_imagen", "analizar_foto", "modelos_excluidos", "candidatos_gemini", "configurar_gemini", "jpeg_sin_metadatos",
    "construir_contenido_ia", "_clasificar_falla_ia", "obtener_explicacion", "asegurar_configuracion_ia",
    "explicacion_para", "LimitePorIP.consumir", "_ip", "usuario_opcional", "_comprobar_contrasena_con_bloqueo",
    "registro", "login", "logout", "eliminar_cuenta", "crear_miniatura", "borrar_miniaturas", "miniatura_de_foto",
    "estado", "_leer_archivo_limitado", "analizar", "analizar_edificio", "historial", "_llenar_hoja", "exportar_historial_excel",
    "_bloque_explicacion_pdf", "exportar_historial_pdf", "texto_explicacion",
]


def tabla_pruebas():
    filas = defaultdict(lambda: {"total": 0, "ok": 0, "falla": 0, "saltada": 0})
    for caso in ET.parse(CARPETA / "junit.xml").getroot().iter("testcase"):
        modulo = caso.get("classname").split(".")[-1]
        funcion = FUNCION_PROBADA.get(re.sub(r"\[.*", "", caso.get("name")), "(sin clasificar)")
        celda = filas[(modulo, funcion)]
        celda["total"] += 1
        if caso.find("failure") is not None or caso.find("error") is not None:
            celda["falla"] += 1
        elif caso.find("skipped") is not None:
            celda["saltada"] += 1
        else:
            celda["ok"] += 1
    lineas = ["| Módulo de pruebas | Función probada (main.py) | Pruebas | Resultado |", "|---|---|---:|---|"]
    totales = {"total": 0, "ok": 0, "falla": 0, "saltada": 0}
    for (modulo, funcion), c in sorted(filas.items()):
        for k in totales:
            totales[k] += c[k]
        resultado = ("Aprobadas" if c["ok"] == c["total"] else
                     ", ".join(t for t in (f"{c['ok']} aprobadas" if c["ok"] else "",
                                          f"{c['falla']} fallidas" if c["falla"] else "",
                                          f"{c['saltada']} saltadas" if c["saltada"] else "") if t))
        lineas.append(f"| {modulo}.py | {funcion} | {c['total']} | {resultado} |")
    lineas.append(f"| **Total** | | **{totales['total']}** | **{totales['ok']} aprobadas, {totales['falla']} fallidas, "
                  f"{totales['saltada']} saltadas** |")
    (CARPETA / "tabla_pruebas.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return totales


def cobertura_por_funcion():
    datos = json.load(open(CARPETA / "coverage.json", encoding="utf-8"))["files"]["main.py"]
    funciones = datos["functions"]
    lineas = ["| Función (main.py) | Sentencias | Sin cubrir | Cobertura |", "|---|---:|---:|---:|"]
    for nombre in FUNCIONES_PRINCIPALES:
        resumen = funciones[nombre]["summary"]
        lineas.append(f"| {nombre} | {resumen['num_statements']} | {resumen['missing_lines']} | "
                      f"{resumen['percent_covered_display']} % |")
    total = datos["summary"]
    lineas.append(f"| **main.py (total)** | **{total['num_statements']}** | **{total['missing_lines']}** | "
                  f"**{total['percent_covered_display']} %** |")
    (CARPETA / "cobertura_por_funcion.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    sin_clasificar = [n for n in funciones if n and n not in FUNCIONES_PRINCIPALES]
    return total, len(funciones), sin_clasificar


if __name__ == "__main__":
    totales = tabla_pruebas()
    total, n_funciones, _ = cobertura_por_funcion()
    print(f"Pruebas: {totales['total']} ({totales['ok']} aprobadas, {totales['falla']} fallidas, {totales['saltada']} saltadas)")
    print(f"Cobertura de main.py: {total['percent_covered_display']} % ({n_funciones} funciones medidas)")
    print(f"Escrito: {CARPETA / 'tabla_pruebas.md'} y {CARPETA / 'cobertura_por_funcion.md'}")
