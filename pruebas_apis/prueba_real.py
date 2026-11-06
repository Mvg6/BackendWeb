"""Prueba real mínima: 3 análisis con fotos de fotos_peru_dataset, forzando una vez cada
proveedor (Gemini y Groq) y una vez la selección automática del pool, para confirmar que
responden con el JSON correcto. Gasta cupo real: solo 3 llamadas de análisis (más el sondeo
del modelo de Gemini al arrancar).

Usa el .env real (claves y POOL_MASTER_KEY) y una COPIA temporal de historial.db: no guarda
nada en el historial real. Nunca muestra claves. Un proveedor sin clave se informa como omitido.

Uso (desde backend/):  python pruebas_apis/prueba_real.py [--dataset RUTA]
"""
import argparse
import asyncio
import csv
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
import main  # noqa: E402

FOTOS = [  # (carpeta, archivo, vista, proveedor forzado o None = automático)
    ("otros_desastres_pisco2007", "pisco2007_04.jpg", "exterior", "gemini"),
    ("ancash_1970", "ancash1970_01.jpg", "exterior", "groq"),
    ("otros_desastres_arequipa1868", "arequipa1868_01.jpg", "exterior", None),
]


async def correr(dataset):
    temporal = tempfile.mkdtemp(prefix="prueba_real_")
    copia = os.path.join(temporal, "historial.db")
    shutil.copy2(main.RUTA_BD, copia)
    main.RUTA_BD = copia
    main.DIR_MINIATURAS = os.path.join(temporal, "miniaturas")
    filas = []
    async with main.lifespan(main.app):
        if main.sesion_onnx is None:
            raise SystemExit("modelo.onnx no se pudo cargar")
        for carpeta, archivo, vista, proveedor in FOTOS:
            ruta = os.path.join(dataset, carpeta, archivo)
            datos = open(ruta, "rb").read()
            resultado = main.aplicar_vista(main.analizar_foto(datos, archivo, content_type="image/jpeg"), vista)
            resultado["recomendacion"] = main._recomendacion_una_foto(resultado, vista)
            _, claves = main.claves_candidatas(proveedor)
            if not claves:
                filas.append({"foto": archivo, "forzado": proveedor or "automático", "proveedor": "",
                              "modelo": "", "segundos": "", "json_valido": "", "nivel_onnx": resultado["nivel_dano"],
                              "tipo_ia": "", "observacion": f"omitido: no hay ninguna clave {'de ' + proveedor + ' ' if proveedor else ''}"
                                               "disponible en el pool"})
                continue
            inicio = time.perf_counter()
            e = await main.obtener_explicacion([{"vista": vista, "bytes": datos, **resultado}], resultado,
                                               solo_proveedor=proveedor)
            segundos = time.perf_counter() - inicio
            valido = e["fuente"] in main.PROVEEDORES_IA
            if valido:  # el mismo esquema para todos los proveedores
                main.ExplicacionIA.model_validate({k: e[k] for k in main.ExplicacionIA.model_fields})
            filas.append({
                "foto": archivo, "forzado": proveedor or "automático", "proveedor": e["fuente"],
                "modelo": e["modelo"] or "", "segundos": f"{segundos:.1f}", "json_valido": "sí" if valido else "no",
                "nivel_onnx": resultado["nivel_dano"],
                "tipo_ia": e.get("tipo_construccion_edificio") or "",
                "observacion": (e["estructura"][0][:90] if valido and e["estructura"] else e.get("mensaje") or ""),
            })
    shutil.rmtree(temporal, ignore_errors=True)
    return filas


def principal():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=os.environ.get(
        "FOTOS_PERU_DATASET", os.path.join(os.path.expanduser("~"), "Downloads", "fotos_peru_dataset")))
    args = parser.parse_args()
    filas = asyncio.run(correr(args.dataset))
    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"resultados_real_{datetime.now():%Y-%m-%d_%H%M%S}.csv")
    with open(salida, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    for fila in filas:
        print(" | ".join(f"{k}: {v}" for k, v in fila.items()))
    print(f"Detalle: {salida}")


if __name__ == "__main__":
    principal()
