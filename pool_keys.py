"""Administra el pool de claves de API de la explicación con IA (sin endpoint web).

Uso (desde backend/, con el .venv):
    python pool_keys.py agregar gemini [--alias NOMBRE] [--prioridad N] [--modelo MODELO]
    python pool_keys.py listar
    python pool_keys.py activar ID
    python pool_keys.py desactivar ID
    python pool_keys.py estado
    python pool_keys.py elegir-modelo-groq

La clave se pide sin eco (getpass) y se guarda cifrada en historial.db con
POOL_MASTER_KEY de .env (se genera si falta). Nunca se muestra completa: solo
los últimos 4 caracteres. Los cambios valen para el siguiente análisis; los
modelos de Gemini y GROQ_MODEL se eligen al arrancar uvicorn.
"""
import argparse
import getpass
import math
import os
import sys
import time
from datetime import datetime

import httpx

import pool_claves

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_BD = os.path.join(BASE_DIR, "historial.db")
RUTA_ENV = os.path.join(BASE_DIR, ".env")


def _fecha(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def _espera(segundos):
    if segundos <= 0:
        return "-"
    if segundos < 120:
        return f"{math.ceil(segundos)} s"
    if segundos < 7200:
        return f"{math.ceil(segundos / 60)} min"
    return f"{segundos / 3600:.1f} h"


def agregar(conn, fernet, args, leer_clave=getpass.getpass):
    clave = leer_clave(f"Clave de {args.proveedor} (no se muestra al escribir): ")
    id_ = pool_claves.agregar(conn, fernet, args.proveedor, clave, args.modelo, args.alias, args.prioridad)
    print(f"Clave {id_} agregada: {args.proveedor} {pool_claves.enmascarar(clave.strip())}, prioridad {args.prioridad}.")


def listar(conn, fernet, _args):
    claves = pool_claves.todas(conn, fernet)
    if not claves:
        print("El pool está vacío. Agrega una clave con: python pool_keys.py agregar gemini")
        return
    print(f"{'ID':>3}  {'Proveedor':<10} {'Clave':<9} {'Activa':<6} {'Prior.':>6}  {'Modelo':<28} Alias")
    for c in claves:
        print(f"{c['id']:>3}  {c['proveedor']:<10} {c['clave']:<9} {'sí' if c['activa'] else 'no':<6} "
              f"{c['prioridad']:>6}  {(c['modelo'] or '(automático)'):<28} {c['alias'] or ''}")


def estado(conn, fernet, _args):
    claves = pool_claves.todas(conn, fernet)
    if not claves:
        print("El pool está vacío.")
        return
    print(f"{'ID':>3}  {'Proveedor':<10} {'Clave':<9} {'Estado':<12} {'Espera':>8} {'Usos hoy':>8} "
          f"{'Errores':>7}  {'Último uso':<16} Último error")
    for c in claves:
        situacion = "desactivada" if not c["activa"] else ("en espera" if c["cooldown_restante"] else "disponible")
        print(f"{c['id']:>3}  {c['proveedor']:<10} {c['clave']:<9} {situacion:<12} {_espera(c['cooldown_restante']):>8} "
              f"{c['usos_hoy']:>8} {c['errores_seguidos']:>7}  {_fecha(c['ultimo_uso']):<16} {c['ultimo_error'] or ''}")
        for modelo, restante in sorted(c["modelos_en_espera"].items()):
            print(f"{'':>5}modelo {modelo} en espera {_espera(restante)}")
    disponibles = sum(1 for c in claves if c["activa"] and not c["cooldown_restante"])
    print(f"\n{len(claves)} clave(s), {disponibles} disponible(s) ahora.")


def cambiar_activa(conn, _fernet, args):
    if not pool_claves.activar(conn, args.id, args.comando == "activar"):
        raise ValueError(f"no existe la clave {args.id}")
    print(f"Clave {args.id} {'activada (sin espera ni errores)' if args.comando == 'activar' else 'desactivada'}.")


def elegir_modelo_groq(conn, fernet, _args, ruta_env=None):
    claves = [c for c in pool_claves.disponibles(conn, fernet, ahora=time.time() + 10 ** 9) if c.proveedor == "groq"]
    if not claves:
        raise ValueError("no hay claves Groq activas en el pool (agrega una con: python pool_keys.py agregar groq)")
    modelo, probados = pool_claves.elegir_modelo_groq(claves[0].clave)
    if not modelo:
        raise ValueError(f"ningún modelo de Groq aceptó imágenes (probados: {', '.join(probados) or 'ninguno'})")
    pool_claves.escribir_variable_env(ruta_env or RUTA_ENV, "GROQ_MODEL", modelo)
    print(f"Modelo de Groq elegido: {modelo} (acepta imágenes). Guardado como GROQ_MODEL en .env; reinicia uvicorn.")


COMANDOS = {"agregar": agregar, "listar": listar, "estado": estado, "activar": cambiar_activa,
            "desactivar": cambiar_activa, "elegir-modelo-groq": elegir_modelo_groq}


def analizador():
    parser = argparse.ArgumentParser(description="Pool de claves de la explicación con IA.")
    parser.add_argument("--bd", default=RUTA_BD, help=argparse.SUPPRESS)
    parser.add_argument("--env", default=RUTA_ENV, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="comando", required=True)
    p = sub.add_parser("agregar", help="agrega una clave (se pide sin eco)")
    p.add_argument("proveedor", choices=pool_claves.PROVEEDORES)
    p.add_argument("--alias")
    p.add_argument("--prioridad", type=int, default=10, help="menor = se usa primero (por defecto 10)")
    p.add_argument("--modelo", help="Groq/OpenRouter: modelo de esta clave (si no, GROQ_MODEL u OPENROUTER_MODEL)")
    sub.add_parser("listar", help="lista las claves (enmascaradas)")
    sub.add_parser("estado", help="disponibilidad, esperas (cooldown), usos de hoy y último error")
    for nombre in ("activar", "desactivar"):
        sub.add_parser(nombre, help=f"{nombre} una clave").add_argument("id", type=int)
    sub.add_parser("elegir-modelo-groq", help="elige un modelo de Groq con imágenes y lo guarda en .env")
    return parser


def main(argv=None, leer_clave=getpass.getpass):
    args = analizador().parse_args(argv)
    try:
        fernet, creada = pool_claves.obtener_fernet(args.env)
        if creada:
            print("Se generó POOL_MASTER_KEY y se agregó a .env (respáldala: sin ella no se pueden descifrar las claves).")
        conn = pool_claves.conectar(args.bd)
        try:
            with conn:
                if args.comando == "agregar":
                    agregar(conn, fernet, args, leer_clave)
                elif args.comando == "elegir-modelo-groq":
                    elegir_modelo_groq(conn, fernet, args, args.env)
                else:
                    COMANDOS[args.comando](conn, fernet, args)
        finally:
            conn.close()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:  # sin el detalle: podría incluir datos de la petición
        print(f"Error al consultar al proveedor ({type(e).__name__}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
