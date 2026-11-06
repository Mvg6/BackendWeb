"""Prueba de carga del pool de claves de IA con proveedores SIMULADOS (no gasta cupo real).

Simula una feria de 5 horas comprimidas: 600 análisis sin cuenta (con consentimiento) en
modo feria, con picos de 10 por minuto, ráfagas de 1 cada 2-3 s y llegadas sueltas el
resto del tiempo. El pool tiene 2 claves Gemini (prioridad 1) con rotación por modelo
(GEMINI_MODELOS_ORDEN: cada modelo con su propio límite por minuto) y 1 Groq (prioridad 2),
todas simuladas, con 503 y 429 aleatorios inyectados.

Todo el tiempo de la simulación (llegadas, latencias, cooldowns, retry-after, ventanas
por minuto) se comprime con el mismo factor; las latencias se informan en segundos de la
simulación. Usa una base y un .env temporales: no toca historial.db ni el .env real.

Uso (desde backend/):  python pruebas_apis/carga.py [--factor 60] [--analisis 600] [--semilla 2026]
Escribe pruebas_apis/resultados_carga_<fecha>.csv (no se sube al repositorio).
"""
import argparse
import asyncio
import csv
import hashlib
import io
import os
import random
import statistics
import sys
import tempfile
import time as tiempo_real
from collections import Counter, deque
from datetime import datetime
from types import SimpleNamespace

import httpx
import numpy as np
from google.genai import errors as errores_genai
from PIL import Image

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
import main  # noqa: E402
import pool_claves  # noqa: E402

DURACION_VIRTUAL = 5 * 3600
LATENCIA_GEMINI = (1.5, 4.0)  # s (simulación)
LATENCIA_GROQ = (0.8, 2.5)
PROB_503 = 0.06
PROB_429 = 0.03
# Rotación por modelo en las claves Gemini: cada modelo tiene su propio cupo por minuto
# (nivel gratuito aproximado) y su latencia (Gemma es más lento: ~9 s en la prueba real).
ORDEN_GEMINI = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemma-4-26b-a4b-it"]
RPM_GEMINI = {"gemini-3.1-flash-lite": 15, "gemini-3.5-flash-lite": 10, "gemma-4-26b-a4b-it": 30}
LATENCIA_POR_MODELO = {"gemma-4-26b-a4b-it": (5.0, 10.0)}
RPM_GROQ = 30


class Reloj:
    """Reloj de la simulación: avanza `factor` veces más rápido que el real."""

    def __init__(self, factor):
        self.factor = factor
        self._inicio_real = tiempo_real.perf_counter()
        self._mono0, self._unix0 = tiempo_real.monotonic(), tiempo_real.time()

    def transcurrido(self):
        return (tiempo_real.perf_counter() - self._inicio_real) * self.factor

    def monotonic(self):
        return self._mono0 + self.transcurrido()

    def time(self):
        return self._unix0 + self.transcurrido()

    def perf_counter(self):
        return self.monotonic()

    async def dormir(self, segundos_virtuales):
        await asyncio.sleep(max(0.0, segundos_virtuales) / self.factor)


class SesionOnnxSimulada:
    """Probabilidades al azar pero deterministas por imagen, como el ONNX real (que no es
    lo que se mide aquí): la misma foto da la misma clasificación."""

    def run(self, _salidas, alimentacion):
        (entrada,) = alimentacion.values()
        semilla = int(hashlib.sha256(entrada.tobytes()).hexdigest()[:8], 16)
        dano = np.random.default_rng(semilla).dirichlet([2, 2, 2])
        return [np.array([[0.2] * 5]), np.array([dano]), np.array([[0.3, 0.3, 0.4]])]


class ProveedorSimulado:
    """Una clave simulada: latencia, límite por minuto (429 al pasarlo) y fallas al azar."""

    def __init__(self, nombre, reloj, azar, latencia, rpm, inyectadas):
        """`rpm`: un número (toda la clave) o {modelo: límite} (el cupo es por modelo)."""
        self.nombre, self.reloj, self.azar, self.latencia, self.rpm = nombre, reloj, azar, latencia, rpm
        self.inyectadas = inyectadas
        self.usos = {}
        self.respuestas = Counter()

    def _decidir(self, modelo=None):
        """None (responde), "503" o ("429", espera en s)."""
        ahora = self.reloj.monotonic()
        clave_cupo = modelo if isinstance(self.rpm, dict) else None
        limite = self.rpm[modelo] if isinstance(self.rpm, dict) else self.rpm
        usos = self.usos.setdefault(clave_cupo, deque())
        while usos and ahora - usos[0] >= 60:
            usos.popleft()
        if len(usos) >= limite:
            self.inyectadas[f"{self.nombre}{'/' + modelo if clave_cupo else ''}: 429 por límite por minuto"] += 1
            return ("429", 60 - (ahora - usos[0]))
        usos.append(ahora)
        sorteo = self.azar.random()
        if sorteo < PROB_503:
            self.inyectadas[f"{self.nombre}: 503 al azar"] += 1
            return "503"
        if sorteo < PROB_503 + PROB_429:
            self.inyectadas[f"{self.nombre}: 429 al azar"] += 1
            return ("429", self.azar.uniform(5, 30))
        return None

    def respuesta_ia(self):
        return main.ExplicacionIA(
            estructura=["Muros de albañilería confinada con columnas de concreto."],
            estado=["Grietas diagonales en el muro del primer piso."],
            viabilidad="Las fotos muestran daño visible; la decisión final es del ingeniero.",
            acciones=["Acordonar el perímetro.", "Solicitar inspección presencial."],
            discrepancia=None, imagen_valida=True,
            tipo_construccion_foto=[main.TipoConstruccionFotoIA(foto=1, tipo_construccion="Ladrillo/Albañilería",
                                                                tipo_construccion_seguro=True)],
            tipo_construccion_edificio="Ladrillo/Albañilería", tipo_construccion_edificio_seguro=True,
        )

    # Gemini (client.aio.models.generate_content)
    async def generate_content(self, model, contents, config):
        await self.reloj.dormir(self.azar.uniform(*LATENCIA_POR_MODELO.get(model, self.latencia)))
        decision = self._decidir(model)
        if decision == "503":
            raise errores_genai.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
        if decision:
            cuerpo = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED", "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{decision[1]:.0f}s"}]}}
            raise errores_genai.ClientError(429, cuerpo)
        self.respuestas[model] += 1
        return SimpleNamespace(parsed=self.respuesta_ia())

    # Groq (POST /chat/completions)
    async def enviar(self, url, clave, cuerpo, timeout):
        await self.reloj.dormir(self.azar.uniform(*self.latencia))
        decision = self._decidir()
        if decision == "503":
            return httpx.Response(503, json={"error": {"message": "service unavailable"}})
        if decision:
            return httpx.Response(429, headers={"retry-after": f"{decision[1]:.0f}"},
                                  json={"error": {"message": "Rate limit reached for requests per minute"}})
        self.respuestas["groq-simulado"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": self.respuesta_ia().model_dump_json()}}]})


def llegadas(azar, total):
    """Instantes (s de la simulación) de `total` análisis en 5 h: 5 picos de 10/min durante
    6 min, 4 ráfagas de 1 cada 2-3 s durante 2 min, y llegadas sueltas el resto."""
    instantes = []
    inicios = sorted(azar.sample(range(600, DURACION_VIRTUAL - 900, 300), 9))
    picos, rafagas = inicios[:5], inicios[5:]
    for inicio in picos:  # 10 por minuto
        instantes += [inicio + minuto * 60 + azar.uniform(0, 60) for minuto in range(6) for _ in range(10)]
    for inicio in rafagas:  # 1 cada 2-3 s
        t = inicio
        while t < inicio + 120:
            instantes.append(t)
            t += azar.uniform(2, 3)
    while len(instantes) < total:
        instantes.append(azar.uniform(0, DURACION_VIRTUAL))
    azar.shuffle(instantes)
    return sorted(instantes[:total])


def fotos(azar, total):
    """Fotos sintéticas distintas; ~5 % repite una anterior (para ver la caché)."""
    lista = []
    for i in range(total):
        if i > 20 and azar.random() < 0.05:
            lista.append(azar.choice(lista))
            continue
        imagen = Image.new("RGB", (320, 240), tuple(azar.randrange(256) for _ in range(3)))
        imagen.putpixel((azar.randrange(320), azar.randrange(240)), (i % 256, (i // 256) % 256, 7))
        salida = io.BytesIO()
        imagen.save(salida, "JPEG")
        lista.append(salida.getvalue())
    return lista


async def correr(factor, total, semilla):
    azar = random.Random(semilla)
    temporal = tempfile.mkdtemp(prefix="carga_pool_")
    ruta_env = os.path.join(temporal, ".env")
    with open(ruta_env, "w", encoding="utf-8") as archivo:
        archivo.write("EXPLICACION_IA=off\nMODO_FERIA=on\nLIMITE_FERIA_HORA=300\nFERIA_IA_RPM=20\n")
    main.RUTA_BD = os.path.join(temporal, "historial_carga.db")
    main.RUTA_ENV = ruta_env
    main.DIR_MINIATURAS = os.path.join(temporal, "miniaturas")
    main.RUTA_MODELO = os.path.join(temporal, "sin_modelo.onnx")
    reloj = Reloj(factor)
    main.time = pool_claves.time = reloj
    main._esperar = reloj.dormir
    inyectadas = Counter()

    async with main.lifespan(main.app):
        main.sesion_onnx, main.entrada_onnx_nombre = SesionOnnxSimulada(), "entrada"
        main.cabezas_no_entrenadas = ["tipo_construccion", "rescatable"]
        conn = pool_claves.conectar(main.RUTA_BD)
        with conn:
            ids = [pool_claves.agregar(conn, main._fernet_pool, "gemini", f"AIzaSimuladaCarga{i:04d}", alias=f"gemini-{i}",
                                       prioridad=1) for i in (1, 2)]
            ids.append(pool_claves.agregar(conn, main._fernet_pool, "groq", "gsk_SimuladaCarga0003", alias="groq-1",
                                           prioridad=2))
        conn.close()
        proveedores = {
            ids[0]: ProveedorSimulado("gemini-1", reloj, azar, LATENCIA_GEMINI, RPM_GEMINI, inyectadas),
            ids[1]: ProveedorSimulado("gemini-2", reloj, azar, LATENCIA_GEMINI, RPM_GEMINI, inyectadas),
            ids[2]: ProveedorSimulado("groq-1", reloj, azar, LATENCIA_GROQ, RPM_GROQ, inyectadas),
        }

        def cliente_de(proveedor):
            return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=proveedor.generate_content)))

        main.cliente_gemini, main._id_clave_gemini = cliente_de(proveedores[ids[0]]), ids[0]
        main._clientes_gemini = {ids[1]: cliente_de(proveedores[ids[1]])}
        main.modelo_gemini, main.modelos_alternativos_gemini = ORDEN_GEMINI[0], ORDEN_GEMINI[1:]
        main.modelos_gemini_orden = list(ORDEN_GEMINI)
        main.motivo_sin_ia, main.reconfigurar_ia_pendiente = None, False
        main.config_ia["GROQ_MODEL"] = "groq-simulado"
        main._enviar_compatible = proveedores[ids[2]].enviar
        main.estadisticas_ia.clear()

        instantes, imagenes = llegadas(azar, total), fotos(azar, total)
        resultados = []
        transporte = httpx.ASGITransport(app=main.app, client=("192.168.1.50", 50000))
        async with httpx.AsyncClient(transport=transporte, base_url="http://feria", timeout=None) as cliente:
            async def analisis(numero, instante, imagen):
                await reloj.dormir(instante - reloj.transcurrido())
                inicio = reloj.transcurrido()
                r = await cliente.post("/analizar", data={"acepta_envio_ia": "true"},
                                       files={"archivo": (f"foto_{numero}.jpg", imagen, "image/jpeg")})
                fin = reloj.transcurrido()
                e = r.json().get("explicacion") if r.status_code == 200 else None
                resultados.append({
                    "numero": numero, "llegada_s": round(instante, 1), "latencia_s": round(fin - inicio, 2),
                    "http": r.status_code, "fuente": e["fuente"] if e else "", "modelo": (e or {}).get("modelo") or "",
                    "motivo_respaldo": (e or {}).get("motivo_respaldo") or "",
                })

            inicio_real = tiempo_real.perf_counter()
            await asyncio.gather(*(analisis(i, t, f) for i, (t, f) in enumerate(zip(instantes, imagenes), 1)))
            duracion_real = tiempo_real.perf_counter() - inicio_real
        conn = pool_claves.conectar(main.RUTA_BD)
        estado_pool = pool_claves.todas(conn, main._fernet_pool, reloj.time())
        conn.close()
    return resultados, inyectadas, proveedores, estado_pool, duracion_real


def resumen(resultados, inyectadas, proveedores, estado_pool, duracion_real, factor):
    total = len(resultados)
    con_ia = [r for r in resultados if r["fuente"] in main.PROVEEDORES_IA]
    latencias = sorted(r["latencia_s"] for r in resultados)
    p95 = latencias[max(0, int(round(0.95 * len(latencias))) - 1)]
    lineas = [
        f"Análisis: {total} en 5 h simuladas (factor {factor}: {duracion_real:.0f} s reales), modo feria on",
        f"HTTP 200: {sum(r['http'] == 200 for r in resultados)} de {total}",
        f"Con explicación de IA: {len(con_ia)} ({100 * len(con_ia) / total:.1f} %) "
        f"- por proveedor: {dict(Counter(r['fuente'] for r in con_ia))}",
        f"Sin explicación (resultado del ONNX y mensaje): {total - len(con_ia)} "
        f"- motivos: {dict(Counter(r['motivo_respaldo'] for r in resultados if r['motivo_respaldo']))}",
        f"Latencia (s de la simulación): media {statistics.mean(latencias):.2f}, p95 {p95:.2f}, máx {latencias[-1]:.2f}",
        f"Rotaciones a otra clave: {main.estadisticas_ia['rotaciones']}; "
        f"a otro modelo de la misma clave: {main.estadisticas_ia['rotaciones_modelo']}; "
        f"desde la caché: {main.estadisticas_ia['cache']}; llamadas a proveedores: {main.estadisticas_ia['llamadas']}",
        f"Fallas inyectadas: {sum(inyectadas.values())} - {dict(sorted(inyectadas.items()))}",
        "Respuestas por clave y modelo: " + "; ".join(
            f"{p.nombre} " + ", ".join(f"{m} {n}" for m, n in sorted(p.respuestas.items()))
            for p in proveedores.values()),
        "Estado final del pool: " + "; ".join(
            f"{c['alias']} activa={'sí' if c['activa'] else 'no'} errores_seguidos={c['errores_seguidos']}"
            for c in estado_pool),
    ]
    return "\n".join(lineas)


def principal():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--factor", type=float, default=60, help="cuántas veces más rápido que el tiempo real")
    parser.add_argument("--analisis", type=int, default=600)
    parser.add_argument("--semilla", type=int, default=2026)
    args = parser.parse_args()
    import logging
    logging.getLogger("fachadas_api").setLevel(logging.ERROR)  # 600 análisis: solo el resumen
    resultados, inyectadas, proveedores, estado_pool, duracion_real = asyncio.run(
        correr(args.factor, args.analisis, args.semilla))
    texto = resumen(resultados, inyectadas, proveedores, estado_pool, duracion_real, args.factor)
    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          f"resultados_carga_{datetime.now():%Y-%m-%d_%H%M%S}.csv")
    with open(salida, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(resultados[0]))
        escritor.writeheader()
        escritor.writerows(sorted(resultados, key=lambda r: r["numero"]))
    print(texto)
    print(f"Detalle por análisis: {salida}")


if __name__ == "__main__":
    principal()
