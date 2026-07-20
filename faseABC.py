"""
Fase A -- Cuantificacion de incertidumbre via muestreo de flujos
=================================================================

Implementa, sobre el GSMM comunitario real (community.xml, 10 especies
hipersalinas ensambladas con CarveMe), el procedimiento descrito en las
Secciones 4.1, 4.3 y 4.5 (Fase A) del manuscrito "Calculo de Schubert
como clasificador combinatorio del reparto de nicho metabolico en
comunidades microbianas hipersalinas".

Requisitos (instalar en tu propio entorno, no en este sandbox):
    pip install cobra numpy scipy statsmodels

Uso tipico:
    python fase_a_muestreo_flujos.py --model community.xml --n-samples 2000

Rendimiento (memoria y tiempo de ejecucion): este script vectoriza con
numpy toda la asignacion de celdas de Schubert (Algoritmo 1) y el bootstrap,
evitando el .iterrows() de pandas fila por fila que domina el costo cuando
se hace "a mano". Ademas recorta el DataFrame de muestras a solo las ~2,000
columnas relevantes (de ~20,784 totales) inmediatamente despues de muestrear.
El paso realmente costoso en tiempo sigue siendo el muestreo OptGP en si;
para paralelizarlo usa --processes N (paralelizacion nativa de cobra):
    python fase_a_muestreo_flujos.py --model community.xml --processes 4
Si el problema es de MEMORIA (no de tiempo), no subas --processes (cada
proceso adicional carga su propia copia del modelo) -- en su lugar reduce
--n-samples.

Si tu licencia de Gurobi es de tamano restringido (error tipico: "Model
too large for size-limited license"), este modelo de 20,784 reacciones
puede excederla. Usa --solver glpk para pruebas rapidas y gratuitas, o
solicita una licencia academica sin restriccion en gurobi.com/academia:
    python fase_a_muestreo_flujos.py --model community.xml --solver glpk

Si el muestreo se ve "colgado" sin ninguna salida (tipico en Colab, donde
ademas puede interrumpirse por timeout de inactividad), este script ahora
muestrea en lotes pequenos con progreso y tiempo estimado restante impreso
entre lotes (--batch-size, default 50). Si se interrumpe a mitad de camino,
las muestras ya recolectadas no se pierden.

GLPK (el solver por defecto, gratuito) es notablemente mas lento que
solvers comerciales para este tamano de modelo. Si tienes muchas muestras
que correr, un solver libre bastante mas rapido es HiGHS:
    pip install highspy   # o !pip install highspy en Colab
    python fase_a_muestreo_flujos.py --model community.xml --solver highs

Nota tecnica importante: cobrapy recorta automaticamente los prefijos
"R_"/"M_" de los ids al leer un SBML (su convencion estandar de import).
Los patrones de este script ya toleran ambas variantes (con y sin el
prefijo "R_"), pero si tu version de cobrapy o tu modelo usan otra
convencion de nombres, revisa los avisos impresos por espacio_ambiente_E()
y reacciones_interfaz_especie() antes de confiar en los resultados.

Estructura del archivo:
    1. Carga del modelo y descubrimiento de la convencion de nombres real
       (E = reacciones R_EX_*_e; interfaz por especie = reacciones *tex*/*texi*)
    2. Muestreo de flujos (OptGP) por condicion ambiental
    3. Construccion de P_i (subespacio muestreado) por especie y muestra
    4. Algoritmo 1 del manuscrito: asignacion de celda de Schubert
    5. Cuantificacion de incertidumbre: distribucion empirica, bootstrap CI,
       regresion logistica ordinal de la codimension sobre la salinidad
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

try:
    import cobra
    from cobra.sampling import OptGPSampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Este script requiere cobrapy. Instala con: pip install cobra"
    ) from exc


SPECIES = [
    "Chromohalobacter",
    "Haloarcula",
    "Halobacterium",
    "Haloferax",
    "Halomonas",
    "Haloquadratum",
    "Halorubrum",
    "Natronomonas",
    "salinibacter",
    "Tetragenococcus",
]

# ---------------------------------------------------------------------------
# Correspondencia taxonomica Santa Pola -> especies del GSMM (Seccion 4.2).
# AJUSTA ESTE DICCIONARIO con tu correspondencia real genero-especie una vez
# resuelta; el orden de las claves (SS13 -> SS19 -> SS33 -> SS37) codifica la
# bandera F_1 subset F_2 subset F_3 subset F_4 por salinidad creciente.
# Los valores por defecto reflejan la dominancia reportada en Ghai2011,
# Fernandez2013 y Fernandez2014a: en SS13 dominan generos distintos a
# Haloquadratum/Salinibacter, que dominan a partir de SS19.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Correspondencia taxonomica Santa Pola -> especies del GSMM (Seccion 4.2),
# CONSTRUIDA A PARTIR DE DATOS REALES E INDEPENDIENTES (no estimacion de
# lectura): Fernandez et al. 2014a (FEMS Microbiol Ecol 88:623-635), Tabla 2 /
# Fig. 3 -- clasificacion taxonomica real de lecturas 16S rRNA de los
# metagenomas SS13, SS19 y SS37 de Santa Pola, con porcentaje de abundancia
# por genero. Estos son los generos que superan el 1% de las lecturas 16S
# clasificadas, tal como reporta el paper:
#
#   Genero            SS13    SS19    SS37
#   Haloquadratum      1.0%    7.9%   58.0%
#   Salinibacter        --     6.4%    9.1%
#   Halorubrum          --    12.5%    3.2%
#   Natronomonas        --     5.6%     --
#
# CORRECCION IMPORTANTE respecto a versiones anteriores de este script: la
# version previa asumia (a partir de una lectura general de los papers, no
# de esta tabla) que Halomonas y Chromohalobacter dominaban en SS13. Los
# datos reales lo contradicen explicitamente -- el propio paper indica:
# "we could not assign any 16S rRNA gene sequence to representatives of the
# genera Halomonas, Chromohalobacter, or Salinicoccus" en ningun estanque.
#
# LIMITACION QUE DEBE DECLARARSE EN EL MANUSCRITO: de las 10 especies del
# GSMM comunitario, SOLO 4 (Haloquadratum, salinibacter, Halorubrum,
# Natronomonas) tienen evidencia real de presencia (>1% de lecturas 16S) en
# al menos un estanque de Santa Pola segun esta tabla. Las otras 6
# (Chromohalobacter, Haloarcula, Halobacterium, Haloferax, Halomonas,
# Tetragenococcus) NO aparecen en la tabla de generos >1% en NINGUN
# estanque -- probablemente por un sesgo conocido de subrepresentacion en
# metagenomica directa de generos que se aislan bien por cultivo (discutido
# en el propio paper). Para esas 6 especies, este script NO tiene evidencia
# independiente de salinidad y su bandera se completa con reacciones sin
# tier asignado (extremo arbitrario de la bandera) -- los resultados de
# Fase A para esas 6 especies deben reportarse con una salvedad explicita
# de que no estan validadas contra datos reales de Santa Pola, o excluirse
# de la prueba principal de H1a/H1b y usarse solo como exploracion.
#
# SS13 no aparece como tier propio: ninguna de las 10 especies del consorcio
# supera el 1% en SS13 segun esta tabla, asi que no hay evidencia real para
# construir ese escalon con estos datos.
# ---------------------------------------------------------------------------
BANDERA_SANTA_POLA: Dict[str, List[str]] = {
    "SS19": ["Halorubrum", "Natronomonas"],
    "SS37": ["Haloquadratum", "salinibacter"],
}

# ---------------------------------------------------------------------------
# CORRECCION IMPORTANTE (bandera degenerada): con la version anterior de este
# diccionario, las especies "nuevas" en SS37 (Haloquadratum, salinibacter,
# Halorubrum) eran un subconjunto de las de SS19 -- ninguna aportaba
# reacciones nuevas al segundo escalon, asi que F_SS19 y F_SS37 resultaban
# EXACTAMENTE iguales en dimension (241 = 241 en una corrida real), la
# bandera tenia en la practica un solo tier util, y todas las especies
# volvian a saturarse cerca del maximo -- el mismo problema de fondo que ya
# se habia corregido, reaparecido de otra forma.
#
# La correccion usa la TENDENCIA de abundancia entre SS19 y SS37 (no solo
# presencia/ausencia) para separar los dos escalones de forma genuinamente
# distinta, aprovechando los porcentajes reales de Fernandez2014a:
#   Halorubrum:    12.5% (SS19) ->  3.2% (SS37)  -- decrece con la salinidad
#   Natronomonas:   5.6% (SS19) ->   --  (SS37)  -- solo presente en SS19
#   Haloquadratum:  7.9% (SS19) -> 58.0% (SS37)  -- crece fuertemente
#   Salinibacter:   6.4% (SS19) ->  9.1% (SS37)  -- crece con la salinidad
# SS19 agrupa ahora a los generos cuya abundancia relativa DECRECE o
# desaparece hacia mayor salinidad (especialistas de salinidad moderada);
# SS37 agrupa a los que CRECEN hacia mayor salinidad (especialistas de
# salinidad extrema). Esto produce dos tiers con membresia de especies
# realmente distinta, en vez de uno anidado trivialmente en el otro.
# ---------------------------------------------------------------------------

# Especies sin evidencia real de presencia en Santa Pola segun esta tabla
# (ver nota arriba) -- exportado para que main() pueda advertir explicitamente
# sobre estas especies en la salida.
ESPECIES_SIN_EVIDENCIA_SANTA_POLA = [
    e for e in SPECIES
    if e not in {"Halorubrum", "Haloquadratum", "salinibacter", "Natronomonas"}
]


# ---------------------------------------------------------------------------
# 1. Carga del modelo y descubrimiento de la convencion de nombres
# ---------------------------------------------------------------------------

def cargar_modelo(path: str, solver: str | None = None) -> cobra.Model:
    """Carga el GSMM comunitario desde SBML.

    Si tu licencia de Gurobi es de tamano restringido (error tipico:
    "Model too large for size-limited license"), este modelo comunitario
    de 20{,}784 reacciones puede exceder ese limite. Opciones:
      1. Solicitar una licencia academica gratuita SIN restriccion de
         tamano en https://www.gurobi.com/academia/academic-program-and-licenses/
         (casi siempre elegible con correo institucional de universidad).
      2. Usar --solver glpk para pruebas rapidas (gratuito, sin limite de
         tamano, pero mas lento para muestreo de flujos en modelos grandes).
      3. Si tienes cplex u otro solver comercial con licencia adecuada,
         pasar --solver cplex.
    """
    print(f"Cargando modelo desde {path} ...")
    modelo = cobra.io.read_sbml_model(path)
    print(f"  {len(modelo.reactions)} reacciones, {len(modelo.metabolites)} metabolitos")
    if solver:
        print(f"  configurando solver: {solver}")
        modelo.solver = solver
    return modelo


def espacio_ambiente_E(modelo: cobra.Model) -> List[str]:
    """
    Devuelve la lista de ids de reacciones que definen la base de E:
    las reacciones de intercambio comunitario unicas R_EX_*_e (m = 544
    en el modelo de referencia de este trabajo). El orden de la lista
    fija la base {e_1, ..., e_m} usada en el resto del script.

    NOTA: cobrapy recorta automaticamente el prefijo "R_" (y "M_" en
    metabolitos) al leer un SBML via read_sbml_model, como parte de su
    manejo estandar de ids validos. Por eso el patron tolera un prefijo
    "R_" opcional: en el XML crudo la reaccion es "R_EX_camp_e", pero en
    el modelo ya cargado en Python su id es "EX_camp_e".
    """
    patron = re.compile(r"^R?_?EX_.*_e$")
    base_E = sorted(rxn.id for rxn in modelo.reactions if patron.match(rxn.id))
    print(f"  dim(E) = m = {len(base_E)} reacciones de intercambio comunitario")
    if not base_E:
        ejemplos = [rxn.id for rxn in modelo.reactions if "EX_" in rxn.id][:5]
        print(f"  AVISO: 0 coincidencias. Ejemplos de ids con 'EX_' en el modelo "
              f"cargado (para depurar el patron manualmente): {ejemplos}")
    return base_E


def reacciones_interfaz_especie(modelo: cobra.Model, especie: str) -> Dict[str, str]:
    """
    Devuelve un diccionario {metabolito_ext_id -> reaccion_tex_id} con las
    reacciones de transporte periplasma->extracelular propias de una especie
    (sufijo tex/texi + nombre de especie; en el XML crudo
    "R_23CAMPtex_Chromohalobacter", pero tras la carga con cobrapy
    "23CAMPtex_Chromohalobacter", sin el prefijo "R_" -- ver nota en
    espacio_ambiente_E).
    """
    patron = re.compile(rf"^R?_?[A-Za-z0-9]+tex[i]?_{re.escape(especie)}$")
    interfaz = {}
    for rxn in modelo.reactions:
        if patron.match(rxn.id):
            # metabolito extracelular producido/consumido por esta reaccion
            met_ext = [m.id for m in rxn.metabolites if m.id.endswith("_e")]
            if met_ext:
                interfaz[met_ext[0]] = rxn.id
    return interfaz


@dataclass
class MapaEspacioAmbiente:
    """Fija la base de E y el mapeo de cada especie a esa base."""
    base_E: List[str]
    indice_metabolito: Dict[str, int]
    interfaz_por_especie: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @property
    def m(self) -> int:
        return len(self.base_E)


def construir_mapa(modelo: cobra.Model) -> MapaEspacioAmbiente:
    base_E = espacio_ambiente_E(modelo)

    # BUG CORREGIDO: antes se construia indice_metabolito recortando el
    # prefijo "R_EX_" del id de la reaccion (p.ej. "R_EX_camp_e" -> "camp_e"),
    # pero el metabolito real tiene el prefijo "M_" (id "M_camp_e"). Como
    # nunca coincidian, el vector v en matriz_en_E() quedaba siempre en cero,
    # R_i(epsilon) resultaba vacio, y por tanto lambda = (0,...,0) y
    # codim = 0 en absolutamente todos los casos. Se corrige construyendo el
    # indice directamente desde el metabolito asociado a cada reaccion de
    # intercambio, sin manipulacion de strings.
    indice_metabolito: Dict[str, int] = {}
    reacciones_sin_metabolito_e = []
    for i, rxn_id in enumerate(base_E):
        rxn = modelo.reactions.get_by_id(rxn_id)
        mets_e = [m.id for m in rxn.metabolites if m.id.endswith("_e")]
        if len(mets_e) == 1:
            indice_metabolito[mets_e[0]] = i
        else:
            reacciones_sin_metabolito_e.append(rxn_id)

    if reacciones_sin_metabolito_e:
        print(f"  AVISO: {len(reacciones_sin_metabolito_e)} reacciones de intercambio "
              f"no tienen exactamente un metabolito '_e' asociado y quedaron fuera "
              f"del indice (revisar manualmente si la cifra es alta): "
              f"{reacciones_sin_metabolito_e[:5]}{'...' if len(reacciones_sin_metabolito_e) > 5 else ''}")
    print(f"  {len(indice_metabolito)}/{len(base_E)} metabolitos de intercambio indexados correctamente")

    mapa = MapaEspacioAmbiente(base_E=base_E, indice_metabolito=indice_metabolito)
    for especie in SPECIES:
        mapa.interfaz_por_especie[especie] = reacciones_interfaz_especie(modelo, especie)
        n_interfaz = len(mapa.interfaz_por_especie[especie])
        n_mapeadas = sum(1 for met in mapa.interfaz_por_especie[especie] if met in indice_metabolito)
        print(f"  {especie}: {n_interfaz} reacciones de interfaz con E "
              f"({n_mapeadas} mapeadas a una coordenada de E)")
    return mapa


# ---------------------------------------------------------------------------
# 2. Muestreo de flujos (OptGP) por condicion ambiental
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parche: evita el error de Windows "archivo de paginacion demasiado pequeno"
# ---------------------------------------------------------------------------

def _evitar_memoria_compartida_innecesaria():
    """
    Monkeypatch de cobra.sampling.hr_sampler.shared_np_array: sustituye los
    arreglos de memoria compartida (multiprocessing.Array, respaldados por
    el archivo de paginacion de Windows) por arreglos numpy normales.

    Por que es seguro: la memoria compartida solo tiene sentido cuando
    varios procesos necesitan leer/escribir el mismo arreglo (--processes
    > 1). Con --processes=1 todo ocurre en un unico proceso, asi que un
    arreglo numpy comun es funcionalmente identico y evita por completo la
    reserva de memoria compartida que dispara OSError [WinError 1455] en
    Windows con modelos grandes (~20,000 reacciones).

    Esta funcion solo se llama cuando processes=1 (ver muestrear_flujos).
    Si tu version de cobra cambio la firma interna de shared_np_array y
    este parche falla, usa --force-shared-memory para desactivarlo y
    recurre a las alternativas de entorno (WSL2 o Google Colab) descritas
    en el docstring del modulo.
    """
    import cobra.sampling.hr_sampler as hr_mod

    def _shared_np_array_local(shape, data=None, shared=False, *_, **__):
        arr = np.zeros(shape)
        if data is not None:
            arr[:] = np.asarray(data).reshape(shape)
        return arr

    hr_mod.shared_np_array = _shared_np_array_local
    print("  parche aplicado: OptGPSampler usara arreglos numpy normales "
          "(no memoria compartida) ya que --processes=1")


def _muestrear_con_progreso(sampler: OptGPSampler, n_samples: int, batch_size: int = 50):
    """
    Llama a sampler.sample() en lotes pequenos en vez de una sola llamada
    bloqueante, imprimiendo progreso y tiempo estimado restante entre
    lotes. Esto resuelve el problema practico de que un muestreo largo
    (minutos a horas con GLPK en un modelo de ~20,000 reacciones) parezca
    "colgado" sin ninguna señal de que sigue trabajando -- que es lo que
    lleva a interrumpirlo (Ctrl+C o timeout de Colab) antes de que termine.

    Si el proceso se interrumpe a mitad de camino, las muestras ya
    recolectadas en lotes anteriores se conservan (se imprimen antes de
    relanzar la interrupcion), en vez de perderse todas.
    """
    import time
    import pandas as pd

    lotes = []
    n_restantes = n_samples
    inicio = time.time()
    n_completadas = 0

    try:
        while n_restantes > 0:
            n_este_lote = min(batch_size, n_restantes)
            t0 = time.time()
            lote = sampler.sample(n_este_lote)
            dt = time.time() - t0
            lotes.append(lote)
            n_completadas += n_este_lote
            n_restantes -= n_este_lote

            transcurrido = time.time() - inicio
            ritmo = n_completadas / transcurrido  # muestras por segundo
            restante_estimado = n_restantes / ritmo if ritmo > 0 else float("nan")
            print(f"    lote de {n_este_lote} muestras en {dt:.1f}s "
                  f"({n_completadas}/{n_samples} completadas, "
                  f"~{restante_estimado/60:.1f} min restantes estimados)")
    except KeyboardInterrupt:
        print(f"\n  AVISO: muestreo interrumpido manualmente en {n_completadas}/{n_samples} "
              f"muestras. Se conservan las {n_completadas} muestras ya recolectadas.")
        if not lotes:
            raise
    return pd.concat(lotes, ignore_index=True)


def muestrear_flujos_random_fba(
    modelo: cobra.Model,
    n_samples: int = 2000,
    fraccion_optimo: float = 0.9,
    columnas_relevantes: List[str] | None = None,
    seed: int = 0,
    batch_size_reporte: int = 50,
    condicion: Dict[str, tuple] | None = None,
) -> "pandas.DataFrame":
    """
    Alternativa APROXIMADA y computacionalmente barata a OptGPSampler:
    en vez de muestrear el politopo de flujos de forma exacta (lo cual
    exige generar puntos de calentamiento tipo FVA -- ~2 x numero de
    reacciones resoluciones LP, del orden de 40,000+ para este modelo,
    independiente de --thinning/--n-samples), se resuelve un FBA distinto
    por muestra con un objetivo lineal ALEATORIO sobre las reacciones de
    interfaz, manteniendo community_growth >= fraccion_optimo * optimo.

    Costo: exactamente UNA resolucion LP por muestra devuelta -- sin
    calentamiento, sin thinning. Esto es lo que nos desbloquea cuando
    GLPK es demasiado lento para las operaciones tipo FVA que exige
    OptGPSampler/find_blocked_reactions a esta escala.

    `condicion`, si se especifica, sobreescribe las cotas de reacciones de
    intercambio especificas para representar una condicion ambiental
    concreta (Seccion 4.5, Fase A) -- p.ej. la restriccion de reacciones
    caracteristicas de un tier de salinidad distinto al que se esta
    simulando (ver construir_condiciones_salinidad). Se aplica y revierte
    automaticamente al salir de este bloque (bloque `with modelo:`).

    LIMITACION IMPORTANTE (reportar si estos datos llegan al manuscrito):
    este metodo NO da una muestra uniforme del politopo de flujos como
    OptGP/ACHR -- los optimos de FBA tienden a caer en vertices o caras
    del politopo, no en su interior, asi que subestima la verdadera
    variabilidad e infla la aparente correlacion entre muestras de un
    mismo objetivo aleatorio. Es apropiado para (a) verificar que el
    resto del pipeline (Seccion 4.3, Algoritmo 1) funciona de punta a
    punta, y (b) una primera exploracion, pero la Fase A tal como se
    describe en el manuscrito (Seccion 4.5) asume muestreo HR real
    (OptGP/ACHR); si estos resultados se usan en el paper, esto debe
    declararse explicitamente como limitacion en la Seccion de Discusion.
    """
    rng = np.random.default_rng(seed)
    columnas_a_usar = columnas_relevantes or [r.id for r in modelo.reactions]
    columnas_presentes = [c for c in columnas_a_usar if c in [r.id for r in modelo.reactions]]

    with modelo:
        if condicion:
            for rxn_id, (lb, ub) in condicion.items():
                if rxn_id in modelo.reactions:
                    modelo.reactions.get_by_id(rxn_id).bounds = (lb, ub)
            print(f"  condicion ambiental aplicada: {len(condicion)} reacciones con cotas modificadas")

        print(f"  solver activo: {type(modelo.solver).__module__}")
        solucion = modelo.optimize()
        if solucion.status != "optimal":
            raise RuntimeError(f"FBA no factible bajo esta condicion (status={solucion.status})")
        f_opt = solucion.objective_value
        modelo.reactions.community_growth.lower_bound = fraccion_optimo * f_opt
        print(f"  optimo de community_growth = {f_opt:.4f}; cota inferior fijada en "
              f"{fraccion_optimo:.0%} de ese valor -- {n_samples} resoluciones LP, "
              f"una por muestra, sin calentamiento")

        filas = []
        rxns_interfaz = [modelo.reactions.get_by_id(c) for c in columnas_presentes]
        import time
        inicio = time.time()
        for t in range(n_samples):
            coeffs = rng.standard_normal(len(rxns_interfaz))
            modelo.objective = {rxn: float(c) for rxn, c in zip(rxns_interfaz, coeffs)}
            modelo.objective_direction = "max" if rng.random() < 0.5 else "min"
            sol = modelo.optimize()
            if sol.status == "optimal":
                filas.append(sol.fluxes.reindex(columnas_presentes).to_numpy())
            else:
                filas.append(np.full(len(columnas_presentes), np.nan))

            if (t + 1) % batch_size_reporte == 0 or (t + 1) == n_samples:
                dt = time.time() - inicio
                ritmo = (t + 1) / dt
                restante = (n_samples - (t + 1)) / ritmo if ritmo > 0 else float("nan")
                print(f"    {t + 1}/{n_samples} muestras ({dt:.1f}s transcurridos, "
                      f"~{restante/60:.1f} min restantes estimados)")

    import pandas as pd
    return pd.DataFrame(filas, columns=columnas_presentes)


def muestrear_flujos(
    modelo: cobra.Model,
    n_samples: int = 2000,
    thinning: int = 100,
    fraccion_optimo: float = 0.9,
    condicion: Dict[str, tuple] | None = None,
    seed: int = 0,
    processes: int = 1,
    columnas_relevantes: List[str] | None = None,
    forzar_memoria_compartida: bool = False,
    tamano_lote: int = 50,
) -> "pandas.DataFrame":
    """
    Muestrea el politopo de flujos factibles del modelo comunitario
    (Seccion 4.5, Fase A), fijando primero una cota inferior sobre
    community_growth a una fraccion del optimo, como es estandar en
    muestreo de flujos (Gelbach2024).

    `condicion` permite sobreescribir cotas de reacciones de intercambio
    especificas para representar un punto del gradiente de salinidad
    (p.ej. disponibilidad de glicerol/DHA); se aplica y se revierte
    automaticamente al salir de este bloque.

    `processes` se pasa directamente a OptGPSampler (paralelizacion nativa
    de cobra para la etapa de muestreo -- la mas costosa en tiempo de todo
    el script). En Windows, cobra maneja internamente los guardas de
    multiprocessing necesarios; no se requiere configuracion adicional
    mas alla de correr el script con `if __name__ == "__main__":` (ya
    presente en este archivo).

    `columnas_relevantes`, si se especifica, recorta el DataFrame resultante
    a solo esas columnas antes de devolverlo -- reduce el uso de memoria de
    ~20,784 columnas a las ~2,000 que realmente se usan en el resto del
    script (Seccion 4.1: E y las interfaces tex/texi por especie).
    """
    with modelo:
        if condicion:
            for rxn_id, (lb, ub) in condicion.items():
                modelo.reactions.get_by_id(rxn_id).bounds = (lb, ub)

        print(f"  solver activo: {type(modelo.solver).__module__}")
        try:
            solucion = modelo.optimize()
        except Exception:
            import traceback
            print("\nERROR durante la optimizacion (FBA) -- traceback completo:")
            print(traceback.format_exc())
            raise RuntimeError(
                "La optimizacion fallo -- ver traceback impreso arriba. Si menciona "
                "Gurobi y 'size-limited license', el solver activo sigue siendo Gurobi: "
                "vuelve a correr con --solver glpk explicitamente (revisa arriba que "
                "aparezca la linea 'configurando solver: glpk'), o solicita una licencia "
                "academica sin restriccion en gurobi.com/academia."
            ) from None

        if solucion.status != "optimal":
            raise RuntimeError(f"FBA no factible bajo esta condicion (status={solucion.status})")
        f_opt = solucion.objective_value
        modelo.reactions.community_growth.lower_bound = fraccion_optimo * f_opt

        print(f"  optimo de community_growth = {f_opt:.4f}; "
              f"muestreando con cota inferior {fraccion_optimo:.0%} de ese valor")
        print(f"  paralelizando muestreo OptGP con processes={processes}, thinning={thinning} "
              f"(cada muestra = {thinning} resoluciones LP internas)")
        if processes == 1 and not forzar_memoria_compartida:
            _evitar_memoria_compartida_innecesaria()
        try:
            import time
            print("  construyendo sampler (esto incluye generar puntos de calentamiento "
                  "tipo FVA sobre el modelo completo -- puede ser el paso mas lento de todos, "
                  "independiente de --thinning/--batch-size/--n-samples)...")
            t0 = time.time()
            sampler = OptGPSampler(modelo, processes=processes, thinning=thinning, seed=seed)
            print(f"  sampler construido en {time.time() - t0:.1f}s -- iniciando muestreo...")
            muestras = _muestrear_con_progreso(sampler, n_samples, batch_size=tamano_lote)
        except Exception:
            import traceback
            print("\nERROR durante el muestreo OptGP -- traceback completo:")
            print(traceback.format_exc())
            raise

    if columnas_relevantes is not None:
        cols_presentes = [c for c in columnas_relevantes if c in muestras.columns]
        antes_mb = muestras.memory_usage(deep=True).sum() / 1e6
        muestras = muestras[cols_presentes].copy()
        despues_mb = muestras.memory_usage(deep=True).sum() / 1e6
        print(f"  recorte de columnas: {antes_mb:.1f} MB -> {despues_mb:.1f} MB "
              f"({len(cols_presentes)}/{len(columnas_relevantes)} columnas relevantes encontradas)")

    return muestras


# ---------------------------------------------------------------------------
# 3. Construccion de P_i (subespacio muestreado) por especie -- VECTORIZADO
# ---------------------------------------------------------------------------

def matriz_en_E(muestras: "pandas.DataFrame", mapa: MapaEspacioAmbiente, especie: str) -> np.ndarray:
    """
    Version vectorizada de la proyeccion a E: en lugar de recorrer fila por
    fila con .iterrows() (lento y costoso en memoria por el overhead de
    convertir cada fila a un objeto Python), construye de una sola vez la
    matriz (n_samples, m) de flujos de la especie en la base de E mediante
    indexado numpy.
    """
    interfaz = mapa.interfaz_por_especie[especie]
    n_samples = len(muestras)
    V = np.zeros((n_samples, mapa.m))
    cols, idxs = [], []
    for met_ext, rxn_id in interfaz.items():
        idx = mapa.indice_metabolito.get(met_ext)
        if idx is not None and rxn_id in muestras.columns:
            cols.append(rxn_id)
            idxs.append(idx)
    if cols:
        V[:, idxs] = muestras[cols].to_numpy()
    return V


# ---------------------------------------------------------------------------
# 4. Algoritmo 1 del manuscrito: asignacion de celda de Schubert -- VECTORIZADO
# ---------------------------------------------------------------------------

def construir_bandera(mapa: MapaEspacioAmbiente,
                       bandera_taxones: Dict[str, List[str]],
                       excluir_especie: str | None = None,
                       verbose: bool = True) -> List[np.ndarray]:
    """
    Construye una bandera COMPLETA F_1 subset F_2 subset ... subset F_m = E,
    con dim(F_j) = j EXACTAMENTE para todo j -- tal como exige la definicion
    formal de bandera completa (Seccion 4 del manuscrito) y como asume el
    Algoritmo 1 (la formula de asignacion de lambda solo es valida si
    dim(F_j) crece de a lo mas 1 por escalon).

    CORRECCION IMPORTANTE (circularidad): BANDERA_SANTA_POLA usa como
    "especies dominantes por estanque" los nombres de las mismas 10
    especies del GSMM comunitario que se estan clasificando -- porque no
    tenemos datos de abundancia de Santa Pola independientes, solo esta
    aproximacion basada en lo que reportan los papers en prosa (ver
    conversacion). Esto significa que, sin correccion, al evaluar la
    especie X la bandera se construye en parte con las propias reacciones
    de X (via mapa.interfaz_por_especie[X]) -- una comparacion circular:
    X coincide con la bandera casi por definicion, no por señal biologica.
    Esto produjo el patron degenerado observado (codimension ~5000 de
    ~5340 para todas las especies, particiones que arrancan casi en el
    maximo).

    `excluir_especie`, si se especifica, remueve esa especie de la lista
    de especies dominantes en TODOS los estanques antes de construir la
    bandera -- asi la especie evaluada nunca se compara contra si misma
    (leave-one-out). Debe llamarse una vez POR ESPECIE, pasando esa misma
    especie como excluir_especie, en vez de compartir una unica bandera
    global entre las 10 especies.

    NOTA: esto mitiga la circularidad usando los datos disponibles, pero
    no la elimina del todo -- las reacciones compartidas entre X y OTRAS
    especies dominantes siguen entrando en la bandera de X a traves de
    esas otras especies (legitimo), pero la bandera sigue derivada del
    mismo GSMM comunitario, no de datos de Santa Pola verdaderamente
    independientes. Si se consiguen datos reales de abundancia/anotacion
    funcional de Santa Pola, deberian reemplazar por completo a
    BANDERA_SANTA_POLA y esta funcion podria simplificarse.
    """
    especies_vistas: set = set()
    especies_nuevas_por_tier = []
    for pond, especies in bandera_taxones.items():
        especies_filtradas = [e for e in especies if e != excluir_especie]
        nuevas = [e for e in especies_filtradas if e not in especies_vistas]
        especies_nuevas_por_tier.append((pond, nuevas))
        especies_vistas.update(especies_filtradas)

    tier_de_reaccion: Dict[int, int] = {}
    for tier_idx, (_, especies_nuevas) in enumerate(especies_nuevas_por_tier, start=1):
        for especie in especies_nuevas:
            interfaz = mapa.interfaz_por_especie.get(especie, {})
            for met_ext in interfaz:
                idx = mapa.indice_metabolito.get(met_ext)
                if idx is not None and idx not in tier_de_reaccion:
                    tier_de_reaccion[idx] = tier_idx

    # criterio secundario: cuantas especies (excluyendo la evaluada) comparten
    # cada reaccion -- mayor conteo = mas "central", entra primero en su tier
    conteo_especies_por_reaccion = np.zeros(mapa.m, dtype=int)
    for especie, interfaz in mapa.interfaz_por_especie.items():
        if especie == excluir_especie:
            continue
        for met_ext in interfaz:
            idx = mapa.indice_metabolito.get(met_ext)
            if idx is not None:
                conteo_especies_por_reaccion[idx] += 1

    indices_con_tier = sorted(
        tier_de_reaccion.keys(),
        key=lambda idx: (tier_de_reaccion[idx], -conteo_especies_por_reaccion[idx], idx)
    )
    indices_sin_tier = sorted(set(range(mapa.m)) - set(tier_de_reaccion.keys()))
    orden_completo = indices_con_tier + indices_sin_tier

    banderas = []
    mask = np.zeros(mapa.m, dtype=bool)
    for idx in orden_completo:
        mask = mask.copy()
        mask[idx] = True
        banderas.append(mask)

    if verbose:
        etiqueta = f" (excluyendo {excluir_especie})" if excluir_especie else ""
        for tier_idx, (pond, _) in enumerate(especies_nuevas_por_tier, start=1):
            dim_acumulada = sum(1 for i in indices_con_tier if tier_de_reaccion[i] <= tier_idx)
            print(f"  F_{pond}{etiqueta}: dim acumulada = {dim_acumulada} de {mapa.m} totales")
        print(f"  bandera completa{etiqueta}: {len(banderas)} escalones, "
              f"{len(indices_sin_tier)} reacciones sin especie dominante asociada al final")

    return banderas


def construir_condiciones_salinidad(
    mapa: MapaEspacioAmbiente,
    bandera_taxones: Dict[str, List[str]],
    factor_restriccion: float = 0.1,
) -> Dict[str, Dict[str, tuple]]:
    """
    Construye las cotas de reacciones de intercambio que representan las
    DOS condiciones ambientales de interes (Seccion 4.5, Fase A): SS19 y
    SS37. Necesarias para poder muestrear flujos POR CONDICION, en vez de
    una sola vez bajo condiciones basales -- sin esto, la regresion
    logistica ordinal sobre s (Seccion 4.5) no tiene datos que ajustar.

    Diseño: bajo la condicion "SS19", se restringen (a `factor_restriccion`
    de su cota original) las reacciones de intercambio asociadas a T_2
    (genros que crecen hacia SS37 -- Haloquadratum, salinibacter), 
    simulando que esos recursos son relativamente escasos en un estanque
    de salinidad moderada; bajo "SS37", se restringen las de T_1
    (Halorubrum, Natronomonas). Esto ata la condicion ambiental
    directamente a la misma evidencia real de Fernandez2014a usada para
    construir la bandera (Seccion 4.2), en vez de inventar una quimica
    ambiental que no tenemos datos para especificar con precision --
    una eleccion de modelado que debe declararse explicitamente, no un
    hecho medido.
    """
    especies_vistas: set = set()
    especies_nuevas_por_tier = []
    for pond, especies in bandera_taxones.items():
        nuevas = [e for e in especies if e not in especies_vistas]
        especies_nuevas_por_tier.append((pond, nuevas))
        especies_vistas.update(especies)

    if len(especies_nuevas_por_tier) != 2:
        raise ValueError("construir_condiciones_salinidad asume exactamente 2 tiers "
                          "(SS19, SS37); BANDERA_SANTA_POLA cambio de estructura.")

    (pond1, especies_t1), (pond2, especies_t2) = especies_nuevas_por_tier

    def reacciones_de(especies):
        ids = set()
        for especie in especies:
            ids.update(mapa.interfaz_por_especie.get(especie, {}).values())
            # tambien las reacciones comunitarias EX_*_e correspondientes
            for met_ext in mapa.interfaz_por_especie.get(especie, {}):
                idx = mapa.indice_metabolito.get(met_ext)
                if idx is not None:
                    ids.add(mapa.base_E[idx])
        return ids

    rxns_t1 = reacciones_de(especies_t1)
    rxns_t2 = reacciones_de(especies_t2)

    condiciones = {}
    for pond_actual, rxns_a_restringir in [(pond1, rxns_t2), (pond2, rxns_t1)]:
        condicion = {}
        for rxn_id in rxns_a_restringir:
            condicion[rxn_id] = (-1000.0 * factor_restriccion, 1000.0 * factor_restriccion)
        condiciones[pond_actual] = condicion
        print(f"  condicion {pond_actual}: {len(condicion)} reacciones restringidas "
              f"a {factor_restriccion:.0%} de su cota original")

    return condiciones


def asignar_celdas_schubert_vectorizado(
    V: np.ndarray,
    bandera_masks: Sequence[np.ndarray],
    epsilon: float,
    r: int,
) -> Dict[str, np.ndarray]:
    """
    Version vectorizada del Algoritmo 1 (Seccion 4.3): procesa TODAS las
    muestras de una especie en un puñado de operaciones numpy, en vez de
    un bucle Python de tamaño n_samples. Sobre un modelo de miles de
    muestras esto es tipicamente 100-1000x mas rapido que el equivalente
    fila por fila, y evita por completo el overhead de memoria de crear
    n_samples objetos Python intermedios.

    Devuelve un diccionario con:
      'lambda' : matriz (n_samples, r) con la particion de cada muestra
      'codim'  : vector (n_samples,) con la codimension de cada muestra
      'n_activas' : numero de muestras con al menos una reaccion activa
    """
    A = np.abs(V) > epsilon  # (n_samples, m): R_i(epsilon) por muestra, vectorizado
    n_samples = V.shape[0]
    n = len(bandera_masks)  # numero de escalones de la bandera

    # d[:, 0] = 0 (por definicion); d[:, j] = dim(P_i cap F_j) para j=1..n
    d = np.zeros((n_samples, n + 1), dtype=int)
    for j, mask in enumerate(bandera_masks, start=1):
        d[:, j] = (A & mask).sum(axis=1)

    lam = np.zeros((n_samples, r), dtype=int)
    for i in range(1, r + 1):
        alcanzado = d >= i
        tiene_alguno = alcanzado.any(axis=1)
        # primer j donde d_j >= i; si ninguno alcanza i, usar n (por defecto)
        j_i = np.where(tiene_alguno, alcanzado.argmax(axis=1), n)
        lam[:, i - 1] = np.clip(n + i - j_i, 0, None)

    return {
        "lambda": lam,
        "codim": lam.sum(axis=1),
        "n_activas": int(A.any(axis=1).sum()),
    }


# ---------------------------------------------------------------------------
# 5. Cuantificacion de incertidumbre
# ---------------------------------------------------------------------------

def distribucion_empirica_lambda(lam_matrix: np.ndarray) -> Dict[tuple, float]:
    """p_hat_i(lambda) de la Seccion 4.5, Fase A. `lam_matrix` es la matriz
    (n_samples, r) devuelta por asignar_celdas_schubert_vectorizado."""
    from collections import Counter
    tuplas = map(tuple, lam_matrix)
    conteo = Counter(tuplas)
    total = len(lam_matrix)
    return {lam: c / total for lam, c in conteo.items()}


def bootstrap_ci_codim(codims: np.ndarray,
                        n_boot: int = 2000,
                        alpha: float = 0.05,
                        seed: int = 0) -> tuple:
    """
    Intervalo de confianza bootstrap sobre E[|lambda_i|]. Vectorizado:
    genera las n_boot remuestras de una sola vez (matriz n_boot x n_samples)
    en vez de un bucle Python de tamaño n_boot, evitando otro cuello de
    botella de tiempo de ejecucion para n_boot grandes.
    """
    rng = np.random.default_rng(seed)
    n = len(codims)
    indices_boot = rng.integers(0, n, size=(n_boot, n))
    medias_boot = codims[indices_boot].mean(axis=1)
    lo, hi = np.percentile(medias_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(codims.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Prueba de H1a: correlacion entre codimension y amplitud de tolerancia a NaCl
# ---------------------------------------------------------------------------

# Amplitud del rango de NaCl (%, w/v) reportado en descripciones taxonomicas
# originales de cepa tipo -- fuente INDEPENDIENTE tanto de Fernandez2014a
# (usado para construir la bandera) como del propio GSMM comunitario. Esta
# amplitud es, literalmente, la definicion fisiologica clasica de
# estenohalino (rango estrecho) vs euryhalino (rango amplio) -- mas fiel al
# termino que la proxy basada en abundancia de Wu2024salinity, que ademas
# resulto inaplicable a estas especies (Wu2024salinity es un estuario de
# salinidad baja-intermedia, no un ambiente hipersalino).
#
# CONFIANZA ALTA (descripcion original de especie o fuente muy especifica):
#   Chromohalobacter salexigens: 0.1-4 M NaCl (~0.6-23.4%), "uno de los
#     rangos mas amplios documentados en la naturaleza" (fuente: estudio de
#     adaptacion osmotica, PMC2553793).
#   Halomonas elongata: 0-32% sal solar, DESCRIPCION ORIGINAL DEL GENERO
#     (Vreeland et al. 1980, IJSEM).
#   Haloquadratum walsbyi: 14% hasta saturacion (~32%) (Bolhuis et al. 2006).
#   Salinibacter ruber: 20-30% optimo (Anton et al. 2013).
#   Halorubrum spp.: ~10-30% (cepas tipo representativas, p.ej. Hrr.
#     salinarum 10-30%, Hrr. halophilum 15-30%; Cui et al. 2022).
# CONFIANZA MEDIA (nivel genero/familia, NO especifico de la especie exacta
# del GSMM comunitario -- Haloarcula, Halobacterium y Haloferax comparten
# aqui el mismo valor aproximado, tomado del rango general reportado para
# Halobacteriaceae/Halobacteriales: minimo ~10%, hasta ~35% de salinidad
# (Matarredona et al. 2024)):
#   Haloarcula, Halobacterium, Haloferax: ~10-35% (aproximacion de familia).
# CONFIANZA BAJA (incompleto):
#   Tetragenococcus halophilus: tolerancia hasta 26% bien documentada, pero
#     no se encontro un minimo de crecimiento claramente establecido en la
#     busqueda realizada; se aproxima el rango como ~5-26%.
# SIN DATO:
#   Natronomonas: minimo 15%, optimo 20-25%, limite superior NO confirmado
#     -- se deja como None, no se asume un valor.
NACL_AMPLITUD_ESPECIES: Dict[str, float | None] = {
    "Chromohalobacter": 4.0 * 5.844 - 0.1 * 5.844,  # 0.1-4 M -> ~0.58-23.38%
    "Halomonas": 32.0 - 0.0,
    "Haloquadratum": 32.0 - 14.0,
    "salinibacter": 30.0 - 20.0,
    "Halorubrum": 30.0 - 10.0,
    "Haloarcula": 35.0 - 10.0,       # confianza media: aproximacion de familia
    "Halobacterium": 35.0 - 10.0,    # confianza media: aproximacion de familia
    "Haloferax": 35.0 - 10.0,        # confianza media: aproximacion de familia
    "Tetragenococcus": 26.0 - 5.0,   # confianza baja: minimo aproximado
    "Natronomonas": None,  # limite superior no confirmado -- excluida de la prueba
}

# Subconjunto de SOLO las especies con dato de CONFIANZA ALTA (descripcion
# original de especie, no aproximacion de genero/familia) -- usado para
# comparar contra el resultado con todas las especies disponibles y exponer
# honestamente si la conclusion depende de incluir datos de menor confianza.
NACL_AMPLITUD_ALTA_CONFIANZA: Dict[str, float | None] = {
    e: v for e, v in NACL_AMPLITUD_ESPECIES.items()
    if e in {"Chromohalobacter", "Halomonas", "Haloquadratum", "salinibacter", "Halorubrum"}
}


def prueba_H1a_correlacion(
    codim_medio_por_especie: Dict[str, float],
    amplitud_por_especie: Dict[str, float | None] = NACL_AMPLITUD_ESPECIES,
    n_perm: int = 10000,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Prueba H1a adaptada: en lugar de la formulacion original (dominancia
    estocastica entre dos categorias estenohalino/euryhalino de
    Wu2024salinity, que resulto no aplicable a estas especies -- ver
    discusion), se usa una version continua: correlacion de Spearman entre
    la codimension media de Schubert y la amplitud del rango de NaCl
    reportado en la literatura taxonomica (NACL_AMPLITUD_ESPECIES), con
    p-valor por permutacion (apropiado dado el n muy pequeno, sin asumir
    normalidad).

    IMPORTANTE: con n=3 especies validas (Natronomonas se excluye por falta
    de dato de limite superior), esta prueba tiene practicamente nulo poder
    estadistico. Se reporta de forma exploratoria, no confirmatoria --
    ver Seccion de Discusion del manuscrito.
    """
    from scipy.stats import spearmanr

    especies_validas = [e for e, a in amplitud_por_especie.items() if a is not None
                        and e in codim_medio_por_especie]
    if len(especies_validas) < 3:
        return {"n": len(especies_validas), "rho": None, "p_valor": None,
                "aviso": "Menos de 3 especies con dato valido -- prueba no informativa."}

    x = np.array([amplitud_por_especie[e] for e in especies_validas])
    y = np.array([codim_medio_por_especie[e] for e in especies_validas])

    rho_obs, _ = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    n = len(x)
    rhos_perm = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        rhos_perm[i], _ = spearmanr(x, y_perm)
    p_valor = float(np.mean(np.abs(rhos_perm) >= np.abs(rho_obs)))

    return {
        "n": n,
        "especies": especies_validas,
        "rho": float(rho_obs),
        "p_valor": p_valor,
        "aviso": ("n < 5: resultado exploratorio, poder estadistico muy bajo; "
                  "no interpretar un p-valor no significativo como evidencia de H0."),
    }


# ---------------------------------------------------------------------------
# Fase B: benchmark contra un clasificador supervisado (Seccion 4.5)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fase C: homologia persistente (Seccion 4.5)
# ---------------------------------------------------------------------------

def persistencia_h0(nube_puntos: np.ndarray) -> np.ndarray:
    """
    Respaldo sin ripser: homologia persistente en dimension 0 (componentes
    conexas) via enlace simple (identica a la usada en
    estudio_validacion_sintetica.py). Exacta para H_0, no cubre H_1.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    distancias = pdist(nube_puntos, metric="euclidean")
    Z = linkage(distancias, method="single")
    return Z[:, 2]


def ejecutar_fase_c(
    muestras_por_condicion: Dict[str, "pandas.DataFrame"],
    mapa: "MapaEspacioAmbiente",
    especies_validas: List[str],
    epsilon: float = 1e-6,
) -> Dict[str, object]:
    """
    Fase C (Seccion 4.5): para cada especie validada, calcula el diagrama
    de persistencia (H0 y H1, via ripser si esta disponible; solo H0 via
    scipy si no) de la nube de muestras de flujo (ambas condiciones
    combinadas), y lo compara contra la codimension de Schubert de esas
    mismas muestras y contra la amplitud de tolerancia a NaCl (el mismo
    Y_i externo usado en H1a, Seccion 4.4).

    Requiere ripser para H1 (ciclos): pip install ripser
    Requiere ademas persim para distancia bottleneck: pip install persim
    Sin ripser, cae automaticamente a la aproximacion H0-only (enlace
    simple via scipy) usada en el estudio de validacion sintetica.
    """
    try:
        from ripser import ripser as ripser_fn
        tiene_ripser = True
        print("  usando ripser para H0 y H1 (ciclos)")
    except ImportError:
        tiene_ripser = False
        print("  AVISO: ripser no disponible -- calculando solo H0 via scipy "
              "(enlace simple). Instala con: pip install ripser")

    try:
        from persim import bottleneck
        tiene_persim = True
    except ImportError:
        tiene_persim = False
        if tiene_ripser:
            print("  AVISO: persim no disponible -- se omite distancia bottleneck "
                  "entre especies. Instala con: pip install persim")

    resultados = {}
    for especie in especies_validas:
        Vs = [matriz_en_E(muestras, mapa, especie) for muestras in muestras_por_condicion.values()]
        V_completo = np.vstack(Vs)

        if tiene_ripser:
            resultado_ripser = ripser_fn(V_completo, maxdim=1)
            dgms = resultado_ripser["dgms"]
            h0, h1 = dgms[0], dgms[1]
            h0_finito = h0[np.isfinite(h0[:, 1])]
            persistencia_h0_total = float((h0_finito[:, 1] - h0_finito[:, 0]).sum())
            persistencia_h1_total = float((h1[:, 1] - h1[:, 0]).sum()) if len(h1) > 0 else 0.0
            n_ciclos_h1 = int(len(h1))
        else:
            tiempos_muerte = persistencia_h0(V_completo)
            dgms = None
            persistencia_h0_total = float(tiempos_muerte.sum())
            persistencia_h1_total = None
            n_ciclos_h1 = None

        bandera_especie = construir_bandera(mapa, BANDERA_SANTA_POLA,
                                             excluir_especie=especie, verbose=False)
        codim = asignar_celdas_schubert_vectorizado(
            V_completo, bandera_especie, epsilon, r=10)["codim"]

        resultados[especie] = {
            "persistencia_h0": persistencia_h0_total,
            "persistencia_h1": persistencia_h1_total,
            "n_ciclos_h1": n_ciclos_h1,
            "codim_media": float(codim.mean()),
            "dgms": dgms,
        }
        h1_str = f"{persistencia_h1_total:.3f} ({n_ciclos_h1} ciclos)" \
            if persistencia_h1_total is not None else "N/D (ripser no disponible)"
        print(f"  {especie}: persistencia H0={persistencia_h0_total:.3f}, "
              f"H1={h1_str}, codim media={codim.mean():.3f}")

    # Correlacion entre resumenes de persistencia y amplitud de NaCl (mismo
    # Y_i externo de H1a, Seccion 4.4), analoga a prueba_H1a_correlacion
    # pero usando persistencia en vez de codimension.
    amplitud_valida = {e: a for e, a in NACL_AMPLITUD_ESPECIES.items()
                       if a is not None and e in resultados}
    if len(amplitud_valida) >= 3:
        especies_amp = list(amplitud_valida.keys())
        x = np.array([amplitud_valida[e] for e in especies_amp])
        y_h0 = np.array([resultados[e]["persistencia_h0"] for e in especies_amp])
        from scipy.stats import spearmanr
        rho_h0, p_h0 = spearmanr(x, y_h0)
        print(f"\n  correlacion persistencia H0 vs. amplitud NaCl (n={len(especies_amp)}): "
              f"rho={rho_h0:.3f}, p={p_h0:.4f}")
        if tiene_ripser:
            y_h1 = np.array([resultados[e]["persistencia_h1"] for e in especies_amp])
            rho_h1, p_h1 = spearmanr(x, y_h1)
            print(f"  correlacion persistencia H1 vs. amplitud NaCl (n={len(especies_amp)}): "
                  f"rho={rho_h1:.3f}, p={p_h1:.4f}")

    # Distancia bottleneck entre pares de especies (solo H1, si disponible)
    if tiene_ripser and tiene_persim:
        print("\n  distancia bottleneck (H1) entre pares de especies:")
        for i, e1 in enumerate(especies_validas):
            for e2 in especies_validas[i + 1:]:
                d = bottleneck(resultados[e1]["dgms"][1], resultados[e2]["dgms"][1])
                print(f"    {e1} vs {e2}: {d:.3f}")

    return resultados


def ejecutar_fase_b(
    muestras_por_condicion: Dict[str, "pandas.DataFrame"],
    mapa: "MapaEspacioAmbiente",
    especies_validas: List[str],
    tier_por_especie: Dict[str, int],
    epsilon: float = 1e-6,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Fase B (Seccion 4.5): compara un clasificador supervisado convencional
    (Random Forest) contra un clasificador trivial basado solo en la
    codimension de Schubert, ambos prediciendo la etiqueta de tier
    (SS19=0, SS37=1) a partir de las muestras de flujo.

    DISEÑO: con solo 4 especies validadas (2 por tier), se usa validacion
    cruzada dejando UNA ESPECIE FUERA en cada pliegue (leave-one-species-out,
    no k-fold estandar sobre las muestras individuales) -- de lo contrario,
    el clasificador podria simplemente aprender a reconocer la identidad de
    cada especie a partir de patrones de flujo caracteristicos, en vez de
    aprender el patron real asociado al tier, y el resultado de validacion
    seria optimista de forma artificial (fuga de informacion por especie).

    ADVERTENCIA: con solo 4 especies (4 pliegues), este es un diseño de
    validacion extremadamente pequeño -- los resultados deben reportarse
    como exploratorios, no como una comparacion de desempeño confiable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score

    # Construir features (muestras de flujo, ambas condiciones) y labels
    X_por_especie = {}
    for especie in especies_validas:
        Vs = []
        for pond, muestras in muestras_por_condicion.items():
            V = matriz_en_E(muestras, mapa, especie)
            Vs.append(V)
        X_por_especie[especie] = np.vstack(Vs)

    y_true_todas, y_pred_ml_todas, y_pred_sch_todas, grupos_todas = [], [], [], []

    for especie_test in especies_validas:
        especies_train = [e for e in especies_validas if e != especie_test]

        X_train = np.vstack([X_por_especie[e] for e in especies_train])
        y_train = np.concatenate([
            np.full(len(X_por_especie[e]), tier_por_especie[e]) for e in especies_train
        ])
        X_test = X_por_especie[especie_test]
        y_test = np.full(len(X_test), tier_por_especie[especie_test])

        # Clasificador ML: Random Forest sobre las muestras de flujo crudas
        clf = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred_ml = clf.predict(X_test)

        # Clasificador trivial: umbral sobre la codimension de Schubert.
        # El umbral se elige por CV interna sobre especies_train (no usa
        # especie_test), maximizando exactitud balanceada en ese subconjunto.
        bandera_test = construir_bandera(mapa, BANDERA_SANTA_POLA,
                                          excluir_especie=especie_test, verbose=False)
        codim_test = asignar_celdas_schubert_vectorizado(
            X_test, bandera_test, epsilon, r=10)["codim"]

        # CORRECCION IMPORTANTE: buscar el umbral en AMBAS direcciones (codim
        # alta = tier 1, o codim baja = tier 1) -- con una sola direccion fija,
        # si la relacion real resulta invertida, el clasificador queda
        # sistematicamente invertido y la exactitud balanceada colapsa a 0 en
        # vez de a ~0.5 (bug verificado empiricamente en el estudio sintetico,
        # ver estudio_validacion_sintetica.py).
        mejores_umbral, mejor_direccion, mejor_score = None, None, -1
        for especie_val in especies_train:
            bandera_val = construir_bandera(mapa, BANDERA_SANTA_POLA,
                                             excluir_especie=especie_val, verbose=False)
            codim_val = asignar_celdas_schubert_vectorizado(
                X_por_especie[especie_val], bandera_val, epsilon, r=10)["codim"]
            for umbral_candidato in np.percentile(codim_val, np.arange(10, 100, 10)):
                for direccion in (">", "<"):
                    pred = (codim_val > umbral_candidato).astype(int) if direccion == ">" \
                        else (codim_val < umbral_candidato).astype(int)
                    score = balanced_accuracy_score(
                        np.full(len(pred), tier_por_especie[especie_val]), pred)
                    if score > mejor_score:
                        mejor_score, mejores_umbral, mejor_direccion = score, umbral_candidato, direccion

        y_pred_sch = (codim_test > mejores_umbral).astype(int) if mejor_direccion == ">" \
            else (codim_test < mejores_umbral).astype(int)

        y_true_todas.append(y_test)
        y_pred_ml_todas.append(y_pred_ml)
        y_pred_sch_todas.append(y_pred_sch)
        grupos_todas.append(np.full(len(y_test), especie_test))

    y_true = np.concatenate(y_true_todas)
    y_pred_ml = np.concatenate(y_pred_ml_todas)
    y_pred_sch = np.concatenate(y_pred_sch_todas)

    return {
        "ml_exactitud_balanceada": balanced_accuracy_score(y_true, y_pred_ml),
        "ml_f1_macro": f1_score(y_true, y_pred_ml, average="macro"),
        "schubert_exactitud_balanceada": balanced_accuracy_score(y_true, y_pred_sch),
        "schubert_f1_macro": f1_score(y_true, y_pred_sch, average="macro"),
        "n_especies": len(especies_validas),
    }


def regresion_logistica_ordinal(codims: np.ndarray, salinidad: np.ndarray, n_categorias: int = 3):
    """
    Regresion logistica ordinal de |lambda| sobre la salinidad (Seccion 4.5,
    Fase A): logit(P(|lambda| <= k | s)) = theta_k - beta * s.
    Requiere statsmodels (no incluido por defecto; instalar en tu entorno).

    ADAPTACION IMPORTANTE respecto a la formula literal del manuscrito: esta
    asume k = 0,...,|lambda|_max-1 categorias, es decir, tantas categorias
    como valores distintos de codimension existan. En la practica, la
    codimension observada toma miles de valores posibles (rango ~3000-5300
    en este trabajo) -- OrderedModel no es utilizable ni estadisticamente
    sensato con esa cardinalidad. Se agrupa la codimension en
    `n_categorias` terciles (por defecto 3: baja/media/alta) antes de
    ajustar, preservando el espiritu de la regresion ordinal (usar el orden,
    no la magnitud exacta) de forma computacionalmente viable. Esto debe
    declararse en el manuscrito si estos resultados se reportan.
    """
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Este paso requiere statsmodels y pandas. Instala con: pip install statsmodels"
        ) from exc

    codims_cat = pd.qcut(codims, q=n_categorias, labels=False, duplicates="drop")
    if len(np.unique(codims_cat)) < 2:
        return {"beta": None, "p_valor": None,
                "aviso": "Menos de 2 categorias distintas tras el binning -- codimension "
                         "casi constante, regresion no informativa."}

    modelo_ordinal = OrderedModel(codims_cat, salinidad.reshape(-1, 1), distr="logit")
    resultado = modelo_ordinal.fit(method="bfgs", disp=False)
    beta = float(resultado.params.iloc[-1]) if hasattr(resultado.params, "iloc") else float(resultado.params[-1])
    p_valor = float(resultado.pvalues.iloc[-1]) if hasattr(resultado.pvalues, "iloc") else float(resultado.pvalues[-1])
    return {"beta": beta, "p_valor": p_valor, "resultado_completo": resultado}


# ---------------------------------------------------------------------------
# Orquestacion principal
# ---------------------------------------------------------------------------

def podar_reacciones_bloqueadas(modelo: cobra.Model) -> cobra.Model:
    """
    Elimina del modelo las reacciones bloqueadas (aquellas cuyo flujo es
    forzosamente cero bajo cualquier condicion factible del modelo actual).
    Esto reduce el tamano real del problema de programacion lineal que
    OptGPSampler debe construir en memoria -- no es solo una optimizacion
    de velocidad, es la forma correcta de reducir el error de Windows
    "El archivo de paginacion es demasiado pequeno" (WinError 1455), que
    ocurre porque cobra reserva un arreglo de memoria compartida del
    tamano del problema COMPLETO incluso con --processes 1.

    Advertencia: find_blocked_reactions ejecuta FVA (Flux Variability
    Analysis) sobre el modelo completo, lo cual puede tardar varios
    minutos en un modelo de ~20,000 reacciones. Es una inversion de
    tiempo que vale la pena una sola vez por modelo, no en cada corrida.
    """
    from cobra.flux_analysis import find_blocked_reactions
    print("  buscando reacciones bloqueadas (puede tardar varios minutos en un "
          "modelo de este tamano)...")
    bloqueadas = find_blocked_reactions(modelo)
    if bloqueadas:
        modelo.remove_reactions(bloqueadas, remove_orphans=True)
        print(f"  se removieron {len(bloqueadas)} reacciones bloqueadas "
              f"({len(modelo.reactions)} reacciones restantes) -- esto reduce "
              f"directamente el tamano del arreglo de memoria compartida que "
              f"construye OptGPSampler")
    else:
        print("  no se encontraron reacciones bloqueadas")
    return modelo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="community.xml", help="Ruta al GSMM comunitario (SBML)")
    parser.add_argument("--n-samples", type=int, default=2000, help="Muestras de flujo por especie/condicion")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                         help="Umbral de actividad metabolica (Seccion 4.1), usado solo si "
                              "--epsilon-sweep no se especifica. El default 1e-6 es esencialmente "
                              "cero dado que los flujos de este modelo estan en la escala de "
                              "cientos/miles (community_growth optimo = 1000) -- usa "
                              "--epsilon-sweep en su lugar para explorar esto correctamente.")
    parser.add_argument("--epsilon-sweep", default=None,
                         help="Lista de valores de epsilon separados por coma (p.ej. "
                              "'1e-6,1e-3,1e-1,1,10') para evaluar la robustez de H1b "
                              "(Seccion 4.5, Fase A) SIN volver a muestrear: el muestreo de "
                              "flujos (el paso caro, ~30 min) se hace una sola vez, y el "
                              "reetiquetado de celdas de Schubert (Algoritmo 1, barato y "
                              "vectorizado) se repite una vez por valor de epsilon sobre las "
                              "MISMAS muestras. Si se especifica, tiene prioridad sobre --epsilon.")
    parser.add_argument("--fraccion-optimo", type=float, default=0.9,
                         help="Fraccion del optimo de community_growth exigida al muestrear")
    parser.add_argument("--solver", default="glpk",
                         help="Solver a usar (default: glpk, gratuito y sin limite de tamano). "
                              "Cambia a --solver gurobi o --solver cplex solo si tienes una "
                              "licencia sin restriccion de tamano; con licencia restringida de "
                              "Gurobi este modelo de 20,784 reacciones fallara.")
    parser.add_argument("--processes", type=int, default=1,
                         help="Procesos paralelos para el muestreo OptGP (paralelizacion nativa "
                              "de cobra). Este es, con diferencia, el paso mas costoso en tiempo "
                              "de todo el script -- sube este numero hasta el numero de nucleos "
                              "disponibles si tienes problemas de tiempo de ejecucion. Cada "
                              "proceso adicional usa memoria propia, asi que si el problema es "
                              "de memoria en vez de tiempo, mantenlo en 1 y reduce --n-samples.")
    parser.add_argument("--remove-blocked", action="store_true",
                         help="Elimina reacciones bloqueadas antes de muestrear. ADVERTENCIA: "
                              "find_blocked_reactions corre FVA sobre el modelo completo, una "
                              "operacion del MISMO orden de costo (miles de resoluciones LP) que "
                              "el calentamiento de OptGP que este flag intenta evitar -- con GLPK "
                              "en un modelo de 20,784 reacciones puede ser igual de lento o peor. "
                              "Considera --metodo random-fba en su lugar si GLPK es demasiado "
                              "lento para tu caso.")
    parser.add_argument("--force-shared-memory", action="store_true",
                         help="Desactiva el parche que evita la memoria compartida de cobra "
                              "cuando --processes=1 (ver _evitar_memoria_compartida_innecesaria). "
                              "Solo util para depurar si el parche mismo causa problemas.")
    parser.add_argument("--batch-size", type=int, default=50,
                         help="Muestras por lote durante el muestreo (default: 50). Lotes mas "
                              "pequenos dan progreso mas frecuente (util en Colab para confirmar "
                              "que no esta 'colgado'); lotes mas grandes tienen menos overhead.")
    parser.add_argument("--thinning", type=int, default=100,
                         help="Pasos internos de OptGP por cada muestra devuelta (default: 100, "
                              "el valor estandar recomendado para reducir autocorrelacion entre "
                              "muestras -- ver Gelbach2024). CADA muestra devuelta cuesta "
                              "'thinning' resoluciones LP completas, asi que con un modelo de "
                              "20,784 reacciones y GLPK esto puede ser muy lento. Para una PRIMERA "
                              "corrida de diagnostico (confirmar que el pipeline funciona de punta "
                              "a punta, no para resultados finales), usa --thinning 1 o --thinning 5 "
                              "junto con --batch-size 1: las muestras individuales siguen siendo "
                              "puntos validos del politopo de flujos, solo estaran mas "
                              "autocorrelacionadas entre si -- no uses un thinning bajo para los "
                              "resultados que vayan al manuscrito.")
    parser.add_argument("--metodo", choices=["optgp", "random-fba"], default="optgp",
                         help="'optgp' (default): muestreo HR exacto (Seccion 4.5 del "
                              "manuscrito), pero requiere generar puntos de calentamiento tipo "
                              "FVA -- muy costoso con GLPK en modelos grandes. 'random-fba': "
                              "alternativa APROXIMADA y barata (un FBA con objetivo aleatorio por "
                              "muestra, sin calentamiento) para desbloquear el pipeline cuando "
                              "'optgp' es demasiado lento; ver la advertencia de limitacion en "
                              "muestrear_flujos_random_fba antes de usar estos resultados en el "
                              "manuscrito.")
    parser.add_argument("--fase-b", action="store_true",
                         help="Corre la Fase B (Seccion 4.5): benchmark leave-one-species-out "
                              "de un Random Forest contra el clasificador trivial de codimension "
                              "de Schubert, sobre las 4 especies validadas. Requiere sklearn.")
    parser.add_argument("--fase-c", action="store_true",
                         help="Corre la Fase C (Seccion 4.5): diagramas de persistencia (H0/H1) "
                              "de las nubes de muestras de flujo, sobre las 4 especies validadas. "
                              "Requiere ripser (pip install ripser) para H1; sin ripser, cae a "
                              "H0 solamente via scipy. Distancia bottleneck requiere ademas persim.")
    parser.add_argument("--output-dir", default="resultados_fase_a",
                         help="Carpeta donde guardar las muestras de flujo (para reutilizar sin "
                              "volver a muestrear) y un reporte de texto con todos los resultados "
                              "de la corrida. Se crea si no existe (default: resultados_fase_a).")
    parser.add_argument("--reusar-muestras", action="store_true",
                         help="Si existen muestras ya guardadas en --output-dir para este mismo "
                              "--metodo y --n-samples, las reutiliza en vez de volver a muestrear "
                              "(ahorra los ~25-35 min de muestreo por condicion).")
    args = parser.parse_args()

    lista_epsilon = (
        [float(e) for e in args.epsilon_sweep.split(",")]
        if args.epsilon_sweep else [args.epsilon]
    )

    # Guarda TODO lo impreso en consola tambien en un archivo de reporte, para
    # no perder los resultados si se cierra la sesion de Colab o hay que
    # revisar la corrida despues sin volver a ejecutar nada.
    import os
    import sys
    import datetime

    os.makedirs(args.output_dir, exist_ok=True)
    marca_tiempo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_reporte = os.path.join(args.output_dir, f"reporte_{marca_tiempo}.txt")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    archivo_reporte = open(ruta_reporte, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, archivo_reporte)

    import atexit
    def _restaurar_stdout():
        sys.stdout = sys.__stdout__
        if not archivo_reporte.closed:
            archivo_reporte.close()
    atexit.register(_restaurar_stdout)

    print(f"(reporte completo de esta corrida se esta guardando en: {ruta_reporte})")

    modelo = cargar_modelo(args.model, solver=args.solver)
    if args.remove_blocked:
        modelo = podar_reacciones_bloqueadas(modelo)
    mapa = construir_mapa(modelo)
    print("  bandera de referencia (diagnostico, SIN exclusion -- la bandera "
          "realmente usada en el algoritmo es leave-one-out, ver mas abajo):")
    construir_bandera(mapa, BANDERA_SANTA_POLA, verbose=True)

    # columnas que realmente se usan en el resto del script: la base de E
    # (para poder leer directamente el flujo de cada reaccion de intercambio
    # comunitario si se necesitara) mas la interfaz tex/texi de cada especie.
    # Recortar el DataFrame de muestras a solo estas columnas, justo despues
    # de muestrear, es la principal palanca de memoria de este script: pasa
    # de ~20,784 columnas a unas ~2,000.
    columnas_relevantes = set(mapa.base_E)
    for interfaz in mapa.interfaz_por_especie.values():
        columnas_relevantes.update(interfaz.values())
    columnas_relevantes = sorted(columnas_relevantes)

    condiciones = construir_condiciones_salinidad(mapa, BANDERA_SANTA_POLA)

    os.makedirs(args.output_dir, exist_ok=True)

    muestras_por_condicion: Dict[str, "pandas.DataFrame"] = {}
    for pond, condicion_dict in condiciones.items():
        ruta_cache = os.path.join(args.output_dir, f"muestras_{pond}_{args.metodo}_n{args.n_samples}.csv")
        if args.reusar_muestras and os.path.exists(ruta_cache):
            print(f"\nReutilizando muestras cacheadas para {pond}: {ruta_cache}")
            import pandas as pd
            muestras_por_condicion[pond] = pd.read_csv(ruta_cache, index_col=0)
            continue

        print(f"\nMuestreando flujos bajo la condicion {pond}...")
        if args.metodo == "random-fba":
            print("  METODO: random-fba (aproximado, ver advertencia de limitacion en el codigo)")
            muestras_por_condicion[pond] = muestrear_flujos_random_fba(
                modelo,
                n_samples=args.n_samples,
                fraccion_optimo=args.fraccion_optimo,
                columnas_relevantes=columnas_relevantes,
                batch_size_reporte=args.batch_size,
                condicion=condicion_dict,
            )
        else:
            muestras_por_condicion[pond] = muestrear_flujos(
                modelo,
                n_samples=args.n_samples,
                fraccion_optimo=args.fraccion_optimo,
                processes=args.processes,
                columnas_relevantes=columnas_relevantes,
                forzar_memoria_compartida=args.force_shared_memory,
                tamano_lote=args.batch_size,
                thinning=args.thinning,
                condicion=condicion_dict,
            )
        muestras_por_condicion[pond].to_csv(ruta_cache)
        print(f"  muestras guardadas en {ruta_cache} (usa --reusar-muestras para reutilizarlas "
              f"sin volver a muestrear)")

    # resultados_por_especie[especie][pond][epsilon] = resultado de asignar_celdas_schubert_vectorizado
    resultados_por_especie: Dict[str, Dict[str, Dict[float, Dict[str, np.ndarray]]]] = {}
    especies_en_cero = []
    print(f"\nBarriendo epsilon en {lista_epsilon} por cada condicion (el muestreo ya esta "
          f"hecho; esto solo reetiqueta celdas de Schubert, es rapido)...")
    for especie in SPECIES:
        bandera_especie = construir_bandera(mapa, BANDERA_SANTA_POLA,
                                             excluir_especie=especie, verbose=False)
        resultados_por_especie[especie] = {}
        print(f"\n{especie}:")
        n_activas_alguna = False
        for pond, muestras in muestras_por_condicion.items():
            V = matriz_en_E(muestras, mapa, especie)
            resultados_por_especie[especie][pond] = {}
            for epsilon in lista_epsilon:
                resultado = asignar_celdas_schubert_vectorizado(V, bandera_especie, epsilon, r=len(SPECIES))
                resultados_por_especie[especie][pond][epsilon] = resultado
                codims = resultado["codim"]
                media, lo, hi = bootstrap_ci_codim(codims)
                print(f"  [{pond}] epsilon={epsilon:g}: {resultado['n_activas']}/{len(codims)} "
                      f"activas, codim media={media:.3f} IC95%=[{lo:.3f},{hi:.3f}]")
                if resultado["n_activas"] > 0:
                    n_activas_alguna = True

        if not n_activas_alguna:
            especies_en_cero.append(especie)

    if especies_en_cero:
        print(f"\nAVISO: las siguientes especies no tuvieron NINGUNA reaccion activa en "
              f"ninguna muestra, en NINGUNA condicion, NINGUN epsilon: {especies_en_cero}. "
              f"Esto casi siempre indica un problema de mapeo, NO que la codimension real "
              f"sea cero -- revisar antes de interpretar cualquier resultado.")

    print("\nRegresion logistica ordinal: codimension ~ salinidad (s in {19,37}), por "
          "especie, usando epsilon="
          f"{lista_epsilon[0]:g} (el primero del barrido; recordar que epsilon apenas "
          "afecta la codimension segun la Seccion 4.5.1):")
    for especie in SPECIES:
        try:
            codims_19 = resultados_por_especie[especie]["SS19"][lista_epsilon[0]]["codim"]
            codims_37 = resultados_por_especie[especie]["SS37"][lista_epsilon[0]]["codim"]
            codims_todas = np.concatenate([codims_19, codims_37])
            s = np.concatenate([np.full(len(codims_19), 19.0), np.full(len(codims_37), 37.0)])
            r = regresion_logistica_ordinal(codims_todas, s)
            if r.get("beta") is None:
                print(f"  {especie}: {r.get('aviso')}")
            else:
                print(f"  {especie}: beta={r['beta']:.4f}, p-valor={r['p_valor']:.4f} "
                      f"({'crece' if r['beta'] > 0 else 'decrece'} con la salinidad)")
        except Exception as e:
            print(f"  {especie}: ERROR al ajustar la regresion ({e})")

    print("\nPrueba H1a (exploratoria): correlacion entre codimension media (promediada "
          "entre las condiciones SS19 y SS37) y amplitud de tolerancia a NaCl (dato "
          "independiente de literatura taxonomica), por cada epsilon del barrido. Se "
          "reportan DOS versiones para exponer la sensibilidad al conjunto de datos usado:")
    for epsilon in lista_epsilon:
        codim_medio = {
            especie: float(np.concatenate([
                resultados_por_especie[especie][pond][epsilon]["codim"]
                for pond in muestras_por_condicion
            ]).mean())
            for especie in SPECIES
            if especie in resultados_por_especie
        }
        r_alta = prueba_H1a_correlacion(codim_medio, NACL_AMPLITUD_ALTA_CONFIANZA)
        r_todas = prueba_H1a_correlacion(codim_medio, NACL_AMPLITUD_ESPECIES)
        print(f"  epsilon={epsilon:g}:")
        print(f"    solo confianza alta (n={r_alta.get('n')}): rho={r_alta.get('rho')}, "
              f"p={r_alta.get('p_valor')}")
        print(f"    todas las especies con dato (n={r_todas.get('n')}): "
              f"rho={r_todas.get('rho')}, p={r_todas.get('p_valor')}")
    print("  AVISO: si el signo o la magnitud de rho cambia sustancialmente entre ambas "
          "versiones, es evidencia de que el resultado es fragil frente a la calidad del "
          "dato externo, no solo frente a n -- reportar ambas versiones, no solo la mas "
          "favorable. Ver Discusion.")

    print(f"\nAVISO METODOLOGICO: {len(ESPECIES_SIN_EVIDENCIA_SANTA_POLA)} de las "
          f"{len(SPECIES)} especies NO tienen evidencia real de presencia (>1% de lecturas "
          f"16S) en ningun estanque de Santa Pola segun Fernandez et al. 2014a: "
          f"{ESPECIES_SIN_EVIDENCIA_SANTA_POLA}. Sus resultados de codimension en esta "
          f"corrida se basan en el extremo arbitrario de la bandera (reacciones sin tier "
          f"asignado) y NO deben usarse para someter a prueba H1a/H1b sin esa salvedad "
          f"explicita -- la prueba principal de hipotesis deberia restringirse a "
          f"{[e for e in SPECIES if e not in ESPECIES_SIN_EVIDENCIA_SANTA_POLA]}.")

    if args.fase_b:
        print("\nFase B: benchmark leave-one-species-out (Random Forest vs. codimension "
              "de Schubert) sobre las 4 especies validadas...")
        especies_validas = ["Halorubrum", "Natronomonas", "Haloquadratum", "salinibacter"]
        tier_por_especie = {"Halorubrum": 0, "Natronomonas": 0,
                             "Haloquadratum": 1, "salinibacter": 1}
        try:
            resultado_b = ejecutar_fase_b(
                muestras_por_condicion, mapa, especies_validas, tier_por_especie,
                epsilon=lista_epsilon[0],
            )
            print(f"  n especies (pliegues de CV) = {resultado_b['n_especies']}")
            print(f"  Random Forest:       exactitud balanceada = "
                  f"{resultado_b['ml_exactitud_balanceada']:.3f}, "
                  f"F1 macro = {resultado_b['ml_f1_macro']:.3f}")
            print(f"  Schubert (trivial):  exactitud balanceada = "
                  f"{resultado_b['schubert_exactitud_balanceada']:.3f}, "
                  f"F1 macro = {resultado_b['schubert_f1_macro']:.3f}")
            print("  AVISO: n=4 especies (4 pliegues de CV) es un diseño extremadamente "
                  "pequeño -- resultado exploratorio, no una comparacion de desempeño "
                  "confiable. Requiere sklearn instalado.")
        except ImportError:
            print("  ERROR: Fase B requiere scikit-learn. Instala con: pip install scikit-learn")
        except Exception as e:
            print(f"  ERROR en Fase B: {e}")

    if args.fase_c:
        print("\nFase C: homologia persistente (H0/H1) sobre las 4 especies validadas...")
        especies_validas = ["Halorubrum", "Natronomonas", "Haloquadratum", "salinibacter"]
        try:
            ejecutar_fase_c(muestras_por_condicion, mapa, especies_validas,
                             epsilon=lista_epsilon[0])
        except Exception as e:
            print(f"  ERROR en Fase C: {e}")

    print("\nListo. `resultados_por_especie` contiene, por especie y por valor de epsilon")
    print("del barrido, las matrices 'lambda' (n_samples x r) y 'codim' (n_samples,) --")
    print("insumo directo para las Fases B y C (Seccion 4.5) y para la validacion")
    print("(Seccion 4.4). Compara los resultados entre valores de epsilon para evaluar H1b.")
    print(f"\nReporte completo guardado en: {ruta_reporte}")

    sys.stdout = sys.__stdout__
    archivo_reporte.close()


if __name__ == "__main__":
    main()
