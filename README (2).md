# Structural ecological modelling of microbial metabolic niches using Grassmannian representations

Code, data, and figures accompanying the manuscript *"Structural ecological
modelling of microbial metabolic niches using Grassmannian representations"*
(A. Centeno-Mejía) and its Supplementary Material. This repository implements
the Schubert-codimension classification framework applied to a ten-species
community genome-scale metabolic model (GSMM) of a talasohaline hypersaline
microbial community, referenced against the Santa Pola salinity gradient.

> **TODO (before making the repo public / archiving on Zenodo):** rename the
> placeholder script names below to match your actual filenames, remove any
> script that does not correspond to a real file, and delete this notice.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── community.xml            # original community GSMM (SBML L3, FBC v2)
│   ├── community_fixed.xml       # after fixing duplicate fbc:label="G_spontaneous"
│   └── santapola_abundancias.csv # Table 3.1 (16S rRNA relative abundances, Fernandez2014a)
├── src/
│   ├── construir_bandera.py      # reference-flag construction (tiers by abundance trend,
│   │                              # leave-one-out, complete-flag ordering)
│   ├── asignar_celdas_schubert.py# Algorithm 1: Schubert cell / codimension assignment
│   ├── fase_a_muestreo.py        # Phase A: flux sampling (random-fba), epsilon sweep,
│   │                              # bootstrap CIs, ordinal logistic regression (OrderedModel)
│   ├── fase_b_ml.py              # Phase B: Random Forest / Schubert classifier benchmark,
│   │                              # leave-one-species-out cross-validation
│   ├── fase_c_homologia.py       # Phase C: persistent homology (ripser), bottleneck distances
│   ├── simulacion_sintetica.py   # synthetic validation study (power analysis, Phases A/B/C
│   │                              # under known ground truth)
│   └── utils.py                  # shared helpers (species-reaction mapping, I/O)
├── results/
│   ├── tabla_codimensiones.csv
│   ├── tabla_potencia_sintetica.csv
│   ├── tabla_faseb_sintetica.csv
│   └── tabla_persistencia.csv
└── figures/
    └── fig_synthetic_combined.pdf
```

## Reproducibility notes and known environment sensitivities

This section documents environment-specific behaviour encountered during
development, so that a reviewer or future user does not mistake a
solver/version difference for a bug.

- **Solver.** All reported results use **GLPK** (via `optlang`/`cobrapy`),
  not Gurobi or HiGHS. A single LP solve on the full 20,783-reaction
  community model took ≈40–44 s on our machine; this dominates the cost of
  every sampling step.
  - `HiGHS` (`highspy`) is **not** registered as an `optlang` backend in the
    environment used for this project (only `glpk_exact`, `glpk`, `hybrid`,
    `osqp`, `scipy` were available); do not expect `--solver highs` to work
    without installing/registering `highspy` separately.
  - `Gurobi` was tried locally and hit a *"Model too large for size-limited
    license"* error on the full community model; it was not used for any
    reported result.
- **Sampling method.** Exact hit-and-run sampling (`OptGPSampler`/ACHR, as
  originally intended) was **not computationally feasible** with GLPK on
  this model: the warm-up stage alone requires ≈2m LP solves (≈1,088 solves,
  several hundred hours). All flux samples reported in the manuscript use
  the **`random-fba` approximation** implemented in `fase_a_muestreo.py`
  (a single FBA solve per sample, with a randomly generated linear objective
  over the interface reactions, subject to
  `community_growth ≥ 0.9 × optimum`). This is a documented limitation
  (Discussion, main manuscript) and preferentially samples vertices/faces of
  the flux polytope rather than its interior.
- **`find_blocked_reactions`** (i.e., any `--remove-blocked`-style flag) also
  hangs on this model under GLPK and was not used in the reported results.
- **Windows-specific issue.** `OptGPSampler`'s shared-memory array allocation
  raised `OSError [WinError 1455]` ("the paging file is too small") on
  Windows even with `processes=1`; this was one additional reason exact
  sampling was abandoned in favour of `random-fba`. Not relevant if running
  on Linux/macOS or in a container.
- **`thinning`.** The upstream sampler's `thinning=100` default is not
  exposed via CLI in some cobrapy versions; if you resurrect exact sampling,
  check this explicitly, since it silently multiplies the number of internal
  LP solves per returned sample.
- **SBML model.** `community.xml` originally failed to load ("No SBML model
  detected") because of ten duplicate `fbc:label="G_spontaneous"` gene
  products (one per species, identical label, distinct `fbc:id`). This is
  fixed in `community_fixed.xml` by making each label unique
  (`G_spontaneous_<Species>`); use the fixed file for all analyses.

## Installation

```bash
conda env create -f environment.yml
conda activate schubert-niches
# or, with pip:
pip install -r requirements.txt
```

Python ≥3.10 is recommended. GLPK must be available to `optlang`/`cobrapy`
(installed automatically with `cobra` via `conda-forge` on most platforms;
on some Linux distributions you may need `apt install glpk-utils
libglpk-dev` first).

## Basic usage

```bash
# 1. Build the species-specific leave-one-out reference flags
python src/construir_bandera.py --model data/community_fixed.xml \
    --abundancias data/santapola_abundancias.csv --out results/banderas/

# 2. Phase A: flux sampling (random-fba) + epsilon sweep + ordinal regression
python src/fase_a_muestreo.py --model data/community_fixed.xml \
    --bandera-dir results/banderas/ --n-samples 50 \
    --epsilon-sweep 1e-6 1e-3 1e-1 1 10 --metodo random-fba \
    --out results/tabla_codimensiones.csv

# 3. Phase B: supervised-learning benchmark
python src/fase_b_ml.py --samples results/fase_a_samples.npz \
    --labels data/tiers.csv --out results/tabla_faseb_sintetica.csv

# 4. Phase C: persistent homology
python src/fase_c_homologia.py --samples results/fase_a_samples.npz \
    --out results/tabla_persistencia.csv

# 5. Synthetic validation study (power analysis, Phases A/B/C under known ground truth)
python src/simulacion_sintetica.py --k-values 4 8 12 20 30 50 \
    --gamma 0.7 --reps 200 --out results/tabla_potencia_sintetica.csv
```

Adjust script names, arguments, and paths above to match the actual files
you commit — these are illustrative of the pipeline described in the
manuscript's Methods section, not a literal API reference.

## Citation

If you use this code or the accompanying community GSMM, please cite:

> Centeno-Mejía, A. *Structural ecological modelling of microbial metabolic
> niches using Grassmannian representations*. [[journal, year, DOI once
> available](https://zenodo.org/me/uploads?q=&f=shared_with_me%3Afalse&l=list&p=1&s=10&sort=newest)].



## License

<!-- TODO: pick a license (e.g. MIT for code, CC-BY-4.0 for data/figures) and add LICENSE file(s) -->

## Contact

Alex Centeno-Mejía — alex.centeno@alumnos.ucm.cl
Doctorado en Modelamiento Matemático Aplicado, Universidad Católica del Maule, Talca, Chile
