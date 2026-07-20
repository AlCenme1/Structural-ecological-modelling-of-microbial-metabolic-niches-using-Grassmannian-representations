"""
Synthetic validation study (Manuscript Section 4.5.1 / Results 5.1)
=====================================================================

Generates synthetic species with a subspace P_i whose position relative to
a reference flag depends, with a controlled effect size, on a known
covariate z_i (analogous to the NaCl tolerance breadth in the real case
study). This script allows us to:

  1. Verify that the framework recovers the known dependency between
     Schubert codimension and z_i when both the effect size and k are
     large.
  2. Estimate the statistical power of the real design (k=4) via a
     simulation-based power analysis, varying k.
  3. Repeat the Phase B benchmark (Random Forest vs. trivial Schubert
     classifier) on synthetic data, to bound whether the collapse observed
     in the real case is due to sample size or to the method itself.

Unlike fase_a_muestreo_flujos.py, this script does NOT require cobra or a
real GSMM -- it is pure numpy/scipy/sklearn and runs in seconds, not
minutes or hours.

Usage:
    python synthetic_validation_study.py
"""

import argparse
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Fixed synthetic flag (arbitrary, but fixed for the whole study)
# ---------------------------------------------------------------------------

def generate_synthetic_flag(m: int, seed: int = 0) -> Tuple[List[np.ndarray], np.ndarray]:
    """Complete flag F_1 subset ... subset F_m = E: a fixed random
    permutation of the m coordinates of E. dim(F_j)=j by construction,
    exactly as required by Algorithm 1."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(m)
    flags = []
    mask = np.zeros(m, dtype=bool)
    for idx in order:
        mask = mask.copy()
        mask[idx] = True
        flags.append(mask)
    return flags, order


# ---------------------------------------------------------------------------
# 2. Algorithm 1 (identical to the one in fase_a_muestreo_flujos.py)
# ---------------------------------------------------------------------------

def assign_schubert_cells_vectorized(
    V: np.ndarray, flag_masks, epsilon: float, r: int
) -> Dict[str, np.ndarray]:
    A = np.abs(V) > epsilon
    n_samples = V.shape[0]
    n = len(flag_masks)
    d = np.zeros((n_samples, n + 1), dtype=int)
    for j, mask in enumerate(flag_masks, start=1):
        d[:, j] = (A & mask).sum(axis=1)
    lam = np.zeros((n_samples, r), dtype=int)
    for i in range(1, r + 1):
        reached = d >= i
        has_any = reached.any(axis=1)
        j_i = np.where(has_any, reached.argmax(axis=1), n)
        lam[:, i - 1] = np.clip(n + i - j_i, 0, None)
    return {"lambda": lam, "codim": lam.sum(axis=1), "n_active": int(A.any(axis=1).sum())}


# ---------------------------------------------------------------------------
# 3. Synthetic species generator with controlled effect size
# ---------------------------------------------------------------------------

def generate_synthetic_species(
    flag_order: np.ndarray,
    z_i: float,
    effect_size: float,
    n_active_reactions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates the set of active reactions for ONE synthetic species/sample
    with covariate z_i in [0,1] and effect size gamma in [0,1].

    gamma=0: active reactions are a UNIFORM random subset, COMPLETELY
             independent of z_i -- corresponds to H0 (no real signal).
             CORRECTION: an earlier version of this generator used a wide
             Gaussian bell instead of an explicit mixture, which still
             carried a spurious correlation with z_i even at gamma=0
             (empirically verified: rho=-0.96, p=0 in the control case
             before this correction, when the expected value is rho
             approx 0).
    gamma=1: active reactions concentrate deterministically around
             position z_i*m in the flag order -- maximal signal.
    Intermediate values explicitly mix a uniform distribution (weight
    1-gamma) with a Gaussian centered at z_i*m (weight gamma), instead of
    just varying the width of a single Gaussian.
    """
    m = len(flag_order)
    if effect_size <= 0:
        chosen_positions = rng.choice(m, size=n_active_reactions, replace=False)
        return flag_order[chosen_positions]

    center = z_i * m
    width = max(m * (1 - effect_size) / 2, 1.0)
    gauss_weights = np.exp(-0.5 * ((np.arange(m) - center) / width) ** 2)
    gauss_weights = gauss_weights / gauss_weights.sum()
    uniform_weights = np.full(m, 1.0 / m)
    weights = effect_size * gauss_weights + (1 - effect_size) * uniform_weights
    weights = weights / weights.sum()
    chosen_positions = rng.choice(m, size=n_active_reactions, replace=False, p=weights)
    return flag_order[chosen_positions]


def build_V(active_indices: np.ndarray, m: int) -> np.ndarray:
    v = np.zeros((1, m))
    v[0, active_indices] = 1.0
    return v


# ---------------------------------------------------------------------------
# 4. Permutation correlation test (identical to prueba_H1a_correlacion)
# ---------------------------------------------------------------------------

def permutation_correlation_test(z: np.ndarray, codim: np.ndarray,
                                  n_perm: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """
    Vectorized version: converts z and codim to ranks ONCE (Spearman
    correlation = Pearson correlation on the ranks), and generates the
    n_perm permutations as a single matrix to compute all null
    correlations at once with numpy, instead of calling
    scipy.stats.spearmanr n_perm times (the actual bottleneck of the power
    analysis, which becomes intractable with thousands of repetitions if
    not vectorized).
    """
    from scipy.stats import rankdata
    rz = rankdata(z)
    rc = rankdata(codim)
    rho_obs = np.corrcoef(rz, rc)[0, 1]

    rng = np.random.default_rng(seed)
    n = len(rc)
    rc_centered = rc - rc.mean()
    rz_centered = rz - rz.mean()
    denom = np.sqrt((rz_centered ** 2).sum() * (rc_centered ** 2).sum())

    perm_indices = np.array([rng.permutation(n) for _ in range(n_perm)])
    rc_perm = rc_centered[perm_indices]  # (n_perm, n)
    rhos_perm = (rc_perm @ rz_centered) / denom

    p_value = float(np.mean(np.abs(rhos_perm) >= np.abs(rho_obs)))
    return float(rho_obs), p_value


# ---------------------------------------------------------------------------
# 5. One full run of the study for a given k
# ---------------------------------------------------------------------------

def run_study(
    k: int,
    m: int = 544,
    effect_size: float = 0.7,
    n_active_reactions: int = 150,
    epsilon: float = 0.5,
    seed: int = 0,
    n_perm: int = 10000,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    flags, order = generate_synthetic_flag(m, seed=seed)

    z = rng.uniform(0, 1, size=k)
    codim = np.zeros(k)
    for i in range(k):
        active_indices = generate_synthetic_species(
            order, z[i], effect_size, n_active_reactions, rng)
        V = build_V(active_indices, m)
        result = assign_schubert_cells_vectorized(V, flags, epsilon, r=k)
        codim[i] = result["codim"][0]

    rho, p_value = permutation_correlation_test(z, codim, n_perm=n_perm, seed=seed)
    return {"k": k, "rho": rho, "p_value": p_value, "z": z, "codim": codim}


# ---------------------------------------------------------------------------
# 6. Simulation-based power analysis: vary k, repeat many times
# ---------------------------------------------------------------------------

def power_analysis(
    k_values: List[int],
    n_repetitions: int = 200,
    effect_size: float = 0.7,
    alpha: float = 0.05,
    m: int = 544,
    n_active_reactions: int = 150,
) -> Dict[int, float]:
    """
    For each k, repeats the study `n_repetitions` times with different
    seeds and computes power = fraction of repetitions where
    p_value < alpha. This directly answers: given the assumed effect size,
    how likely is the real design (k=4) to detect a signal that DOES
    exist?
    """
    power_by_k = {}
    for k in k_values:
        p_values = []
        for rep in range(n_repetitions):
            result = run_study(
                k, m=m, effect_size=effect_size,
                n_active_reactions=n_active_reactions, seed=rep,
                n_perm=500,
            )
            p_values.append(result["p_value"])
        power = float(np.mean(np.array(p_values) < alpha))
        power_by_k[k] = power
        print(f"  k={k:3d}: power = {power:.3f} "
              f"({n_repetitions} repetitions, effect size={effect_size})")
    return power_by_k


# ---------------------------------------------------------------------------
# 7. Phase-B-style benchmark on synthetic data (leave-one-out)
# ---------------------------------------------------------------------------

def synthetic_phase_b_benchmark(
    k: int,
    m: int = 544,
    effect_size: float = 0.7,
    n_active_reactions: int = 150,
    n_samples_per_species: int = 50,
    epsilon: float = 0.5,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Replicates the leave-one-out design of ejecutar_fase_b on synthetic
    species with a binary tier (z_i > median -> tier 1, else tier 0), to
    bound whether the Random Forest collapse observed in the real case
    (k=4) is a sample-size artifact.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    flags, order = generate_synthetic_flag(m, seed=seed)
    z = rng.uniform(0, 1, size=k)
    tier = (z > np.median(z)).astype(int)

    # N flux samples per species (with additional sample-to-sample noise)
    X_per_species = []
    codim_per_species = []
    for i in range(k):
        Vs = np.zeros((n_samples_per_species, m))
        for t in range(n_samples_per_species):
            active_indices = generate_synthetic_species(
                order, z[i], effect_size, n_active_reactions, rng)
            Vs[t, active_indices] = 1.0
        X_per_species.append(Vs)
        result = assign_schubert_cells_vectorized(Vs, flags, epsilon, r=k)
        codim_per_species.append(result["codim"])

    y_true_all, y_pred_ml_all, y_pred_sch_all = [], [], []
    for test_species in range(k):
        train_species = [i for i in range(k) if i != test_species]
        X_train = np.vstack([X_per_species[i] for i in train_species])
        y_train = np.concatenate([
            np.full(n_samples_per_species, tier[i]) for i in train_species
        ])
        X_test = X_per_species[test_species]
        y_test = np.full(n_samples_per_species, tier[test_species])

        clf = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred_ml = clf.predict(X_test)

        codim_train = np.concatenate([codim_per_species[i] for i in train_species])
        tier_train = np.concatenate([
            np.full(n_samples_per_species, tier[i]) for i in train_species
        ])
        # IMPORTANT FIX: the threshold is searched in BOTH directions
        # (codim > threshold => class 1, or codim < threshold => class 1),
        # not just one. With a single fixed direction, if the true
        # relationship between codimension and tier turns out to be
        # inverted relative to the assumed one, the classifier ends up
        # systematically inverted and balanced accuracy collapses to 0
        # instead of ~0.5 (empirically verified: accuracy = 0.000 at k=50
        # before this fix).
        best_threshold, best_direction, best_score = None, None, -1
        for candidate in np.percentile(codim_train, np.arange(10, 100, 10)):
            for direction in (">", "<"):
                pred = (codim_train > candidate).astype(int) if direction == ">" \
                    else (codim_train < candidate).astype(int)
                score = balanced_accuracy_score(tier_train, pred)
                if score > best_score:
                    best_score, best_threshold, best_direction = score, candidate, direction
        y_pred_sch = (codim_per_species[test_species] > best_threshold).astype(int) \
            if best_direction == ">" else (codim_per_species[test_species] < best_threshold).astype(int)

        y_true_all.append(y_test)
        y_pred_ml_all.append(y_pred_ml)
        y_pred_sch_all.append(y_pred_sch)

    y_true = np.concatenate(y_true_all)
    y_pred_ml = np.concatenate(y_pred_ml_all)
    y_pred_sch = np.concatenate(y_pred_sch_all)

    return {
        "k": k,
        "ml_balanced_accuracy": balanced_accuracy_score(y_true, y_pred_ml),
        "schubert_balanced_accuracy": balanced_accuracy_score(y_true, y_pred_sch),
    }


# ---------------------------------------------------------------------------
# 8. Phase C on synthetic data: dimension-0 persistent homology
# ---------------------------------------------------------------------------

def h0_persistence(point_cloud: np.ndarray) -> np.ndarray:
    """
    Computes dimension-0 persistent homology (connected components) of a
    point cloud, via single-linkage hierarchical clustering: the merge
    heights of the dendrogram are EXACTLY the death times of the
    Vietoris--Rips filtration in dimension 0 (each point is born at
    delta=0; it dies when it merges with another component). This is a
    legitimate and exact implementation of H_0, but NOT of H_1 (cycles) --
    computing H_1 would require a dedicated library (ripser/gudhi), not
    available in this environment without network access.

    Returns the vector of death times (equivalent to the persistence
    diagram in dimension 0, since birth is always 0).
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    distances = pdist(point_cloud, metric="euclidean")
    Z = linkage(distances, method="single")
    death_times = Z[:, 2]
    return death_times


def synthetic_phase_c(
    k: int,
    m: int = 544,
    effect_size: float = 0.7,
    n_active_reactions: int = 150,
    n_samples_per_species: int = 50,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Replicates the Phase C logic (manuscript Section 4.5) on synthetic
    data: for each of k synthetic species, computes an H_0 persistence
    summary (total persistence = sum of death times) of the cloud of N
    samples for that species, and tests whether that topological summary
    --computed INDEPENDENTLY of the flag and of Algorithm 1-- also
    recovers the known dependency on z_i, as a cross-check of whether
    Phase A (codimension) and Phase C (topology) converge on the same
    structure when it exists.
    """
    rng = np.random.default_rng(seed)
    flags, order = generate_synthetic_flag(m, seed=seed)
    z = rng.uniform(0, 1, size=k)

    total_persistence = np.zeros(k)
    mean_codim = np.zeros(k)
    for i in range(k):
        Vs = np.zeros((n_samples_per_species, m))
        for t in range(n_samples_per_species):
            active_indices = generate_synthetic_species(
                order, z[i], effect_size, n_active_reactions, rng)
            Vs[t, active_indices] = 1.0
        death_times = h0_persistence(Vs)
        total_persistence[i] = death_times.sum()
        result = assign_schubert_cells_vectorized(Vs, flags, 0.5, r=k)
        mean_codim[i] = result["codim"].mean()

    rho_c, p_c = permutation_correlation_test(z, total_persistence, n_perm=5000, seed=seed)
    rho_a, p_a = permutation_correlation_test(z, mean_codim, n_perm=5000, seed=seed)
    rho_ac, _ = permutation_correlation_test(total_persistence, mean_codim, n_perm=5000, seed=seed)

    return {
        "k": k,
        "rho_persistence_vs_z": rho_c, "p_persistence_vs_z": p_c,
        "rho_codim_vs_z": rho_a, "p_codim_vs_z": p_a,
        "rho_persistence_vs_codim": rho_ac,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(power: Dict[int, float], phase_b: Dict[int, dict],
                  phase_c: Dict[int, dict], directory: str = "figures",
                  combined: bool = True):
    """
    Generates the synthetic validation study figures in PDF (vector
    format, ready to include in the LaTeX manuscript), with a professional
    style: colorblind-friendly palette, direct annotations on the key
    points, and a combined 3-panel (a)(b)(c) version in addition to the
    individual figures, since many journals (including Elsevier's)
    prefer composite figures when results are related.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(directory, exist_ok=True)

    # Colorblind-friendly palette (Wong, 2011)
    COLOR_SCHUBERT = "#0072B2"
    COLOR_ML = "#D55E00"
    COLOR_PERSISTENCE = "#009E73"
    COLOR_REF = "#999999"
    COLOR_K4 = "#CC79A7"

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })

    ks_p = sorted(power.keys())
    ks_b = sorted(phase_b.keys())
    ks_c = sorted(phase_c.keys())

    def _panel_power(ax):
        ax.axhspan(0, 0.8, color=COLOR_REF, alpha=0.08, zorder=0)
        ax.plot(ks_p, [power[k] for k in ks_p], marker="o", markersize=6,
                linewidth=2, color=COLOR_SCHUBERT, zorder=3)
        ax.axhline(0.8, linestyle="--", color=COLOR_REF, linewidth=1,
                   label="conventional power = 0.8")
        ax.axvline(4, linestyle=":", color=COLOR_K4, linewidth=1.4, zorder=2)
        ax.annotate(f"$k=4$\npower = {power[4]:.3f}",
                    xy=(4, power[4]), xytext=(9, 0.28),
                    fontsize=9, color=COLOR_K4,
                    arrowprops=dict(arrowstyle="->", color=COLOR_K4, lw=1.2))
        ax.set_xlabel("$k$ (number of species)")
        ax.set_ylabel("Statistical power")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, max(ks_p) + 3)
        ax.legend(loc="lower right", fontsize=8.5)

    def _panel_phase_b(ax):
        ax.axhline(0.5, linestyle="--", color=COLOR_REF, linewidth=1, label="chance (0.5)")
        ax.plot(ks_b, [phase_b[k]["ml_balanced_accuracy"] for k in ks_b],
                marker="s", markersize=6, linewidth=2, color=COLOR_ML,
                label="Random Forest")
        ax.plot(ks_b, [phase_b[k]["schubert_balanced_accuracy"] for k in ks_b],
                marker="o", markersize=6, linewidth=2, color=COLOR_SCHUBERT,
                label="Schubert (trivial)")
        ax.axvline(4, linestyle=":", color=COLOR_K4, linewidth=1.4, zorder=1)
        ax.set_xlabel("$k$ (number of species)")
        ax.set_ylabel("Balanced accuracy (leave-one-out)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(ks_b) + 3)
        ax.legend(loc="lower right", fontsize=8.5)

    def _panel_phase_c(ax):
        ax.plot(ks_c, [abs(phase_c[k]["rho_codim_vs_z"]) for k in ks_c],
                marker="o", markersize=6, linewidth=2, color=COLOR_SCHUBERT,
                label="Schubert codimension")
        ax.plot(ks_c, [abs(phase_c[k]["rho_persistence_vs_z"]) for k in ks_c],
                marker="^", markersize=7, linewidth=2, color=COLOR_PERSISTENCE,
                label="$H_0$ persistence")
        ax.axvline(4, linestyle=":", color=COLOR_K4, linewidth=1.4, zorder=1)
        ax.set_xlabel("$k$ (number of species)")
        ax.set_ylabel("$|\\rho|$ Spearman vs. $z$")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, max(ks_c) + 3)
        ax.legend(loc="center right", fontsize=8.5)

    directory_abs = os.path.abspath(directory)
    print(f"\nSaving figures to absolute path: {directory_abs}")

    def _save(fig, filename):
        path = os.path.join(directory, filename)
        try:
            fig.savefig(path)
            print(f"  saved: {os.path.abspath(path)}")
        except Exception as e:
            print(f"  ERROR saving {path}: {e}")
        plt.close(fig)

    # --- Individual figures ---
    fig, ax = plt.subplots(figsize=(5, 3.6))
    _panel_power(ax)
    _save(fig, "fig_power.pdf")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    _panel_phase_b(ax)
    _save(fig, "fig_phase_b.pdf")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    _panel_phase_c(ax)
    _save(fig, "fig_phase_c.pdf")

    # --- Combined 3-panel (a)(b)(c) figure ---
    if combined:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        _panel_power(axes[0])
        _panel_phase_b(axes[1])
        _panel_phase_c(axes[2])
        for ax, letter in zip(axes, "abc"):
            ax.text(-0.14, 1.08, f"({letter})", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="top")
        fig.tight_layout()
        _save(fig, "fig_synthetic_combined.pdf")

    print(f"\nDone. If you don't see these files in your Drive/local folder, "
          f"double-check the ABSOLUTE path printed above -- a relative "
          f"--figures-dir resolves against the current working directory "
          f"of the process running this script (often /content/ in Colab, "
          f"NOT your Drive folder, unless you pass a full path).")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=544, help="Dimension of E (default: same as real case)")
    parser.add_argument("--effect-size", type=float, default=0.7)
    parser.add_argument("--n-active-reactions", type=int, default=150)
    parser.add_argument("--n-repetitions", type=int, default=200)
    parser.add_argument("--k-values", default="4,8,12,20,30,50")
    parser.add_argument("--plot", action="store_true",
                         help="Generate PDF figures of the results in the 'figures/' folder")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]

    print("=" * 70)
    print("PART 1: recovery of the known signal with large k")
    print("=" * 70)
    large_k_result = run_study(k=50, m=args.m, effect_size=args.effect_size,
                                n_active_reactions=args.n_active_reactions)
    print(f"k=50, effect size={args.effect_size}: "
          f"rho={large_k_result['rho']:.3f}, p={large_k_result['p_value']:.4g}")

    control_result = run_study(k=50, m=args.m, effect_size=0.0,
                                n_active_reactions=args.n_active_reactions)
    print(f"k=50, effect size=0 (control, NO real signal): "
          f"rho={control_result['rho']:.3f}, p={control_result['p_value']:.4g}")

    print("\n" + "=" * 70)
    print(f"PART 2: power analysis (effect size={args.effect_size}, "
          f"{args.n_repetitions} repetitions per k)")
    print("=" * 70)
    power = power_analysis(k_values, n_repetitions=args.n_repetitions,
                            effect_size=args.effect_size, m=args.m,
                            n_active_reactions=args.n_active_reactions)

    print("\n" + "=" * 70)
    print("PART 3: Phase-B-style benchmark (Random Forest vs. trivial Schubert)")
    print("=" * 70)
    phase_b_results = {}
    for k in [4, 8, 20, 50]:
        r = synthetic_phase_b_benchmark(k, m=args.m, effect_size=args.effect_size,
                                         n_active_reactions=args.n_active_reactions)
        phase_b_results[k] = r
        print(f"  k={k:3d}: Random Forest balanced accuracy = "
              f"{r['ml_balanced_accuracy']:.3f}, "
              f"Schubert trivial = {r['schubert_balanced_accuracy']:.3f}")

    print("\n" + "=" * 70)
    print("PART 4: Phase C (H0 persistent homology) on synthetic data")
    print("=" * 70)
    phase_c_results = {}
    for k in [4, 8, 20, 50]:
        r = synthetic_phase_c(k, m=args.m, effect_size=args.effect_size,
                               n_active_reactions=args.n_active_reactions)
        phase_c_results[k] = r
        print(f"  k={k:3d}: H0 persistence vs z: rho={r['rho_persistence_vs_z']:.3f} "
              f"(p={r['p_persistence_vs_z']:.4g}); codim vs z: rho={r['rho_codim_vs_z']:.3f} "
              f"(p={r['p_codim_vs_z']:.4g}); persistence vs codim: rho={r['rho_persistence_vs_codim']:.3f}")

    if args.plot:
        plot_results(power, phase_b_results, phase_c_results,
                      directory=args.figures_dir)
    else:
        print("\nNOTE: figures were NOT generated or saved because --plot was not "
              "passed. Re-run with --plot --figures-dir <absolute_path> to save them.")

    return {"power": power, "phase_b": phase_b_results, "phase_c": phase_c_results}


if __name__ == "__main__":
    main()