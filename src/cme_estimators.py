"""
Common-mode error estimators — frozen Step 6 implementation.

Data-agnostic: every function operates on error/prediction matrices or geometry,
so the scientific safeguards can be validated on synthetic ground truth without
real MatPES data or model weights. The same functions run unchanged on the real
configuration-by-model error matrices produced by the inference pipeline.

Conventions:
  E : signed error matrix, shape (n_config, M). e[i,m] = pred_{i,m} - ref_i.
  For forces, the per-atom vector form is reduced to a per-atom squared error and
  aggregated (mean-over-atoms = dispersion; max-over-atoms = extreme) BEFORE entering
  these functions as a scalar-per-config channel.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------- decomposition
def per_config_components(E: np.ndarray):
    """F1 identity pieces. Returns ebar (n,), s2_pop (n,, ddof=0), sig2_unb (n,, ddof=1), mse (n,)."""
    E = np.asarray(E, float)
    ebar = E.mean(1)
    s2_pop = E.var(1, ddof=0)
    sig2_unb = E.var(1, ddof=1)
    mse = (E ** 2).mean(1)
    return ebar, s2_pop, sig2_unb, mse


def fcm_biascorrected(E: np.ndarray):
    """
    Bias-corrected common-mode fraction (F3), stratum-level scalar.
    mu2_hat = mean_i ebar_i^2 - mean_i sig2_unb_i / M   (unbiased for mu^2)
    sig2_hat = mean_i sig2_unb_i                        (unbiased for sigma^2)
    f_CM = [mu2_hat]_+ / ([mu2_hat]_+ + sig2_hat)
    """
    E = np.asarray(E, float)
    M = E.shape[1]
    ebar, _, sig2_unb, _ = per_config_components(E)
    mu2_hat = (ebar ** 2).mean() - sig2_unb.mean() / M
    sig2_hat = sig2_unb.mean()
    cm = max(0.0, mu2_hat)
    return cm / (cm + sig2_hat) if (cm + sig2_hat) > 0 else np.nan


def fcm_naive(E: np.ndarray):
    """Uncorrected f_CM = mean(ebar^2)/mean(mse). Retained only to demonstrate the finite-M floor."""
    ebar, _, _, mse = per_config_components(E)
    num = (ebar ** 2).mean()
    return num / mse.mean() if mse.mean() > 0 else np.nan


def intermodel_corr(E: np.ndarray):
    """Independent validator: mean pairwise Pearson correlation of signed errors across configs."""
    E = np.asarray(E, float)
    M = E.shape[1]
    if E.shape[0] < 3:
        return np.nan
    C = np.corrcoef(E.T)  # M x M
    iu = np.triu_indices(M, k=1)
    vals = C[iu]
    return np.nanmean(vals)


# ---------------------------------------------------------------- ensemble UQ
def ensemble_spread(E: np.ndarray, ddof: int = 0):
    """
    Across-model SD per configuration.

    ddof=0 is used for the exact F2 algebra with TOTAL model MSE
    and the NAIVE common-mode fraction.
    """
    E = np.asarray(E, float)
    return E.std(axis=1, ddof=ddof)


def underestimation_ratio_total(E: np.ndarray):
    """
    Exact F2 diagnostic:

        R_total^2 =
            mean_{i,m}(e_im^2) /
            mean_i(sigma_ens,i^2)

    With ddof=0:
        R_total^2 = 1 / (1 - f_CM_naive)

    This identity does NOT apply exactly to bias-corrected f_CM.
    """
    E = np.asarray(E, float)

    sigma = ensemble_spread(E, ddof=0)
    denom = float(np.mean(sigma**2))

    if denom <= 0:
        return np.nan

    total_mse = float(np.mean(E**2))
    return float(np.sqrt(total_mse / denom))


def ensemble_mean_to_spread_ratio(E: np.ndarray):
    """
    Calibration diagnostic for the ensemble-mean predictor.

    This is scientifically useful but is NOT the exact F2 identity.
    """
    E = np.asarray(E, float)

    ebar = np.mean(E, axis=1)
    sigma = ensemble_spread(E, ddof=0)

    denom = float(np.mean(sigma**2))

    if denom <= 0:
        return np.nan

    return float(
        np.sqrt(np.mean(ebar**2) / denom)
    )


def underestimation_ratio(E: np.ndarray):
    """
    Backward-compatible name for the corrected exact-F2 R_total.
    """
    return underestimation_ratio_total(E)


# ---------------------------------------------------------------- rank / calibration
def spearman_rank(sigma: np.ndarray, abs_err: np.ndarray):
    if len(sigma) < 3:
        return np.nan
    return spearmanr(sigma, abs_err).statistic


def coverage(abs_err: np.ndarray, sigma: np.ndarray, k: float):
    return float(np.mean(abs_err <= k * sigma))


def ence(abs_err: np.ndarray, sigma: np.ndarray, n_bins: int = 10):
    """Expected Normalized Calibration Error: bin by sigma, |RMSE_b - RMSsigma_b|/RMSsigma_b."""
    order = np.argsort(sigma)
    a, s = abs_err[order], sigma[order]
    idx = np.array_split(np.arange(len(s)), n_bins)
    terms = []
    for b in idx:
        if len(b) == 0:
            continue
        rmse = np.sqrt(np.mean(a[b] ** 2))
        rmv = np.sqrt(np.mean(s[b] ** 2))
        if rmv > 0:
            terms.append(abs(rmse - rmv) / rmv)
    return float(np.mean(terms)) if terms else np.nan


def envelope_violation(abs_err: np.ndarray, sigma: np.ndarray, k: float = 1.0):
    """Fraction of points outside the k-sigma predicted envelope (Perez/POPS style)."""
    return float(np.mean(abs_err > k * sigma))


# ---------------------------------------------------------------- per-atom statistic representations
def reduce_per_atom(err_atoms: np.ndarray, how: str):
    """err_atoms: (n_config, n_atoms) magnitude of per-atom error for one model. Returns (n_config,)."""
    if how == "mean":       # dispersion statistic
        return err_atoms.mean(1)
    if how == "max":        # extreme-value statistic
        return err_atoms.max(1)
    raise ValueError(how)


# ---------------------------------------------------------------- structure-only d_eq (leakage-proof)
def d_eq_components(pos_config, cell_config, pos_parent, cell_parent):
    """
    Structure-only d_eq components.

    Primary Step-6 implementation:
      u = RMS minimum-image displacement / mean parent NN distance
      epsV = |log(V/V0)|

    No energies, forces, or model outputs enter this function.
    """
    import numpy as np
    from ase.geometry import find_mic

    pos_config = np.asarray(pos_config, dtype=float)
    pos_parent = np.asarray(pos_parent, dtype=float)
    cell_config = np.asarray(cell_config, dtype=float)
    cell_parent = np.asarray(cell_parent, dtype=float)

    if pos_config.shape != pos_parent.shape:
        raise ValueError("Config/parent coordinate shape mismatch")

    N = len(pos_parent)
    if N < 2:
        raise ValueError("Need at least 2 sites for NN normalization")

    # Corresponding-site displacement, using parent lattice and full MIC.
    raw_cart = (pos_config - pos_parent) @ cell_parent
    mic_cart, _ = find_mic(
        raw_cart,
        cell_parent,
        pbc=[True, True, True],
    )
    rms_disp = float(
        np.sqrt(np.mean(np.sum(mic_cart**2, axis=1)))
    )

    # Mean nearest-neighbour distance in relaxed parent.
    parent_cart = pos_parent @ cell_parent
    pair_vec = (
        parent_cart[None, :, :] - parent_cart[:, None, :]
    ).reshape(-1, 3)

    _, pair_dist = find_mic(
        pair_vec,
        cell_parent,
        pbc=[True, True, True],
    )

    D = np.asarray(pair_dist).reshape(N, N)
    np.fill_diagonal(D, np.inf)

    nn = np.min(D, axis=1)
    d_nn0 = float(np.mean(nn))

    if not np.isfinite(d_nn0) or d_nn0 <= 0:
        raise ValueError("Invalid parent nearest-neighbour distance")

    V0 = abs(float(np.linalg.det(cell_parent)))
    V = abs(float(np.linalg.det(cell_config)))

    if V <= 0 or V0 <= 0:
        raise ValueError("Non-positive cell volume")

    u = rms_disp / d_nn0
    epsV = abs(np.log(V / V0))

    return float(u), float(epsV)


def robust_scale(x):
    """
    Robust scale WITHOUT centering.

    Keeping the physical origin fixed is essential:
    exact equilibrium u=epsV=0 must map to d_eq=0.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))

    if np.isfinite(mad) and mad > 0:
        return float(mad)

    # Robust fallbacks for degenerate distributions.
    q25, q75 = np.percentile(x, [25, 75])
    iqr_scale = (q75 - q25) / 1.349

    if np.isfinite(iqr_scale) and iqr_scale > 0:
        return float(iqr_scale)

    positive = np.abs(x[np.abs(x) > 0])
    if len(positive):
        return float(np.median(positive))

    return 1.0


def fit_deq_scaler(u_raw, epsV_raw):
    """
    Fit scale parameters on the frozen cohort.
    No centering is performed.
    """
    return {
        "u_scale": robust_scale(u_raw),
        "epsV_scale": robust_scale(epsV_raw),
    }


def d_eq_from_components(
    u_raw,
    epsV_raw,
    scaler=None,
    return_scaler=False,
):
    """
    Frozen primary coordinate after amendment:
        d_eq = sqrt((u/s_u)^2 + (epsV/s_eps)^2)

    scaler must ultimately be fitted once on the frozen full cohort
    and reused unchanged for pilot/evaluation subsets.
    """
    import numpy as np

    u_raw = np.asarray(u_raw, dtype=float)
    epsV_raw = np.asarray(epsV_raw, dtype=float)

    if scaler is None:
        scaler = fit_deq_scaler(u_raw, epsV_raw)

    su = float(scaler["u_scale"])
    se = float(scaler["epsV_scale"])

    if su <= 0 or se <= 0:
        raise ValueError("Invalid d_eq scale")

    d = np.sqrt((u_raw / su)**2 + (epsV_raw / se)**2)

    if return_scaler:
        return d, scaler
    return d


# ---------------------------------------------------------------- protostructure-aware bootstrap
def cluster_bootstrap(values_by_cluster, statistic, n_boot=2000, seed=0):
    """
    values_by_cluster: list of arrays; each array = the per-config values for one cluster
                       (cluster = protostructure). Resample CLUSTERS with replacement.
    statistic: callable(concatenated_values) -> scalar.
    Returns (point, lo, hi) with 95% percentile CI.
    """
    rng = np.random.default_rng(seed)
    clusters = list(values_by_cluster)
    K = len(clusters)
    point = statistic(np.concatenate(clusters))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, K, K)
        boots[b] = statistic(np.concatenate([clusters[j] for j in pick]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def naive_bootstrap(values, statistic, n_boot=2000, seed=0):
    """Config-level iid resample (WRONG for clustered data; used only to show it under-covers)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    n = len(v)
    point = statistic(v)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = statistic(v[rng.integers(0, n, n)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


# ============================================================
# Signed vector-force common-mode decomposition
# ============================================================

def fcm_biascorrected_vector(F: np.ndarray):
    """
    Bias-corrected common-mode fraction for signed VECTOR errors.

    F shape:
        (n_observations, M, 3)

    Each observation may be an atom-level force-error vector.
    """
    F = np.asarray(F, dtype=float)

    if F.ndim != 3 or F.shape[2] != 3:
        raise ValueError("F must have shape (n_obs, M, 3)")

    M = F.shape[1]
    if M < 2:
        return np.nan

    mean_vec = F.mean(axis=1)

    # unbiased across-model vector variance for each observation
    dev = F - mean_vec[:, None, :]
    sig2_unb = np.sum(dev**2, axis=(1, 2)) / (M - 1)

    mean_vec_sq = np.sum(mean_vec**2, axis=1)

    cm_hat = mean_vec_sq.mean() - sig2_unb.mean() / M
    idio_hat = sig2_unb.mean()

    cm = max(0.0, float(cm_hat))
    denom = cm + float(idio_hat)

    return cm / denom if denom > 0 else np.nan


def force_ensemble_spread(F: np.ndarray):
    """
    RMS vector spread across models per atom/observation.

    F shape: (n_obs, M, 3)
    returns: (n_obs,)
    """
    F = np.asarray(F, dtype=float)

    mean_vec = F.mean(axis=1, keepdims=True)
    sq = np.sum((F - mean_vec)**2, axis=2)

    return np.sqrt(np.mean(sq, axis=1))
