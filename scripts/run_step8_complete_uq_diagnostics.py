#!/usr/bin/env python3

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


INPUT = Path(
    "step8/analysis/"
    "PRIMARY_SIGNED_ERROR_DECOMPOSITION_v1.csv.gz"
)

POINT = Path(
    "step8/analysis/"
    "PRIMARY_WEIGHTED_POINT_ESTIMATES_v1.json"
)

PRED_ROOT = Path("step8/predictions")
OUTDIR = Path("step8/analysis")

SUMMARY = OUTDIR / (
    "PRIMARY_COMPLETE_UQ_DIAGNOSTICS_v1.json"
)

REPS = OUTDIR / (
    "PRIMARY_COMPLETE_UQ_DIAGNOSTICS_BOOTSTRAP_REPLICATES_v1.csv"
)

CI_OUT = OUTDIR / (
    "PRIMARY_COMPLETE_UQ_DIAGNOSTICS_BOOTSTRAP_CI_v1.json"
)

AUDIT = OUTDIR / (
    "PRIMARY_COMPLETE_UQ_DIAGNOSTICS_AUDIT_v1.json"
)

B = 2000
SEED = 20260812
N_BINS = 10
EPS = 1e-15


MODELS = [
    "CHGNet",
    "MACE-MP-0",
    "SevenNet-l3i5",
    "ORB-v2-MPtrj",
    "GRACE-2L-MPtrj",
    "eqV2-S-DeNS",
    "eSEN-30M-OAM",
    "MACE-MPA-0",
]


ENERGY_COLS = {
    "CHGNet":
        "energy_error__CHGNet",

    "MACE-MP-0":
        "energy_error__MACE_MP_0",

    "SevenNet-l3i5":
        "energy_error__SevenNet_l3i5",

    "ORB-v2-MPtrj":
        "energy_error__ORB_v2_MPtrj",

    "GRACE-2L-MPtrj":
        "energy_error__GRACE_2L_MPtrj",

    "eqV2-S-DeNS":
        "energy_error__eqV2_S_DeNS",

    "eSEN-30M-OAM":
        "energy_error__eSEN_30M_OAM",

    "MACE-MPA-0":
        "energy_error__MACE_MPA_0",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def wmean(x, w):
    return float(
        np.sum(w * x)
        / np.sum(w)
    )


def wcorr(x, y, w):
    mx = wmean(x, w)
    my = wmean(y, w)

    dx = x - mx
    dy = y - my

    den = math.sqrt(
        np.sum(w * dx * dx)
        *
        np.sum(w * dy * dy)
    )

    if den <= 0:
        return float("nan")

    return float(
        np.sum(w * dx * dy)
        / den
    )


def weighted_spearman(x, y, w):
    return wcorr(
        rankdata(
            x,
            method="average",
        ),
        rankdata(
            y,
            method="average",
        ),
        w,
    )


def weighted_coverage(
    abs_error,
    sigma,
    k,
    w,
):
    return wmean(
        (
            abs_error
            <= k * sigma
        ).astype(float),
        w,
    )


def weighted_envelope_violation(
    abs_error,
    sigma,
    k,
    w,
):
    return wmean(
        (
            abs_error
            > k * sigma
        ).astype(float),
        w,
    )


def weighted_ence(
    abs_error,
    sigma,
    w,
):
    order = np.argsort(
        sigma,
        kind="mergesort",
    )

    a = abs_error[order]
    s = sigma[order]
    ww = w[order]

    total_w = np.sum(ww)

    cumulative = (
        np.cumsum(ww)
        / total_w
    )

    labels = np.minimum(
        (
            cumulative * N_BINS
        ).astype(int),
        N_BINS - 1,
    )

    ence = 0.0

    for b in range(N_BINS):
        mask = labels == b

        if not np.any(mask):
            continue

        wb = ww[mask]

        rmse = math.sqrt(
            wmean(
                a[mask] ** 2,
                wb,
            )
        )

        rmv = math.sqrt(
            wmean(
                s[mask] ** 2,
                wb,
            )
        )

        gap = (
            abs(rmse - rmv)
            / max(rmv, EPS)
        )

        ence += (
            np.sum(wb)
            / total_w
        ) * gap

    return float(ence)


print("=" * 78)
print(
    "STEP-8 COMPLETE UQ "
    "MAGNITUDE/CALIBRATION DIAGNOSTICS"
)
print("=" * 78)


# ==========================================================
# Frozen 2998 configurations
# ==========================================================

with gzip.open(
    INPUT,
    "rt",
    newline="",
) as f:
    rows = list(
        csv.DictReader(f)
    )

assert len(rows) == 2998

N = len(rows)
M = len(MODELS)

weights = np.asarray(
    [
        float(r["sampling_weight"])
        for r in rows
    ],
    dtype=float,
)

prototypes = np.asarray(
    [
        r["full_protostructure_label"]
        for r in rows
    ],
    dtype=object,
)


# ==========================================================
# ENERGY — exact 2998 x 8 signed matrix
# ==========================================================

E = np.column_stack([
    np.asarray(
        [
            float(
                r[
                    ENERGY_COLS[model]
                ]
            )
            for r in rows
        ],
        dtype=float,
    )
    for model in MODELS
])

assert E.shape == (N, M)
assert np.isfinite(E).all()


energy_mean_signed = np.mean(
    E,
    axis=1,
)

energy_abs_error = np.abs(
    energy_mean_signed
)

energy_total = np.mean(
    E ** 2,
    axis=1,
)

energy_popvar = np.var(
    E,
    axis=1,
    ddof=0,
)

energy_samplevar = np.var(
    E,
    axis=1,
    ddof=1,
)

energy_sigma_pop = np.sqrt(
    np.maximum(
        energy_popvar,
        0.0,
    )
)

energy_sigma_uq = np.sqrt(
    np.maximum(
        energy_samplevar,
        0.0,
    )
)


# ==========================================================
# FORCE
#
# Reconstruct exact atomwise channels from frozen predictions.
#
# UQ representation:
#   error_mean = mean_atom ||Fbar||
#   error_max  = max_atom  ||Fbar||
#   sigma_mean = mean_atom vector spread
#   sigma_max  = max_atom  vector spread
#
# Rank/calibration: unbiased ddof=1 spread.
# Exact F2 total/spread: population ddof=0 variance.
# ==========================================================

force_mean_error = np.empty(N)
force_max_error = np.empty(N)

force_mean_sigma_uq = np.empty(N)
force_max_sigma_uq = np.empty(N)

force_mean_sigma_pop = np.empty(N)
force_max_sigma_pop = np.empty(N)

force_mean_total = np.empty(N)
force_max_total = np.empty(N)

force_mean_popvar = np.empty(N)
force_max_popvar = np.empty(N)

force_mean_cm_naive = np.empty(N)
force_max_cm_naive = np.empty(N)


max_force_crosscheck = 0.0


for ii, row in enumerate(rows):

    mid = row["matpes_id"]

    ferr = []

    for model in MODELS:

        path = (
            PRED_ROOT
            / model
            / f"{mid}.npz"
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            fm = np.asarray(
                z[
                    "model_forces_eV_per_A"
                ],
                dtype=float,
            )

            fdft = np.asarray(
                z[
                    "dft_forces_eV_per_A"
                ],
                dtype=float,
            )

        ferr.append(
            fm - fdft
        )


    # M x atoms x xyz
    F = np.stack(
        ferr,
        axis=0,
    )

    Fbar = np.mean(
        F,
        axis=0,
    )

    error_atom = np.linalg.norm(
        Fbar,
        axis=1,
    )

    popvar_atom = np.sum(
        np.var(
            F,
            axis=0,
            ddof=0,
        ),
        axis=1,
    )

    samplevar_atom = np.sum(
        np.var(
            F,
            axis=0,
            ddof=1,
        ),
        axis=1,
    )

    sigma_pop_atom = np.sqrt(
        np.maximum(
            popvar_atom,
            0.0,
        )
    )

    sigma_uq_atom = np.sqrt(
        np.maximum(
            samplevar_atom,
            0.0,
        )
    )

    total_atom = np.mean(
        np.sum(
            F ** 2,
            axis=2,
        ),
        axis=0,
    )

    cm_atom = np.sum(
        Fbar ** 2,
        axis=1,
    )


    force_mean_error[ii] = (
        np.mean(error_atom)
    )

    force_max_error[ii] = (
        np.max(error_atom)
    )

    force_mean_sigma_uq[ii] = (
        np.mean(sigma_uq_atom)
    )

    force_max_sigma_uq[ii] = (
        np.max(sigma_uq_atom)
    )

    force_mean_sigma_pop[ii] = (
        np.mean(sigma_pop_atom)
    )

    force_max_sigma_pop[ii] = (
        np.max(sigma_pop_atom)
    )

    force_mean_total[ii] = (
        np.mean(total_atom)
    )

    force_max_total[ii] = (
        np.max(total_atom)
    )

    force_mean_popvar[ii] = (
        np.mean(popvar_atom)
    )

    force_max_popvar[ii] = (
        np.max(popvar_atom)
    )

    force_mean_cm_naive[ii] = (
        np.mean(cm_atom)
    )

    force_max_cm_naive[ii] = (
        np.max(cm_atom)
    )


    # Cross-check frozen H4 scalar channels.
    checks = [
        (
            force_mean_error[ii],
            float(
                row[
                    "force_ensemble_error_mean_atom"
                ]
            ),
        ),
        (
            force_max_error[ii],
            float(
                row[
                    "force_ensemble_error_max_atom"
                ]
            ),
        ),
        (
            force_mean_sigma_uq[ii],
            float(
                row[
                    "force_ensemble_spread_mean_atom"
                ]
            ),
        ),
        (
            force_max_sigma_uq[ii],
            float(
                row[
                    "force_ensemble_spread_max_atom"
                ]
            ),
        ),
        (
            force_mean_total[ii],
            float(
                row[
                    "force_total_mse_mean_atom"
                ]
            ),
        ),
        (
            force_max_total[ii],
            float(
                row[
                    "force_total_mse_max_atom"
                ]
            ),
        ),
    ]

    for a, b in checks:
        max_force_crosscheck = max(
            max_force_crosscheck,
            abs(a - b),
        )


    if (ii + 1) % 250 == 0:
        print(
            f"Force reconstruction "
            f"{ii + 1:,}/{N:,}"
        )


assert max_force_crosscheck < 1e-9

print(
    "Frozen force-channel cross-check: PASS"
)


# ==========================================================
# Metric function
# ==========================================================

def uq_metrics(
    idx,
    *,
    abs_error,
    sigma_uq,
    total_error_sq,
    popvar_sq_channel,
    sigma_pop_scalar,
):

    w = weights[idx]

    a = abs_error[idx]
    suq = sigma_uq[idx]

    total = total_error_sq[idx]
    popvar = popvar_sq_channel[idx]

    spop = sigma_pop_scalar[idx]


    # Exact/total squared-error to population-spread ratio.
    R_total = math.sqrt(
        wmean(
            total,
            w,
        )
        /
        max(
            wmean(
                popvar,
                w,
            ),
            EPS,
        )
    )


    # Ensemble-mean prediction error / scalar ensemble spread.
    # Same scientific role as
    # cme.ensemble_mean_to_spread_ratio().
    R_mean = math.sqrt(
        wmean(
            a ** 2,
            w,
        )
        /
        max(
            wmean(
                spop ** 2,
                w,
            ),
            EPS,
        )
    )


    cov1 = weighted_coverage(
        a,
        suq,
        1.0,
        w,
    )

    cov196 = weighted_coverage(
        a,
        suq,
        1.96,
        w,
    )

    env1 = (
        weighted_envelope_violation(
            a,
            suq,
            1.0,
            w,
        )
    )

    env196 = (
        weighted_envelope_violation(
            a,
            suq,
            1.96,
            w,
        )
    )


    return {
        "rank_spearman":
            weighted_spearman(
                suq,
                a,
                w,
            ),

        "ENCE":
            weighted_ence(
                a,
                suq,
                w,
            ),

        "coverage_k1":
            cov1,

        "coverage_k1_96":
            cov196,

        "envelope_violation_k1":
            env1,

        "envelope_violation_k1_96":
            env196,

        "R_total":
            R_total,

        "R_ensemble_mean":
            R_mean,

        "coverage_envelope_identity_k1":
            abs(
                cov1 + env1 - 1.0
            ),

        "coverage_envelope_identity_k1_96":
            abs(
                cov196 + env196 - 1.0
            ),
    }


full_idx = np.arange(
    N,
    dtype=int,
)


def all_metrics(idx):

    return {
        "energy":
            uq_metrics(
                idx,
                abs_error=
                    energy_abs_error,
                sigma_uq=
                    energy_sigma_uq,
                total_error_sq=
                    energy_total,
                popvar_sq_channel=
                    energy_popvar,
                sigma_pop_scalar=
                    energy_sigma_pop,
            ),

        "force_mean_over_atoms":
            uq_metrics(
                idx,
                abs_error=
                    force_mean_error,
                sigma_uq=
                    force_mean_sigma_uq,
                total_error_sq=
                    force_mean_total,
                popvar_sq_channel=
                    force_mean_popvar,
                sigma_pop_scalar=
                    force_mean_sigma_pop,
            ),

        "force_max_over_atoms":
            uq_metrics(
                idx,
                abs_error=
                    force_max_error,
                sigma_uq=
                    force_max_sigma_uq,
                total_error_sq=
                    force_max_total,
                popvar_sq_channel=
                    force_max_popvar,
                sigma_pop_scalar=
                    force_max_sigma_pop,
            ),
    }


point = all_metrics(
    full_idx
)


# ==========================================================
# Exact F2 consistency QC where decomposition identity
# remains valid: energy and mean-over-atoms force.
# ==========================================================

energy_naive_fcm = (
    wmean(
        energy_mean_signed ** 2,
        weights,
    )
    /
    wmean(
        energy_total,
        weights,
    )
)

force_mean_naive_fcm = (
    wmean(
        force_mean_cm_naive,
        weights,
    )
    /
    wmean(
        force_mean_total,
        weights,
    )
)

energy_F2_residual = abs(
    point["energy"]["R_total"] ** 2
    -
    1.0 / (
        1.0 - energy_naive_fcm
    )
)

force_F2_residual = abs(
    point[
        "force_mean_over_atoms"
    ]["R_total"] ** 2
    -
    1.0 / (
        1.0
        - force_mean_naive_fcm
    )
)


# ==========================================================
# Cross-check already frozen point estimates
# ==========================================================

frozen = json.loads(
    POINT.read_text()
)

crosschecks = [
    (
        point["energy"][
            "rank_spearman"
        ],
        frozen["energy"][
            "weighted_spearman_error_vs_spread"
        ],
    ),
    (
        point["energy"]["ENCE"],
        frozen["energy"][
            "calibration"
        ]["ENCE"],
    ),
    (
        point["energy"][
            "coverage_k1"
        ],
        frozen["energy"][
            "calibration"
        ]["coverage_1sigma"],
    ),
    (
        point["energy"][
            "R_total"
        ],
        frozen["energy"][
            "R_total_direct"
        ],
    ),
    (
        point[
            "force_mean_over_atoms"
        ]["rank_spearman"],
        frozen[
            "force_mean_over_atoms"
        ][
            "weighted_spearman_error_vs_spread"
        ],
    ),
    (
        point[
            "force_mean_over_atoms"
        ]["ENCE"],
        frozen[
            "force_mean_over_atoms"
        ]["calibration"]["ENCE"],
    ),
    (
        point[
            "force_mean_over_atoms"
        ]["coverage_k1"],
        frozen[
            "force_mean_over_atoms"
        ][
            "calibration"
        ]["coverage_1sigma"],
    ),
    (
        point[
            "force_mean_over_atoms"
        ]["R_total"],
        frozen[
            "force_mean_over_atoms"
        ]["R_total_direct"],
    ),
    (
        point[
            "force_max_over_atoms"
        ]["rank_spearman"],
        frozen[
            "force_max_over_atoms_control"
        ][
            "weighted_spearman_error_vs_spread"
        ],
    ),
    (
        point[
            "force_max_over_atoms"
        ]["ENCE"],
        frozen[
            "force_max_over_atoms_control"
        ]["calibration"]["ENCE"],
    ),
]


max_point_crosscheck = max(
    abs(
        float(a)
        - float(b)
    )
    for a, b
    in crosschecks
)

assert max_point_crosscheck < 1e-9

print(
    "Frozen UQ point-estimate cross-check: PASS"
)


# ==========================================================
# Prototype blocks
# ==========================================================

block_map = defaultdict(list)

for i, g in enumerate(
    prototypes
):
    block_map[g].append(i)

block_labels = sorted(
    block_map.keys()
)

blocks = [
    np.asarray(
        block_map[g],
        dtype=int,
    )
    for g in block_labels
]

G = len(blocks)

assert G == 2674


# ==========================================================
# Bootstrap — deterministic, same prototype-block convention
# ==========================================================

flat_metric_names = []

for channel in [
    "energy",
    "force_mean_over_atoms",
    "force_max_over_atoms",
]:
    for metric in point[channel]:
        if metric.startswith(
            "coverage_envelope_identity"
        ):
            continue

        flat_metric_names.append(
            f"{channel}__{metric}"
        )


with open(
    REPS,
    "w",
    newline="",
) as f:

    fields = [
        "replicate",
        "n_rows",
        "n_unique_blocks_drawn",
    ] + flat_metric_names

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    writer.writeheader()

    for rep in range(
        1,
        B + 1,
    ):

        rng = np.random.default_rng(
            SEED + rep
        )

        draw = rng.integers(
            0,
            G,
            size=G,
        )

        idx = np.concatenate([
            blocks[j]
            for j in draw
        ])

        m = all_metrics(idx)

        rec = {
            "replicate":
                rep,

            "n_rows":
                len(idx),

            "n_unique_blocks_drawn":
                len(
                    set(
                        draw.tolist()
                    )
                ),
        }

        for channel in [
            "energy",
            "force_mean_over_atoms",
            "force_max_over_atoms",
        ]:
            for metric, value in (
                m[channel].items()
            ):

                if metric.startswith(
                    "coverage_envelope_identity"
                ):
                    continue

                rec[
                    f"{channel}__{metric}"
                ] = value

        writer.writerow(rec)

        if rep % 200 == 0:
            print(
                f"Bootstrap "
                f"{rep:,}/{B:,}"
            )


# ==========================================================
# Confidence intervals
# ==========================================================

with open(
    REPS,
    "r",
    newline="",
) as f:
    rep_rows = list(
        csv.DictReader(f)
    )

assert len(rep_rows) == B


ci = {}

for name in flat_metric_names:

    vals = np.asarray(
        [
            float(r[name])
            for r in rep_rows
        ],
        dtype=float,
    )

    q025, q50, q975 = (
        np.percentile(
            vals,
            [2.5, 50, 97.5],
        )
    )

    channel, metric = (
        name.split(
            "__",
            1,
        )
    )

    ci.setdefault(
        channel,
        {},
    )

    ci[channel][metric] = {
        "point_estimate":
            float(
                point[channel][metric]
            ),

        "bootstrap_median":
            float(q50),

        "CI95_lower":
            float(q025),

        "CI95_upper":
            float(q975),

        "bootstrap_sd":
            float(
                np.std(
                    vals,
                    ddof=1,
                )
            ),
    }


CI_OUT.write_text(
    json.dumps(
        {
            "stage":
                "STEP8_COMPLETE_UQ_DIAGNOSTICS_BOOTSTRAP",

            "n_configs":
                N,

            "prototype_blocks":
                G,

            "bootstrap_replicates":
                B,

            "seed":
                SEED,

            "bootstrap_unit":
                "full_protostructure_label",

            "metrics":
                ci,
        },
        indent=2,
    )
)


# ==========================================================
# Summary
# ==========================================================

summary = {
    "stage":
        "STEP8_COMPLETE_UQ_MAGNITUDE_CALIBRATION",

    "n_configs":
        N,

    "M_core":
        M,

    "calibration_spread":
        (
            "unbiased across-model vector/scalar "
            "spread, ddof=1, matching frozen "
            "primary rank/ENCE channels"
        ),

    "R_total_spread":
        (
            "population across-model variance, "
            "ddof=0, matching exact F2 definition"
        ),

    "R_ensemble_mean":
        (
            "RMS ensemble-mean actual error divided "
            "by RMS population ensemble spread"
        ),

    "coverage_levels":
        [
            1.0,
            1.96,
        ],

    "ENCE_bins":
        N_BINS,

    "weighting":
        "frozen inverse-inclusion sampling weights",

    "point_estimates":
        point,

    "energy_F2_identity_abs_residual":
        energy_F2_residual,

    "force_mean_F2_identity_abs_residual":
        force_F2_residual,

    "force_max_F2_identity_not_asserted":
        (
            "max-over-atoms reduction need not preserve "
            "the exact additive common-mode identity "
            "because extrema of total, common-mode and "
            "model-specific components may occur at "
            "different atoms"
        ),

    "hypothesis_decision_performed":
        False,
}

SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


# ==========================================================
# Audit
# ==========================================================

identity_max = max(
    energy_F2_residual,
    force_F2_residual,
)

coverage_identity_max = max(
    point[c][k]
    for c in point
    for k in point[c]
    if k.startswith(
        "coverage_envelope_identity"
    )
)

status = (
    "PASS"
    if (
        max_force_crosscheck < 1e-9
        and max_point_crosscheck < 1e-9
        and identity_max < 1e-10
        and coverage_identity_max < 1e-12
        and len(rep_rows) == B
    )
    else "REVISE"
)


audit = {
    "stage":
        "STEP8_COMPLETE_UQ_MAGNITUDE_CALIBRATION",

    "status":
        status,

    "n_configs":
        N,

    "M_core":
        M,

    "force_channel_crosscheck_max_abs_difference":
        max_force_crosscheck,

    "frozen_point_crosscheck_max_abs_difference":
        max_point_crosscheck,

    "energy_F2_identity_abs_residual":
        energy_F2_residual,

    "force_mean_F2_identity_abs_residual":
        force_F2_residual,

    "coverage_plus_envelope_identity_max_abs_residual":
        coverage_identity_max,

    "bootstrap_replicates":
        B,

    "summary_file":
        str(SUMMARY),

    "summary_sha256":
        sha256(SUMMARY),

    "replicates_file":
        str(REPS),

    "replicates_sha256":
        sha256(REPS),

    "ci_file":
        str(CI_OUT),

    "ci_sha256":
        sha256(CI_OUT),

    "source":
        str(INPUT),

    "source_sha256":
        sha256(INPUT),

    "sample_membership_changed":
        False,

    "primary_estimates_overwritten":
        False,

    "scientific_hypothesis_decision_performed":
        False,
}

AUDIT.write_text(
    json.dumps(
        audit,
        indent=2,
    )
)


# ==========================================================
# Console
# ==========================================================

print("\n" + "=" * 78)
print("COMPLETE UQ DIAGNOSTICS")
print("=" * 78)

for channel, label in [
    (
        "energy",
        "ENERGY",
    ),
    (
        "force_mean_over_atoms",
        "FORCE MEAN-OVER-ATOMS",
    ),
    (
        "force_max_over_atoms",
        "FORCE MAX-OVER-ATOMS",
    ),
]:

    x = point[channel]

    print(f"\n{label}")

    print(
        "  Rank rho          :",
        f"{x['rank_spearman']:.6f}"
    )

    print(
        "  ENCE              :",
        f"{x['ENCE']:.6f}"
    )

    print(
        "  Coverage 1sigma   :",
        f"{x['coverage_k1']:.6f}"
    )

    print(
        "  Coverage 1.96sigma:",
        f"{x['coverage_k1_96']:.6f}"
    )

    print(
        "  Envelope viol. 1  :",
        f"{x['envelope_violation_k1']:.6f}"
    )

    print(
        "  Envelope viol.1.96:",
        f"{x['envelope_violation_k1_96']:.6f}"
    )

    print(
        "  R_total           :",
        f"{x['R_total']:.6f}"
    )

    print(
        "  R_ensemble_mean   :",
        f"{x['R_ensemble_mean']:.6f}"
    )


print("\nQC")

print(
    "  Force channel max diff :",
    f"{max_force_crosscheck:.3e}"
)

print(
    "  Frozen point max diff  :",
    f"{max_point_crosscheck:.3e}"
)

print(
    "  Energy F2 residual     :",
    f"{energy_F2_residual:.3e}"
)

print(
    "  Force-mean F2 residual :",
    f"{force_F2_residual:.3e}"
)

print(
    "  Coverage identity max  :",
    f"{coverage_identity_max:.3e}"
)


print("\nFiles")
print("Summary   :", SUMMARY)
print("Replicates:", REPS)
print("CI        :", CI_OUT)
print("Audit     :", AUDIT)

print("\nSHA256")
print("Summary   :", sha256(SUMMARY))
print("Replicates:", sha256(REPS))
print("CI        :", sha256(CI_OUT))
print("Audit     :", sha256(AUDIT))

print("\n" + "=" * 78)

if status == "PASS":
    print(
        "STEP8 COMPLETE UQ DIAGNOSTICS: PASS"
    )
else:
    print(
        "STEP8 COMPLETE UQ DIAGNOSTICS: REVISE"
    )
