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

POINT_FILE = Path(
    "step8/analysis/"
    "PRIMARY_WEIGHTED_POINT_ESTIMATES_v1.json"
)

OUTDIR = Path("step8/analysis")

REPS = OUTDIR / (
    "PRIMARY_PROTOTYPE_BLOCK_BOOTSTRAP_REPLICATES_v1.csv"
)

CI_OUT = OUTDIR / (
    "PRIMARY_PROTOTYPE_BLOCK_BOOTSTRAP_CI_v1.json"
)

AUDIT_OUT = OUTDIR / (
    "PRIMARY_PROTOTYPE_BLOCK_BOOTSTRAP_AUDIT_v1.json"
)

B = 2000
SEED = 20260812
N_CAL_BINS = 10
EPS = 1e-15


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
        np.sum(w * x) / np.sum(w)
    )


def wcorr(x, y, w):
    mx = wmean(x, w)
    my = wmean(y, w)

    dx = x - mx
    dy = y - my

    num = np.sum(w * dx * dy)

    den = math.sqrt(
        np.sum(w * dx * dx)
        * np.sum(w * dy * dy)
    )

    if den <= 0:
        return float("nan")

    return float(num / den)


def weighted_spearman(x, y, w):
    rx = rankdata(
        x,
        method="average",
    )

    ry = rankdata(
        y,
        method="average",
    )

    return wcorr(
        rx,
        ry,
        w,
    )


def calibration(error, sigma, w):
    error = np.asarray(
        error,
        dtype=float,
    )

    sigma = np.asarray(
        sigma,
        dtype=float,
    )

    w = np.asarray(
        w,
        dtype=float,
    )

    order = np.argsort(
        sigma,
        kind="mergesort",
    )

    error = error[order]
    sigma = sigma[order]
    w = w[order]

    total_w = np.sum(w)

    cumulative = (
        np.cumsum(w) / total_w
    )

    labels = np.minimum(
        (
            cumulative * N_CAL_BINS
        ).astype(int),
        N_CAL_BINS - 1,
    )

    ence = 0.0

    for b in range(N_CAL_BINS):
        mask = labels == b

        if not np.any(mask):
            continue

        wb = w[mask]

        rmse = math.sqrt(
            wmean(
                error[mask] ** 2,
                wb,
            )
        )

        rmv = math.sqrt(
            wmean(
                sigma[mask] ** 2,
                wb,
            )
        )

        gap = abs(
            rmse - rmv
        ) / max(rmv, EPS)

        ence += (
            np.sum(wb) / total_w
        ) * gap

    coverage1 = wmean(
        (
            error <= sigma
        ).astype(float),
        w,
    )

    coverage2 = wmean(
        (
            error <= 2.0 * sigma
        ).astype(float),
        w,
    )

    return {
        "ENCE":
            float(ence),

        "coverage_1sigma":
            coverage1,

        "coverage_2sigma":
            coverage2,
    }


print("=" * 78)
print(
    "STEP-8 PRIMARY PROTOTYPE-BLOCK BOOTSTRAP"
)
print("=" * 78)


# ==========================================================
# Load frozen decomposition
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


def A(key):
    return np.asarray(
        [
            float(row[key])
            for row in rows
        ],
        dtype=float,
    )


weights = A("sampling_weight")
deq = A("d_eq")

decile = np.asarray(
    [
        int(row["deq_decile"])
        for row in rows
    ],
    dtype=int,
)

prototype = np.asarray(
    [
        row[
            "full_protostructure_label"
        ]
        for row in rows
    ],
    dtype=object,
)


# ENERGY

e_total = A(
    "energy_total_mse_mean_models"
)

e_cm_naive = A(
    "energy_cm_naive"
)

e_cm_corr = A(
    "energy_cm_corrected"
)

e_specific = A(
    "energy_model_specific_popvar"
)

e_error = A(
    "energy_error_mean_abs"
)

e_spread = A(
    "energy_ensemble_spread"
)


# FORCE mean-over-atoms

f_total = A(
    "force_total_mse_mean_atom"
)

f_cm_naive = A(
    "force_cm_naive_mean_atom"
)

f_cm_corr = A(
    "force_cm_corrected_mean_atom"
)

f_specific = A(
    "force_model_specific_popvar_mean_atom"
)

f_error = A(
    "force_ensemble_error_mean_atom"
)

f_spread = A(
    "force_ensemble_spread_mean_atom"
)


# MAX-FORCE CONTROL

fmax_error = A(
    "force_ensemble_error_max_atom"
)

fmax_spread = A(
    "force_ensemble_spread_max_atom"
)


assert np.isfinite(weights).all()
assert np.all(weights > 0)


# ==========================================================
# Prototype blocks
# ==========================================================

block_map = defaultdict(list)

for i, label in enumerate(prototype):
    block_map[label].append(i)

block_labels = sorted(
    block_map.keys()
)

blocks = [
    np.asarray(
        block_map[label],
        dtype=int,
    )
    for label in block_labels
]

G = len(blocks)

print("Configurations :", len(rows))
print("Prototype blocks:", G)
print("Bootstrap reps  :", B)
print("Master seed     :", SEED)


# ==========================================================
# Metrics
# ==========================================================

def decomposition_metrics(
    total,
    cm_naive,
    cm_corr,
    specific,
    error,
    spread,
    idx,
):
    w = weights[idx]

    total_w = wmean(
        total[idx],
        w,
    )

    naive_w = wmean(
        cm_naive[idx],
        w,
    )

    corr_w = wmean(
        cm_corr[idx],
        w,
    )

    specific_w = wmean(
        specific[idx],
        w,
    )

    f_naive = (
        naive_w / total_w
    )

    f_corr = (
        corr_w / total_w
    )

    R = math.sqrt(
        total_w
        / max(specific_w, EPS)
    )

    rho = weighted_spearman(
        error[idx],
        spread[idx],
        w,
    )

    cal = calibration(
        error[idx],
        spread[idx],
        w,
    )

    return {
        "f_naive": f_naive,
        "f_corr": f_corr,
        "R": R,
        "rho": rho,
        "ENCE": cal["ENCE"],
        "coverage1":
            cal["coverage_1sigma"],
        "coverage2":
            cal["coverage_2sigma"],
    }


def compute_metrics(idx):
    w = weights[idx]

    E = decomposition_metrics(
        e_total,
        e_cm_naive,
        e_cm_corr,
        e_specific,
        e_error,
        e_spread,
        idx,
    )

    F = decomposition_metrics(
        f_total,
        f_cm_naive,
        f_cm_corr,
        f_specific,
        f_error,
        f_spread,
        idx,
    )

    max_rho = weighted_spearman(
        fmax_error[idx],
        fmax_spread[idx],
        w,
    )

    max_cal = calibration(
        fmax_error[idx],
        fmax_spread[idx],
        w,
    )


    # Continuous d_eq associations

    e_R_i = np.sqrt(
        e_total[idx]
        /
        np.maximum(
            e_specific[idx],
            EPS,
        )
    )

    f_R_i = np.sqrt(
        f_total[idx]
        /
        np.maximum(
            f_specific[idx],
            EPS,
        )
    )


    # Extreme deciles

    idx1 = idx[
        decile[idx] == 1
    ]

    idx10 = idx[
        decile[idx] == 10
    ]

    if (
        len(idx1) == 0
        or len(idx10) == 0
    ):
        raise RuntimeError(
            "Bootstrap replicate missing D01 or D10"
        )

    E1 = decomposition_metrics(
        e_total,
        e_cm_naive,
        e_cm_corr,
        e_specific,
        e_error,
        e_spread,
        idx1,
    )

    E10 = decomposition_metrics(
        e_total,
        e_cm_naive,
        e_cm_corr,
        e_specific,
        e_error,
        e_spread,
        idx10,
    )

    F1 = decomposition_metrics(
        f_total,
        f_cm_naive,
        f_cm_corr,
        f_specific,
        f_error,
        f_spread,
        idx1,
    )

    F10 = decomposition_metrics(
        f_total,
        f_cm_naive,
        f_cm_corr,
        f_specific,
        f_error,
        f_spread,
        idx10,
    )


    return {
        "energy_f_CM_corrected":
            E["f_corr"],

        "energy_f_CM_naive":
            E["f_naive"],

        "energy_R_total":
            E["R"],

        "force_f_CM_corrected":
            F["f_corr"],

        "force_f_CM_naive":
            F["f_naive"],

        "force_R_total":
            F["R"],

        "energy_rank_rho":
            E["rho"],

        "energy_ENCE":
            E["ENCE"],

        "energy_coverage_1sigma":
            E["coverage1"],

        "energy_coverage_2sigma":
            E["coverage2"],

        "force_rank_rho":
            F["rho"],

        "force_ENCE":
            F["ENCE"],

        "force_coverage_1sigma":
            F["coverage1"],

        "force_coverage_2sigma":
            F["coverage2"],

        "force_max_rank_rho":
            max_rho,

        "force_max_ENCE":
            max_cal["ENCE"],

        "deq_vs_energy_corrected_CM":
            weighted_spearman(
                deq[idx],
                e_cm_corr[idx],
                w,
            ),

        "deq_vs_force_corrected_CM":
            weighted_spearman(
                deq[idx],
                f_cm_corr[idx],
                w,
            ),

        "deq_vs_energy_error":
            weighted_spearman(
                deq[idx],
                e_error[idx],
                w,
            ),

        "deq_vs_energy_spread":
            weighted_spearman(
                deq[idx],
                e_spread[idx],
                w,
            ),

        "deq_vs_force_error":
            weighted_spearman(
                deq[idx],
                f_error[idx],
                w,
            ),

        "deq_vs_force_spread":
            weighted_spearman(
                deq[idx],
                f_spread[idx],
                w,
            ),

        "deq_vs_energy_R_config":
            weighted_spearman(
                deq[idx],
                e_R_i,
                w,
            ),

        "deq_vs_force_R_config":
            weighted_spearman(
                deq[idx],
                f_R_i,
                w,
            ),

        "energy_f_CM_D10_minus_D01":
            E10["f_corr"]
            - E1["f_corr"],

        "force_f_CM_D10_minus_D01":
            F10["f_corr"]
            - F1["f_corr"],

        "energy_R_D10_minus_D01":
            E10["R"]
            - E1["R"],

        "force_R_D10_minus_D01":
            F10["R"]
            - F1["R"],

        "energy_rho_D10_minus_D01":
            E10["rho"]
            - E1["rho"],

        "force_rho_D10_minus_D01":
            F10["rho"]
            - F1["rho"],

        "energy_ENCE_D10_minus_D01":
            E10["ENCE"]
            - E1["ENCE"],

        "force_ENCE_D10_minus_D01":
            F10["ENCE"]
            - F1["ENCE"],
    }


# ==========================================================
# Full-sample point estimate
# ==========================================================

full_idx = np.arange(
    len(rows),
    dtype=int,
)

point = compute_metrics(
    full_idx
)


# ==========================================================
# Cross-check frozen point estimates
# ==========================================================

frozen = json.loads(
    POINT_FILE.read_text()
)

checks = {
    "energy_f_CM_corrected":
        frozen["energy"][
            "f_CM_corrected"
        ],

    "energy_f_CM_naive":
        frozen["energy"][
            "f_CM_naive"
        ],

    "energy_R_total":
        frozen["energy"][
            "R_total_direct"
        ],

    "energy_rank_rho":
        frozen["energy"][
            "weighted_spearman_error_vs_spread"
        ],

    "energy_ENCE":
        frozen["energy"][
            "calibration"
        ]["ENCE"],

    "force_f_CM_corrected":
        frozen[
            "force_mean_over_atoms"
        ]["f_CM_corrected"],

    "force_R_total":
        frozen[
            "force_mean_over_atoms"
        ]["R_total_direct"],

    "force_rank_rho":
        frozen[
            "force_mean_over_atoms"
        ][
            "weighted_spearman_error_vs_spread"
        ],

    "force_ENCE":
        frozen[
            "force_mean_over_atoms"
        ][
            "calibration"
        ]["ENCE"],
}

max_point_difference = 0.0

for key, frozen_value in checks.items():
    diff = abs(
        point[key]
        - float(frozen_value)
    )

    max_point_difference = max(
        max_point_difference,
        diff,
    )

assert max_point_difference < 1e-12

print(
    "Frozen point-estimate cross-check: PASS"
)


# ==========================================================
# Resumable bootstrap
# ==========================================================

metric_names = list(
    point.keys()
)

fieldnames = [
    "replicate",
    "n_rows",
    "n_unique_blocks_drawn",
] + metric_names


completed = set()

if REPS.exists():
    with open(
        REPS,
        "r",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != fieldnames:
            raise RuntimeError(
                "Existing bootstrap file schema mismatch"
            )

        for row in reader:
            completed.add(
                int(row["replicate"])
            )


print(
    "Existing completed reps:",
    len(completed),
)


need_header = not REPS.exists()

with open(
    REPS,
    "a",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    if need_header:
        writer.writeheader()
        f.flush()

    for rep in range(1, B + 1):

        if rep in completed:
            continue

        rng = np.random.default_rng(
            SEED + rep
        )

        draw = rng.integers(
            low=0,
            high=G,
            size=G,
        )

        idx = np.concatenate(
            [
                blocks[j]
                for j in draw
            ]
        )

        metrics = compute_metrics(
            idx
        )

        result = {
            "replicate":
                rep,

            "n_rows":
                len(idx),

            "n_unique_blocks_drawn":
                len(
                    set(draw.tolist())
                ),

            **metrics,
        }

        if not all(
            np.isfinite(
                float(result[k])
            )
            for k in metric_names
        ):
            raise RuntimeError(
                f"Nonfinite metric at rep {rep}"
            )

        writer.writerow(result)
        f.flush()

        if rep % 100 == 0:
            print(
                f"Bootstrap {rep:,}/{B:,}"
            )


# ==========================================================
# Reload complete replicates
# ==========================================================

with open(
    REPS,
    "r",
    newline="",
) as f:
    rep_rows = list(
        csv.DictReader(f)
    )

rep_numbers = [
    int(x["replicate"])
    for x in rep_rows
]

assert len(rep_rows) == B
assert len(set(rep_numbers)) == B
assert set(rep_numbers) == set(
    range(1, B + 1)
)


# ==========================================================
# 95% percentile confidence intervals
# ==========================================================

contrast_metrics = {
    "energy_f_CM_D10_minus_D01",
    "force_f_CM_D10_minus_D01",
    "energy_R_D10_minus_D01",
    "force_R_D10_minus_D01",
    "energy_rho_D10_minus_D01",
    "force_rho_D10_minus_D01",
    "energy_ENCE_D10_minus_D01",
    "force_ENCE_D10_minus_D01",

    "deq_vs_energy_corrected_CM",
    "deq_vs_force_corrected_CM",
    "deq_vs_energy_error",
    "deq_vs_energy_spread",
    "deq_vs_force_error",
    "deq_vs_force_spread",
    "deq_vs_energy_R_config",
    "deq_vs_force_R_config",
}


CI = {}

for key in metric_names:

    values = np.asarray(
        [
            float(x[key])
            for x in rep_rows
        ],
        dtype=float,
    )

    q025, q50, q975 = np.percentile(
        values,
        [2.5, 50.0, 97.5],
    )

    entry = {
        "point_estimate":
            float(point[key]),

        "bootstrap_median":
            float(q50),

        "CI95_percentile_lower":
            float(q025),

        "CI95_percentile_upper":
            float(q975),

        "bootstrap_sd":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
    }

    if key in contrast_metrics:

        entry[
            "bootstrap_fraction_positive"
        ] = float(
            np.mean(values > 0)
        )

        entry[
            "bootstrap_fraction_negative"
        ] = float(
            np.mean(values < 0)
        )

        entry[
            "CI_excludes_zero"
        ] = bool(
            q025 > 0
            or q975 < 0
        )

    CI[key] = entry


ci_record = {
    "stage":
        "STEP8_PRIMARY_PROTOTYPE_BLOCK_BOOTSTRAP",

    "status":
        "PASS",

    "n_configs":
        len(rows),

    "n_prototype_blocks":
        G,

    "bootstrap_replicates":
        B,

    "seed":
        SEED,

    "bootstrap_unit":
        "full_protostructure_label",

    "sampling_weights":
        "frozen inverse-inclusion weights retained",

    "confidence_interval":
        "95% percentile prototype-block bootstrap",

    "metrics":
        CI,

    "hypothesis_decisions":
        "NOT_RECORDED_IN_THIS_STAGE",
}

CI_OUT.write_text(
    json.dumps(
        ci_record,
        indent=2,
    )
)


audit = {
    "stage":
        "STEP8_PRIMARY_PROTOTYPE_BLOCK_BOOTSTRAP",

    "status":
        "PASS",

    "source_decomposition":
        str(INPUT),

    "source_decomposition_sha256":
        sha256(INPUT),

    "source_point_estimates":
        str(POINT_FILE),

    "source_point_estimates_sha256":
        sha256(POINT_FILE),

    "n_configs":
        len(rows),

    "n_prototype_blocks":
        G,

    "bootstrap_replicates":
        B,

    "master_seed":
        SEED,

    "point_estimate_max_abs_crosscheck_difference":
        max_point_difference,

    "point_estimate_crosscheck_pass":
        True,

    "replicates_file":
        str(REPS),

    "replicates_sha256":
        sha256(REPS),

    "ci_file":
        str(CI_OUT),

    "ci_sha256":
        sha256(CI_OUT),

    "prototype_blocking_preserved":
        True,

    "frozen_sampling_weights_preserved":
        True,

    "sample_membership_changed":
        False,

    "model_roster_changed":
        False,

    "scientific_hypothesis_decision_performed":
        False,
}

AUDIT_OUT.write_text(
    json.dumps(
        audit,
        indent=2,
    )
)


# ==========================================================
# Console
# ==========================================================

print("\n" + "=" * 78)
print("PROTOTYPE-BLOCK BOOTSTRAP AUDIT")
print("=" * 78)

print("Configs          :", len(rows))
print("Prototype blocks :", G)
print("Replicates       :", B)


print("\nGLOBAL C1")

for key in [
    "energy_f_CM_corrected",
    "force_f_CM_corrected",
    "energy_R_total",
    "force_R_total",
]:
    x = CI[key]

    print(
        f"{key:32s} "
        f"{x['point_estimate']:.6f} "
        f"[{x['CI95_percentile_lower']:.6f}, "
        f"{x['CI95_percentile_upper']:.6f}]"
    )


print("\nGLOBAL C2/C3")

for key in [
    "energy_rank_rho",
    "energy_ENCE",
    "force_rank_rho",
    "force_ENCE",
    "force_max_rank_rho",
    "force_max_ENCE",
]:
    x = CI[key]

    print(
        f"{key:32s} "
        f"{x['point_estimate']:.6f} "
        f"[{x['CI95_percentile_lower']:.6f}, "
        f"{x['CI95_percentile_upper']:.6f}]"
    )


print("\nCONTINUOUS d_eq")

for key in [
    "deq_vs_energy_corrected_CM",
    "deq_vs_force_corrected_CM",
    "deq_vs_energy_R_config",
    "deq_vs_force_R_config",
]:
    x = CI[key]

    print(
        f"{key:32s} "
        f"{x['point_estimate']:.6f} "
        f"[{x['CI95_percentile_lower']:.6f}, "
        f"{x['CI95_percentile_upper']:.6f}]"
    )


print("\nD10 - D01")

for key in [
    "energy_f_CM_D10_minus_D01",
    "force_f_CM_D10_minus_D01",
    "energy_R_D10_minus_D01",
    "force_R_D10_minus_D01",
    "energy_ENCE_D10_minus_D01",
    "force_ENCE_D10_minus_D01",
]:
    x = CI[key]

    print(
        f"{key:32s} "
        f"{x['point_estimate']:.6f} "
        f"[{x['CI95_percentile_lower']:.6f}, "
        f"{x['CI95_percentile_upper']:.6f}]"
    )


print("\nFiles")
print("Replicates:", REPS)
print("CI        :", CI_OUT)
print("Audit     :", AUDIT_OUT)

print("\nSHA256")
print("Replicates:", sha256(REPS))
print("CI        :", sha256(CI_OUT))
print("Audit     :", sha256(AUDIT_OUT))

print("\n" + "=" * 78)
print(
    "STEP8 PRIMARY PROTOTYPE-BLOCK "
    "BOOTSTRAP: PASS"
)
