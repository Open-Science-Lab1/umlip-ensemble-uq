#!/usr/bin/env python3

import csv
import gzip
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


INPUT = Path(
    "step8/controls/r2scan/analysis/"
    "R2SCAN_SIGNED_ENERGY_ERROR_DECOMPOSITION_v1.csv.gz"
)

EXPECTED_INPUT_SHA256 = (
    "3936d268f34c2046577ea2e47559a4ed"
    "679fb7507e4c8de1d5b7c2fca31e8ed8"
)

POINT_FILE = Path(
    "step8/controls/r2scan/analysis/"
    "R2SCAN_WEIGHTED_POINT_ESTIMATES_v1.json"
)

EXPECTED_POINT_SHA256 = (
    "385bd90555ab521d55bfad1d441ce300"
    "0635d3670308ea736a7cf605a95f7671"
)

DECILE_FILE = Path(
    "step8/controls/r2scan/analysis/"
    "R2SCAN_DEQ_DECILE_SUMMARY_v1.csv"
)

EXPECTED_DECILE_SHA256 = (
    "2f5f5d0c7cfbda5166090d9c9db27801"
    "a1526620b15bd5493e6fac163f2f4c37"
)

OUTDIR = Path(
    "step8/controls/r2scan/analysis"
)

REPS = OUTDIR / (
    "R2SCAN_PROTOTYPE_BLOCK_BOOTSTRAP_REPLICATES_v1.csv"
)

CI_OUT = OUTDIR / (
    "R2SCAN_PROTOTYPE_BLOCK_BOOTSTRAP_CI_v1.json"
)

AUDIT_OUT = OUTDIR / (
    "R2SCAN_PROTOTYPE_BLOCK_BOOTSTRAP_AUDIT_v1.json"
)

B = 2000
SEED = 20260812
N_CAL_BINS = 10
EPS = 1e-15

N_EXPECTED = 2996
M_CORE = 8


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def A(rows, key):
    return np.asarray(
        [
            float(row[key])
            for row in rows
        ],
        dtype=float,
    )


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

    num = np.sum(
        w * dx * dy
    )

    den = math.sqrt(
        np.sum(w * dx * dx)
        *
        np.sum(w * dy * dy)
    )

    if den <= 0:
        return float("nan")

    return float(
        num / den
    )


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
        np.cumsum(w)
        / total_w
    )

    labels = np.minimum(
        (
            cumulative
            * N_CAL_BINS
        ).astype(int),
        N_CAL_BINS - 1,
    )

    ence = 0.0

    for b in range(
        N_CAL_BINS
    ):
        mask = (
            labels == b
        )

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

        gap = (
            abs(rmse - rmv)
            / max(
                rmv,
                EPS,
            )
        )

        ence += (
            np.sum(wb)
            / total_w
        ) * gap

    coverage1 = wmean(
        (
            error <= sigma
        ).astype(float),
        w,
    )

    # Exact primary prototype-bootstrap convention.
    coverage2 = wmean(
        (
            error
            <= 2.0 * sigma
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
    "STEP-8 r2SCAN ENERGY "
    "PROTOTYPE-BLOCK BOOTSTRAP"
)
print("=" * 78)


# ==========================================================
# Frozen-source integrity
# ==========================================================

assert INPUT.exists(), INPUT
assert POINT_FILE.exists(), POINT_FILE
assert DECILE_FILE.exists(), DECILE_FILE

input_sha = sha256(INPUT)
point_sha = sha256(POINT_FILE)
decile_sha = sha256(DECILE_FILE)

assert (
    input_sha
    == EXPECTED_INPUT_SHA256
)

assert (
    point_sha
    == EXPECTED_POINT_SHA256
)

assert (
    decile_sha
    == EXPECTED_DECILE_SHA256
)


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

assert len(rows) == N_EXPECTED


weights = A(
    rows,
    "sampling_weight",
)

deq = A(
    rows,
    "d_eq",
)

decile = np.asarray(
    [
        int(
            row["deq_decile"]
        )
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


e_total = A(
    rows,
    "energy_total_mse_mean_models",
)

e_cm_naive = A(
    rows,
    "energy_cm_naive",
)

e_cm_corr = A(
    rows,
    "energy_cm_corrected",
)

e_specific = A(
    rows,
    "energy_model_specific_popvar",
)

e_error = A(
    rows,
    "energy_error_mean_abs",
)

e_spread = A(
    rows,
    "energy_ensemble_spread",
)


assert np.isfinite(
    weights
).all()

assert np.all(
    weights > 0
)

for x in [
    deq,
    e_total,
    e_cm_naive,
    e_cm_corr,
    e_specific,
    e_error,
    e_spread,
]:
    assert np.isfinite(x).all()


# ==========================================================
# Prototype blocks
# ==========================================================

block_map = defaultdict(
    list
)

for i, label in enumerate(
    prototype
):
    block_map[
        label
    ].append(i)


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

assert G == N_EXPECTED

# r2SCAN sampling selected one configuration
# per distinct prototype.
assert all(
    len(block) == 1
    for block in blocks
)


print(
    "Configurations :",
    len(rows),
)

print(
    "Prototype blocks:",
    G,
)

print(
    "Bootstrap reps :",
    B,
)

print(
    "Master seed    :",
    SEED,
)


# ==========================================================
# Metrics
# ==========================================================

def decomposition_metrics(idx):
    w = weights[idx]

    total_w = wmean(
        e_total[idx],
        w,
    )

    naive_w = wmean(
        e_cm_naive[idx],
        w,
    )

    corr_w = wmean(
        e_cm_corr[idx],
        w,
    )

    specific_w = wmean(
        e_specific[idx],
        w,
    )

    f_naive = (
        naive_w
        / total_w
    )

    f_corr = (
        corr_w
        / total_w
    )

    R = math.sqrt(
        total_w
        / max(
            specific_w,
            EPS,
        )
    )

    rho = weighted_spearman(
        e_error[idx],
        e_spread[idx],
        w,
    )

    cal = calibration(
        e_error[idx],
        e_spread[idx],
        w,
    )

    return {
        "f_naive":
            f_naive,

        "f_corr":
            f_corr,

        "R":
            R,

        "rho":
            rho,

        "ENCE":
            cal["ENCE"],

        "coverage1":
            cal[
                "coverage_1sigma"
            ],

        "coverage2":
            cal[
                "coverage_2sigma"
            ],
    }


def compute_metrics(idx):
    w = weights[idx]

    E = decomposition_metrics(
        idx
    )

    e_R_i = np.sqrt(
        e_total[idx]
        /
        np.maximum(
            e_specific[idx],
            EPS,
        )
    )

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
        idx1
    )

    E10 = decomposition_metrics(
        idx10
    )

    return {
        "energy_f_CM_corrected":
            E["f_corr"],

        "energy_f_CM_naive":
            E["f_naive"],

        "energy_R_total":
            E["R"],

        "energy_rank_rho":
            E["rho"],

        "energy_ENCE":
            E["ENCE"],

        "energy_coverage_1sigma":
            E["coverage1"],

        "energy_coverage_2sigma":
            E["coverage2"],

        "deq_vs_energy_corrected_CM":
            weighted_spearman(
                deq[idx],
                e_cm_corr[idx],
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

        "deq_vs_energy_R_config":
            weighted_spearman(
                deq[idx],
                e_R_i,
                w,
            ),

        "energy_f_CM_D10_minus_D01":
            (
                E10["f_corr"]
                - E1["f_corr"]
            ),

        "energy_R_D10_minus_D01":
            (
                E10["R"]
                - E1["R"]
            ),

        "energy_rho_D10_minus_D01":
            (
                E10["rho"]
                - E1["rho"]
            ),

        "energy_ENCE_D10_minus_D01":
            (
                E10["ENCE"]
                - E1["ENCE"]
            ),
    }


# ==========================================================
# Full-sample point estimate
# ==========================================================

full_idx = np.arange(
    N_EXPECTED,
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
        frozen[
            "energy"
        ][
            "f_CM_corrected"
        ],

    "energy_f_CM_naive":
        frozen[
            "energy"
        ][
            "f_CM_naive"
        ],

    "energy_R_total":
        frozen[
            "energy"
        ][
            "R_total_direct"
        ],

    "energy_rank_rho":
        frozen[
            "energy"
        ][
            "weighted_spearman_error_vs_spread"
        ],

    "energy_ENCE":
        frozen[
            "energy"
        ][
            "calibration"
        ][
            "ENCE"
        ],

    "energy_coverage_1sigma":
        frozen[
            "energy"
        ][
            "calibration"
        ][
            "coverage_1sigma"
        ],

    "energy_coverage_2sigma":
        frozen[
            "energy"
        ][
            "calibration"
        ][
            "coverage_2sigma"
        ],

    "deq_vs_energy_corrected_CM":
        frozen[
            "continuous_d_eq_descriptive_associations"
        ][
            "energy_d_eq_vs_corrected_CM"
        ],

    "deq_vs_energy_error":
        frozen[
            "continuous_d_eq_descriptive_associations"
        ][
            "energy_d_eq_vs_abs_ensemble_error"
        ],

    "deq_vs_energy_spread":
        frozen[
            "continuous_d_eq_descriptive_associations"
        ][
            "energy_d_eq_vs_spread"
        ],
}


max_point_difference = max(
    abs(
        float(
            point[key]
        )
        - float(value)
    )
    for key, value
    in checks.items()
)

assert (
    max_point_difference
    < 1e-10
), max_point_difference


# ==========================================================
# D01 / D10 cross-check against frozen decile table
# ==========================================================

with open(
    DECILE_FILE,
    "r",
    newline="",
) as f:
    dec_rows = list(
        csv.DictReader(f)
    )

assert len(dec_rows) == 10

d1_row = next(
    x
    for x in dec_rows
    if int(
        x["deq_decile"]
    ) == 1
)

d10_row = next(
    x
    for x in dec_rows
    if int(
        x["deq_decile"]
    ) == 10
)


decile_checks = {
    "energy_f_CM_D10_minus_D01":
        (
            float(
                d10_row[
                    "energy_f_CM_corrected"
                ]
            )
            -
            float(
                d1_row[
                    "energy_f_CM_corrected"
                ]
            )
        ),

    "energy_R_D10_minus_D01":
        (
            float(
                d10_row[
                    "energy_R_total"
                ]
            )
            -
            float(
                d1_row[
                    "energy_R_total"
                ]
            )
        ),

    "energy_rho_D10_minus_D01":
        (
            float(
                d10_row[
                    "energy_rank_error_vs_spread"
                ]
            )
            -
            float(
                d1_row[
                    "energy_rank_error_vs_spread"
                ]
            )
        ),

    "energy_ENCE_D10_minus_D01":
        (
            float(
                d10_row[
                    "energy_ENCE"
                ]
            )
            -
            float(
                d1_row[
                    "energy_ENCE"
                ]
            )
        ),
}


max_decile_crosscheck = max(
    abs(
        point[key]
        - value
    )
    for key, value
    in decile_checks.items()
)

assert (
    max_decile_crosscheck
    < 1e-10
), max_decile_crosscheck


print(
    "Frozen point-estimate cross-check: PASS"
)

print(
    "Frozen D01/D10 cross-check      : PASS"
)


# ==========================================================
# Bootstrap — exact master-RNG block convention
# Resumable without changing RNG sequence.
# ==========================================================

metric_names = list(
    point.keys()
)

fields = [
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

        if (
            reader.fieldnames
            != fields
        ):
            raise RuntimeError(
                "Existing bootstrap file schema mismatch"
            )

        for row in reader:
            completed.add(
                int(
                    row[
                        "replicate"
                    ]
                )
            )

    print(
        "Existing completed reps:",
        len(completed),
    )


need_header = (
    not REPS.exists()
    or REPS.stat().st_size == 0
)


rng = np.random.default_rng(
    SEED
)


with open(
    REPS,
    "a",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields,
    )

    if need_header:
        writer.writeheader()

    for rep in range(
        1,
        B + 1,
    ):

        # Always generate the draw, even for an already
        # completed replicate, so resuming preserves the
        # exact master-RNG sequence.
        draw = rng.integers(
            0,
            G,
            size=G,
        )

        if rep in completed:
            continue

        idx = np.concatenate(
            [
                blocks[j]
                for j in draw
            ]
        )

        metrics = compute_metrics(
            idx
        )

        if not all(
            np.isfinite(
                float(value)
            )
            for value in metrics.values()
        ):
            raise RuntimeError(
                f"Nonfinite metric at replicate {rep}"
            )

        rec = {
            "replicate":
                rep,

            "n_rows":
                len(idx),

            "n_unique_blocks_drawn":
                int(
                    len(
                        np.unique(
                            draw
                        )
                    )
                ),
        }

        rec.update(
            metrics
        )

        writer.writerow(
            rec
        )

        f.flush()

        if rep % 200 == 0:
            print(
                f"Bootstrap {rep:,}/{B:,}"
            )


# ==========================================================
# Validate completed bootstrap
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
    int(
        row["replicate"]
    )
    for row in rep_rows
]

assert len(rep_rows) == B
assert len(set(rep_numbers)) == B
assert set(rep_numbers) == set(
    range(
        1,
        B + 1,
    )
)


# ==========================================================
# Confidence intervals
# ==========================================================

CI = {}

for metric in metric_names:

    values = np.asarray(
        [
            float(
                row[metric]
            )
            for row in rep_rows
        ],
        dtype=float,
    )

    assert np.isfinite(
        values
    ).all()

    q025, q50, q975 = np.percentile(
        values,
        [
            2.5,
            50.0,
            97.5,
        ],
    )

    point_value = float(
        point[metric]
    )

    CI[metric] = {
        "point_estimate":
            point_value,

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

        "bootstrap_fraction_positive":
            float(
                np.mean(
                    values > 0
                )
            ),

        "bootstrap_fraction_negative":
            float(
                np.mean(
                    values < 0
                )
            ),

        "CI_excludes_zero":
            bool(
                q025 > 0
                or q975 < 0
            ),
    }


ci_record = {
    "stage":
        "STEP8_R2SCAN_PROTOTYPE_BLOCK_BOOTSTRAP",

    "scope":
        "secondary_reference_functional_energy_only_control",

    "reference_functional":
        "r2SCAN",

    "n_configs":
        N_EXPECTED,

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

    "scientific_hypothesis_decision":
        "NOT_YET_AUTHORIZED",
}


tmp = CI_OUT.with_suffix(
    ".json.tmp"
)

tmp.write_text(
    json.dumps(
        ci_record,
        indent=2,
    )
)

os.replace(
    tmp,
    CI_OUT,
)


# ==========================================================
# Audit
# ==========================================================

status = "PASS"

audit = {
    "stage":
        "STEP8_R2SCAN_PROTOTYPE_BLOCK_BOOTSTRAP",

    "status":
        status,

    "scope":
        "secondary_reference_functional_energy_only_control",

    "n_configs":
        N_EXPECTED,

    "M_core":
        M_CORE,

    "n_prototype_blocks":
        G,

    "one_configuration_per_prototype":
        True,

    "bootstrap_replicates":
        B,

    "master_seed":
        SEED,

    "point_estimate_max_abs_crosscheck_difference":
        float(
            max_point_difference
        ),

    "point_estimate_crosscheck_pass":
        bool(
            max_point_difference
            < 1e-10
        ),

    "D01_D10_max_abs_crosscheck_difference":
        float(
            max_decile_crosscheck
        ),

    "D01_D10_crosscheck_pass":
        bool(
            max_decile_crosscheck
            < 1e-10
        ),

    "prototype_blocking_preserved":
        True,

    "frozen_sampling_weights_preserved":
        True,

    "sample_membership_changed":
        False,

    "sample_replacements":
        0,

    "model_roster_changed":
        False,

    "primary_PBE_results_changed":
        False,

    "forces_used":
        False,

    "scientific_hypothesis_decision_performed":
        False,

    "source_decomposition":
        str(INPUT),

    "source_decomposition_sha256":
        input_sha,

    "source_point_estimates":
        str(POINT_FILE),

    "source_point_estimates_sha256":
        point_sha,

    "source_decile_summary":
        str(DECILE_FILE),

    "source_decile_summary_sha256":
        decile_sha,

    "replicates_file":
        str(REPS),

    "replicates_sha256":
        sha256(REPS),

    "ci_file":
        str(CI_OUT),

    "ci_sha256":
        sha256(CI_OUT),
}


tmp = AUDIT_OUT.with_suffix(
    ".json.tmp"
)

tmp.write_text(
    json.dumps(
        audit,
        indent=2,
    )
)

os.replace(
    tmp,
    AUDIT_OUT,
)


# ==========================================================
# Console
# ==========================================================

print()
print("=" * 78)
print(
    "r2SCAN PROTOTYPE-BLOCK BOOTSTRAP AUDIT"
)
print("=" * 78)

print(
    "Configs          :",
    N_EXPECTED,
)

print(
    "Prototype blocks :",
    G,
)

print(
    "Replicates       :",
    B,
)

print()
print(
    "GLOBAL ENERGY"
)

for key in [
    "energy_f_CM_corrected",
    "energy_R_total",
    "energy_rank_rho",
    "energy_ENCE",
]:
    x = CI[key]

    print(
        f"{key:34s} "
        f"{x['point_estimate']:+.6f} "
        f"[{x['CI95_percentile_lower']:+.6f}, "
        f"{x['CI95_percentile_upper']:+.6f}]"
    )


print()
print(
    "CONTINUOUS d_eq"
)

for key in [
    "deq_vs_energy_corrected_CM",
    "deq_vs_energy_error",
    "deq_vs_energy_spread",
    "deq_vs_energy_R_config",
]:
    x = CI[key]

    print(
        f"{key:34s} "
        f"{x['point_estimate']:+.6f} "
        f"[{x['CI95_percentile_lower']:+.6f}, "
        f"{x['CI95_percentile_upper']:+.6f}]"
    )


print()
print(
    "D10 - D01"
)

for key in [
    "energy_f_CM_D10_minus_D01",
    "energy_R_D10_minus_D01",
    "energy_rho_D10_minus_D01",
    "energy_ENCE_D10_minus_D01",
]:
    x = CI[key]

    print(
        f"{key:34s} "
        f"{x['point_estimate']:+.6f} "
        f"[{x['CI95_percentile_lower']:+.6f}, "
        f"{x['CI95_percentile_upper']:+.6f}]"
    )


print()
print(
    "Replicates SHA256:",
    sha256(REPS),
)

print(
    "CI SHA256        :",
    sha256(CI_OUT),
)

print(
    "Audit SHA256     :",
    sha256(AUDIT_OUT),
)

print(
    "\nr2SCAN PROTOTYPE-BLOCK BOOTSTRAP:",
    status,
)
