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

OUTDIR = Path("step8/analysis")

TABLE = OUTDIR / (
    "PRIMARY_COMPOSITION_CLASS_SENSITIVITY_v1.csv"
)

CI_OUT = OUTDIR / (
    "PRIMARY_COMPOSITION_CLASS_BOOTSTRAP_CI_v1.json"
)

REPS_OUT = OUTDIR / (
    "PRIMARY_COMPOSITION_CLASS_BOOTSTRAP_REPLICATES_v1.csv.gz"
)

DEVIATION_OUT = OUTDIR / (
    "CHEMICAL_FAMILY_PREREGISTRATION_DEVIATION_v1.json"
)

AUDIT_OUT = OUTDIR / (
    "PRIMARY_COMPOSITION_CLASS_SENSITIVITY_AUDIT_v1.json"
)

B = 2000
SEED = 20260812
EPS = 1e-15
N_CAL_BINS = 10

CLASSES = [
    "binary",
    "ternary",
    "quaternary",
    "5plus",
]

CLASS_OFFSET = {
    "binary": 10000,
    "ternary": 20000,
    "quaternary": 30000,
    "5plus": 40000,
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
        rankdata(x, method="average"),
        rankdata(y, method="average"),
        w,
    )


def kish_ess(w):
    w = np.asarray(
        w,
        dtype=float,
    )

    return float(
        np.sum(w) ** 2
        / np.sum(w ** 2)
    )


def prototype_ess(w, groups):
    totals = defaultdict(float)

    for wi, gi in zip(w, groups):
        totals[gi] += float(wi)

    return kish_ess(
        np.asarray(
            list(totals.values()),
            dtype=float,
        )
    )


def calibration(error, sigma, w):
    error = np.asarray(error, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w = np.asarray(w, dtype=float)

    order = np.argsort(
        sigma,
        kind="mergesort",
    )

    error = error[order]
    sigma = sigma[order]
    w = w[order]

    total_w = np.sum(w)

    cum = (
        np.cumsum(w)
        / total_w
    )

    labels = np.minimum(
        (
            cum * N_CAL_BINS
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

        ence += (
            np.sum(wb)
            / total_w
        ) * (
            abs(rmse - rmv)
            / max(rmv, EPS)
        )

    return float(ence)


print("=" * 78)
print(
    "STEP-8 COMPOSITION-CLASS "
    "ROBUSTNESS SENSITIVITY"
)
print("=" * 78)


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
            float(x[key])
            for x in rows
        ],
        dtype=float,
    )


weights = A("sampling_weight")
deq = A("d_eq")

classes = np.asarray(
    [
        x["composition_class"]
        for x in rows
    ],
    dtype=object,
)

prototypes = np.asarray(
    [
        x["full_protostructure_label"]
        for x in rows
    ],
    dtype=object,
)


# Energy
e_total = A(
    "energy_total_mse_mean_models"
)

e_cm = A(
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


# Force mean
f_total = A(
    "force_total_mse_mean_atom"
)

f_cm = A(
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


# Force max control
fmax_error = A(
    "force_ensemble_error_max_atom"
)

fmax_spread = A(
    "force_ensemble_spread_max_atom"
)


def metrics(idx):
    w = weights[idx]

    et = wmean(
        e_total[idx],
        w,
    )

    ft = wmean(
        f_total[idx],
        w,
    )

    espec = wmean(
        e_specific[idx],
        w,
    )

    fspec = wmean(
        f_specific[idx],
        w,
    )

    e_R_cfg = np.sqrt(
        e_total[idx]
        /
        np.maximum(
            e_specific[idx],
            EPS,
        )
    )

    f_R_cfg = np.sqrt(
        f_total[idx]
        /
        np.maximum(
            f_specific[idx],
            EPS,
        )
    )

    return {
        "energy_f_CM_corrected":
            wmean(
                e_cm[idx],
                w,
            ) / et,

        "force_f_CM_corrected":
            wmean(
                f_cm[idx],
                w,
            ) / ft,

        "energy_R_total":
            math.sqrt(
                et / max(espec, EPS)
            ),

        "force_R_total":
            math.sqrt(
                ft / max(fspec, EPS)
            ),

        "energy_rank_rho":
            weighted_spearman(
                e_error[idx],
                e_spread[idx],
                w,
            ),

        "force_rank_rho":
            weighted_spearman(
                f_error[idx],
                f_spread[idx],
                w,
            ),

        "energy_ENCE":
            calibration(
                e_error[idx],
                e_spread[idx],
                w,
            ),

        "force_ENCE":
            calibration(
                f_error[idx],
                f_spread[idx],
                w,
            ),

        "force_max_rank_rho":
            weighted_spearman(
                fmax_error[idx],
                fmax_spread[idx],
                w,
            ),

        "force_max_ENCE":
            calibration(
                fmax_error[idx],
                fmax_spread[idx],
                w,
            ),

        "deq_vs_energy_corrected_CM":
            weighted_spearman(
                deq[idx],
                e_cm[idx],
                w,
            ),

        "deq_vs_force_corrected_CM":
            weighted_spearman(
                deq[idx],
                f_cm[idx],
                w,
            ),

        "deq_vs_energy_R_config":
            weighted_spearman(
                deq[idx],
                e_R_cfg,
                w,
            ),

        "deq_vs_force_R_config":
            weighted_spearman(
                deq[idx],
                f_R_cfg,
                w,
            ),
    }


# ==========================================================
# Point estimates + support
# ==========================================================

point = {}
table_rows = []

for cls in CLASSES:

    idx = np.flatnonzero(
        classes == cls
    )

    assert len(idx) > 0

    w = weights[idx]
    g = prototypes[idx]

    m = metrics(idx)

    p_ess = prototype_ess(
        w,
        g,
    )

    record = {
        "composition_class":
            cls,

        "n_configs":
            len(idx),

        "n_prototypes":
            len(np.unique(g)),

        "row_level_kish_ESS":
            kish_ess(w),

        "prototype_level_kish_ESS":
            p_ess,

        "ESS_ge_30":
            bool(p_ess >= 30.0),

        **m,
    }

    point[cls] = record
    table_rows.append(record)


with open(
    TABLE,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            table_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        table_rows
    )


# ==========================================================
# Prototype-block bootstrap within each class
# ==========================================================

metric_names = list(
    metrics(
        np.arange(len(rows))
    ).keys()
)

rep_rows = []

for cls in CLASSES:

    idx_cls = np.flatnonzero(
        classes == cls
    )

    groups = defaultdict(list)

    for i in idx_cls:
        groups[
            prototypes[i]
        ].append(i)

    block_labels = sorted(
        groups.keys()
    )

    blocks = [
        np.asarray(
            groups[g],
            dtype=int,
        )
        for g in block_labels
    ]

    G = len(blocks)

    print(
        f"\n{cls}: "
        f"n={len(idx_cls)}, "
        f"prototypes={G}, "
        f"protoESS="
        f"{point[cls]['prototype_level_kish_ESS']:.2f}"
    )

    for rep in range(1, B + 1):

        rng = np.random.default_rng(
            SEED
            + CLASS_OFFSET[cls]
            + rep
        )

        draw = rng.integers(
            0,
            G,
            size=G,
        )

        idx = np.concatenate(
            [
                blocks[j]
                for j in draw
            ]
        )

        m = metrics(idx)

        rep_rows.append({
            "composition_class":
                cls,

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

            **m,
        })

        if rep % 500 == 0:
            print(
                f"  bootstrap "
                f"{rep:,}/{B:,}"
            )


with gzip.open(
    REPS_OUT,
    "wt",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rep_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rep_rows
    )


# ==========================================================
# Bootstrap CIs
# ==========================================================

ci_record = {}

for cls in CLASSES:

    rr = [
        x
        for x in rep_rows
        if x[
            "composition_class"
        ] == cls
    ]

    ci_record[cls] = {
        "n_configs":
            point[cls][
                "n_configs"
            ],

        "n_prototypes":
            point[cls][
                "n_prototypes"
            ],

        "prototype_level_kish_ESS":
            point[cls][
                "prototype_level_kish_ESS"
            ],

        "ESS_ge_30":
            point[cls][
                "ESS_ge_30"
            ],

        "metrics": {},
    }

    for metric in metric_names:

        vals = np.asarray(
            [
                float(x[metric])
                for x in rr
            ],
            dtype=float,
        )

        q025, q50, q975 = (
            np.percentile(
                vals,
                [2.5, 50, 97.5],
            )
        )

        ci_record[cls][
            "metrics"
        ][metric] = {
            "point_estimate":
                float(
                    point[cls][metric]
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
                "STEP8_COMPOSITION_CLASS_SENSITIVITY",

            "bootstrap_unit":
                "full_protostructure_label",

            "bootstrap_replicates":
                B,

            "seed":
                SEED,

            "classes":
                ci_record,

            "interpretation":
                (
                    "Composition-class robustness "
                    "analysis; must not be renamed "
                    "chemical-family sensitivity."
                ),
        },
        indent=2,
    )
)


# ==========================================================
# Document preregistration deviation
# ==========================================================

deviation = {
    "issue":
        (
            "Preregistration requested a "
            "chemical-family sensitivity, but no "
            "chemical-family variable, mapping or "
            "operational definition was frozen."
        ),

    "evidence":
        [
            (
                "PRE_STEP8_PREREGISTRATION_v1.md "
                "lists chemical-family sensitivity."
            ),
            (
                "The same preregistration separately "
                "lists chemical family and composition "
                "class in the missingness audit."
            ),
            (
                "The frozen primary sample contains "
                "composition_class but no "
                "chemical_family column."
            ),
            (
                "The compute-feasibility amendment "
                "changed sampling strata to "
                "composition_class x d_eq decile, "
                "but did not define composition_class "
                "as chemical_family."
            ),
        ],

    "chemical_family_sensitivity_status":
        "NOT_OPERATIONALIZED_FROM_FROZEN_DESIGN",

    "action":
        (
            "Do not construct a post-hoc chemical "
            "family definition. Report the deviation "
            "transparently and perform the frozen "
            "composition-class robustness analysis "
            "as a separate sensitivity."
        ),

    "sample_changed":
        False,

    "hypothesis_threshold_changed":
        False,

    "posthoc_family_definition_created":
        False,
}

DEVIATION_OUT.write_text(
    json.dumps(
        deviation,
        indent=2,
    )
)


# ==========================================================
# Audit
# ==========================================================

all_finite = all(
    np.isfinite(
        float(r[m])
    )
    for r in table_rows
    for m in metric_names
)

support = {
    r["composition_class"]:
        bool(r["ESS_ge_30"])
    for r in table_rows
}

if all(support.values()):
    status = "PASS"
else:
    status = (
        "PASS_WITH_LIMITED_STRATUM_SUPPORT"
    )


audit = {
    "stage":
        "STEP8_COMPOSITION_CLASS_SENSITIVITY",

    "status":
        status,

    "n_configs":
        len(rows),

    "composition_classes":
        CLASSES,

    "bootstrap_replicates_per_class":
        B,

    "prototype_blocking_preserved":
        True,

    "sampling_weights_preserved":
        True,

    "sample_membership_changed":
        False,

    "chemical_family_redefined_posthoc":
        False,

    "chemical_family_requirement_status":
        "DOCUMENTED_PREREGISTRATION_DEVIATION",

    "composition_class_support":
        support,

    "table":
        str(TABLE),

    "table_sha256":
        sha256(TABLE),

    "ci_file":
        str(CI_OUT),

    "ci_file_sha256":
        sha256(CI_OUT),

    "replicates_file":
        str(REPS_OUT),

    "replicates_file_sha256":
        sha256(REPS_OUT),

    "deviation_file":
        str(DEVIATION_OUT),

    "deviation_file_sha256":
        sha256(DEVIATION_OUT),

    "source":
        str(INPUT),

    "source_sha256":
        sha256(INPUT),

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
print("COMPOSITION-CLASS SUMMARY")
print("=" * 78)

print(
    "Class        n    proto  ESS    "
    "E_fCM   F_fCM   E_R    F_R    "
    "E_rho  F_rho"
)

for r in table_rows:

    print(
        f"{r['composition_class']:10s} "
        f"{r['n_configs']:4d} "
        f"{r['n_prototypes']:6d} "
        f"{r['prototype_level_kish_ESS']:6.1f} "
        f"{r['energy_f_CM_corrected']:7.3f} "
        f"{r['force_f_CM_corrected']:7.3f} "
        f"{r['energy_R_total']:6.3f} "
        f"{r['force_R_total']:6.3f} "
        f"{r['energy_rank_rho']:6.3f} "
        f"{r['force_rank_rho']:6.3f}"
    )


print("\nCONTINUOUS d_eq")

for r in table_rows:

    print(
        f"{r['composition_class']:10s} "
        f"E_CM="
        f"{r['deq_vs_energy_corrected_CM']:+.3f} "
        f"F_CM="
        f"{r['deq_vs_force_corrected_CM']:+.3f} "
        f"E_R="
        f"{r['deq_vs_energy_R_config']:+.3f} "
        f"F_R="
        f"{r['deq_vs_force_R_config']:+.3f}"
    )


print("\nChemical-family sensitivity:")
print(
    "  NOT OPERATIONALIZED — documented preregistration deviation"
)

print("\nFiles")
print("Table    :", TABLE)
print("SHA256   :", sha256(TABLE))
print("CI       :", CI_OUT)
print("SHA256   :", sha256(CI_OUT))
print("Replicates:", REPS_OUT)
print("SHA256   :", sha256(REPS_OUT))
print("Deviation:", DEVIATION_OUT)
print("SHA256   :", sha256(DEVIATION_OUT))
print("Audit    :", AUDIT_OUT)
print("SHA256   :", sha256(AUDIT_OUT))

print("\n" + "=" * 78)
print(
    "STEP8 COMPOSITION-CLASS "
    f"SENSITIVITY: {status}"
)
