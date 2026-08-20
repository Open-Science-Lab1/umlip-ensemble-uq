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


MEMBERSHIP = Path(
    "step8/sensitivity/"
    "STEP8_ALL_CANDIDATE_VALIDITY_MEMBERSHIP_v1.csv.gz"
)

MEMBERSHIP_AUDIT = Path(
    "step8/sensitivity/"
    "STEP8_ALL_CANDIDATE_VALIDITY_MEMBERSHIP_AUDIT_v1.json"
)

PRED = Path("step8/predictions")

PRIMARY_POINT = Path(
    "step8/analysis/"
    "PRIMARY_WEIGHTED_POINT_ESTIMATES_v1.json"
)

OUTDIR = Path("step8/sensitivity")
OUTDIR.mkdir(parents=True, exist_ok=True)

POINT_OUT = OUTDIR / (
    "STEP8_SENSITIVITY_COMPLETENESS_POINT_ESTIMATES_v1.json"
)

DECILE_OUT = OUTDIR / (
    "STEP8_SENSITIVITY_COMPLETENESS_DECILES_v1.csv"
)

BOOT_OUT = OUTDIR / (
    "STEP8_SENSITIVITY_COMPLETENESS_BOOTSTRAP_REPLICATES_v1.csv.gz"
)

CI_OUT = OUTDIR / (
    "STEP8_SENSITIVITY_COMPLETENESS_BOOTSTRAP_CI_v1.json"
)

AUDIT_OUT = OUTDIR / (
    "STEP8_SENSITIVITY_COMPLETENESS_AUDIT_v1.json"
)


PRIMARY8 = [
    "CHGNet",
    "MACE-MP-0",
    "SevenNet-l3i5",
    "ORB-v2-MPtrj",
    "GRACE-2L-MPtrj",
    "eqV2-S-DeNS",
    "eSEN-30M-OAM",
    "MACE-MPA-0",
]

FULL10 = PRIMARY8 + [
    "ORB-v2-MPA",
    "GRACE-2L-OAM",
]

ALL11 = FULL10 + [
    "PET-OAM-XL",
]

SCENARIOS = [
    "variable_M",
    "strict_10",
    "strict_11_PET_subset",
]

B = 2000
SEED = 20260812
N_BINS = 10
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


def as_bool(x):
    return str(x).strip().lower() in {
        "1", "true", "yes"
    }


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
        return np.nan

    return float(
        np.sum(w * dx * dy)
        / den
    )


def weighted_spearman(x, y, w):
    if len(x) < 3:
        return np.nan

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

    total = 0.0

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

        if rmv <= 0:
            continue

        gap = abs(
            rmse - rmv
        ) / rmv

        total += (
            np.sum(wb)
            / total_w
        ) * gap

    return float(total)


# =========================================================
# Frozen membership
# =========================================================

with gzip.open(
    MEMBERSHIP,
    "rt",
    newline="",
) as f:
    rows = list(
        csv.DictReader(f)
    )

assert len(rows) == 3000

membership_audit = json.loads(
    MEMBERSHIP_AUDIT.read_text()
)

assert membership_audit["status"] == "PASS"


scenario_records = {
    name: []
    for name in SCENARIOS
}

primary_qc_records = []


# =========================================================
# Per-configuration decomposition
# =========================================================

def load_errors(mid, models):

    energy = []
    force = []

    dft_force_reference = None

    for model in models:

        path = (
            PRED
            / model
            / f"{mid}.npz"
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            n_atoms = int(
                np.asarray(
                    z["n_atoms"]
                ).item()
            )

            Em = float(
                np.asarray(
                    z[
                        "model_config_energy_eV"
                    ]
                ).item()
            )

            Ep = float(
                np.asarray(
                    z[
                        "model_parent_energy_eV"
                    ]
                ).item()
            )

            Ed = float(
                np.asarray(
                    z[
                        "dft_config_energy_eV"
                    ]
                ).item()
            )

            Edp = float(
                np.asarray(
                    z[
                        "dft_parent_energy_eV"
                    ]
                ).item()
            )

            Fm = np.asarray(
                z[
                    "model_forces_eV_per_A"
                ],
                dtype=float,
            )

            Fd = np.asarray(
                z[
                    "dft_forces_eV_per_A"
                ],
                dtype=float,
            )

        e = (
            (Em - Ep) / n_atoms
            -
            (Ed - Edp) / n_atoms
        )

        energy.append(e)

        force.append(
            Fm - Fd
        )

        if dft_force_reference is None:
            dft_force_reference = Fd
        else:
            assert np.allclose(
                Fd,
                dft_force_reference,
                rtol=0,
                atol=1e-12,
            )

    return (
        np.asarray(
            energy,
            dtype=float,
        ),
        np.stack(
            force,
            axis=0,
        ),
    )


def build_record(
    base,
    models,
):

    E, F = load_errors(
        base["matpes_id"],
        models,
    )

    M = len(models)

    assert M >= 2

    # ---------------- ENERGY ----------------

    ebar = float(
        np.mean(E)
    )

    e_samplevar = float(
        np.var(
            E,
            ddof=1,
        )
    )

    e_popvar = float(
        np.var(
            E,
            ddof=0,
        )
    )

    e_total = float(
        np.mean(
            E ** 2
        )
    )

    e_cm_naive = (
        ebar ** 2
    )

    # Variable-M finite-M correction.
    e_cm_corr = (
        e_cm_naive
        -
        e_samplevar / M
    )

    e_sigma_uq = math.sqrt(
        max(
            e_samplevar,
            0.0,
        )
    )

    e_error = abs(
        ebar
    )

    e_R_config = (
        math.sqrt(
            e_total / e_popvar
        )
        if e_popvar > EPS
        else np.nan
    )


    # ---------------- FORCE ----------------

    # M x atom x xyz
    Fbar = np.mean(
        F,
        axis=0,
    )

    error_atom = np.linalg.norm(
        Fbar,
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

    popvar_atom = np.sum(
        np.var(
            F,
            axis=0,
            ddof=0,
        ),
        axis=1,
    )

    cm_naive_atom = np.sum(
        Fbar ** 2,
        axis=1,
    )

    cm_corr_atom = (
        cm_naive_atom
        -
        samplevar_atom / M
    )

    total_atom = np.mean(
        np.sum(
            F ** 2,
            axis=2,
        ),
        axis=0,
    )

    sigma_uq_atom = np.sqrt(
        np.maximum(
            samplevar_atom,
            0.0,
        )
    )

    f_error_mean = float(
        np.mean(
            error_atom
        )
    )

    f_error_max = float(
        np.max(
            error_atom
        )
    )

    f_sigma_mean = float(
        np.mean(
            sigma_uq_atom
        )
    )

    f_sigma_max = float(
        np.max(
            sigma_uq_atom
        )
    )

    f_samplevar_mean = float(
        np.mean(
            samplevar_atom
        )
    )

    f_popvar_mean = float(
        np.mean(
            popvar_atom
        )
    )

    f_total_mean = float(
        np.mean(
            total_atom
        )
    )

    f_cm_naive_mean = float(
        np.mean(
            cm_naive_atom
        )
    )

    f_cm_corr_mean = float(
        np.mean(
            cm_corr_atom
        )
    )

    f_R_config = (
        math.sqrt(
            f_total_mean
            / f_popvar_mean
        )
        if f_popvar_mean > EPS
        else np.nan
    )


    return {
        "matpes_id":
            base["matpes_id"],

        "original_mp_id":
            base["original_mp_id"],

        "prototype":
            base[
                "full_protostructure_label"
            ],

        "deq_decile":
            int(
                base["deq_decile"]
            ),

        "d_eq":
            float(
                base["d_eq"]
            ),

        "weight":
            float(
                base["sampling_weight"]
            ),

        "M":
            M,

        "energy_cm_naive":
            e_cm_naive,

        "energy_cm_corrected":
            e_cm_corr,

        "energy_samplevar":
            e_samplevar,

        "energy_popvar":
            e_popvar,

        "energy_total":
            e_total,

        "energy_abs_error":
            e_error,

        "energy_sigma":
            e_sigma_uq,

        "energy_R_config":
            e_R_config,

        "force_cm_naive_mean":
            f_cm_naive_mean,

        "force_cm_corrected_mean":
            f_cm_corr_mean,

        "force_samplevar_mean":
            f_samplevar_mean,

        "force_popvar_mean":
            f_popvar_mean,

        "force_total_mean":
            f_total_mean,

        "force_abs_error_mean":
            f_error_mean,

        "force_sigma_mean":
            f_sigma_mean,

        "force_R_config_mean":
            f_R_config,

        "force_abs_error_max":
            f_error_max,

        "force_sigma_max":
            f_sigma_max,
    }


for i, row in enumerate(
    rows,
    1,
):

    valid_models = (
        row["valid_models"]
        .split(";")
    )

    valid_models = [
        x
        for x in valid_models
        if x
    ]


    # Primary QC only.
    if as_bool(
        row[
            "strict_10_fullsample_complete_case"
        ]
    ):

        # Strict10 implies primary8 valid.
        primary_qc_records.append(
            build_record(
                row,
                PRIMARY8,
            )
        )


    # Variable M_i >= 8.
    assert int(
        row["M_i"]
    ) >= 8

    scenario_records[
        "variable_M"
    ].append(
        build_record(
            row,
            valid_models,
        )
    )


    # Strict full-sample 10-model.
    if as_bool(
        row[
            "strict_10_fullsample_complete_case"
        ]
    ):

        scenario_records[
            "strict_10"
        ].append(
            build_record(
                row,
                FULL10,
            )
        )


    # Strict all-11, only PET frozen subset.
    if as_bool(
        row[
            "strict_11_PET_subset_complete_case"
        ]
    ):

        scenario_records[
            "strict_11_PET_subset"
        ].append(
            build_record(
                row,
                ALL11,
            )
        )


    if i % 250 == 0:
        print(
            f"Built sensitivity statistics "
            f"{i:,}/3,000"
        )


assert len(
    scenario_records["variable_M"]
) == 3000

assert len(
    scenario_records["strict_10"]
) == 2998

assert len(
    scenario_records[
        "strict_11_PET_subset"
    ]
) == 299


# =========================================================
# Array conversion
# =========================================================

NUMERIC_FIELDS = [
    "M",
    "deq_decile",
    "d_eq",
    "weight",
    "energy_cm_naive",
    "energy_cm_corrected",
    "energy_samplevar",
    "energy_popvar",
    "energy_total",
    "energy_abs_error",
    "energy_sigma",
    "energy_R_config",
    "force_cm_naive_mean",
    "force_cm_corrected_mean",
    "force_samplevar_mean",
    "force_popvar_mean",
    "force_total_mean",
    "force_abs_error_mean",
    "force_sigma_mean",
    "force_R_config_mean",
    "force_abs_error_max",
    "force_sigma_max",
]


def make_arrays(records):

    out = {}

    for key in NUMERIC_FIELDS:
        out[key] = np.asarray(
            [
                r[key]
                for r in records
            ],
            dtype=float,
        )

    out["prototype"] = np.asarray(
        [
            r["prototype"]
            for r in records
        ],
        dtype=object,
    )

    return out


arrays = {
    name:
        make_arrays(records)
    for name, records
    in scenario_records.items()
}

primary_qc = make_arrays(
    primary_qc_records
)


# =========================================================
# Point metrics
# =========================================================

def core_metrics(a, idx=None):

    if idx is None:
        idx = np.arange(
            len(a["weight"])
        )

    w = a["weight"][idx]

    # Energy corrected fCM.
    e_cm = wmean(
        a[
            "energy_cm_corrected"
        ][idx],
        w,
    )

    e_sig = wmean(
        a[
            "energy_samplevar"
        ][idx],
        w,
    )

    e_cm_pos = max(
        0.0,
        e_cm,
    )

    e_fcm = (
        e_cm_pos
        /
        (
            e_cm_pos
            + e_sig
        )
    )

    e_naive_num = wmean(
        a[
            "energy_cm_naive"
        ][idx],
        w,
    )

    e_total = wmean(
        a[
            "energy_total"
        ][idx],
        w,
    )

    e_naive = (
        e_naive_num
        / e_total
    )

    e_pop = wmean(
        a[
            "energy_popvar"
        ][idx],
        w,
    )

    e_R = math.sqrt(
        e_total / e_pop
    )


    # Force mean-over-atoms corrected fCM.
    f_cm = wmean(
        a[
            "force_cm_corrected_mean"
        ][idx],
        w,
    )

    f_sig = wmean(
        a[
            "force_samplevar_mean"
        ][idx],
        w,
    )

    f_cm_pos = max(
        0.0,
        f_cm,
    )

    f_fcm = (
        f_cm_pos
        /
        (
            f_cm_pos
            + f_sig
        )
    )

    f_naive_num = wmean(
        a[
            "force_cm_naive_mean"
        ][idx],
        w,
    )

    f_total = wmean(
        a[
            "force_total_mean"
        ][idx],
        w,
    )

    f_naive = (
        f_naive_num
        / f_total
    )

    f_pop = wmean(
        a[
            "force_popvar_mean"
        ][idx],
        w,
    )

    f_R = math.sqrt(
        f_total / f_pop
    )


    return {
        "energy_fCM_corrected":
            e_fcm,

        "energy_fCM_naive":
            e_naive,

        "energy_R_total":
            e_R,

        "force_fCM_corrected":
            f_fcm,

        "force_fCM_naive":
            f_naive,

        "force_R_total":
            f_R,

        "energy_F2_identity_residual":
            abs(
                e_R ** 2
                -
                1.0
                / (
                    1.0 - e_naive
                )
            ),

        "force_F2_identity_residual":
            abs(
                f_R ** 2
                -
                1.0
                / (
                    1.0 - f_naive
                )
            ),
    }


def full_metrics(a, idx=None):

    if idx is None:
        idx = np.arange(
            len(a["weight"])
        )

    w = a["weight"][idx]

    out = core_metrics(
        a,
        idx,
    )

    out.update({
        "energy_rank_spearman":
            weighted_spearman(
                a["energy_sigma"][idx],
                a[
                    "energy_abs_error"
                ][idx],
                w,
            ),

        "energy_ENCE":
            weighted_ence(
                a[
                    "energy_abs_error"
                ][idx],
                a[
                    "energy_sigma"
                ][idx],
                w,
            ),

        "energy_coverage_1sigma":
            weighted_coverage(
                a[
                    "energy_abs_error"
                ][idx],
                a["energy_sigma"][idx],
                1.0,
                w,
            ),

        "energy_coverage_1_96sigma":
            weighted_coverage(
                a[
                    "energy_abs_error"
                ][idx],
                a["energy_sigma"][idx],
                1.96,
                w,
            ),

        "force_rank_spearman":
            weighted_spearman(
                a[
                    "force_sigma_mean"
                ][idx],
                a[
                    "force_abs_error_mean"
                ][idx],
                w,
            ),

        "force_ENCE":
            weighted_ence(
                a[
                    "force_abs_error_mean"
                ][idx],
                a[
                    "force_sigma_mean"
                ][idx],
                w,
            ),

        "force_coverage_1sigma":
            weighted_coverage(
                a[
                    "force_abs_error_mean"
                ][idx],
                a[
                    "force_sigma_mean"
                ][idx],
                1.0,
                w,
            ),

        "force_coverage_1_96sigma":
            weighted_coverage(
                a[
                    "force_abs_error_mean"
                ][idx],
                a[
                    "force_sigma_mean"
                ][idx],
                1.96,
                w,
            ),

        "force_max_rank_spearman":
            weighted_spearman(
                a[
                    "force_sigma_max"
                ][idx],
                a[
                    "force_abs_error_max"
                ][idx],
                w,
            ),

        "force_max_ENCE":
            weighted_ence(
                a[
                    "force_abs_error_max"
                ][idx],
                a[
                    "force_sigma_max"
                ][idx],
                w,
            ),

        "deq_vs_energy_CM_corrected_spearman":
            weighted_spearman(
                a["d_eq"][idx],
                a[
                    "energy_cm_corrected"
                ][idx],
                w,
            ),

        "deq_vs_force_CM_corrected_spearman":
            weighted_spearman(
                a["d_eq"][idx],
                a[
                    "force_cm_corrected_mean"
                ][idx],
                w,
            ),

        "deq_vs_energy_R_config_spearman":
            weighted_spearman(
                a["d_eq"][idx],
                a[
                    "energy_R_config"
                ][idx],
                w,
            ),

        "deq_vs_force_R_config_spearman":
            weighted_spearman(
                a["d_eq"][idx],
                a[
                    "force_R_config_mean"
                ][idx],
                w,
            ),
    })


    d1 = idx[
        a["deq_decile"][idx]
        == 1
    ]

    d10 = idx[
        a["deq_decile"][idx]
        == 10
    ]

    m1 = core_metrics(
        a,
        d1,
    )

    m10 = core_metrics(
        a,
        d10,
    )

    out.update({
        "D10_minus_D01_energy_fCM_corrected":
            (
                m10[
                    "energy_fCM_corrected"
                ]
                -
                m1[
                    "energy_fCM_corrected"
                ]
            ),

        "D10_minus_D01_force_fCM_corrected":
            (
                m10[
                    "force_fCM_corrected"
                ]
                -
                m1[
                    "force_fCM_corrected"
                ]
            ),

        "D10_minus_D01_energy_R_total":
            (
                m10[
                    "energy_R_total"
                ]
                -
                m1[
                    "energy_R_total"
                ]
            ),

        "D10_minus_D01_force_R_total":
            (
                m10[
                    "force_R_total"
                ]
                -
                m1[
                    "force_R_total"
                ]
            ),
    })

    return out


point = {
    name:
        full_metrics(a)
    for name, a
    in arrays.items()
}


# =========================================================
# Primary 8-model QC
# =========================================================

frozen = json.loads(
    PRIMARY_POINT.read_text()
)

qc_point = full_metrics(
    primary_qc
)

qc_checks = {
    "energy_R_total":
        abs(
            qc_point[
                "energy_R_total"
            ]
            -
            float(
                frozen[
                    "energy"
                ][
                    "R_total_direct"
                ]
            )
        ),

    "energy_rank":
        abs(
            qc_point[
                "energy_rank_spearman"
            ]
            -
            float(
                frozen[
                    "energy"
                ][
                    "weighted_spearman_error_vs_spread"
                ]
            )
        ),

    "energy_ENCE":
        abs(
            qc_point[
                "energy_ENCE"
            ]
            -
            float(
                frozen[
                    "energy"
                ][
                    "calibration"
                ][
                    "ENCE"
                ]
            )
        ),

    "force_R_total":
        abs(
            qc_point[
                "force_R_total"
            ]
            -
            float(
                frozen[
                    "force_mean_over_atoms"
                ][
                    "R_total_direct"
                ]
            )
        ),

    "force_rank":
        abs(
            qc_point[
                "force_rank_spearman"
            ]
            -
            float(
                frozen[
                    "force_mean_over_atoms"
                ][
                    "weighted_spearman_error_vs_spread"
                ]
            )
        ),

    "force_ENCE":
        abs(
            qc_point[
                "force_ENCE"
            ]
            -
            float(
                frozen[
                    "force_mean_over_atoms"
                ][
                    "calibration"
                ][
                    "ENCE"
                ]
            )
        ),
}

max_primary_qc = max(
    qc_checks.values()
)

assert max_primary_qc < 1e-9


# =========================================================
# Decile summaries
# =========================================================

decile_fields = [
    "scenario",
    "deq_decile",
    "n",
    "weighted_mean_d_eq",
    "energy_fCM_corrected",
    "energy_R_total",
    "energy_rank_spearman",
    "energy_ENCE",
    "force_fCM_corrected",
    "force_R_total",
    "force_rank_spearman",
    "force_ENCE",
    "force_max_rank_spearman",
    "force_max_ENCE",
]

with open(
    DECILE_OUT,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=decile_fields,
    )

    writer.writeheader()

    for scenario, a in arrays.items():

        for d in range(1, 11):

            idx = np.where(
                a["deq_decile"] == d
            )[0]

            m = full_metrics(
                a,
                idx,
            )

            writer.writerow({
                "scenario":
                    scenario,

                "deq_decile":
                    d,

                "n":
                    len(idx),

                "weighted_mean_d_eq":
                    wmean(
                        a["d_eq"][idx],
                        a["weight"][idx],
                    ),

                "energy_fCM_corrected":
                    m[
                        "energy_fCM_corrected"
                    ],

                "energy_R_total":
                    m[
                        "energy_R_total"
                    ],

                "energy_rank_spearman":
                    m[
                        "energy_rank_spearman"
                    ],

                "energy_ENCE":
                    m[
                        "energy_ENCE"
                    ],

                "force_fCM_corrected":
                    m[
                        "force_fCM_corrected"
                    ],

                "force_R_total":
                    m[
                        "force_R_total"
                    ],

                "force_rank_spearman":
                    m[
                        "force_rank_spearman"
                    ],

                "force_ENCE":
                    m[
                        "force_ENCE"
                    ],

                "force_max_rank_spearman":
                    m[
                        "force_max_rank_spearman"
                    ],

                "force_max_ENCE":
                    m[
                        "force_max_ENCE"
                    ],
            })


# =========================================================
# Prototype block bootstrap
# =========================================================

BOOT_METRICS = [
    "energy_fCM_corrected",
    "force_fCM_corrected",
    "energy_R_total",
    "force_R_total",
    "energy_rank_spearman",
    "force_rank_spearman",
    "energy_ENCE",
    "force_ENCE",
    "force_max_rank_spearman",
    "force_max_ENCE",
    "deq_vs_energy_CM_corrected_spearman",
    "deq_vs_force_CM_corrected_spearman",
    "deq_vs_energy_R_config_spearman",
    "deq_vs_force_R_config_spearman",
    "D10_minus_D01_energy_fCM_corrected",
    "D10_minus_D01_force_fCM_corrected",
    "D10_minus_D01_energy_R_total",
    "D10_minus_D01_force_R_total",
]


replicates = []

for sidx, scenario in enumerate(
    SCENARIOS
):

    a = arrays[scenario]

    block_map = defaultdict(
        list
    )

    for i, proto in enumerate(
        a["prototype"]
    ):
        block_map[
            str(proto)
        ].append(i)

    blocks = [
        np.asarray(v, dtype=int)
        for _, v
        in sorted(
            block_map.items()
        )
    ]

    G = len(blocks)

    rng = np.random.default_rng(
        SEED + 10000 * sidx
    )

    print(
        f"\nBootstrap {scenario}: "
        f"{G} prototype blocks"
    )

    for rep in range(
        1,
        B + 1,
    ):

        draw = rng.integers(
            0,
            G,
            size=G,
        )

        idx = np.concatenate([
            blocks[j]
            for j in draw
        ])

        m = full_metrics(
            a,
            idx,
        )

        rec = {
            "scenario":
                scenario,

            "replicate":
                rep,

            "n_rows":
                len(idx),

            "n_prototype_blocks":
                G,

            "n_unique_blocks_drawn":
                len(
                    set(
                        draw.tolist()
                    )
                ),
        }

        for metric in BOOT_METRICS:
            rec[metric] = (
                m[metric]
            )

        replicates.append(
            rec
        )

        if rep % 200 == 0:
            print(
                f"  {rep:,}/{B:,}"
            )


boot_fields = [
    "scenario",
    "replicate",
    "n_rows",
    "n_prototype_blocks",
    "n_unique_blocks_drawn",
] + BOOT_METRICS

with gzip.open(
    BOOT_OUT,
    "wt",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=boot_fields,
    )

    writer.writeheader()
    writer.writerows(
        replicates
    )


# =========================================================
# Bootstrap CIs
# =========================================================

ci = {}

for scenario in SCENARIOS:

    subset = [
        r
        for r in replicates
        if r["scenario"]
        == scenario
    ]

    ci[scenario] = {}

    for metric in BOOT_METRICS:

        vals = np.asarray(
            [
                float(
                    r[metric]
                )
                for r in subset
            ],
            dtype=float,
        )

        vals = vals[
            np.isfinite(vals)
        ]

        q = np.percentile(
            vals,
            [2.5, 50, 97.5],
        )

        ci[scenario][metric] = {
            "point_estimate":
                float(
                    point[
                        scenario
                    ][metric]
                ),

            "bootstrap_median":
                float(q[1]),

            "CI95_lower":
                float(q[0]),

            "CI95_upper":
                float(q[2]),

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
                "STEP8_SENSITIVITY_COMPLETENESS_BOOTSTRAP",

            "bootstrap_unit":
                "full_protostructure_label",

            "replicates_per_scenario":
                B,

            "seed":
                SEED,

            "scenarios":
                ci,
        },
        indent=2,
    )
)


# =========================================================
# Point-estimate output
# =========================================================

summary = {
    "stage":
        "STEP8_SENSITIVITY_COMPLETENESS_ANALYSIS",

    "status":
        "PASS",

    "primary_analysis_replaced":
        False,

    "interpretation":
        (
            "Sensitivity analyses only. "
            "The frozen fixed-8 primary result "
            "remains the primary estimator."
        ),

    "scenarios": {
        "variable_M": {
            "definition":
                (
                    "All 3000 frozen configurations; "
                    "use every technically valid "
                    "executed candidate prediction "
                    "with M_i >= 8 and apply the "
                    "finite-M_i bias correction."
                ),

            "n":
                len(
                    scenario_records[
                        "variable_M"
                    ]
                ),

            "M_i_distribution":
                membership_audit[
                    "M_i_distribution"
                ],

            "metrics":
                point[
                    "variable_M"
                ],
        },

        "strict_10": {
            "definition":
                (
                    "Strict complete case for the "
                    "10 models inferred on the full "
                    "3000-configuration frozen sample."
                ),

            "n":
                len(
                    scenario_records[
                        "strict_10"
                    ]
                ),

            "metrics":
                point[
                    "strict_10"
                ],
        },

        "strict_11_PET_subset": {
            "definition":
                (
                    "Strict all-11-model complete case "
                    "inside the pre-inference frozen "
                    "PET 300-configuration sensitivity "
                    "subset."
                ),

            "n":
                len(
                    scenario_records[
                        "strict_11_PET_subset"
                    ]
                ),

            "metrics":
                point[
                    "strict_11_PET_subset"
                ],
        },
    },

    "strict_11_full_3000_status":
        (
            "NOT_FEASIBLE_BY_FROZEN_"
            "COMPUTE_FEASIBILITY_AMENDMENT"
        ),

    "primary_8_model_QC":
        {
            "n":
                len(
                    primary_qc_records
                ),

            "max_abs_difference_from_frozen_primary":
                max_primary_qc,

            "individual_checks":
                qc_checks,
        },

    "hypothesis_decision_performed":
        False,
}


POINT_OUT.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


# =========================================================
# Final audit
# =========================================================

all_F2 = []

for scenario in SCENARIOS:
    all_F2.extend([
        point[scenario][
            "energy_F2_identity_residual"
        ],
        point[scenario][
            "force_F2_identity_residual"
        ],
    ])

max_F2 = max(
    all_F2
)

status = (
    "PASS"
    if (
        len(
            scenario_records[
                "variable_M"
            ]
        ) == 3000
        and
        len(
            scenario_records[
                "strict_10"
            ]
        ) == 2998
        and
        len(
            scenario_records[
                "strict_11_PET_subset"
            ]
        ) == 299
        and
        max_primary_qc < 1e-9
        and
        max_F2 < 1e-10
        and
        len(replicates)
        == B * len(SCENARIOS)
    )
    else "REVISE"
)

audit = {
    "stage":
        "STEP8_SENSITIVITY_COMPLETENESS_ANALYSIS",

    "status":
        status,

    "variable_M_n":
        3000,

    "strict_10_n":
        2998,

    "strict_11_PET_subset_n":
        299,

    "bootstrap_replicates_per_scenario":
        B,

    "total_bootstrap_rows":
        len(replicates),

    "primary_QC_max_abs_difference":
        max_primary_qc,

    "max_F2_identity_residual":
        max_F2,

    "membership_file":
        str(MEMBERSHIP),

    "membership_sha256":
        sha256(MEMBERSHIP),

    "point_file":
        str(POINT_OUT),

    "point_sha256":
        sha256(POINT_OUT),

    "decile_file":
        str(DECILE_OUT),

    "decile_sha256":
        sha256(DECILE_OUT),

    "bootstrap_file":
        str(BOOT_OUT),

    "bootstrap_sha256":
        sha256(BOOT_OUT),

    "CI_file":
        str(CI_OUT),

    "CI_sha256":
        sha256(CI_OUT),

    "primary_sample_membership_changed":
        False,

    "primary_results_overwritten":
        False,

    "hypothesis_decision_performed":
        False,
}

AUDIT_OUT.write_text(
    json.dumps(
        audit,
        indent=2,
    )
)


# =========================================================
# Console
# =========================================================

print("\n" + "=" * 78)
print("STEP-8 SENSITIVITY COMPLETENESS RESULTS")
print("=" * 78)

for scenario in SCENARIOS:

    m = point[scenario]

    print(
        f"\n{scenario}"
    )

    print(
        "  N              :",
        len(
            scenario_records[
                scenario
            ]
        ),
    )

    print(
        "  Energy fCMcorr :",
        f"{m['energy_fCM_corrected']:.6f}",
    )

    print(
        "  Force fCMcorr  :",
        f"{m['force_fCM_corrected']:.6f}",
    )

    print(
        "  Energy R       :",
        f"{m['energy_R_total']:.6f}",
    )

    print(
        "  Force R        :",
        f"{m['force_R_total']:.6f}",
    )

    print(
        "  Energy rho     :",
        f"{m['energy_rank_spearman']:.6f}",
    )

    print(
        "  Force rho      :",
        f"{m['force_rank_spearman']:.6f}",
    )

    print(
        "  Energy ENCE    :",
        f"{m['energy_ENCE']:.6f}",
    )

    print(
        "  Force ENCE     :",
        f"{m['force_ENCE']:.6f}",
    )

    print(
        "  d_eq→E CM rho  :",
        f"{m['deq_vs_energy_CM_corrected_spearman']:.6f}",
    )

    print(
        "  d_eq→F CM rho  :",
        f"{m['deq_vs_force_CM_corrected_spearman']:.6f}",
    )


print("\nQC")
print(
    "  Primary reproduction max diff:",
    f"{max_primary_qc:.3e}",
)

print(
    "  Max F2 residual             :",
    f"{max_F2:.3e}",
)

print("\nFiles")
print("Point :", POINT_OUT)
print("Decile:", DECILE_OUT)
print("Boot  :", BOOT_OUT)
print("CI    :", CI_OUT)
print("Audit :", AUDIT_OUT)

print("\nAudit SHA256:")
print(
    sha256(
        AUDIT_OUT
    )
)

print("\nSTATUS:", status)
