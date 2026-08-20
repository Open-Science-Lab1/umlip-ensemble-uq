#!/usr/bin/env python3

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


META = Path(
    "step8/analysis/"
    "PRIMARY_SIGNED_ERROR_DECOMPOSITION_v1.csv.gz"
)

POINT = Path(
    "step8/analysis/"
    "PRIMARY_WEIGHTED_POINT_ESTIMATES_v1.json"
)

PRED = Path("step8/predictions")

OUTDIR = Path("step8/analysis")

TABLE = OUTDIR / (
    "PRIMARY_LEAVE_ONE_ARCHITECTURE_FAMILY_OUT_v1.csv"
)

AUDIT = OUTDIR / (
    "PRIMARY_LEAVE_ONE_ARCHITECTURE_FAMILY_OUT_AUDIT_v1.json"
)

EPS = 1e-15
N_CAL_BINS = 10


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


FAMILIES = {
    "CHGNet": [
        "CHGNet",
    ],

    "MACE": [
        "MACE-MP-0",
        "MACE-MPA-0",
    ],

    "SevenNet": [
        "SevenNet-l3i5",
    ],

    "ORB": [
        "ORB-v2-MPtrj",
    ],

    "GRACE": [
        "GRACE-2L-MPtrj",
    ],

    "EquiformerV2": [
        "eqV2-S-DeNS",
    ],

    "eSEN": [
        "eSEN-30M-OAM",
    ],
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


def scalar(x):
    a = np.asarray(x)

    if a.size != 1:
        raise ValueError(
            f"Expected scalar; shape={a.shape}"
        )

    return float(a.reshape(-1)[0])


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
        * np.sum(w * dy * dy)
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
            np.sum(wb)
            / total_w
        ) * gap

    return float(ence)


print("=" * 78)
print(
    "STEP-8 LEAVE-ONE-ARCHITECTURE-"
    "FAMILY-OUT SENSITIVITY"
)
print("=" * 78)


# ==========================================================
# Frozen 2998-config metadata
# ==========================================================

with gzip.open(
    META,
    "rt",
    newline="",
) as f:
    rows = list(
        csv.DictReader(f)
    )

assert len(rows) == 2998

n_cfg = len(rows)

weights = np.asarray(
    [
        float(x["sampling_weight"])
        for x in rows
    ],
    dtype=float,
)

deq = np.asarray(
    [
        float(x["d_eq"])
        for x in rows
    ],
    dtype=float,
)

decile = np.asarray(
    [
        int(x["deq_decile"])
        for x in rows
    ],
    dtype=int,
)


# ==========================================================
# Variants
# ==========================================================

VARIANTS = {
    "FULL_8_MODEL_CORE":
        list(MODELS)
}

for family, members in FAMILIES.items():

    VARIANTS[
        f"DROP_{family}"
    ] = [
        m
        for m in MODELS
        if m not in members
    ]


print("Configurations:", n_cfg)
print("Families      :", len(FAMILIES))
print("Variants      :", len(VARIANTS))


# ==========================================================
# Storage for configuration-level scalar metrics
# ==========================================================

store = {}

for variant, roster in VARIANTS.items():

    store[variant] = {
        "M": len(roster),

        "energy_total": [],
        "energy_cm_naive": [],
        "energy_cm_corrected": [],
        "energy_specific": [],
        "energy_error": [],
        "energy_spread": [],
        "energy_R_config": [],

        "force_total": [],
        "force_cm_naive": [],
        "force_cm_corrected": [],
        "force_specific": [],
        "force_error": [],
        "force_spread": [],
        "force_R_config": [],

        "force_max_error": [],
        "force_max_spread": [],
    }


# ==========================================================
# Read each frozen config once
# ==========================================================

for ii, row in enumerate(
    rows,
    1,
):

    mid = row["matpes_id"]

    energy_errors = {}
    force_errors = {}

    for model in MODELS:

        path = (
            PRED
            / model
            / f"{mid}.npz"
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            Em = scalar(
                z[
                    "model_config_energy_eV"
                ]
            )

            Epm = scalar(
                z[
                    "model_parent_energy_eV"
                ]
            )

            Edft = scalar(
                z[
                    "dft_config_energy_eV"
                ]
            )

            Epdft = scalar(
                z[
                    "dft_parent_energy_eV"
                ]
            )

            Fm = np.asarray(
                z[
                    "model_forces_eV_per_A"
                ],
                dtype=float,
            )

            Fdft = np.asarray(
                z[
                    "dft_forces_eV_per_A"
                ],
                dtype=float,
            )

            N = int(
                round(
                    scalar(z["n_atoms"])
                )
            )

        energy_errors[model] = (
            (
                Em - Epm
            )
            - (
                Edft - Epdft
            )
        ) / N

        force_errors[model] = (
            Fm - Fdft
        )


    # ======================================================
    # Every frozen architecture-family sensitivity variant
    # ======================================================

    for variant, roster in VARIANTS.items():

        M = len(roster)

        E = np.asarray(
            [
                energy_errors[m]
                for m in roster
            ],
            dtype=float,
        )

        F = np.stack(
            [
                force_errors[m]
                for m in roster
            ],
            axis=0,
        )

        # ---------------- ENERGY ----------------

        ebar = float(
            np.mean(E)
        )

        e_total = float(
            np.mean(E ** 2)
        )

        e_popvar = float(
            np.mean(
                (
                    E - ebar
                ) ** 2
            )
        )

        e_samplevar = float(
            np.var(
                E,
                ddof=1,
            )
        )

        e_cm_naive = (
            ebar ** 2
        )

        e_cm_corr = (
            e_cm_naive
            - e_samplevar / M
        )

        e_spread = math.sqrt(
            max(
                e_samplevar,
                0.0,
            )
        )

        e_R_cfg = math.sqrt(
            e_total
            / max(
                e_popvar,
                EPS,
            )
        )


        # ---------------- FORCE ----------------

        Fbar = np.mean(
            F,
            axis=0,
        )

        dev = (
            F
            - Fbar[None, :, :]
        )

        total_atom = np.mean(
            np.sum(
                F ** 2,
                axis=2,
            ),
            axis=0,
        )

        cm_naive_atom = np.sum(
            Fbar ** 2,
            axis=1,
        )

        popvar_atom = np.mean(
            np.sum(
                dev ** 2,
                axis=2,
            ),
            axis=0,
        )

        samplevar_atom = np.sum(
            np.var(
                F,
                axis=0,
                ddof=1,
            ),
            axis=1,
        )

        cm_corr_atom = (
            cm_naive_atom
            - samplevar_atom / M
        )

        error_atom = np.linalg.norm(
            Fbar,
            axis=1,
        )

        spread_atom = np.sqrt(
            np.maximum(
                samplevar_atom,
                0.0,
            )
        )

        f_total = float(
            np.mean(
                total_atom
            )
        )

        f_specific = float(
            np.mean(
                popvar_atom
            )
        )

        f_R_cfg = math.sqrt(
            f_total
            / max(
                f_specific,
                EPS,
            )
        )


        s = store[variant]

        s["energy_total"].append(
            e_total
        )

        s["energy_cm_naive"].append(
            e_cm_naive
        )

        s[
            "energy_cm_corrected"
        ].append(
            e_cm_corr
        )

        s["energy_specific"].append(
            e_popvar
        )

        s["energy_error"].append(
            abs(ebar)
        )

        s["energy_spread"].append(
            e_spread
        )

        s["energy_R_config"].append(
            e_R_cfg
        )


        s["force_total"].append(
            f_total
        )

        s["force_cm_naive"].append(
            float(
                np.mean(
                    cm_naive_atom
                )
            )
        )

        s[
            "force_cm_corrected"
        ].append(
            float(
                np.mean(
                    cm_corr_atom
                )
            )
        )

        s["force_specific"].append(
            f_specific
        )

        s["force_error"].append(
            float(
                np.mean(
                    error_atom
                )
            )
        )

        s["force_spread"].append(
            float(
                np.mean(
                    spread_atom
                )
            )
        )

        s["force_R_config"].append(
            f_R_cfg
        )

        s["force_max_error"].append(
            float(
                np.max(
                    error_atom
                )
            )
        )

        s["force_max_spread"].append(
            float(
                np.max(
                    spread_atom
                )
            )
        )


    if ii % 250 == 0:
        print(
            f"Processed "
            f"{ii:,}/{n_cfg:,}"
        )


# ==========================================================
# Aggregate each roster
# ==========================================================

def aggregate(s):

    for key in list(s.keys()):

        if key == "M":
            continue

        s[key] = np.asarray(
            s[key],
            dtype=float,
        )


    et = wmean(
        s["energy_total"],
        weights,
    )

    ecm = wmean(
        s["energy_cm_corrected"],
        weights,
    )

    enaive = wmean(
        s["energy_cm_naive"],
        weights,
    )

    espec = wmean(
        s["energy_specific"],
        weights,
    )


    ft = wmean(
        s["force_total"],
        weights,
    )

    fcm = wmean(
        s["force_cm_corrected"],
        weights,
    )

    fnaive = wmean(
        s["force_cm_naive"],
        weights,
    )

    fspec = wmean(
        s["force_specific"],
        weights,
    )


    # D01 / D10 aggregate contrasts

    d1 = (
        decile == 1
    )

    d10 = (
        decile == 10
    )


    def fcm_subset(
        total_key,
        cm_key,
        mask,
    ):
        total = wmean(
            s[total_key][mask],
            weights[mask],
        )

        cm = wmean(
            s[cm_key][mask],
            weights[mask],
        )

        return (
            cm / total
        )


    def R_subset(
        total_key,
        spec_key,
        mask,
    ):
        total = wmean(
            s[total_key][mask],
            weights[mask],
        )

        spec = wmean(
            s[spec_key][mask],
            weights[mask],
        )

        return math.sqrt(
            total / max(spec, EPS)
        )


    result = {
        "M":
            s["M"],

        "energy_f_CM_corrected":
            ecm / et,

        "energy_f_CM_naive":
            enaive / et,

        "energy_R_total":
            math.sqrt(
                et / max(espec, EPS)
            ),

        "energy_rank_rho":
            weighted_spearman(
                s["energy_error"],
                s["energy_spread"],
                weights,
            ),

        "energy_ENCE":
            calibration(
                s["energy_error"],
                s["energy_spread"],
                weights,
            ),

        "force_f_CM_corrected":
            fcm / ft,

        "force_f_CM_naive":
            fnaive / ft,

        "force_R_total":
            math.sqrt(
                ft / max(fspec, EPS)
            ),

        "force_rank_rho":
            weighted_spearman(
                s["force_error"],
                s["force_spread"],
                weights,
            ),

        "force_ENCE":
            calibration(
                s["force_error"],
                s["force_spread"],
                weights,
            ),

        "force_max_rank_rho":
            weighted_spearman(
                s["force_max_error"],
                s["force_max_spread"],
                weights,
            ),

        "force_max_ENCE":
            calibration(
                s["force_max_error"],
                s["force_max_spread"],
                weights,
            ),

        "deq_vs_energy_corrected_CM":
            weighted_spearman(
                deq,
                s[
                    "energy_cm_corrected"
                ],
                weights,
            ),

        "deq_vs_force_corrected_CM":
            weighted_spearman(
                deq,
                s[
                    "force_cm_corrected"
                ],
                weights,
            ),

        "deq_vs_energy_R_config":
            weighted_spearman(
                deq,
                s["energy_R_config"],
                weights,
            ),

        "deq_vs_force_R_config":
            weighted_spearman(
                deq,
                s["force_R_config"],
                weights,
            ),
    }


    result[
        "energy_f_CM_D10_minus_D01"
    ] = (
        fcm_subset(
            "energy_total",
            "energy_cm_corrected",
            d10,
        )
        -
        fcm_subset(
            "energy_total",
            "energy_cm_corrected",
            d1,
        )
    )


    result[
        "force_f_CM_D10_minus_D01"
    ] = (
        fcm_subset(
            "force_total",
            "force_cm_corrected",
            d10,
        )
        -
        fcm_subset(
            "force_total",
            "force_cm_corrected",
            d1,
        )
    )


    result[
        "energy_R_D10_minus_D01"
    ] = (
        R_subset(
            "energy_total",
            "energy_specific",
            d10,
        )
        -
        R_subset(
            "energy_total",
            "energy_specific",
            d1,
        )
    )


    result[
        "force_R_D10_minus_D01"
    ] = (
        R_subset(
            "force_total",
            "force_specific",
            d10,
        )
        -
        R_subset(
            "force_total",
            "force_specific",
            d1,
        )
    )


    return result


results = {}

for variant, s in store.items():
    results[variant] = aggregate(s)


# ==========================================================
# Cross-check full 8-model results
# ==========================================================

frozen = json.loads(
    POINT.read_text()
)

full = results[
    "FULL_8_MODEL_CORE"
]

checks = {
    "energy_f_CM_corrected":
        frozen[
            "energy"
        ][
            "f_CM_corrected"
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

    "force_f_CM_corrected":
        frozen[
            "force_mean_over_atoms"
        ][
            "f_CM_corrected"
        ],

    "force_R_total":
        frozen[
            "force_mean_over_atoms"
        ][
            "R_total_direct"
        ],

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
        ][
            "ENCE"
        ],

    "force_max_rank_rho":
        frozen[
            "force_max_over_atoms_control"
        ][
            "weighted_spearman_error_vs_spread"
        ],

    "force_max_ENCE":
        frozen[
            "force_max_over_atoms_control"
        ][
            "calibration"
        ][
            "ENCE"
        ],
}


max_crosscheck = max(
    abs(
        full[key]
        - float(value)
    )
    for key, value
    in checks.items()
)

assert max_crosscheck < 1e-10

print(
    "\nFull-core cross-check: PASS"
)


# ==========================================================
# Build table with changes relative to full core
# ==========================================================

metric_names = [
    key
    for key in full
    if key != "M"
]

table_rows = []

for variant, result in results.items():

    dropped_family = (
        ""
        if variant
        == "FULL_8_MODEL_CORE"
        else variant.replace(
            "DROP_",
            "",
            1,
        )
    )

    row = {
        "variant":
            variant,

        "dropped_family":
            dropped_family,

        "M":
            result["M"],
    }

    for metric in metric_names:

        row[metric] = (
            result[metric]
        )

        row[
            f"delta__{metric}"
        ] = (
            result[metric]
            - full[metric]
        )

    table_rows.append(row)


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
# Stability envelope
# ==========================================================

family_rows = [
    r
    for r in table_rows
    if r["variant"]
    != "FULL_8_MODEL_CORE"
]

stability = {}

for metric in metric_names:

    vals = np.asarray(
        [
            float(r[metric])
            for r in family_rows
        ]
    )

    deltas = np.asarray(
        [
            float(
                r[
                    f"delta__{metric}"
                ]
            )
            for r in family_rows
        ]
    )

    stability[metric] = {
        "full_core":
            float(full[metric]),

        "leave_one_family_out_min":
            float(np.min(vals)),

        "leave_one_family_out_max":
            float(np.max(vals)),

        "max_absolute_change":
            float(
                np.max(
                    np.abs(deltas)
                )
            ),

        "all_same_sign_as_full":
            bool(
                np.all(
                    np.sign(vals)
                    == np.sign(
                        full[metric]
                    )
                )
            )
            if full[metric] != 0
            else None,
    }


all_finite = all(
    np.isfinite(
        float(r[m])
    )
    for r in table_rows
    for m in metric_names
)


status = (
    "PASS"
    if (
        all_finite
        and len(family_rows) == 7
        and max_crosscheck < 1e-10
    )
    else "REVISE"
)


audit = {
    "stage":
        "STEP8_PRIMARY_LEAVE_ONE_ARCHITECTURE_FAMILY_OUT",

    "status":
        status,

    "n_configs":
        n_cfg,

    "fixed_full_M_core":
        8,

    "architecture_families":
        FAMILIES,

    "n_architecture_families":
        len(FAMILIES),

    "sensitivity_variants":
        {
            k: v
            for k, v
            in VARIANTS.items()
        },

    "finite_M_correction_recomputed_for_each_variant":
        True,

    "MACE_family_omission_M":
        6,

    "single_model_family_omission_M":
        7,

    "sample_membership_changed":
        False,

    "prediction_values_changed":
        False,

    "full_core_point_estimate_crosscheck_max_abs_difference":
        max_crosscheck,

    "full_core_crosscheck_pass":
        bool(
            max_crosscheck
            < 1e-10
        ),

    "stability_envelope":
        stability,

    "table":
        str(TABLE),

    "table_sha256":
        sha256(TABLE),

    "source_metadata":
        str(META),

    "source_metadata_sha256":
        sha256(META),

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
print("LEAVE-ONE-FAMILY-OUT SUMMARY")
print("=" * 78)

header = (
    "Variant               M   "
    "E_fCM    F_fCM    "
    "E_R      F_R      "
    "E_rho    F_rho"
)

print(header)

for r in table_rows:

    print(
        f"{r['variant'][:21]:21s} "
        f"{int(r['M']):1d} "
        f"{r['energy_f_CM_corrected']:8.4f} "
        f"{r['force_f_CM_corrected']:8.4f} "
        f"{r['energy_R_total']:8.4f} "
        f"{r['force_R_total']:8.4f} "
        f"{r['energy_rank_rho']:8.4f} "
        f"{r['force_rank_rho']:8.4f}"
    )


print("\nContinuous d_eq sensitivity")

for r in table_rows:

    print(
        f"{r['variant'][:21]:21s} "
        f"E_CM={r['deq_vs_energy_corrected_CM']:+.4f} "
        f"F_CM={r['deq_vs_force_corrected_CM']:+.4f} "
        f"E_R={r['deq_vs_energy_R_config']:+.4f} "
        f"F_R={r['deq_vs_force_R_config']:+.4f}"
    )


print("\nD10-D01 sensitivity")

for r in table_rows:

    print(
        f"{r['variant'][:21]:21s} "
        f"E_fCM={r['energy_f_CM_D10_minus_D01']:+.4f} "
        f"F_fCM={r['force_f_CM_D10_minus_D01']:+.4f} "
        f"E_R={r['energy_R_D10_minus_D01']:+.4f} "
        f"F_R={r['force_R_D10_minus_D01']:+.4f}"
    )


print("\nFiles")
print("Table :", TABLE)
print("SHA256:", sha256(TABLE))
print("Audit :", AUDIT)
print("SHA256:", sha256(AUDIT))

print("\n" + "=" * 78)

if status == "PASS":
    print(
        "STEP8 LEAVE-ONE-ARCHITECTURE-"
        "FAMILY-OUT: PASS"
    )
else:
    print(
        "STEP8 LEAVE-ONE-ARCHITECTURE-"
        "FAMILY-OUT: REVISE"
    )
