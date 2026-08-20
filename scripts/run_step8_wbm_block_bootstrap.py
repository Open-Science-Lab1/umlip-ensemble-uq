#!/usr/bin/env python3

import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.stats import rankdata


ROOT = Path("step8/controls/wbm")

INDEX = (
    ROOT
    / "WBM_CONTROL_STRUCTURE_REFERENCE_INDEX_v1.csv.gz"
)

DECOMP = (
    ROOT
    / "WBM_SIGNED_ENERGY_ERROR_DECOMPOSITION_v1.csv.gz"
)

DECOMP_AUDIT = (
    ROOT
    / "WBM_SIGNED_ENERGY_ERROR_DECOMPOSITION_AUDIT_v1.json"
)

POINT = (
    ROOT
    / "WBM_WEIGHTED_POINT_DIAGNOSTICS_v1.json"
)

POINT_AUDIT = (
    ROOT
    / "WBM_WEIGHTED_POINT_DIAGNOSTICS_AUDIT_v1.json"
)


EXPECTED_INDEX_SHA = (
    "b339e4db292cb709b808f634716a91fe"
    "705129e94f488fe92d04be568deafefa"
)

EXPECTED_DECOMP_SHA = (
    "627a4ca42414cfefc4be13dfc77eca1a"
    "15c57fbc757025e571a34fec91f36f40"
)

EXPECTED_DECOMP_AUDIT_SHA = (
    "a0d239795ad425799357a8e32009deb4"
    "ed25addac354fbe815d267dcd3785d7c"
)

EXPECTED_POINT_SHA = (
    "8950a81a2a592e5386ca2110766f35cb"
    "ecdf71625cbd5f97d415445da55f812d"
)

EXPECTED_POINT_AUDIT_SHA = (
    "5ab0e10dd474d35889624bccf315256e"
    "fbb43dac125434993e65548339be06de"
)


SEED = 20260812
N_BOOT = 2000
N_CAL_BINS = 10
EPS = 1e-15


PRIMARY_BLOCK = "relaxed_protostructure"
SENSITIVITY_BLOCK = "initial_protostructure"

EXPECTED_RELAXED_BLOCKS = 2483
EXPECTED_INITIAL_BLOCKS = 2486


OUT_PRIMARY = (
    ROOT
    / "WBM_RELAXED_PROTOTYPE_BOOTSTRAP_REPLICATES_v1.csv.gz"
)

OUT_SENS = (
    ROOT
    / "WBM_INITIAL_PROTOTYPE_BOOTSTRAP_REPLICATES_v1.csv.gz"
)

CHECK_PRIMARY = (
    ROOT
    / "WBM_RELAXED_PROTOTYPE_BOOTSTRAP_CHECKPOINT_v1.npz"
)

CHECK_SENS = (
    ROOT
    / "WBM_INITIAL_PROTOTYPE_BOOTSTRAP_CHECKPOINT_v1.npz"
)

SUMMARY = (
    ROOT
    / "WBM_BLOCK_BOOTSTRAP_SUMMARY_v1.json"
)

AUDIT = (
    ROOT
    / "WBM_BLOCK_BOOTSTRAP_AUDIT_v1.json"
)


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

    coverage2 = wmean(
        (
            error <= 2.0 * sigma
        ).astype(float),
        w,
    )

    return {
        "ENCE":
            float(ence),

        "coverage1":
            coverage1,

        "coverage2":
            coverage2,
    }


# ==========================================================
# Provenance
# ==========================================================

print("=" * 78)
print(
    "STEP-8 WBM 2000-REPLICATE BLOCK BOOTSTRAP"
)
print("=" * 78)


for path, expected in {
    INDEX:
        EXPECTED_INDEX_SHA,

    DECOMP:
        EXPECTED_DECOMP_SHA,

    DECOMP_AUDIT:
        EXPECTED_DECOMP_AUDIT_SHA,

    POINT:
        EXPECTED_POINT_SHA,

    POINT_AUDIT:
        EXPECTED_POINT_AUDIT_SHA,
}.items():

    assert path.exists(), path

    actual = sha256(path)

    assert actual == expected, (
        path,
        actual,
        expected,
    )


with open(
    POINT_AUDIT,
    "r",
) as f:

    point_audit = json.load(f)


assert (
    point_audit["status"]
    == "PASS"
)


print(
    "Frozen point diagnostics: PASS"
)


# ==========================================================
# Load decomposition preserving exact row order
# ==========================================================

with gzip.open(
    DECOMP,
    "rt",
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


assert len(rows) == 2500


material_id = np.asarray(
    [
        str(
            row["material_id"]
        )
        for row in rows
    ],
    dtype=object,
)


def A(name):
    return np.asarray(
        [
            float(
                row[name]
            )
            for row in rows
        ],
        dtype=float,
    )


round_id = np.asarray(
    [
        int(
            row["wbm_round"]
        )
        for row in rows
    ],
    dtype=int,
)


weights = A(
    "sampling_weight"
)

e_total = A(
    "energy_total_mse_across_models_eV2_per_atom2"
)

e_cm_naive = A(
    "energy_cm_naive_eV2_per_atom2"
)

e_cm_corr = A(
    "energy_cm_corrected_eV2_per_atom2"
)

e_specific = A(
    "energy_population_variance_eV2_per_atom2"
)

e_mean_signed = A(
    "ensemble_mean_signed_error_eV_per_atom"
)

e_error = np.abs(
    e_mean_signed
)

e_spread = A(
    "energy_ensemble_spread_eV_per_atom"
)


for x in [
    weights,
    e_total,
    e_cm_naive,
    e_cm_corr,
    e_specific,
    e_error,
    e_spread,
]:
    assert np.isfinite(x).all()


assert np.all(
    weights > 0
)


# ==========================================================
# Join prototype block labels
# ==========================================================

with gzip.open(
    INDEX,
    "rt",
    newline="",
) as f:

    index_rows = list(
        csv.DictReader(f)
    )


assert len(index_rows) == 2500


index_lookup = {
    str(
        row["material_id"]
    ):
        row
    for row in index_rows
}


assert set(
    material_id.tolist()
) == set(
    index_lookup
)


relaxed_proto = np.asarray(
    [
        str(
            index_lookup[mid][
                "relaxed_protostructure"
            ]
        )
        for mid in material_id
    ],
    dtype=object,
)


initial_proto = np.asarray(
    [
        str(
            index_lookup[mid][
                "initial_protostructure"
            ]
        )
        for mid in material_id
    ],
    dtype=object,
)


assert all(
    x.strip()
    for x in relaxed_proto
)

assert all(
    x.strip()
    for x in initial_proto
)


n_relaxed = len(
    np.unique(
        relaxed_proto
    )
)

n_initial = len(
    np.unique(
        initial_proto
    )
)


assert (
    n_relaxed
    == EXPECTED_RELAXED_BLOCKS
)

assert (
    n_initial
    == EXPECTED_INITIAL_BLOCKS
)


print(
    "Relaxed prototype blocks:",
    n_relaxed,
)

print(
    "Initial prototype blocks:",
    n_initial,
)

print(
    "Relaxed duplicate occurrences:",
    2500 - n_relaxed,
)

print(
    "Initial duplicate occurrences:",
    2500 - n_initial,
)


# ==========================================================
# Metric implementation
# ==========================================================

def decomposition_metrics(idx):

    idx = np.asarray(
        idx,
        dtype=int,
    )

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
        "fCM_corrected":
            f_corr,

        "fCM_naive":
            f_naive,

        "R_total":
            R,

        "rho":
            rho,

        "ENCE":
            cal["ENCE"],

        "coverage1":
            cal["coverage1"],

        "coverage2":
            cal["coverage2"],
    }


def compute_metrics(idx):

    idx = np.asarray(
        idx,
        dtype=int,
    )

    w = weights[idx]

    E = decomposition_metrics(
        idx
    )

    out = {
        "overall_fCM_corrected":
            E["fCM_corrected"],

        "overall_fCM_naive":
            E["fCM_naive"],

        "overall_R_total":
            E["R_total"],

        "overall_rho":
            E["rho"],

        "overall_ENCE":
            E["ENCE"],

        "overall_coverage1":
            E["coverage1"],

        "overall_coverage2":
            E["coverage2"],
    }


    per_round = {}

    for r in range(
        1,
        6,
    ):

        ridx = idx[
            round_id[idx] == r
        ]

        if len(ridx) == 0:
            raise RuntimeError(
                f"bootstrap replicate missing R{r}"
            )

        er = decomposition_metrics(
            ridx
        )

        per_round[r] = er

        out[
            f"R{r}_fCM_corrected"
        ] = er[
            "fCM_corrected"
        ]

        out[
            f"R{r}_R_total"
        ] = er[
            "R_total"
        ]

        out[
            f"R{r}_rho"
        ] = er[
            "rho"
        ]

        out[
            f"R{r}_ENCE"
        ] = er[
            "ENCE"
        ]


    R_config = np.sqrt(
        e_total[idx]
        /
        np.maximum(
            e_specific[idx],
            EPS,
        )
    )


    round_numeric = round_id[
        idx
    ].astype(float)


    out[
        "round_vs_corrected_CM"
    ] = weighted_spearman(
        round_numeric,
        e_cm_corr[idx],
        w,
    )

    out[
        "round_vs_abs_error"
    ] = weighted_spearman(
        round_numeric,
        e_error[idx],
        w,
    )

    out[
        "round_vs_spread"
    ] = weighted_spearman(
        round_numeric,
        e_spread[idx],
        w,
    )

    out[
        "round_vs_R_config"
    ] = weighted_spearman(
        round_numeric,
        R_config,
        w,
    )


    out[
        "R5_minus_R1_fCM_corrected"
    ] = (
        per_round[5][
            "fCM_corrected"
        ]
        -
        per_round[1][
            "fCM_corrected"
        ]
    )

    out[
        "R5_minus_R1_R_total"
    ] = (
        per_round[5][
            "R_total"
        ]
        -
        per_round[1][
            "R_total"
        ]
    )

    out[
        "R5_minus_R1_rho"
    ] = (
        per_round[5][
            "rho"
        ]
        -
        per_round[1][
            "rho"
        ]
    )

    out[
        "R5_minus_R1_ENCE"
    ] = (
        per_round[5][
            "ENCE"
        ]
        -
        per_round[1][
            "ENCE"
        ]
    )


    vals = np.asarray(
        list(
            out.values()
        ),
        dtype=float,
    )

    if not np.isfinite(
        vals
    ).all():

        raise RuntimeError(
            "nonfinite bootstrap metric"
        )


    return out


# ==========================================================
# Point-estimate crosscheck
# ==========================================================

point_calc = compute_metrics(
    np.arange(
        2500,
        dtype=int,
    )
)


with open(
    POINT,
    "r",
) as f:

    frozen_point = json.load(f)


crosschecks = {}


crosschecks[
    "overall_fCM_corrected"
] = abs(
    point_calc[
        "overall_fCM_corrected"
    ]
    -
    frozen_point[
        "overall"
    ][
        "energy_f_CM_corrected"
    ]
)

crosschecks[
    "overall_fCM_naive"
] = abs(
    point_calc[
        "overall_fCM_naive"
    ]
    -
    frozen_point[
        "overall"
    ][
        "energy_f_CM_naive"
    ]
)

crosschecks[
    "overall_R_total"
] = abs(
    point_calc[
        "overall_R_total"
    ]
    -
    frozen_point[
        "overall"
    ][
        "energy_R_total"
    ]
)

crosschecks[
    "overall_rho"
] = abs(
    point_calc[
        "overall_rho"
    ]
    -
    frozen_point[
        "overall"
    ][
        "energy_rank_rho"
    ]
)

crosschecks[
    "overall_ENCE"
] = abs(
    point_calc[
        "overall_ENCE"
    ]
    -
    frozen_point[
        "overall"
    ][
        "energy_ENCE"
    ]
)


for r in range(
    1,
    6,
):

    fp = frozen_point[
        "per_WBM_round"
    ][
        str(r)
    ]

    for local, frozen_key in [
        (
            "fCM_corrected",
            "energy_f_CM_corrected",
        ),
        (
            "R_total",
            "energy_R_total",
        ),
        (
            "rho",
            "energy_rank_rho",
        ),
        (
            "ENCE",
            "energy_ENCE",
        ),
    ]:

        crosschecks[
            f"R{r}_{local}"
        ] = abs(
            point_calc[
                f"R{r}_{local}"
            ]
            -
            fp[
                frozen_key
            ]
        )


trend_map = {
    "round_vs_corrected_CM":
        "WBM_round_vs_energy_corrected_CM_spearman",

    "round_vs_abs_error":
        "WBM_round_vs_energy_abs_ensemble_error_spearman",

    "round_vs_spread":
        "WBM_round_vs_energy_ensemble_spread_spearman",

    "round_vs_R_config":
        "WBM_round_vs_energy_R_config_spearman",
}


for local, frozen_key in trend_map.items():

    crosschecks[
        local
    ] = abs(
        point_calc[
            local
        ]
        -
        frozen_point[
            "ordinal_round_trends"
        ][
            frozen_key
        ]
    )


contrast_map = {
    "R5_minus_R1_fCM_corrected":
        "energy_f_CM_corrected",

    "R5_minus_R1_R_total":
        "energy_R_total",

    "R5_minus_R1_rho":
        "energy_rank_rho",

    "R5_minus_R1_ENCE":
        "energy_ENCE",
}


for local, frozen_key in contrast_map.items():

    crosschecks[
        local
    ] = abs(
        point_calc[
            local
        ]
        -
        frozen_point[
            "R5_minus_R1"
        ][
            frozen_key
        ]
    )


max_crosscheck = max(
    crosschecks.values()
)


assert (
    max_crosscheck
    <= 1e-12
), max_crosscheck


print(
    "Point-estimate max crosscheck:",
    f"{max_crosscheck:.3e}",
)

print(
    "POINT CROSSCHECK: PASS"
)


METRIC_NAMES = list(
    point_calc.keys()
)


# ==========================================================
# Resumable deterministic block bootstrap
# ==========================================================

def atomic_npz(
    path,
    values,
    completed,
):

    with NamedTemporaryFile(
        suffix=".npz",
        dir=ROOT,
        delete=False,
    ) as tmp:

        tmp_path = Path(
            tmp.name
        )

    try:

        np.savez_compressed(
            tmp_path,

            values=
                values,

            completed=
                np.asarray(
                    completed,
                    dtype=int,
                ),

            metric_names=
                np.asarray(
                    METRIC_NAMES,
                    dtype="U100",
                ),
        )

        os.replace(
            tmp_path,
            path,
        )

    finally:

        if tmp_path.exists():
            tmp_path.unlink()


def write_replicates(
    path,
    values,
):

    buffer = io.StringIO(
        newline=""
    )

    fieldnames = [
        "bootstrap_rep"
    ] + METRIC_NAMES

    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        lineterminator="\n",
    )

    writer.writeheader()

    for rep in range(
        N_BOOT
    ):

        row = {
            "bootstrap_rep":
                rep + 1
        }

        for j, name in enumerate(
            METRIC_NAMES
        ):

            row[name] = (
                f"{values[rep, j]:.17g}"
            )

        writer.writerow(
            row
        )

    raw = buffer.getvalue().encode(
        "utf-8"
    )

    with open(
        path,
        "wb",
    ) as raw_f:

        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_f,
            mtime=0,
        ) as gz_f:

            gz_f.write(
                raw
            )


def run_scenario(
    scenario_name,
    labels,
    scenario_code,
    checkpoint,
    outfile,
):

    unique_blocks = np.asarray(
        sorted(
            set(
                labels.tolist()
            )
        ),
        dtype=object,
    )

    n_blocks = len(
        unique_blocks
    )


    block_to_idx = {
        block:
            np.flatnonzero(
                labels == block
            )

        for block in unique_blocks
    }


    if checkpoint.exists():

        with np.load(
            checkpoint,
            allow_pickle=False,
        ) as z:

            values = np.asarray(
                z["values"],
                dtype=float,
            )

            completed = int(
                np.asarray(
                    z["completed"]
                ).item()
            )

            old_names = (
                np.asarray(
                    z["metric_names"]
                ).astype(str).tolist()
            )


        assert (
            old_names
            == METRIC_NAMES
        )

        assert values.shape == (
            N_BOOT,
            len(
                METRIC_NAMES
            ),
        )

    else:

        values = np.full(
            (
                N_BOOT,
                len(
                    METRIC_NAMES
                ),
            ),
            np.nan,
            dtype=float,
        )

        completed = 0


    print()
    print(
        f"{scenario_name}: "
        f"{n_blocks} global blocks"
    )

    print(
        f"Starting from replicate "
        f"{completed + 1}"
        if completed < N_BOOT
        else
        "All replicates already complete"
    )


    for rep in range(
        completed,
        N_BOOT,
    ):

        # Deterministic independent stream per replicate.
        # Makes reruns/resume invariant to interruption.
        ss = np.random.SeedSequence(
            [
                SEED,
                scenario_code,
                rep,
            ]
        )

        rng = np.random.default_rng(
            ss
        )


        draws = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )


        idx = np.concatenate(
            [
                block_to_idx[
                    unique_blocks[j]
                ]
                for j in draws
            ]
        )


        metrics = compute_metrics(
            idx
        )


        values[
            rep,
            :
        ] = [
            metrics[
                name
            ]
            for name in METRIC_NAMES
        ]


        done = rep + 1


        if (
            done % 100 == 0
            or done == N_BOOT
        ):

            atomic_npz(
                checkpoint,
                values,
                done,
            )

            print(
                f"{scenario_name}: "
                f"{done:,}/{N_BOOT:,}"
            )


    assert np.isfinite(
        values
    ).all()


    write_replicates(
        outfile,
        values,
    )


    return (
        values,
        n_blocks,
    )


primary_values, primary_blocks = run_scenario(
    "PRIMARY relaxed-protostructure",
    relaxed_proto,
    1,
    CHECK_PRIMARY,
    OUT_PRIMARY,
)


sensitivity_values, sensitivity_blocks = run_scenario(
    "SENSITIVITY initial-protostructure",
    initial_proto,
    2,
    CHECK_SENS,
    OUT_SENS,
)


# ==========================================================
# Percentile CI summaries
# ==========================================================

def summarize(
    values,
):

    out = {}

    for j, name in enumerate(
        METRIC_NAMES
    ):

        x = values[
            :,
            j
        ]

        q = np.quantile(
            x,
            [
                0.025,
                0.5,
                0.975,
            ],
            method="linear",
        )

        out[
            name
        ] = {
            "point_estimate":
                float(
                    point_calc[
                        name
                    ]
                ),

            "bootstrap_median":
                float(
                    q[1]
                ),

            "CI95_lower":
                float(
                    q[0]
                ),

            "CI95_upper":
                float(
                    q[2]
                ),
        }

    return out


primary_summary = summarize(
    primary_values
)

sensitivity_summary = summarize(
    sensitivity_values
)


summary = {
    "stage":
        "STEP8_WBM_BLOCK_BOOTSTRAP",

    "status":
        "PASS",

    "bootstrap_replicates":
        N_BOOT,

    "seed":
        SEED,

    "confidence_interval":
        "percentile_2.5_97.5",

    "chemical_novelty_coordinate":
        "WBM_substitution_round_1_to_5",

    "primary_block":
        "global_relaxed_protostructure",

    "primary_number_of_blocks":
        primary_blocks,

    "secondary_sensitivity_block":
        "global_initial_protostructure",

    "secondary_number_of_blocks":
        sensitivity_blocks,

    "resampling":
        (
            "sample global prototype blocks "
            "with replacement; retain every row "
            "belonging to each selected block"
        ),

    "sample_membership_changed":
        False,

    "sample_replacements":
        0,

    "d_eq_used":
        False,

    "N_CAL_BINS":
        N_CAL_BINS,

    "EPS":
        EPS,

    "point_estimate_max_abs_crosscheck_difference":
        max_crosscheck,

    "primary_relaxed_prototype":
        primary_summary,

    "secondary_initial_prototype":
        sensitivity_summary,

    "input_hashes": {
        "index":
            sha256(
                INDEX
            ),

        "decomposition":
            sha256(
                DECOMP
            ),

        "decomposition_audit":
            sha256(
                DECOMP_AUDIT
            ),

        "point_diagnostics":
            sha256(
                POINT
            ),

        "point_diagnostics_audit":
            sha256(
                POINT_AUDIT
            ),
    },

    "replicate_file_hashes": {
        "primary_relaxed":
            sha256(
                OUT_PRIMARY
            ),

        "secondary_initial":
            sha256(
                OUT_SENS
            ),
    },
}


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


audit = {
    "stage":
        "STEP8_WBM_BLOCK_BOOTSTRAP",

    "status":
        "PASS",

    "bootstrap_replicates_per_scenario":
        N_BOOT,

    "seed":
        SEED,

    "primary_block":
        PRIMARY_BLOCK,

    "primary_unique_blocks":
        primary_blocks,

    "primary_duplicate_occurrences":
        2500
        - primary_blocks,

    "sensitivity_block":
        SENSITIVITY_BLOCK,

    "sensitivity_unique_blocks":
        sensitivity_blocks,

    "sensitivity_duplicate_occurrences":
        2500
        - sensitivity_blocks,

    "all_primary_metrics_finite":
        bool(
            np.isfinite(
                primary_values
            ).all()
        ),

    "all_sensitivity_metrics_finite":
        bool(
            np.isfinite(
                sensitivity_values
            ).all()
        ),

    "point_crosscheck_max_abs_difference":
        max_crosscheck,

    "primary_replicates":
        str(
            OUT_PRIMARY
        ),

    "primary_replicates_sha256":
        sha256(
            OUT_PRIMARY
        ),

    "sensitivity_replicates":
        str(
            OUT_SENS
        ),

    "sensitivity_replicates_sha256":
        sha256(
            OUT_SENS
        ),

    "summary":
        str(
            SUMMARY
        ),

    "summary_sha256":
        sha256(
            SUMMARY
        ),

    "checkpoint_primary":
        str(
            CHECK_PRIMARY
        ),

    "checkpoint_sensitivity":
        str(
            CHECK_SENS
        ),
}


AUDIT.write_text(
    json.dumps(
        audit,
        indent=2,
    )
)


# ==========================================================
# Compact terminal report
# ==========================================================

print()
print("=" * 78)
print(
    "WBM BLOCK BOOTSTRAP SUMMARY"
)
print("=" * 78)


KEYS = [
    "overall_fCM_corrected",
    "overall_R_total",
    "overall_rho",
    "overall_ENCE",

    "round_vs_corrected_CM",
    "round_vs_abs_error",
    "round_vs_spread",
    "round_vs_R_config",

    "R5_minus_R1_fCM_corrected",
    "R5_minus_R1_R_total",
    "R5_minus_R1_rho",
    "R5_minus_R1_ENCE",
]


def compact(
    label,
    summary_obj,
):

    print()
    print(label)

    for key in KEYS:

        x = summary_obj[
            key
        ]

        print(
            f"{key:31s} "
            f"{x['point_estimate']:+.6f}  "
            f"["
            f"{x['CI95_lower']:+.6f}, "
            f"{x['CI95_upper']:+.6f}"
            f"]"
        )


compact(
    "PRIMARY — relaxed protostructure",
    primary_summary,
)

compact(
    "SENSITIVITY — initial protostructure",
    sensitivity_summary,
)


print()
print(
    "Primary replicate SHA256 :",
    sha256(
        OUT_PRIMARY
    ),
)

print(
    "Sensitivity replicate SHA:",
    sha256(
        OUT_SENS
    ),
)

print(
    "Summary SHA256           :",
    sha256(
        SUMMARY
    ),
)

print(
    "Audit SHA256             :",
    sha256(
        AUDIT
    ),
)

print()
print(
    "WBM 2000-REPLICATE BLOCK BOOTSTRAP: PASS"
)
