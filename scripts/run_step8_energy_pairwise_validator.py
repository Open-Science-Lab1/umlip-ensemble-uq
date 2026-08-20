#!/usr/bin/env python3

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


INPUT = Path(
    "step8/analysis/"
    "PRIMARY_SIGNED_ERROR_DECOMPOSITION_v1.csv.gz"
)

OUTDIR = Path("step8/analysis")

OUT = OUTDIR / (
    "PRIMARY_ENERGY_PAIRWISE_SIGNED_CORRELATION_VALIDATOR_v1.json"
)

NULLCSV = OUTDIR / (
    "PRIMARY_ENERGY_PAIRWISE_SHUFFLE_NULL_v1.csv"
)

SEED = 20260812
B = 2000


MODELS = [
    "CHGNet",
    "MACE_MP_0",
    "SevenNet_l3i5",
    "ORB_v2_MPtrj",
    "GRACE_2L_MPtrj",
    "eqV2_S_DeNS",
    "eSEN_30M_OAM",
    "MACE_MPA_0",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def wcorr(x, y, w):
    sw = np.sum(w)

    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw

    dx = x - mx
    dy = y - my

    den = np.sqrt(
        np.sum(w * dx * dx)
        * np.sum(w * dy * dy)
    )

    if den <= 0:
        return np.nan

    return float(
        np.sum(w * dx * dy)
        / den
    )


print("=" * 78)
print(
    "STEP-8 ENERGY SIGNED PAIRWISE "
    "CORRELATION VALIDATOR"
)
print("=" * 78)


with gzip.open(
    INPUT,
    "rt",
    newline="",
) as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 2998


w = np.asarray(
    [
        float(x["sampling_weight"])
        for x in rows
    ],
    dtype=float,
)


E = np.column_stack([
    np.asarray(
        [
            float(
                x[
                    f"energy_error__{model}"
                ]
            )
            for x in rows
        ],
        dtype=float,
    )
    for model in MODELS
])


assert E.shape == (2998, 8)
assert np.isfinite(E).all()
assert np.isfinite(w).all()
assert np.all(w > 0)


# ==========================================================
# Observed pairwise signed-error correlations
# ==========================================================

pairwise = []

for i in range(8):
    for j in range(i + 1, 8):

        r = wcorr(
            E[:, i],
            E[:, j],
            w,
        )

        pairwise.append({
            "model_1": MODELS[i],
            "model_2": MODELS[j],
            "weighted_signed_error_correlation":
                r,
        })


observed_mean = float(
    np.mean([
        x[
            "weighted_signed_error_correlation"
        ]
        for x in pairwise
    ])
)

observed_median = float(
    np.median([
        x[
            "weighted_signed_error_correlation"
        ]
        for x in pairwise
    ])
)


print("Configurations :", len(rows))
print("Model pairs    :", len(pairwise))
print(
    "Observed mean pairwise correlation:",
    f"{observed_mean:.6f}"
)
print(
    "Observed median pairwise correlation:",
    f"{observed_median:.6f}"
)


# ==========================================================
# Correct shuffle null
#
# Independently permute each model's errors ACROSS
# configurations.
# ==========================================================

rng = np.random.default_rng(SEED)

null_mean = np.empty(
    B,
    dtype=float,
)

with open(
    NULLCSV,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "replicate",
            "mean_pairwise_signed_error_correlation",
        ],
    )

    writer.writeheader()

    for b in range(B):

        Ep = np.empty_like(E)

        for m in range(8):
            Ep[:, m] = E[
                rng.permutation(len(E)),
                m,
            ]

        vals = []

        for i in range(8):
            for j in range(i + 1, 8):

                vals.append(
                    wcorr(
                        Ep[:, i],
                        Ep[:, j],
                        w,
                    )
                )

        null_mean[b] = np.mean(vals)

        writer.writerow({
            "replicate": b + 1,
            "mean_pairwise_signed_error_correlation":
                null_mean[b],
        })

        if (b + 1) % 200 == 0:
            print(
                f"Shuffle {b + 1:,}/{B:,}"
            )


q025, q50, q975 = np.percentile(
    null_mean,
    [2.5, 50, 97.5],
)


# One-sided empirical tail probability for
# greater shared signed correlation.

p_greater = float(
    (
        1
        + np.sum(
            null_mean >= observed_mean
        )
    )
    / (B + 1)
)


record = {
    "stage":
        "STEP8_ENERGY_PAIRWISE_SIGNED_CORRELATION_VALIDATOR",

    "status":
        "PASS",

    "n_configs":
        2998,

    "n_models":
        8,

    "n_model_pairs":
        28,

    "weighting":
        "frozen inverse-inclusion sampling weights",

    "observed_mean_pairwise_signed_error_correlation":
        observed_mean,

    "observed_median_pairwise_signed_error_correlation":
        observed_median,

    "pairwise_correlations":
        pairwise,

    "shuffle_null": {
        "definition":
            "independently permute each model's signed errors across configurations",

        "replicates":
            B,

        "seed":
            SEED,

        "mean":
            float(np.mean(null_mean)),

        "median":
            float(q50),

        "CI95_lower":
            float(q025),

        "CI95_upper":
            float(q975),

        "empirical_one_sided_p_greater":
            p_greater,
    },

    "source":
        str(INPUT),

    "source_sha256":
        sha256(INPUT),

    "shuffle_file":
        str(NULLCSV),

    "shuffle_file_sha256":
        sha256(NULLCSV),

    "role":
        (
            "Independent validator of common-mode "
            "signed error; does not replace primary "
            "bias-corrected f_CM estimator."
        ),

    "H1_H5_decision_performed":
        False,
}

OUT.write_text(
    json.dumps(
        record,
        indent=2,
    )
)


print("\n" + "=" * 78)
print("ENERGY VALIDATOR SUMMARY")
print("=" * 78)

print(
    "Observed mean r :",
    f"{observed_mean:.6f}"
)

print(
    "Shuffle mean    :",
    f"{np.mean(null_mean):.6f}"
)

print(
    "Shuffle 95% CI  :",
    f"[{q025:.6f}, {q975:.6f}]"
)

print(
    "Empirical p     :",
    f"{p_greater:.6g}"
)

print("\nOutput:", OUT)
print("SHA256:", sha256(OUT))
print("Null  :", NULLCSV)
print("SHA256:", sha256(NULLCSV))

print("\n" + "=" * 78)
print(
    "STEP8 ENERGY SIGNED-ERROR "
    "VALIDATOR: PASS"
)
