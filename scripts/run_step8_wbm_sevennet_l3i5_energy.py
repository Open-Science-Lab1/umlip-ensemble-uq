#!/usr/bin/env python3

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sevenn.calculator import SevenNetCalculator


BUNDLE = Path(
    "step8/controls/wbm/"
    "WBM_CONTROL_STRUCTURE_REFERENCE_BUNDLE_v1.json.gz"
)

EXPECTED_BUNDLE_SHA = (
    "a66fc8c9d65d791cb5dca1c462d9998d"
    "f00b308e3bf6e794028f8be4a1bf7bb0"
)

INDEX = Path(
    "step8/controls/wbm/"
    "WBM_CONTROL_STRUCTURE_REFERENCE_INDEX_v1.csv.gz"
)

EXPECTED_INDEX_SHA = (
    "b339e4db292cb709b808f634716a91fe"
    "705129e94f488fe92d04be568deafefa"
)

DESIGN_LOCK = Path(
    "step8/controls/wbm/"
    "WBM_INFERENCE_DESIGN_LOCK_v1.json"
)

EXPECTED_DESIGN_SHA = (
    "c8893d0a3200834993d2502c664a87c7"
    "dc2f2d50f71bd4c1a59f8791de967907"
)

CHECKPOINT = Path(
    "model_weights/sevennet/"
    "sevennet_l3i5.pth"
)

EXPECTED_CHECKPOINT_SHA = (
    "a7ff190d41efe5b8317a03d20a468e40"
    "36a1a3e64e0b14fdbab64ac3f4bfc09d"
)

OUTDIR = Path(
    "step8/controls/wbm/"
    "predictions/SevenNet-l3i5"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT = OUTDIR / (
    "energy_inference_audit_v1.json"
)

THREADS = 2

torch.set_num_threads(
    THREADS
)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
        )
    )

    os.replace(
        tmp,
        path,
    )


def to_atoms(struct_dict):
    structure = Structure.from_dict(
        struct_dict
    )

    return AseAtomsAdaptor.get_atoms(
        structure
    )


def valid_prediction(path):
    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            required = {
                "model_config_energy_eV",
                "dft_config_energy_eV",
                "n_atoms",
                "wbm_round",
                "sampling_weight",
            }

            if not required.issubset(
                z.files
            ):
                return False

            for key in required:

                if not np.isfinite(
                    np.asarray(
                        z[key]
                    )
                ).all():
                    return False

            n_atoms = int(
                np.asarray(
                    z["n_atoms"]
                ).item()
            )

            wbm_round = int(
                np.asarray(
                    z["wbm_round"]
                ).item()
            )

            if n_atoms <= 0:
                return False

            if wbm_round not in {
                1, 2, 3, 4, 5
            }:
                return False

            return True

    except Exception:
        return False


print("=" * 78)
print(
    "STEP-8 WBM ENERGY CONTROL — SevenNet-l3i5"
)
print("=" * 78)


# ==========================================================
# Frozen provenance
# ==========================================================

sources = {
    BUNDLE:
        EXPECTED_BUNDLE_SHA,

    INDEX:
        EXPECTED_INDEX_SHA,

    DESIGN_LOCK:
        EXPECTED_DESIGN_SHA,

    CHECKPOINT:
        EXPECTED_CHECKPOINT_SHA,
}


for path, expected in sources.items():

    assert path.exists(), path

    actual = sha256(
        path
    )

    assert actual == expected, (
        path,
        actual,
        expected,
    )


checkpoint_sha = sha256(
    CHECKPOINT
)

print(
    "Checkpoint SHA256:",
    checkpoint_sha,
)


# ==========================================================
# Load frozen WBM bundle
# ==========================================================

with gzip.open(
    BUNDLE,
    "rt",
) as f:

    bundle = json.load(
        f
    )


records = bundle[
    "records"
]

assert len(records) == 2500


record_lookup = {
    str(
        rec["material_id"]
    ):
        rec
    for rec in records
}

assert len(
    record_lookup
) == 2500


# ==========================================================
# Frozen reference index
# ==========================================================

with gzip.open(
    INDEX,
    "rt",
    newline="",
) as f:

    index_rows = list(
        csv.DictReader(
            f
        )
    )


assert len(
    index_rows
) == 2500


index_lookup = {
    str(
        row["material_id"]
    ):
        row
    for row in index_rows
}


assert set(
    index_lookup
) == set(
    record_lookup
)


print(
    "Frozen WBM rows:",
    len(records),
)

print(
    "Threads         :",
    THREADS,
)


# ==========================================================
# Exact verified SevenNet loader
# ==========================================================

print(
    "\nLoading SevenNet-l3i5..."
)


calc = SevenNetCalculator(
    model=str(
        CHECKPOINT
    ),
    device="cpu",
    enable_cueq=False,
    enable_flash=False,
    enable_oeq=False,
)


print(
    "SevenNet-l3i5 LOAD: PASS"
)


# ==========================================================
# Static relaxed-structure inference
# ==========================================================

success = 0
skipped = 0
failures = []


for i, rec in enumerate(
    records,
    1,
):

    mid = str(
        rec["material_id"]
    )

    idx = index_lookup[
        mid
    ]

    out = OUTDIR / (
        f"{mid}.npz"
    )


    # Resumable run
    if (
        out.exists()
        and valid_prediction(
            out
        )
    ):

        success += 1
        skipped += 1

    else:

        try:

            atoms = to_atoms(
                rec[
                    "relaxed_structure_opt"
                ]
            )

            n_atoms = len(
                atoms
            )

            expected_n_atoms = int(
                idx[
                    "n_sites"
                ]
            )

            if (
                n_atoms
                != expected_n_atoms
            ):
                raise ValueError(
                    "bundle/index atom-count mismatch: "
                    f"{n_atoms} vs "
                    f"{expected_n_atoms}"
                )


            # Static single-point only.
            # NO geometry relaxation.
            atoms.calc = calc

            E_model = float(
                atoms.get_potential_energy()
            )

            E_dft = float(
                idx[
                    "uncorrected_energy"
                ]
            )

            wbm_round = int(
                idx[
                    "wbm_round"
                ]
            )

            sampling_weight = float(
                idx[
                    "sampling_weight"
                ]
            )


            for name, value in [
                (
                    "model_energy",
                    E_model,
                ),
                (
                    "DFT_energy",
                    E_dft,
                ),
                (
                    "sampling_weight",
                    sampling_weight,
                ),
            ]:

                if not np.isfinite(
                    value
                ):
                    raise ValueError(
                        f"nonfinite {name}"
                    )


            if wbm_round not in {
                1, 2, 3, 4, 5
            }:
                raise ValueError(
                    "invalid WBM round: "
                    f"{wbm_round}"
                )


            with NamedTemporaryFile(
                suffix=".npz",
                dir=OUTDIR,
                delete=False,
            ) as tmp:

                tmp_path = Path(
                    tmp.name
                )


            try:

                np.savez_compressed(
                    tmp_path,

                    model_config_energy_eV=
                        np.asarray(
                            E_model,
                            dtype=float,
                        ),

                    dft_config_energy_eV=
                        np.asarray(
                            E_dft,
                            dtype=float,
                        ),

                    n_atoms=
                        np.asarray(
                            n_atoms,
                            dtype=int,
                        ),

                    wbm_round=
                        np.asarray(
                            wbm_round,
                            dtype=int,
                        ),

                    sampling_weight=
                        np.asarray(
                            sampling_weight,
                            dtype=float,
                        ),
                )

                os.replace(
                    tmp_path,
                    out,
                )

            finally:

                if tmp_path.exists():
                    tmp_path.unlink()


            if not valid_prediction(
                out
            ):
                raise RuntimeError(
                    "written NPZ failed validation"
                )


            success += 1


        except Exception as exc:

            failures.append({
                "material_id":
                    mid,

                "wbm_round":
                    int(
                        idx[
                            "wbm_round"
                        ]
                    ),

                "error":
                    repr(
                        exc
                    ),
            })


    # ======================================================
    # Progressive resumable audit
    # ======================================================

    if (
        i % 50 == 0
        or i == 2500
    ):

        progress = {
            "stage":
                "STEP8_WBM_SEVENNET_L3I5_ENERGY_INFERENCE",

            "model":
                "SevenNet-l3i5",

            "family":
                "SevenNet",

            "tier":
                "T1",

            "checkpoint":
                str(
                    CHECKPOINT
                ),

            "checkpoint_sha256":
                checkpoint_sha,

            "environment":
                ".sevennet-venv",

            "device":
                "cpu",

            "expected_configs":
                2500,

            "processed_rows":
                i,

            "successful_rows":
                success,

            "skipped_existing":
                skipped,

            "failures":
                failures,

            "complete":
                (
                    success == 2500
                    and
                    len(
                        failures
                    ) == 0
                ),

            "inference_geometry":
                "DFT-relaxed WBM opt",

            "model_relaxation":
                False,

            "forces_computed_for_analysis":
                False,

            "forces_stored":
                False,

            "DFT_reference_field":
                "uncorrected_energy",

            "d_eq_used":
                False,

            "sample_membership_changed":
                False,
        }


        atomic_json(
            AUDIT,
            progress,
        )


        print(
            f"{i:,}/2,500 "
            f"success={success:,} "
            f"skipped={skipped:,} "
            f"failures={len(failures):,}"
        )


# ==========================================================
# Final validation
# ==========================================================

available = sum(
    valid_prediction(
        OUTDIR / (
            f"{mid}.npz"
        )
    )
    for mid in record_lookup
)


round_available = {
    r: 0
    for r in range(
        1,
        6,
    )
}


for mid in record_lookup:

    path = OUTDIR / (
        f"{mid}.npz"
    )

    if not valid_prediction(
        path
    ):
        continue

    with np.load(
        path,
        allow_pickle=False,
    ) as z:

        r = int(
            np.asarray(
                z[
                    "wbm_round"
                ]
            ).item()
        )

    round_available[
        r
    ] += 1


status = (
    "PASS"
    if (
        available == 2500
        and
        len(
            failures
        ) == 0
        and
        round_available
        == {
            1: 500,
            2: 500,
            3: 500,
            4: 500,
            5: 500,
        }
    )
    else "REVISE"
)


final = {
    "stage":
        "STEP8_WBM_SEVENNET_L3I5_ENERGY_INFERENCE",

    "status":
        status,

    "model":
        "SevenNet-l3i5",

    "family":
        "SevenNet",

    "tier":
        "T1",

    "checkpoint":
        str(
            CHECKPOINT
        ),

    "checkpoint_sha256":
        checkpoint_sha,

    "environment":
        ".sevennet-venv",

    "device":
        "cpu",

    "frozen_sample_rows":
        2500,

    "available_prediction_files":
        int(
            available
        ),

    "available_by_WBM_round": {
        str(k):
            int(v)
        for k, v in
        round_available.items()
    },

    "successful_rows_this_run":
        success,

    "skipped_existing":
        skipped,

    "failures":
        failures,

    "inference_geometry":
        "DFT-relaxed WBM opt",

    "model_relaxation":
        False,

    "forces_computed_for_analysis":
        False,

    "forces_stored":
        False,

    "DFT_reference_field":
        "uncorrected_energy",

    "error_definition_for_later_analysis":
        (
            "(E_model_total_eV - "
            "E_DFT_uncorrected_eV) / "
            "n_atoms"
        ),

    "d_eq_used":
        False,

    "sample_membership_changed":
        False,

    "sample_replacements":
        0,

    "bundle_sha256":
        sha256(
            BUNDLE
        ),

    "index_sha256":
        sha256(
            INDEX
        ),

    "design_lock_sha256":
        sha256(
            DESIGN_LOCK
        ),
}


atomic_json(
    AUDIT,
    final,
)


print()
print("=" * 78)
print(
    "WBM SevenNet-l3i5 ENERGY INFERENCE AUDIT"
)
print("=" * 78)

print(
    "Expected :",
    2500,
)

print(
    "Available:",
    available,
)

print(
    "Failures :",
    len(
        failures
    ),
)

print(
    "By round :",
    round_available,
)

print(
    "Audit SHA256:",
    sha256(
        AUDIT
    ),
)

print(
    "\nSevenNet-l3i5 WBM ENERGY CONTROL:",
    status,
)
