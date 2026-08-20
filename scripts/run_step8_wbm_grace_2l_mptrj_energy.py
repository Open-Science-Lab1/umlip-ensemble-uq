#!/usr/bin/env python3

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault(
    "TF_USE_LEGACY_KERAS",
    "1",
)

import csv
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf
from ase import Atoms
from tensorpotential.calculator import grace_fm


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

MODEL_KEY = "GRACE-2L-MP-r6"

FROZEN_CACHE_MANIFEST_SHA256 = (
    "2e749888093c9e27f1a7075559b5dd0f"
    "e7e3fa4328b81d7cf0d2edc41c60164f"
)

CACHE = Path.home() / (
    ".cache/grace/GRACE-2L-MP-r6"
)

CACHE_FILES = {
    CACHE / "fingerprint.pb":
        (
            "cad0d4d98c99bfd5ed3cab7f89a6e8c7"
            "7ac07450ba7f33d84d348e42327453b7"
        ),

    CACHE / "variables/variables.index":
        (
            "fc27ac659508d63786043d6bc8500057"
            "c4e588197fc6bf18b60f4308cd7bc9ce"
        ),

    CACHE / (
        "variables/"
        "variables.data-00000-of-00001"
    ):
        (
            "7a7415f0a5df18a27784a65b68457923"
            "1d232c013d8b32ceb8e0a98bbfffb451"
        ),

    CACHE / "saved_model.pb":
        (
            "7150c2f19d6d046c7adc71eb452dbf42"
            "8b6bbd136696e406ae95d2a60836ecd9"
        ),

    CACHE / "metadata.yaml":
        (
            "4f5f4d3adf228da543fd58ea0a634412"
            "58526d3cf3d0c65e57fd844c1bd10336"
        ),
}

OUTDIR = Path(
    "step8/controls/wbm/"
    "predictions/GRACE-2L-MPtrj"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT = OUTDIR / (
    "energy_inference_audit_v1.json"
)

THREADS = 2


try:
    tf.config.threading.set_intra_op_parallelism_threads(
        THREADS
    )

    tf.config.threading.set_inter_op_parallelism_threads(
        1
    )

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


def structure_dict_to_atoms(sd):

    cell = np.asarray(
        sd["lattice"]["matrix"],
        dtype=float,
    )

    symbols = []
    scaled = []

    for site in sd["sites"]:

        species = site[
            "species"
        ]

        if len(species) != 1:
            raise ValueError(
                "disordered site not supported"
            )

        sp = species[0]

        occu = float(
            sp.get(
                "occu",
                1.0,
            )
        )

        if not np.isclose(
            occu,
            1.0,
        ):
            raise ValueError(
                "partial occupancy not supported"
            )

        symbol = (
            sp.get("element")
            or sp.get("name")
        )

        if symbol is None:
            raise ValueError(
                "site element missing"
            )

        symbols.append(
            str(symbol)
        )

        if "abc" in site:
            frac = site["abc"]

        elif "frac_coords" in site:
            frac = site[
                "frac_coords"
            ]

        else:
            raise ValueError(
                "fractional coordinates missing"
            )

        scaled.append(
            frac
        )


    atoms = Atoms(
        symbols=symbols,
        cell=cell,
        scaled_positions=np.asarray(
            scaled,
            dtype=float,
        ),
        pbc=True,
    )

    return atoms


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

            return (
                n_atoms > 0
                and
                wbm_round
                in {
                    1, 2, 3, 4, 5
                }
            )

    except Exception:
        return False


print("=" * 78)
print(
    "STEP-8 WBM ENERGY CONTROL — "
    "GRACE-2L-MPtrj"
)
print("=" * 78)


# ==========================================================
# Frozen provenance
# ==========================================================

for path, expected in {
    BUNDLE:
        EXPECTED_BUNDLE_SHA,

    INDEX:
        EXPECTED_INDEX_SHA,

    DESIGN_LOCK:
        EXPECTED_DESIGN_SHA,
}.items():

    assert path.exists(), path

    actual = sha256(
        path
    )

    assert actual == expected, (
        path,
        actual,
        expected,
    )


print(
    "TensorFlow :",
    tf.__version__,
)

print(
    "GPU        :",
    tf.config.list_physical_devices(
        "GPU"
    ),
)

print(
    "Threads    :",
    THREADS,
)

print(
    "Runtime key:",
    MODEL_KEY,
)


# ==========================================================
# Exact GRACE cache provenance
# ==========================================================

print(
    "\nVerifying frozen GRACE cache..."
)


for path, expected in (
    CACHE_FILES.items()
):

    assert path.exists(), path

    actual = sha256(
        path
    )

    assert actual == expected, (
        path,
        actual,
        expected,
    )


print(
    "GRACE CACHE PROVENANCE: PASS"
)


# ==========================================================
# Frozen WBM bundle
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

assert len(
    records
) == 2500


record_lookup = {
    str(
        rec[
            "material_id"
        ]
    ):
        rec
    for rec in records
}

assert len(
    record_lookup
) == 2500


# ==========================================================
# Frozen WBM index
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
        row[
            "material_id"
        ]
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


# ==========================================================
# Exact verified GRACE loader
# ==========================================================

print(
    "\nLoading GRACE-2L-MPtrj..."
)


calc = grace_fm(
    MODEL_KEY,
    float_dtype="float64",
)


print(
    "GRACE-2L-MPtrj LOAD: PASS"
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
        rec[
            "material_id"
        ]
    )

    idx = index_lookup[
        mid
    ]

    out = OUTDIR / (
        f"{mid}.npz"
    )


    if (
        out.exists()
        and
        valid_prediction(
            out
        )
    ):

        success += 1
        skipped += 1

    else:

        try:

            atoms = structure_dict_to_atoms(
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
                    "bundle/index atom-count "
                    "mismatch: "
                    f"{n_atoms} vs "
                    f"{expected_n_atoms}"
                )


            # Static single point only.
            # No GRACE geometry relaxation.
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
                    f"invalid WBM round "
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
    # Resumable progress audit
    # ======================================================

    if (
        i % 50 == 0
        or i == 2500
    ):

        progress = {
            "stage":
                "STEP8_WBM_GRACE_2L_MPTRJ_ENERGY_INFERENCE",

            "model":
                "GRACE-2L-MPtrj",

            "runtime_model":
                MODEL_KEY,

            "family":
                "GRACE",

            "tier":
                "T1",

            "environment":
                ".grace-venv",

            "tensorflow":
                tf.__version__,

            "float_dtype":
                "float64",

            "device":
                "cpu",

            "frozen_cache_manifest_sha256":
                FROZEN_CACHE_MANIFEST_SHA256,

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
            f"failures="
            f"{len(failures):,}"
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
        "STEP8_WBM_GRACE_2L_MPTRJ_ENERGY_INFERENCE",

    "status":
        status,

    "model":
        "GRACE-2L-MPtrj",

    "runtime_model":
        MODEL_KEY,

    "family":
        "GRACE",

    "tier":
        "T1",

    "environment":
        ".grace-venv",

    "tensorflow":
        tf.__version__,

    "float_dtype":
        "float64",

    "device":
        "cpu",

    "frozen_cache_manifest_sha256":
        FROZEN_CACHE_MANIFEST_SHA256,

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

    "cache_file_sha256": {
        str(path):
            sha256(
                path
            )
        for path in CACHE_FILES
    },
}


atomic_json(
    AUDIT,
    final,
)


print()
print("=" * 78)
print(
    "WBM GRACE-2L-MPtrj ENERGY INFERENCE AUDIT"
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
    "\nGRACE-2L-MPtrj WBM ENERGY CONTROL:",
    status,
)
