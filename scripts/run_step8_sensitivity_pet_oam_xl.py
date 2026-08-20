#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import csv
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from ase import Atoms
from upet.calculator import UPETCalculator


BUNDLE = Path(
    "step8/data/primary_inference_bundle_v1.json.gz"
)

SUBSET = Path(
    "step8/preflight/PET_SENSITIVITY_SUBSET_v1.csv.gz"
)

CHECKPOINT = Path(
    "model_weights/pet/models/pet-oam-xl-v1.0.0.ckpt"
)

EXPECTED_SHA256 = (
    "c3a67cd019969dfd4dcabe9574682fe0"
    "35f861d3f1c10190989b36c983699409"
)

OUTDIR = Path(
    "step8/predictions/PET-OAM-XL"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

PARENT_CACHE = OUTDIR / "parent_energies.json"
AUDIT = OUTDIR / "inference_audit.json"

THREADS = 2

torch.set_num_threads(THREADS)

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


def ase_from_structure_dict(d):
    cell = np.asarray(
        d["lattice"]["matrix"],
        dtype=float,
    )

    symbols = []
    positions = []

    for site in d["sites"]:

        species = site["species"]

        elem = max(
            species,
            key=lambda z: z.get(
                "occu",
                1.0,
            ),
        )["element"]

        symbols.append(elem)

        if "xyz" in site:
            positions.append(
                site["xyz"]
            )
        else:
            abc = np.asarray(
                site["abc"],
                dtype=float,
            )
            positions.append(
                abc @ cell
            )

    return Atoms(
        symbols=symbols,
        positions=np.asarray(
            positions,
            dtype=float,
        ),
        cell=cell,
        pbc=True,
    )


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        "is not JSON serializable"
    )


def atomic_json(path, obj):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            default=json_default,
        )
    )

    os.replace(
        tmp,
        path,
    )


def valid_prediction(path):
    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            required = {
                "model_config_energy_eV",
                "model_parent_energy_eV",
                "model_forces_eV_per_A",
                "dft_config_energy_eV",
                "dft_parent_energy_eV",
                "dft_forces_eV_per_A",
                "n_atoms",
                "d_eq",
                "sampling_weight",
            }

            if not required.issubset(
                z.files
            ):
                return False

            return (
                np.isfinite(
                    z[
                        "model_config_energy_eV"
                    ]
                ).all()
                and
                np.isfinite(
                    z[
                        "model_parent_energy_eV"
                    ]
                ).all()
                and
                np.isfinite(
                    z[
                        "model_forces_eV_per_A"
                    ]
                ).all()
            )

    except Exception:
        return False


print("=" * 76)
print(
    "STEP-8 SENSITIVITY INFERENCE — "
    "PET-OAM-XL"
)
print("=" * 76)


# ---------------------------------------------------------
# Checkpoint provenance
# ---------------------------------------------------------

assert CHECKPOINT.exists(), CHECKPOINT

actual_sha = sha256(
    CHECKPOINT
)

print(
    "Checkpoint SHA256:",
    actual_sha,
)

assert actual_sha == EXPECTED_SHA256, (
    "PET checkpoint SHA256 mismatch"
)


# ---------------------------------------------------------
# Frozen primary bundle
# ---------------------------------------------------------

with gzip.open(
    BUNDLE,
    "rt",
) as f:
    bundle = json.load(f)

primary_manifest = (
    bundle["sample_manifest"]
)

configs = bundle["configs"]
parents = bundle["parents"]

primary_by_id = {
    str(r["matpes_id"]): r
    for r in primary_manifest
}


# ---------------------------------------------------------
# Frozen PET sensitivity subset
# ---------------------------------------------------------

with gzip.open(
    SUBSET,
    "rt",
    newline="",
) as f:
    subset = list(
        csv.DictReader(f)
    )

assert len(subset) == 300

subset_ids = [
    str(r["matpes_id"])
    for r in subset
]

assert len(
    set(subset_ids)
) == 300

missing_from_primary = [
    mid
    for mid in subset_ids
    if mid not in primary_by_id
]

assert not missing_from_primary, (
    missing_from_primary[:10]
)


# Cross-check subset metadata against frozen primary manifest.
for row in subset:

    mid = str(
        row["matpes_id"]
    )

    p = primary_by_id[mid]

    assert str(
        p["original_mp_id"]
    ) == str(
        row["original_mp_id"]
    )

    assert np.isclose(
        float(p["d_eq"]),
        float(row["d_eq"]),
        rtol=0,
        atol=1e-12,
    )

    assert np.isclose(
        float(p["sampling_weight"]),
        float(row["sampling_weight"]),
        rtol=0,
        atol=1e-12,
    )


required_parents = sorted({
    str(
        r["original_mp_id"]
    )
    for r in subset
})

assert len(
    required_parents
) == 294


print(
    "Configs    :",
    len(subset),
)

print(
    "Parents    :",
    len(required_parents),
)

print(
    "Threads    :",
    THREADS,
)

print(
    "Device     : cpu"
)


# ---------------------------------------------------------
# Verified PET loader
# ---------------------------------------------------------

print(
    "\nLoading PET-OAM-XL..."
)

calc = UPETCalculator(
    model="pet-oam-xl",
    checkpoint_path=str(
        CHECKPOINT
    ),
    device="cpu",
)

print(
    "PET-OAM-XL LOAD: PASS"
)


# ---------------------------------------------------------
# Parent-energy cache
# ---------------------------------------------------------

if PARENT_CACHE.exists():

    parent_energy = json.loads(
        PARENT_CACHE.read_text()
    )

else:

    parent_energy = {}


# Remove any irrelevant cached keys.
parent_energy = {
    str(k): float(v)
    for k, v in parent_energy.items()
    if str(k) in required_parents
}


print(
    "\nCached parent energies:",
    len(parent_energy),
    "/",
    len(required_parents),
)


for i, pid in enumerate(
    required_parents,
    1,
):

    if pid in parent_energy:
        continue

    rec = parents[pid]

    atoms = ase_from_structure_dict(
        rec["structure"]
    )

    atoms.calc = calc

    E_parent = float(
        atoms.get_potential_energy()
    )

    if not np.isfinite(
        E_parent
    ):
        raise RuntimeError(
            f"Nonfinite parent energy: {pid}"
        )

    parent_energy[pid] = (
        E_parent
    )

    if i % 10 == 0:
        atomic_json(
            PARENT_CACHE,
            parent_energy,
        )

    if i % 25 == 0:
        print(
            f"Parents {i:,}/"
            f"{len(required_parents):,}"
        )


atomic_json(
    PARENT_CACHE,
    parent_energy,
)

assert len(
    parent_energy
) == len(
    required_parents
)

print(
    "PARENT ENERGY CACHE: PASS"
)


# ---------------------------------------------------------
# Configuration inference
# ---------------------------------------------------------

success = 0
skipped = 0
failures = []


for i, subset_row in enumerate(
    subset,
    1,
):

    mid = str(
        subset_row["matpes_id"]
    )

    pid = str(
        subset_row[
            "original_mp_id"
        ]
    )

    manifest_row = (
        primary_by_id[mid]
    )

    out = (
        OUTDIR
        / f"{mid}.npz"
    )


    if (
        out.exists()
        and valid_prediction(out)
    ):

        success += 1
        skipped += 1

    else:

        try:

            rec = configs[mid]

            atoms = (
                ase_from_structure_dict(
                    rec["structure"]
                )
            )

            atoms.calc = calc


            E_model = float(
                atoms.get_potential_energy()
            )

            F_model = np.asarray(
                atoms.get_forces(),
                dtype=float,
            )


            E_parent_model = float(
                parent_energy[pid]
            )


            E_dft = float(
                rec["energy"]
            )

            E_parent_dft = float(
                parents[pid][
                    "energy"
                ]
            )

            F_dft = np.asarray(
                rec["forces"],
                dtype=float,
            )


            n_atoms = len(
                atoms
            )


            if not np.isfinite(
                E_model
            ):
                raise ValueError(
                    "nonfinite model energy"
                )


            if not np.isfinite(
                F_model
            ).all():
                raise ValueError(
                    "nonfinite model forces"
                )


            if (
                F_model.shape
                != F_dft.shape
            ):
                raise ValueError(
                    "force shape mismatch: "
                    f"{F_model.shape} "
                    f"vs {F_dft.shape}"
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
                        E_model,

                    model_parent_energy_eV=
                        E_parent_model,

                    model_forces_eV_per_A=
                        F_model,

                    dft_config_energy_eV=
                        E_dft,

                    dft_parent_energy_eV=
                        E_parent_dft,

                    dft_forces_eV_per_A=
                        F_dft,

                    n_atoms=
                        n_atoms,

                    d_eq=
                        float(
                            manifest_row[
                                "d_eq"
                            ]
                        ),

                    sampling_weight=
                        float(
                            manifest_row[
                                "sampling_weight"
                            ]
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
                "matpes_id":
                    mid,

                "original_mp_id":
                    pid,

                "deq_decile":
                    subset_row[
                        "deq_decile"
                    ],

                "error":
                    repr(exc),
            })


    if (
        i % 10 == 0
        or i == len(subset)
    ):
        print(
            f"{i:,}/{len(subset)} "
            f"success={success:,} "
            f"skipped={skipped:,} "
            f"failures={len(failures):,}"
        )


# ---------------------------------------------------------
# Final availability audit
# ---------------------------------------------------------

available = sum(
    valid_prediction(
        OUTDIR / f"{mid}.npz"
    )
    for mid in subset_ids
)


decile_available = {}

for d in range(
    1,
    11,
):

    ids = [
        str(r["matpes_id"])
        for r in subset
        if int(
            r["deq_decile"]
        ) == d
    ]

    decile_available[
        str(d)
    ] = {
        "expected":
            len(ids),

        "available":
            sum(
                valid_prediction(
                    OUTDIR
                    / f"{mid}.npz"
                )
                for mid in ids
            ),
    }


status = (
    "PASS"
    if (
        available == 300
        and len(failures) == 0
        and all(
            x["available"]
            == x["expected"]
            == 30
            for x
            in decile_available.values()
        )
    )
    else "REVISE"
)


audit = {
    "stage":
        "STEP8_PET_OAM_XL_SENSITIVITY_INFERENCE",

    "status":
        status,

    "model":
        "PET-OAM-XL",

    "runtime_model":
        "pet-oam-xl",

    "environment":
        ".pet-venv",

    "device":
        "cpu",

    "expected_configs":
        300,

    "available_configs":
        available,

    "expected_parents":
        294,

    "cached_parent_energies":
        len(parent_energy),

    "failures":
        failures,

    "decile_availability":
        decile_available,

    "checkpoint":
        str(CHECKPOINT),

    "checkpoint_sha256":
        actual_sha,

    "subset":
        str(SUBSET),

    "subset_sha256":
        sha256(SUBSET),

    "primary_bundle":
        str(BUNDLE),

    "primary_bundle_sha256":
        sha256(BUNDLE),

    "primary_sample_membership_changed":
        False,

    "hypothesis_decision_performed":
        False,
}


atomic_json(
    AUDIT,
    audit,
)


print(
    "\n" + "=" * 76
)

print(
    "PET-OAM-XL SENSITIVITY INFERENCE AUDIT"
)

print(
    "=" * 76
)

print(
    "Expected :",
    300,
)

print(
    "Available:",
    available,
)

print(
    "Parents  :",
    len(parent_energy),
)

print(
    "Failures :",
    len(failures),
)


if status == "PASS":

    print(
        "\nPET-OAM-XL STEP8 "
        "SENSITIVITY INFERENCE: PASS"
    )

else:

    print(
        "\nPET-OAM-XL STEP8 "
        "SENSITIVITY INFERENCE: REVISE"
    )
