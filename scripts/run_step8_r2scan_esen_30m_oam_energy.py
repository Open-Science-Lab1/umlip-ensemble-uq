#!/usr/bin/env python3

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
from fairchem.core import OCPCalculator


BUNDLE = Path(
    "step8/controls/r2scan/"
    "R2SCAN_CONTROL_INFERENCE_BUNDLE_v1.json.gz"
)

CHECKPOINT = Path(
    "model_weights/esen/esen_30m_oam.pt"
)

EXPECTED_SHA256 = (
    "adf7d38e5bccb8e0334434c0bd65ac75"
    "661fb646891df17ecc89c19d111efde1"
)

OUTDIR = Path(
    "step8/controls/r2scan/"
    "predictions/eSEN-30M-OAM"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

PARENT_CACHE = OUTDIR / "parent_energies.json"
AUDIT = OUTDIR / "energy_inference_audit_v1.json"

SEED = 20260812
THREADS = 2

np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def to_atoms(rec):
    structure = Structure.from_dict(
        rec["structure"]
    )

    return AseAtomsAdaptor.get_atoms(
        structure
    )


def atomic_json(path, obj):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
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


def valid_prediction(path):
    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as z:

            required = {
                "model_config_energy_eV",
                "model_parent_energy_eV",
                "dft_config_energy_eV",
                "dft_parent_energy_eV",
                "n_atoms",
                "d_eq",
                "sampling_weight",
            }

            if not required.issubset(
                z.files
            ):
                return False

            for key in required:
                if not np.isfinite(
                    np.asarray(z[key])
                ).all():
                    return False

            return True

    except Exception:
        return False


print("=" * 78)
print(
    "STEP-8 r2SCAN ENERGY CONTROL — "
    "eSEN-30M-OAM"
)
print("=" * 78)


# =========================================================
# Checkpoint provenance
# =========================================================

assert CHECKPOINT.exists(), CHECKPOINT

actual_sha = sha256(
    CHECKPOINT
)

print(
    "Checkpoint SHA256:",
    actual_sha,
)

assert actual_sha == EXPECTED_SHA256, (
    "eSEN checkpoint hash mismatch"
)


# =========================================================
# Frozen control bundle
# =========================================================

with gzip.open(
    BUNDLE,
    "rt",
) as f:
    bundle = json.load(f)

manifest = bundle["sample_manifest"]
configs = bundle["configs"]
parents = bundle["parents"]

assert len(manifest) == 3000
assert len(configs) == 3000
assert len(parents) == 3000

print("Configs    :", len(manifest))
print("Parents    :", len(parents))
print("Checkpoint :", CHECKPOINT)
print("Threads    :", THREADS)
print("Seed       :", SEED)


# =========================================================
# Exact verified loader
# =========================================================

print(
    "\nLoading eSEN-30M-OAM..."
)

calc = OCPCalculator(
    checkpoint_path=str(CHECKPOINT),
    cpu=True,
)

print(
    "eSEN-30M-OAM LOAD: PASS"
)


# =========================================================
# Parent-energy cache
# =========================================================

required_parents = sorted({
    str(row["original_mp_id"])
    for row in manifest
})

assert len(
    required_parents
) == 3000


if PARENT_CACHE.exists():

    parent_energy = json.loads(
        PARENT_CACHE.read_text()
    )

else:

    parent_energy = {}


parent_energy = {
    str(k): float(v)
    for k, v
    in parent_energy.items()
    if str(k)
    in required_parents
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

    atoms = to_atoms(
        parents[pid]
    )

    atoms.calc = calc

    energy = float(
        atoms.get_potential_energy()
    )

    if not np.isfinite(
        energy
    ):
        raise RuntimeError(
            f"Nonfinite parent energy: {pid}"
        )

    parent_energy[
        pid
    ] = energy

    if i % 25 == 0:
        atomic_json(
            PARENT_CACHE,
            parent_energy,
        )

    if i % 100 == 0:
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
) == 3000

print(
    "PARENT ENERGY CACHE: PASS"
)


# =========================================================
# Configuration inference
# =========================================================

success = 0
skipped = 0
failures = []


for i, row in enumerate(
    manifest,
    1,
):

    mid = str(
        row["matpes_id"]
    )

    pid = str(
        row["original_mp_id"]
    )

    out = (
        OUTDIR
        / f"{mid}.npz"
    )


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

            rec = configs[
                mid
            ]

            atoms = to_atoms(
                rec
            )

            parent_atoms = to_atoms(
                parents[pid]
            )


            if len(atoms) != len(
                parent_atoms
            ):
                raise ValueError(
                    "config-parent atom-count mismatch"
                )


            if (
                atoms.get_chemical_symbols()
                !=
                parent_atoms.get_chemical_symbols()
            ):
                raise ValueError(
                    "config-parent species/order mismatch"
                )


            atoms.calc = calc

            E_model = float(
                atoms.get_potential_energy()
            )

            E_parent_model = float(
                parent_energy[
                    pid
                ]
            )

            E_ref = float(
                rec["energy"]
            )

            E_parent_ref = float(
                parents[
                    pid
                ]["energy"]
            )

            n_atoms = len(
                atoms
            )


            for name, value in [
                (
                    "model_config",
                    E_model,
                ),
                (
                    "model_parent",
                    E_parent_model,
                ),
                (
                    "r2scan_config",
                    E_ref,
                ),
                (
                    "r2scan_parent",
                    E_parent_ref,
                ),
            ]:

                if not np.isfinite(
                    value
                ):
                    raise ValueError(
                        f"nonfinite energy: {name}"
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
                            E_model
                        ),

                    model_parent_energy_eV=
                        np.asarray(
                            E_parent_model
                        ),

                    dft_config_energy_eV=
                        np.asarray(
                            E_ref
                        ),

                    dft_parent_energy_eV=
                        np.asarray(
                            E_parent_ref
                        ),

                    n_atoms=
                        np.asarray(
                            n_atoms
                        ),

                    d_eq=
                        np.asarray(
                            float(
                                row["d_eq"]
                            )
                        ),

                    sampling_weight=
                        np.asarray(
                            float(
                                row[
                                    "sampling_weight"
                                ]
                            )
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

                "error":
                    repr(exc),
            })


    if (
        i % 50 == 0
        or i == 3000
    ):

        print(
            f"{i:,}/3000 "
            f"success={success:,} "
            f"skipped={skipped:,} "
            f"failures={len(failures):,}"
        )


# =========================================================
# Final integrity audit
# =========================================================

available = sum(
    valid_prediction(
        OUTDIR
        / f"{row['matpes_id']}.npz"
    )
    for row in manifest
)


status = (
    "PASS"
    if (
        available == 3000
        and len(failures) == 0
    )
    else "REVISE"
)


audit = {
    "stage":
        "STEP8_R2SCAN_ESEN_30M_OAM_ENERGY_INFERENCE",

    "status":
        status,

    "model":
        "eSEN-30M-OAM",

    "runtime_model":
        "esen_30m_oam.pt",

    "family":
        "eSEN",

    "tier":
        "T3",

    "environment":
        ".esen-venv",

    "checkpoint":
        str(CHECKPOINT),

    "checkpoint_sha256":
        actual_sha,

    "seed":
        SEED,

    "cpu":
        True,

    "cpu_bitwise_determinism_claimed":
        False,

    "scope":
        "r2SCAN_secondary_energy_only_control",

    "expected_configs":
        3000,

    "available_prediction_files":
        int(available),

    "successful_rows":
        success,

    "skipped_existing":
        skipped,

    "failures":
        failures,

    "forces_computed_for_analysis":
        False,

    "forces_stored":
        False,

    "sample_membership_changed":
        False,

    "replacement_configs_used":
        False,

    "primary_PBE_results_changed":
        False,

    "bundle":
        str(BUNDLE),

    "bundle_sha256":
        sha256(BUNDLE),
}


atomic_json(
    AUDIT,
    audit,
)


print("\n" + "=" * 78)
print(
    "r2SCAN eSEN-30M-OAM "
    "ENERGY INFERENCE AUDIT"
)
print("=" * 78)

print(
    "Expected :",
    3000,
)

print(
    "Available:",
    available,
)

print(
    "Failures :",
    len(failures),
)

print(
    "\nAudit:",
    AUDIT,
)

print(
    "Audit SHA256:",
    sha256(AUDIT),
)

print(
    "\neSEN-30M-OAM "
    "r2SCAN ENERGY CONTROL:",
    status,
)
