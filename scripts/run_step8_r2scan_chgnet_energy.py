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
from chgnet.model.dynamics import CHGNetCalculator


BUNDLE = Path(
    "step8/controls/r2scan/"
    "R2SCAN_CONTROL_INFERENCE_BUNDLE_v1.json.gz"
)

MASK = Path(
    "step8/controls/r2scan/predictions/CHGNet/"
    "TECHNICAL_VALIDITY_MASK_v1.csv.gz"
)

OUTDIR = Path(
    "step8/controls/r2scan/predictions/CHGNet"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

PARENT_CACHE = OUTDIR / "parent_energies.json"
AUDIT = OUTDIR / "energy_inference_audit_v1.json"

CHECKPOINT = Path(
    ".pilot-venv/lib/python3.12/site-packages/"
    "chgnet/pretrained/0.3.0/"
    "chgnet_0.3.0_e29f68s314m37.pth.tar"
)

EXPECTED_CHECKPOINT_SHA = (
    "d14ab7c0f093efe64b60a7bcd540bca1"
    "0e74fb7f46c86108a079af60524659d1"
)

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


def to_atoms(rec):
    s = Structure.from_dict(
        rec["structure"]
    )
    return AseAtomsAdaptor.get_atoms(s)


def atomic_json(path, obj):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )
    tmp.write_text(
        json.dumps(obj, indent=2)
    )
    os.replace(tmp, path)


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

            for key in [
                "model_config_energy_eV",
                "model_parent_energy_eV",
                "dft_config_energy_eV",
                "dft_parent_energy_eV",
                "n_atoms",
                "d_eq",
                "sampling_weight",
            ]:
                if not np.isfinite(
                    np.asarray(z[key])
                ).all():
                    return False

            return True

    except Exception:
        return False


print("=" * 78)
print("STEP-8 r2SCAN ENERGY CONTROL — CHGNet")
print("=" * 78)


# =========================================================
# Provenance
# =========================================================

assert CHECKPOINT.exists(), CHECKPOINT

checkpoint_sha = sha256(
    CHECKPOINT
)

print(
    "Checkpoint SHA256:",
    checkpoint_sha,
)

assert (
    checkpoint_sha
    == EXPECTED_CHECKPOINT_SHA
)


# =========================================================
# Frozen bundle
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


# =========================================================
# Frozen technical mask
# =========================================================

with gzip.open(
    MASK,
    "rt",
    newline="",
) as f:
    mask_rows = list(
        csv.DictReader(f)
    )

assert len(mask_rows) == 3000

technical_valid = {
    str(r["matpes_id"]):
        str(r["technical_valid"])
        .strip().lower()
        in {"true", "1", "yes"}
    for r in mask_rows
}

assert set(technical_valid) == {
    str(r["matpes_id"])
    for r in manifest
}

valid_manifest = [
    r
    for r in manifest
    if technical_valid[
        str(r["matpes_id"])
    ]
]

excluded_manifest = [
    r
    for r in manifest
    if not technical_valid[
        str(r["matpes_id"])
    ]
]

assert len(valid_manifest) == 2996
assert len(excluded_manifest) == 4

print("Frozen sample       :", 3000)
print("Technical valid     :", len(valid_manifest))
print("Technical excluded  :", len(excluded_manifest))
print("Threads             :", THREADS)


# =========================================================
# Load verified model
# =========================================================

print("\nLoading CHGNet 0.3.0...")

calc = CHGNetCalculator(
    use_device="cpu"
)

print("CHGNet LOAD: PASS")


# =========================================================
# Required parents for valid rows only
# =========================================================

required_parents = sorted({
    str(r["original_mp_id"])
    for r in valid_manifest
})

print(
    "\nRequired valid parents:",
    len(required_parents),
)


# =========================================================
# Parent cache
# =========================================================

if PARENT_CACHE.exists():
    parent_energy = json.loads(
        PARENT_CACHE.read_text()
    )
else:
    parent_energy = {}


# Keep only currently required parents.
parent_energy = {
    str(k): float(v)
    for k, v in parent_energy.items()
    if str(k) in required_parents
}


print(
    "Cached parent energies:",
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

    E = float(
        atoms.get_potential_energy()
    )

    if not np.isfinite(E):
        raise RuntimeError(
            f"Nonfinite parent energy: {pid}"
        )

    parent_energy[pid] = E

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

assert len(parent_energy) == len(
    required_parents
)

print("PARENT ENERGY CACHE: PASS")


# =========================================================
# Config inference
# =========================================================

success = 0
skipped = 0
failures = []


for i, row in enumerate(
    valid_manifest,
    1,
):

    mid = str(
        row["matpes_id"]
    )

    pid = str(
        row["original_mp_id"]
    )

    out = OUTDIR / f"{mid}.npz"


    if (
        out.exists()
        and valid_prediction(out)
    ):
        success += 1
        skipped += 1

    else:

        try:

            rec = configs[mid]

            atoms = to_atoms(rec)

            # Composition/site-count consistency
            # for displacement-energy definition.
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
                parent_energy[pid]
            )

            E_ref = float(
                rec["energy"]
            )

            E_parent_ref = float(
                parents[pid]["energy"]
            )

            n_atoms = len(atoms)

            if not np.isfinite(
                E_model
            ):
                raise ValueError(
                    "nonfinite model config energy"
                )

            if not np.isfinite(
                E_parent_model
            ):
                raise ValueError(
                    "nonfinite model parent energy"
                )

            if not np.isfinite(
                E_ref
            ):
                raise ValueError(
                    "nonfinite r2SCAN config energy"
                )

            if not np.isfinite(
                E_parent_ref
            ):
                raise ValueError(
                    "nonfinite r2SCAN parent energy"
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
                        np.asarray(E_model),

                    model_parent_energy_eV=
                        np.asarray(
                            E_parent_model
                        ),

                    dft_config_energy_eV=
                        np.asarray(E_ref),

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
                    "written prediction failed validation"
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
        or i == len(valid_manifest)
    ):

        print(
            f"{i:,}/{len(valid_manifest):,} "
            f"success={success:,} "
            f"skipped={skipped:,} "
            f"failures={len(failures):,}"
        )


# =========================================================
# Final availability
# =========================================================

available = sum(
    valid_prediction(
        OUTDIR
        / f"{r['matpes_id']}.npz"
    )
    for r in valid_manifest
)

excluded_files_present = sum(
    (
        OUTDIR
        / f"{r['matpes_id']}.npz"
    ).exists()
    for r in excluded_manifest
)


status = (
    "PASS"
    if (
        available == 2996
        and len(failures) == 0
        and excluded_files_present == 0
    )
    else "REVISE"
)


audit = {
    "stage":
        "STEP8_R2SCAN_CHGNET_ENERGY_INFERENCE",

    "status":
        status,

    "model":
        "CHGNet",

    "runtime_model":
        "CHGNet-0.3.0",

    "checkpoint_sha256":
        checkpoint_sha,

    "scope":
        "r2SCAN_secondary_energy_only_control",

    "frozen_sample_rows":
        3000,

    "technical_valid_rows":
        2996,

    "technical_excluded_rows":
        4,

    "available_prediction_files":
        int(available),

    "excluded_prediction_files_present":
        int(
            excluded_files_present
        ),

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

    "technical_exclusions_defined_before_prediction":
        True,

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

    "technical_mask":
        str(MASK),

    "technical_mask_sha256":
        sha256(MASK),
}


atomic_json(
    AUDIT,
    audit,
)


print("\n" + "=" * 78)
print("r2SCAN CHGNet ENERGY INFERENCE AUDIT")
print("=" * 78)

print(
    "Frozen sample       :",
    3000,
)

print(
    "Technical valid     :",
    2996,
)

print(
    "Available           :",
    available,
)

print(
    "Technical excluded  :",
    4,
)

print(
    "Excluded files made :",
    excluded_files_present,
)

print(
    "Failures            :",
    len(failures),
)

print("\nAudit:", AUDIT)
print("Audit SHA256:", sha256(AUDIT))

print(
    "\nCHGNet r2SCAN ENERGY CONTROL:",
    status,
)
