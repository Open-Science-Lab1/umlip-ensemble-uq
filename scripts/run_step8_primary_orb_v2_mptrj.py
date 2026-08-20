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

from orb_models.forcefield import pretrained
from orb_models.forcefield.inference.calculator import ORBCalculator


BUNDLE = Path(
    "step8/data/primary_inference_bundle_v1.json.gz"
)

KEY = "orb-mptraj-only-v2"

CHECKPOINT = Path(
    "model_weights/orb/orb_v2_mptrj.ckpt"
)

EXPECTED_SHA256 = (
    "9e6722a31b0c274f7fe3bc37e06f7899"
    "1dcb29e07001e76184c61da0f32c43b9"
)

OUTDIR = Path(
    "step8/predictions/ORB-v2-MPtrj"
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
                    z["model_config_energy_eV"]
                ).all()
                and
                np.isfinite(
                    z["model_parent_energy_eV"]
                ).all()
                and
                np.isfinite(
                    z["model_forces_eV_per_A"]
                ).all()
            )

    except Exception:
        return False


print("=" * 76)
print(
    "STEP-8 PRIMARY INFERENCE — "
    "ORB-v2-MPtrj"
)
print("=" * 76)

assert CHECKPOINT.exists(), CHECKPOINT

actual_sha = sha256(CHECKPOINT)

print("Checkpoint SHA256:", actual_sha)

assert actual_sha == EXPECTED_SHA256, (
    "ORB checkpoint hash mismatch"
)

with gzip.open(
    BUNDLE,
    "rt",
) as f:
    bundle = json.load(f)

manifest = bundle["sample_manifest"]
configs = bundle["configs"]
parents = bundle["parents"]

assert len(manifest) == 3000

print("Configs    :", len(manifest))
print("Parents    :", len(parents))
print("Model key  :", KEY)
print("Checkpoint :", CHECKPOINT)
print("Threads    :", THREADS)


# =========================================================
# Exact loader already verified in real TiO test
# =========================================================

print("\nLoading ORB-v2-MPtrj...")

loader = pretrained.ORB_PRETRAINED_MODELS[
    KEY
]

orbff, atoms_adapter = loader(
    weights_path=str(CHECKPOINT),
    device="cpu",
)

calc = ORBCalculator(
    orbff,
    atoms_adapter=atoms_adapter,
    device="cpu",
)

print("ORB-v2-MPtrj LOAD: PASS")


# =========================================================
# Parent-energy cache
# =========================================================

if PARENT_CACHE.exists():
    parent_energy = json.loads(
        PARENT_CACHE.read_text()
    )
else:
    parent_energy = {}

required_parents = sorted({
    str(row["original_mp_id"])
    for row in manifest
})

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

    if not np.isfinite(energy):
        raise RuntimeError(
            f"Nonfinite parent energy: {pid}"
        )

    parent_energy[pid] = energy

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
# Configuration inference
# =========================================================

success = 0
skipped = 0
failures = []

for i, row in enumerate(
    manifest,
    1,
):

    mid = str(row["matpes_id"])
    pid = str(row["original_mp_id"])

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
                parents[pid]["energy"]
            )

            F_dft = np.asarray(
                rec["forces"],
                dtype=float,
            )

            n_atoms = len(atoms)

            if not np.isfinite(E_model):
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
                        np.asarray(
                            E_model
                        ),

                    model_parent_energy_eV=
                        np.asarray(
                            E_parent_model
                        ),

                    model_forces_eV_per_A=
                        F_model,

                    dft_config_energy_eV=
                        np.asarray(
                            E_dft
                        ),

                    dft_parent_energy_eV=
                        np.asarray(
                            E_parent_dft
                        ),

                    dft_forces_eV_per_A=
                        F_dft,

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

            success += 1

        except Exception as exc:

            failures.append({
                "matpes_id": mid,
                "original_mp_id": pid,
                "error": repr(exc),
            })

    if (
        i % 50 == 0
        or i == 3000
    ):

        audit = {
            "model":
                "ORB-v2-MPtrj",

            "model_key":
                KEY,

            "checkpoint":
                str(CHECKPOINT),

            "checkpoint_sha256":
                actual_sha,

            "expected_configs":
                3000,

            "processed_manifest_rows":
                i,

            "successful_configs":
                success,

            "skipped_existing":
                skipped,

            "failures":
                failures,

            "complete": (
                success == 3000
                and len(failures) == 0
            ),
        }

        atomic_json(
            AUDIT,
            audit,
        )

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

final = {
    "model": "ORB-v2-MPtrj",
    "model_key": KEY,
    "checkpoint": str(CHECKPOINT),
    "checkpoint_sha256": actual_sha,
    "expected_configs": 3000,
    "available_prediction_files":
        int(available),
    "failures": failures,
    "complete": (
        available == 3000
        and len(failures) == 0
    ),
}

atomic_json(
    AUDIT,
    final,
)

print("\n" + "=" * 76)
print(
    "ORB-v2-MPtrj PRIMARY "
    "INFERENCE AUDIT"
)
print("=" * 76)

print("Expected :", 3000)
print("Available:", available)
print("Failures :", len(failures))

if final["complete"]:

    print(
        "\nORB-v2-MPtrj STEP8 "
        "PRIMARY INFERENCE: PASS"
    )

else:

    print(
        "\nORB-v2-MPtrj STEP8 "
        "PRIMARY INFERENCE: REVISE"
    )
