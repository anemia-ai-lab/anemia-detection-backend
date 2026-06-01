"""Split train/val por paciente (sin filtración entre crops)."""

from __future__ import annotations

import numpy as np

from ml.baseline.dataops import (
    patient_id_from_crop_path,
    stratified_train_val_paths,
    stratified_train_val_paths_by_patient,
)


def test_patient_id_from_crop_stem() -> None:
    assert patient_id_from_crop_path("data/positive/P001_0.png") == "P001"
    assert patient_id_from_crop_path("data/negative/xyz.png") == "xyz"


def test_patient_split_keeps_same_patient_in_one_fold() -> None:
    paths = np.array(
        [
            "a/P1_0.png",
            "a/P1_1.png",
            "a/P2_0.png",
            "a/P2_1.png",
        ],
        dtype=object,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    tr_p, tr_l, va_p, va_l = stratified_train_val_paths_by_patient(
        paths,
        labels,
        validation_split=0.5,
        seed=42,
    )
    crop_split = stratified_train_val_paths(
        paths,
        labels,
        validation_split=0.5,
        seed=42,
    )
    # Con 2 pacientes por clase y 50% val, cada fold tiene 1 paciente (2 crops).
    assert len(tr_p) == 2
    assert len(va_p) == 2
    pids_train = {patient_id_from_crop_path(p) for p in tr_p}
    pids_val = {patient_id_from_crop_path(p) for p in va_p}
    assert pids_train.isdisjoint(pids_val)
    # El split por crop puede mezclar crops del mismo paciente (filtración).
    tr_crop, _, va_crop, _ = crop_split
    pids_tr_crop = {patient_id_from_crop_path(p) for p in tr_crop}
    pids_va_crop = {patient_id_from_crop_path(p) for p in va_crop}
    assert len(pids_tr_crop & pids_va_crop) >= 1
