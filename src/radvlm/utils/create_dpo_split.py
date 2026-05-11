import json
import os
import random
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_JSON   = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model16-p18reports-1e-2lambda-with-study-ids.json"
OUTPUT_JSON  = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/split_model16-p18reports-1e-2lambda-with-study-ids.json"
TRAIN_RATIO  = 0.7
TEST_RATIO   = 0.1
VAL_RATIO    = 0.2
RANDOM_SEED  = 42

def extract_study_id(filename: str) -> str:
    """
    Derive a study ID from a filename.

    Assumption: the study ID is every part of the stem *except* the last
    underscore-separated token.
      e.g.  "STUDY001_slice03.png"  →  "STUDY001"
            "patient42_scan2_0005.dcm" →  "patient42_scan2"

    Adjust this function to match your actual naming convention.
    """
    stem = os.path.splitext(filename)[0]          # strip extension
    parts = stem.rsplit("_", maxsplit=1)           # split off last segment
    return parts[0] if len(parts) > 1 else stem   # fallback: whole stem


def assign_splits(image_paths: list[str]) -> dict[str, str]:
    """
    Returns a dict mapping each image basename → split label
    ('train' | 'test' | 'validate'), keeping every image of the
    same study in the same split.
    """
    # 1. Group basenames by study
    study_to_files: dict[str, list[str]] = defaultdict(list)
    for path in image_paths:
        basename = os.path.basename(path).replace(".png", "")
        study_id = path.split("/")[-2]
        study_to_files[study_id].append(basename)

    studies = list(study_to_files.keys())

    # 2. Shuffle studies (not individual images) so the split is study-level
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(studies)

    # 3. Compute cut-points on the study list
    n = len(studies)
    n_train = round(n * TRAIN_RATIO)
    n_test  = round(n * TEST_RATIO)
    # validate gets whatever remains, preserving the exact total

    train_studies    = set(studies[:n_train])
    test_studies     = set(studies[n_train : n_train + n_test])
    validate_studies = set(studies[n_train + n_test :])

    print(f"Studies  → train: {len(train_studies)}, "
          f"test: {len(test_studies)}, "
          f"validate: {len(validate_studies)}")

    # 4. Build the per-image assignment
    assignments: dict[str, str] = {}
    for study_id, files in study_to_files.items():
        if study_id in train_studies:
            label = "train"
        elif study_id in test_studies:
            label = "test"
        else:
            label = "validate"
        for f in files:
            assignments[f] = label

    return assignments


def main() -> None:
    # Load input
    with open(INPUT_JSON, "r") as fh:
        data = json.load(fh)

    image_paths: list[str] = data["image_paths"]
    print(f"Loaded {len(image_paths)} image paths from '{INPUT_JSON}'.")

    # Compute splits
    assignments = assign_splits(image_paths)

    # Summary
    counts = {"train": 0, "test": 0, "validate": 0}
    for label in assignments.values():
        counts[label] += 1
    total = len(assignments)
    print(f"Images   → train: {counts['train']} ({counts['train']/total:.1%}), "
          f"test: {counts['test']} ({counts['test']/total:.1%}), "
          f"validate: {counts['validate']} ({counts['validate']/total:.1%})")

    # Save output
    output = {"splits": assignments}
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Assignments written to '{OUTPUT_JSON}'.")


if __name__ == "__main__":
    main()