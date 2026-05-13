import json
import os
import csv
import random
import argparse
from collections import defaultdict

def split_dataset(
    input_path: str,
    output_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split the RadGraph preferences dataset into train, val, and test splits.

    Each item in the dataset contains a list of image_paths (an image pair).
    All images belonging to the same item are guaranteed to land in the same split.

    Args:
        input_path:   Path to the input JSON file (radgraph preferences dataset).
        output_path:  Path to write the output CSV (columns: dicom_id, split).
        train_ratio:  Fraction of items assigned to train  (default 0.8).
        val_ratio:    Fraction of items assigned to val    (default 0.1).
        test_ratio:   Fraction of items assigned to test   (default 0.1).
        seed:         Random seed for reproducibility      (default 42).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio + test_ratio must sum to 1.0"
    )

    # ── Load dataset ────────────────────────────────────────────────────────────
    with open(input_path, "r") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} items from {input_path}")

    # ── Shuffle items (keep image pairs together) ────────────────────────────────
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    # test gets whatever is left so the counts always sum exactly to n
    n_test  = n - n_train - n_val

    split_map = {}
    for i, idx in enumerate(indices):
        if i < n_train:
            split_map[idx] = "train"
        elif i < n_train + n_val:
            split_map[idx] = "validate"
        else:
            split_map[idx] = "test"

    print(f"Split sizes  →  train: {n_train}  |  val: {n_val}  |  test: {n_test}")

    # ── Build (dicom_id, split) rows ─────────────────────────────────────────────
    rows = []
    seen_dicom_ids = defaultdict(set)  # dicom_id → set of splits (sanity check)

    for idx, item in enumerate(dataset):
        image_paths = item.get("image_paths", [])
        if not image_paths:
            print(f"  WARNING: item {idx} has no image_paths – skipping.")
            continue

        split = split_map[idx]

        for path in image_paths:
            dicom_id = os.path.splitext(os.path.basename(path))[0]
            rows.append({"dicom_id": dicom_id, "split": split})
            seen_dicom_ids[dicom_id].add(split)

    # ── Sanity-check: no dicom_id should appear in more than one split ───────────
    conflicts = {did: splits for did, splits in seen_dicom_ids.items() if len(splits) > 1}
    if conflicts:
        print(
            f"\n  WARNING: {len(conflicts)} dicom_id(s) appear in multiple splits "
            f"(same image referenced by items in different splits):"
        )
        for did, splits in list(conflicts.items())[:5]:
            print(f"    {did}: {splits}")
    else:
        print("Sanity check passed: every dicom_id appears in exactly one split.")

    # ── Write CSV ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dicom_id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {output_path}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    split_counts = defaultdict(int)
    for row in rows:
        split_counts[row["split"]] += 1
    print("\nImage-level counts per split:")
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {split_counts[s]}")


if __name__ == "__main__":
    json_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model16-p18reports-1e-2lambda-with-study-ids.json"
    output_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/split_model16-p18reports-1e-2lambda-with-study-ids.csv"
        
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    seed = 42

    split_dataset(
        input_path=json_path,
        output_path=output_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )