"""
Recompute rewards from already-computed METEOR and RadGraph scores stored in the
preference JSON, using a different lambda — no model inference required.
 
Usage:
    python recompute_rewards.py \
        --input  path/to/original_preferences.json \
        --output path/to/recomputed_preferences.json \
        --lam    0.5
"""
 
import argparse
import json
from pathlib import Path
 
 
def recompute_item(item: dict, lam: float) -> dict:
    m1  = item["meteor_report_1"]
    m2  = item["meteor_report_2"]

    if m1 >= m2:
        m1_discrete = 1
        m2_discrete = 0
    else:
        m1_discrete = 0
        m2_discrete = 1
        
    rg1 = item["radgraph_complete_report_1"]
    rg2 = item["radgraph_complete_report_2"]
 
    reward_1 = lam * m1_discrete + (1 - lam) * rg1
    reward_2 = lam * m2_discrete + (1 - lam) * rg2
 
    return {
        # keep original text fields
        "study_id":               item["study_id"],
        "image_paths":            item["image_paths"],
        "report_1":               item["report_1"],
        "report_2":               item["report_2"],
        "ground_truth":           item["ground_truth"],
        # keep original radiologist label (human gold standard, unchanged)
        "radiologist_preference": "report_1" if reward_1 >= reward_2 else "report_2",
        # recomputed
        "reward_report_1":        reward_1,
        "reward_report_2":        reward_2,
        # cached scores (unchanged)
        "meteor_report_1":        m1,
        "meteor_report_2":        m2,
        "radgraph_complete_report_1": rg1,
        "radgraph_complete_report_2": rg2,
    }
 
 
def main():
 
    input_path  = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model23-10pctdatasetreports-5e-1lambda.json"
    output_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model23-10pctdatasetreports-5e-1lambda-discrete.json"
    lam         = 0.5
 
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"--lambda must be in [0, 1], got {lam}")
 
    print(f"Loading  : {input_path}")
    with open(input_path) as f:
        dataset = json.load(f)
 
    recomputed = [recompute_item(item, lam) for item in dataset]
    n_total = len(recomputed)
    
    n_metric_flipped = sum(
        1 for orig, new in zip(dataset, recomputed)
        # compare old metric preference (derived from stored rewards) vs new
        if (orig["radiologist_preference"] != new["radiologist_preference"])
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(recomputed, f, indent=4)
 
    print(f"Saved    : {output_path}  ({n_total} items)")
    print(f"Lambda   : {lam}")
    print(f"Flipped  : {n_metric_flipped} items ({n_metric_flipped / len(dataset):.2%})")
 
if __name__ == "__main__":
    main()