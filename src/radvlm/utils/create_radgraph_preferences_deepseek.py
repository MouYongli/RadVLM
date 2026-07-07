from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import json, nltk
from nltk.translate.meteor_score import meteor_score
from radgraph import F1RadGraph


def compute_meteor_pair(item):
    gt = item['ground_truth'].split()
    m1 = meteor_score([gt], item['report_1'].split())
    m2 = meteor_score([gt], item['report_2'].split())
    return m1, m2


def run_batched_radgraph(scorer, hyps, refs, batch_size):
    results = []
    for i in tqdm(range(0, len(hyps), batch_size), desc="RadGraph"):
        _, reward_list, _, _ = scorer(
            hyps=hyps[i:i+batch_size],
            refs=refs[i:i+batch_size]
        )
        simple, partial, complete = reward_list
        results.extend(zip(simple, partial, complete))
    return results


def build_preference(item, meteor_1, meteor_2, rg1, rg2, lam):
    if meteor_1 >= meteor_2:
        m1_discrete = 1
        m2_discrete = 0
    else:
        m1_discrete = 0
        m2_discrete = 1
        
 
    reward_1 = lam * m1_discrete + (1 - lam) * rg1[2]
    reward_2 = lam * m2_discrete + (1 - lam) * rg2[2]
    # reward_1 = lam * meteor_1 + (1 - lam) * rg1[2]  # rg[2] = complete score
    # reward_2 = lam * meteor_2 + (1 - lam) * rg2[2]
    return {
        "study_id": item['study_id'],
        "image_paths": item['image_paths'],
        "report_1": item['report_1'],
        "report_2": item['report_2'],
        "ground_truth": item['ground_truth'],
        "radiologist_preference": "report_1" if reward_1 >= reward_2 else "report_2",
        "reward_report_1": reward_1,
        "reward_report_2": reward_2,
        "meteor_report_1": meteor_1,
        "meteor_report_2": meteor_2,
        "radgraph_complete_report_1": rg1[2],
        "radgraph_complete_report_2": rg2[2],
    }


def create_radgraph_preferences_deepseek(preference_dataset_path, output_path):
    for resource in ('wordnet', 'omw-1.4'):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)

    with open(preference_dataset_path) as f:
        preference_dataset = json.load(f)

    BATCH_SIZE = 64   # increase from 16 — tune to your VRAM
    LAMBDA = 0.01

    # --- METEOR: parallel CPU execution ---
    meteor_scores = [compute_meteor_pair(item) for item in tqdm(preference_dataset, desc="METEOR")]

    # --- RadGraph: deduplicate refs ---
    # Each ground_truth was being annotated twice. By separating report_1 and
    # report_2 passes we make it trivial to cache ref annotations if radgraph
    # exposes that API in future; for now we at least keep the structure clean.
    refs  = [item['ground_truth'] for item in preference_dataset]
    hyps1 = [item['report_1']     for item in preference_dataset]
    hyps2 = [item['report_2']     for item in preference_dataset]

    f1radgraph = F1RadGraph(
        reward_level="all",
        model_type="radgraph-xl",
        model_cache_dir="/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/.cache/radgraph/0.1.2"
    )

    rg1 = run_batched_radgraph(f1radgraph, hyps1, refs, BATCH_SIZE)
    rg2 = run_batched_radgraph(f1radgraph, hyps2, refs, BATCH_SIZE)

    # --- Build preferences ---
    preferences = [
        build_preference(item, m1, m2, r1, r2, LAMBDA)
        for item, (m1, m2), r1, r2
        in zip(preference_dataset, meteor_scores, rg1, rg2)
    ]

    # remove items, where item[radgraph_complete_report_1]==0 and item[radgraph_complete_report_2]==0
    preferences = [item for item in preferences if item["radgraph_complete_report_1"] != 0 or item["radgraph_complete_report_2"] != 0]
    preferences = [
        data for data in preferences
        if not (
            (data["radiologist_preference"] == "report_1" and data["radgraph_complete_report_1"] == 0)
            or
            (data["radiologist_preference"] == "report_2" and data["radgraph_complete_report_2"] == 0)
        )
    ]
    
    with open(output_path, 'w') as f:
        json.dump(preferences, f, indent=4)

    print(f"Saved {len(preferences)} preferences to {output_path}.")
    
if __name__ == "__main__":
    preference_dataset_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-60pctdata-allsubsets-final-generated-report-pairs-10pctdataset-correct-sampling.json"
    output_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model23-10pctdatasetreports-1e-2lambda-discrete-processed-corrected-sampling.json"
    create_radgraph_preferences_deepseek(preference_dataset_path, output_path)