from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from radgraph import F1RadGraph
import nltk
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel
import json

import sys
sys.path.append("/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM")

from src.radvlm.utils.evaluation_utils import compute_metrics

def create_radgraph_preferences_deepseek(preference_dataset_path, output_path):
    """
    Create RadGraph preferences for DeepSeek evaluation dataset.
    
    Args:
        preference_dataset_path: Path to preference dataset (json file with pairs of reports and images)
        output_path: Path to save the generated RadGraph preferences (json file)
    """
    # Load preference dataset
    with open(preference_dataset_path, 'r') as f:
        preference_dataset = json.load(f)
    
    preferences = []

    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl", model_cache_dir="/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/.cache/radgraph/0.1.2")
    
    
    BATCH_SIZE = 16
    LAMBDA = 0.01


    all_hyps, all_refs = [], []
    meteor_scores = []
    for item in preference_dataset:
        all_hyps.extend([item['report_1'], item['report_2']])
        all_refs.extend([item['ground_truth'], item['ground_truth']])

        # Compute meteor scores for each report against the ground truth
        gt_report = item['ground_truth']
        report_1 = item['report_1']
        report_2 = item['report_2']

        meteor_1 = meteor_score([gt_report.split()], report_1.split())
        meteor_2 = meteor_score([gt_report.split()], report_2.split())
        meteor_scores.append((meteor_1, meteor_2))
    
    print(meteor_scores)

    radgraph_results = []
    for i in tqdm(range(0, len(all_hyps), BATCH_SIZE), desc="Computing radgraph scores"):
        hyps_batch = all_hyps[i:i+BATCH_SIZE]
        refs_batch = all_refs[i:i+BATCH_SIZE]
        _, reward_list, _, _ = f1radgraph(hyps=hyps_batch, refs=refs_batch)
        radgraph_results.extend(reward_list)

        print("Reward list for current batch:", reward_list)


    # Generate preference dataset
    for idx, item in enumerate(preference_dataset):
        report_1 = item['report_1']
        report_2 = item['report_2']
        gt_report = item['ground_truth']
        meteor_1, meteor_2 = meteor_scores[idx]
        radgraph_1_simple = radgraph_results[2*idx][0]
        radgraph_1_partial = radgraph_results[2*idx][1]
        radgraph_1_complete = radgraph_results[2*idx][2]
        radgraph_2_simple = radgraph_results[2*idx + 1][0]
        radgraph_2_partial = radgraph_results[2*idx + 1][1]
        radgraph_2_complete = radgraph_results[2*idx + 1][2]

        print("Report pair index:", idx)
        print("Meteor scores - Report 1:", meteor_1, "Report 2:", meteor_2)
        print("RadGraph F1 complete scores - Report 1:", radgraph_1_complete, "Report 2:", radgraph_2_complete)

        # Compute overall reward for each report
        reward_1 = LAMBDA * meteor_1 + (1 - LAMBDA) * radgraph_1_complete
        reward_2 = LAMBDA * meteor_2 + (1 - LAMBDA) * radgraph_2_complete

        print("Overall rewards - Report 1:", reward_1, "Report 2:", reward_2)

        if reward_1 > reward_2:
            preferences.append({
                "report_1": report_1,
                "report_2": report_2,
                "ground_truth": gt_report,
                "radiologist_preference": "report_1",
                "reward_report_1": reward_1,
                "reward_report_2": reward_2,
                "meteor_report_1": meteor_1,
                "meteor_report_2": meteor_2,
                "radgraph_complete_report_1": radgraph_1_complete,
                "radgraph_complete_report_2": radgraph_2_complete
            })
        elif reward_2 > reward_1:
            preferences.append({
                "report_1": report_1,
                "report_2": report_2,
                "ground_truth": gt_report,
                "radiologist_preference": "report_2",
                "reward_report_1": reward_1,
                "reward_report_2": reward_2,
                "meteor_report_1": meteor_1,
                "meteor_report_2": meteor_2,
                "radgraph_complete_report_1": radgraph_1_complete,
                "radgraph_complete_report_2": radgraph_2_complete
            })
        else:
            preferences.append({
                "report_1": report_1,
                "report_2": report_2,
                "ground_truth": gt_report,
                "radiologist_preference": "report_1", # In case of tie, we can arbitrarily choose one as the preferred report (here we choose report_1)
                "reward_report_1": reward_1,
                "reward_report_2": reward_2,
                "meteor_report_1": meteor_1,
                "meteor_report_2": meteor_2,
                "radgraph_complete_report_1": radgraph_1_complete,
                "radgraph_complete_report_2": radgraph_2_complete
            })

        
        
        
    # Save the RadGraph preferences to a json file
    with open(output_path, 'w') as f:
        json.dump(preferences, f, indent=4)
    
if __name__ == "__main__":
    preference_dataset_path = "/home/ug301051/jupyterlab/RadVLM/results/dpo_dataset/example_report_pairs.json"
    output_path = "/home/ug301051/jupyterlab/RadVLM/results/dpo_dataset/example_report_pairs_preference_dataset.json"
    create_radgraph_preferences_deepseek(preference_dataset_path, output_path)