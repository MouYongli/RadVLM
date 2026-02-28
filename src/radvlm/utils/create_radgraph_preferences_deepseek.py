from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from radgraph import F1RadGraph
import nltk
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

    for item in tqdm(preference_dataset, desc="Creating RadGraph preferences"):
        report_1 = item['report_1']
        report_2 = item['report_2']
        ground_truth_report = item['ground_truth']
        
        f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl", model_cache_dir="/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/.cache/radgraph/0.1.2")
        _, reward_list, _, _ = f1radgraph(hyps=[report_1, report_2], refs=[ground_truth_report, ground_truth_report])
        # print(reward_list)
        
        if reward_list[2][0] > reward_list[2][1]:  # Compare complete F1 scores
            preferences.append({
                "image_paths": item['image_paths'],
                "report_1": report_1,
                "report_2": report_2,
                "ground_truth": ground_truth_report,
                "radiologist_preference": "report_1",
                "radgraph_scores": {
                    "report_1": {
                        "simple": reward_list[0][0],
                        "partial": reward_list[1][0],
                        "complete": reward_list[2][0],
                    },
                    "report_2": {
                        "simple": reward_list[0][1],
                        "partial": reward_list[1][1],
                        "complete": reward_list[2][1],
                    }
                }
            })
        elif reward_list[2][0] < reward_list[2][1]:
            preferences.append({
                "image_paths": item['image_paths'],
                "report_1": report_1,
                "report_2": report_2,
                "ground_truth": ground_truth_report,
                "radiologist_preference": "report_2",
                "radgraph_scores": {
                    "report_1": {
                        "simple": reward_list[0][0],
                        "partial": reward_list[1][0],
                        "complete": reward_list[2][0],
                    },
                    "report_2": {
                        "simple": reward_list[0][1],
                        "partial": reward_list[1][1],
                        "complete": reward_list[2][1],
                    }
                }
            })
        else:
            preferences.append({
                "image_paths": item['image_paths'],
                "report_1": report_1,
                "report_2": report_2,
                "ground_truth": ground_truth_report,
                "radiologist_preference": "report_1",  # If scores are equal, default to report_1 (greedy decoding)
                "radgraph_scores": {
                    "report_1": {
                        "simple": reward_list[0][0],
                        "partial": reward_list[1][0],
                        "complete": reward_list[2][0],
                    },
                    "report_2": {
                        "simple": reward_list[0][1],
                        "partial": reward_list[1][1],
                        "complete": reward_list[2][1],
                    }
                }
            })
        
        
    # Save the RadGraph preferences to a json file
    with open(output_path, 'w') as f:
        json.dump(preferences, f, indent=4)
    
if __name__ == "__main__":
    preference_dataset_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/deepseek-vl2-mimic-cxr-lora-r16-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision-proj-final-generated-report-pairs.json"
    output_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/deepseek-vl2-mimic-cxr-lora-r16-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision-proj-final_radgraph_preferences.json"
    create_radgraph_preferences_deepseek(preference_dataset_path, output_path)