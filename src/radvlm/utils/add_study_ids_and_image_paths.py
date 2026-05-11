from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import json, nltk
from nltk.translate.meteor_score import meteor_score
from radgraph import F1RadGraph


def add_study_ids_and_image_paths(candidates_dataset_path, preference_dataset_path, output_path):
    
    with open(candidates_dataset_path) as f:
        candidates_dataset = json.load(f)

    with open(preference_dataset_path) as f:
        preference_dataset = json.load(f)

    for candidate_item, preference_item in zip(candidates_dataset, preference_dataset):
        if candidate_item['report_1'] != preference_item['report_1'] or candidate_item['report_2'] != preference_item['report_2']:
            raise ValueError("Mismatch between candidates and preferences datasets. Ensure they are aligned and contain the same report pairs.")
        if candidate_item['ground_truth'] != preference_item['ground_truth']:
            raise ValueError("Mismatch between candidates and preferences datasets. Ensure they are aligned and contain the same ground truth values.")
        
        preference_item['study_id'] = candidate_item['study_id']
        preference_item['image_paths'] = candidate_item['image_paths']

    with open(output_path, 'w') as f:
        json.dump(preference_dataset, f, indent=4)

if __name__ == "__main__":
    candidates_dataset_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-p13-p15-6vision-final-generated-report-pairs-p18.json"
    preference_dataset_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model16-p18reports-1e-2lambda.json"
    output_path = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model16-p18reports-1e-2lambda-with-study-ids.json"
    add_study_ids_and_image_paths(candidates_dataset_path, preference_dataset_path, output_path)