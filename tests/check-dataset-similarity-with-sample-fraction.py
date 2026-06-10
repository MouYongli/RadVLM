import torch
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.radvlm.data.build_dataset import load_dataset
from src.radvlm.data.medgemma_dataset import RadVLMDatasetMedGemma, create_collate_fn_medgemma
from src.radvlm.utils.evaluation_utils_medgemma import MedGemmaEvaluator
from src.radvlm.utils.config import MEDGEMMA_BASE_MODEL_PATH

MedGemmaPretrainingEvaluator = MedGemmaEvaluator


def evaluate_pre_training():
    """Evaluate the pre-trained MedGemma model from file on a validate set"""
    
    print("Checking dataset similarity with data fraction...", flush=True)
    
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(here, "../../results/pretraining/medgemma-1.5-mimic-cxr-poc-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-40pctdata-bf16-final"))
    evaluator = MedGemmaPretrainingEvaluator(model_path=model_path, base_model_path=MEDGEMMA_BASE_MODEL_PATH)
    raw_data = load_dataset(["p10"])

    study_ids_all = []
    for n in range(5):
        train_dataset = RadVLMDatasetMedGemma(raw_data, evaluator.processor, evaluator.tokenizer, split='train', mode="eval", sample_fraction=0.1)
    
        study_ids = []
        for i in range(len(train_dataset)):
            study_ids.append(train_dataset[i]["study_id"])
            
        study_ids_all.append(study_ids)
    

    print(len(study_ids_all[0]), len(study_ids_all[1]), len(study_ids_all[2]), len(study_ids_all[3]), len(study_ids_all[4]))
    
    # Convert each list to a set
    sets = [set(lst) for lst in study_ids_all]
    
    # Elements present in every list
    common_to_all = set.intersection(*sets)
    
    # Elements that are missing from at least one list
    not_in_all_five = set.union(*sets) - common_to_all
    
    print(f"Common to all lists: {len(common_to_all)}")
    print(f"Not present in every list: {len(not_in_all_five)}")
    print(not_in_all_five)
        
    
    print("Check completed")

if __name__ == "__main__":
    
    evaluate_pre_training()
    