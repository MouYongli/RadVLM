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
    """Evaluate the DPO-trained MedGemma model from file on a validate set"""
    
    print("Evaluating DPO-trained MedGemma model...", flush=True)
    
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(here, "../../results/dpo/medgemma-1.5-mimic-cxr-dpo-lora-r16-lr5e-5-beta0.1-basemodel-dataset-small-1e-2lambda/best_model"))
    evaluator = MedGemmaPretrainingEvaluator(model_path=model_path, base_model_path=MEDGEMMA_BASE_MODEL_PATH)
    raw_data = load_dataset(["p10"])
    
    val_dataset = RadVLMDatasetMedGemma(raw_data, evaluator.processor, evaluator.tokenizer, split='validate', mode="eval")
    # print("val dataset item example:", val_dataset[0], flush=True)
    collate_fn = create_collate_fn_medgemma(evaluator.processor)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    study_ids, generated_reports, ground_truth_reports, losses, perplexities = evaluator.evaluate_reports(val_dataloader, max_samples=500)
    # save the study_ids and generated reports to text file for further inspection
    
    output_file = '/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/generated_reports/generated_reports_dpo-lora-r16-lr5e-5-beta0.1-basemodel-dataset-small-1e-2lambda.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GENERATED REPORTS EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        
        for study_id, gen_report, gt_report in zip(study_ids, generated_reports, ground_truth_reports):
            f.write(f"Study ID: {study_id}\n")
            f.write("-" * 80 + "\n")
            f.write("Generated Report:\n")
            f.write(f"{gen_report}\n\n")
            f.write("Ground Truth Report:\n")
            f.write(f"{gt_report}\n")
            f.write("=" * 80 + "\n\n")

    print(f"Reports saved to {output_file}", flush=True)

    metrics_dict = evaluator.compute_metrics(generated_reports, ground_truth_reports, losses, perplexities)

    print("Evaluation Metrics:", metrics_dict, flush=True)
    print(f"\nCross-Entropy Loss: {metrics_dict['cross_entropy_loss']:.4f}", flush=True)
    print(f"Perplexity: {metrics_dict['perplexity']:.4f}", flush=True)
    print("Evaluation complete!", flush=True)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    evaluate_pre_training()
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Total evaluation time: {end_time - start_time}", flush=True)