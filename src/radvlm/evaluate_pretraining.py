import torch
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.radvlm.data.build_dataset import load_dataset
from src.radvlm.data.deepseek_dataset import RadVLMDatasetDeepseek, collate_fn
from src.radvlm.utils.evaluation_utils_deepseek import DeepSeekVL2Evaluator

DeepSeekVL2PretrainingEvaluator = DeepSeekVL2Evaluator


def evaluate_pre_training():
    """Evaluate the pre-trained DeepSeek VL2 model from file on a validate set"""
    
    print("Evaluating pre-trained DeepSeek VL2 model...", flush=True)
    
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "../../results/pretraining/deepseek-vl2-mimic-cxr-poc-lora-r8-lr1e-4-3epochs-linear-5pctwarmup-3earlystop-100pct-final")
    evaluator = DeepSeekVL2PretrainingEvaluator(model_path=model_path)
    raw_data = load_dataset()
    # print("Raw data item example:", raw_data[0], flush=True)
    val_dataset = RadVLMDatasetDeepseek(raw_data, evaluator.processor, evaluator.tokenizer, split='validate', mode="eval")
    # print("val dataset item example:", val_dataset[0], flush=True)

    # Custom collate function that includes all necessary fields
    # def collate_fn(batch):
    #     return {
    #         "study_id": [item['study_id'] for item in batch],
    #         "input_ids": torch.stack([item['input_ids'] for item in batch]),
    #         "attention_mask": torch.stack([item['attention_mask'] for item in batch]),
    #         "labels": torch.stack([item['labels'] for item in batch]),
    #         "images": [item['images'] for item in batch],
    #         "report": [evaluator.tokenizer.decode(item['labels'], skip_special_tokens=True) for item in batch]  # Decode labels to get ground truth text
    #     }

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    study_ids, generated_reports, ground_truth_reports, losses, perplexities = evaluator.evaluate_reports(val_dataloader, max_samples=500)
    # save the study_ids and generated reports to text file for further inspection
    output_file = 'generated_reports_poc-lora-r8-lr1e-4-3epochs-linear-5pctwarmup-3earlystop-100pct.txt'
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
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    evaluate_pre_training()
    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total evaluation time: {end_time - start_time}")