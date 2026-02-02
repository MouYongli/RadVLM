import torch
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from radgraph import F1RadGraph
import nltk

def compute_metrics(generated_reports, ground_truth_reports, losses=None, perplexities=None):
    """
    Compute evaluation metrics for generated reports
    
    Args:
        generated_reports: List of generated report strings
        ground_truth_reports: List of ground truth report strings
        losses: List of cross-entropy losses (optional)
        perplexities: List of perplexity values (optional)
    
    Returns:
        Dictionary containing average scores and individual scores for:
        - BLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR
        - RadGraph (F1, Precision, Recall)
        - Cross-Entropy Loss, Perplexity
    """
    
    # Download required NLTK data
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4', quiet=True)
    
    print("\nComputing evaluation metrics...", flush=True)
    
    # Handle single string inputs by converting to lists
    if isinstance(generated_reports, str):
        generated_reports = [generated_reports]
    if isinstance(ground_truth_reports, str):
        ground_truth_reports = [ground_truth_reports]
    
    # Validate inputs
    assert len(generated_reports) == len(ground_truth_reports), \
        f"Mismatch: {len(generated_reports)} generated vs {len(ground_truth_reports)} ground truth reports"
    
    # Initialize score lists
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute BLEU, ROUGE, and METEOR scores for each pair
    print("Computing BLEU, ROUGE, and METEOR scores...", flush=True)
    for gen_report, gt_report in zip(generated_reports, ground_truth_reports):
        # BLEU
        reference = gt_report.split()
        hypothesis = gen_report.split()
        bleu1 = sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
        bleu2 = sentence_bleu([reference], hypothesis, weights=(0.5, 0.5, 0, 0))
        bleu3 = sentence_bleu([reference], hypothesis, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = sentence_bleu([reference], hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)
        
        # ROUGE
        rouge_scores = scorer.score(gt_report, gen_report)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        
        # METEOR
        meteor = meteor_score([gt_report.split()], gen_report.split())
        meteor_scores.append(meteor)
    
    # Compute RadGraph scores in batches to avoid OOM
    print("Computing RadGraph scores...", flush=True)
    try:          
        f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
        
        # Process in batches to avoid OOM
        batch_size = 16
        all_f1_simple_scores = []
        all_f1_partial_scores = []
        all_f1_complete_scores = []
        
        for i in range(0, len(generated_reports), batch_size):
            batch_end = min(i + batch_size, len(generated_reports))
            batch_hyps = generated_reports[i:batch_end]
            batch_refs = ground_truth_reports[i:batch_end]
            
            print(f"Processing RadGraph batch {i//batch_size + 1}/{(len(generated_reports) + batch_size - 1)//batch_size} (samples {i+1}-{batch_end})...", flush=True)
            
            _, reward_list, _, _ = f1radgraph(hyps=batch_hyps, refs=batch_refs)
            print(f"RadGraph batch {i//batch_size + 1} processed.", flush=True)
            all_f1_simple_scores.extend(reward_list[0])
            all_f1_partial_scores.extend(reward_list[1])
            all_f1_complete_scores.extend(reward_list[2])
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Compute mean scores
        radgraph_f1_mean = (
            sum(all_f1_simple_scores) / len(all_f1_simple_scores),
            sum(all_f1_partial_scores) / len(all_f1_partial_scores),
            sum(all_f1_complete_scores) / len(all_f1_complete_scores)
        )
        
        # Combine individual scores
        radgraph_individual = []
        for i in range(len(generated_reports)):
            t = (all_f1_simple_scores[i], all_f1_partial_scores[i], all_f1_complete_scores[i])
            radgraph_individual.append(t)          
    except Exception as e:
        print(f"Warning: RadGraph computation failed: {e}", flush=True)
        radgraph_f1_mean = (0.0, 0.0, 0.0)
        radgraph_individual = [(0.0, 0.0, 0.0)] * len(generated_reports)
    
    # Calculate averages
    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0
    avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0.0
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    # Calculate loss and perplexity averages if provided
    avg_loss = None
    avg_perplexity = None
    if losses is not None and len(losses) > 0:
        valid_losses = [l for l in losses if not torch.isnan(torch.tensor(l))]
        avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0.0
    if perplexities is not None and len(perplexities) > 0:
        valid_perplexities = [p for p in perplexities if not torch.isnan(torch.tensor(p))]
        avg_perplexity = sum(valid_perplexities) / len(valid_perplexities) if valid_perplexities else 0.0
    
    return {
        "bleu1": avg_bleu1,
        "bleu2": avg_bleu2,
        "bleu3": avg_bleu3,
        "bleu4": avg_bleu4,
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "meteor": avg_meteor,
        "radgraph_f1": radgraph_f1_mean,
        "cross_entropy_loss": avg_loss,
        "perplexity": avg_perplexity,
        "bleu1_scores": bleu1_scores,
        "bleu2_scores": bleu2_scores,
        "bleu3_scores": bleu3_scores,
        "bleu4_scores": bleu4_scores,
        "rouge1_scores": rouge1_scores,
        "rouge2_scores": rouge2_scores,
        "rougeL_scores": rougeL_scores,
        "meteor_scores": meteor_scores,
        "radgraph_individual": radgraph_individual,
        "losses": losses if losses is not None else [],
        "perplexities": perplexities if perplexities is not None else [],
        "eval_samples": len(generated_reports)
    }