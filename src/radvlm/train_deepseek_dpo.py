import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer
from functools import partial
import math
import argparse
import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import AdamW
from tqdm import tqdm
import json
import sys
# sys.path.append('/home/gustke/Projects/RadVLM')

here = os.path.dirname(os.path.abspath(__file__))

from src.radvlm.data.build_dataset import load_preference_dataset
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from src.radvlm.data.deepseek_dpo_dataset import RadVLMDPODataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Paths
    
    model_path: str = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pct-vision-final"
    base_model_path: str = "deepseek-ai/deepseek-vl2-small"
    output_dir: str = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo/deepseek-vl2-mimic-cxr-dpo-lora-r16-lr5e-5-beta0.1"

    # Training
    num_train_epochs:            int   = 3
    gradient_accumulation_steps: int   = 4
    learning_rate:               float = 5e-5
    weight_decay:                float = 0.01
    warmup_ratio:                float = 0.1
    max_grad_norm:               float = 1.0
    max_seq_length:              int   = 3072
    beta:                        float = 0.1

    # LoRA
    lora_r:              int   = 16
    lora_alpha:          int   = 32
    lora_dropout:        float = 0.05
    lora_target_modules: list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Logging / checkpointing
    logging_steps: int = 1
    eval_steps:    int = 1
    save_steps:    int = 2
    eval_samples:  int = 64
    seed:          int = 42

    # W&B
    wandb_project: str = "deepseek-vl2-mimic-cxr-dpo"
    wandb_run:     str = "dpo-lora-r16-lr5e-5-beta0.1"


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    p = argparse.ArgumentParser()
    for f_name, f_val in vars(defaults).items():
        p.add_argument(f"--{f_name}", type=type(f_val) if not isinstance(f_val, list) else None,
                       default=f_val)
    ns = p.parse_args()
    return TrainingConfig(**{k: v for k, v in vars(ns).items()})


# ── Model loading ──────────────────────────────────────────────────────────────
def setup_model(model_path: str, base_model_path="deepseek-ai/deepseek-vl2-small"):
    """
    Load DeepSeek-VL2 with optional LoRA adapter
    
    Args:
        model_path: HuggingFace model ID or local path to base model
        adapter_path: Path to pretrained LoRA adapter (from SFT training)
    """
    
    print("Loading base model...", flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if we're loading an existing adapter
    has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if has_adapter:
        # Load existing PEFT model with adapter
        print(f"Loading PEFT model with existing adapter from {model_path}...", flush=True)
        
        # Read base model path from adapter config
        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            adapter_config = json.load(f)
            if "base_model_name_or_path" in adapter_config:
                base_model_path = adapter_config["base_model_name_or_path"]
                print(f"Base model path found in adapter config: {base_model_path}", flush=True)
        
        # Load base model first
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Load PEFT adapter on top
        model = PeftModel.from_pretrained(
            base_model,
            model_path,
            is_trainable=True
        )
        print("PEFT adapter loaded successfully.", flush=True)
    else:
        # Load base model and apply new LoRA
        print(f"Loading base model from {model_path}...", flush=True)
        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        ).to(device)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            inference_mode=False,
        )
        
        print("Applying new LoRA adapter...", flush=True)
        model = get_peft_model(model, lora_config)

    print(f"Loading processor and tokenizer from {base_model_path}...", flush=True)
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(base_model_path)
    tokenizer = processor.tokenizer
    
    # Freeze vision encoder
    for name, param in model.named_parameters():
        if "vision_tower" in name or "visual" in name or "vision_model" in name:
            param.requires_grad = False
    
    print("Vision encoder frozen.", flush=True)
    
    model.print_trainable_parameters()
    print("Model setup complete.", flush=True)
    
    return model, processor, tokenizer


# ── Encoding ───────────────────────────────────────────────────────────────────
def encode_single(
    processor,
    prompt_text: str,
    completion_text: str,
    pil_images: list,
    max_length: int,
    device: torch.device,
):
    """
    Tokenise `prompt + completion` for one sample with any number of images.

    Prompt-masking strategy: find the assistant token in the sequence to
    determine where the completion starts. Everything before is masked.
    """
    images = pil_images if pil_images else None

    # Process the full conversation (prompt + completion)
    full_enc = processor(
        conversations=prompt_text + completion_text,
        images=images,
        force_batchify=True,
        system_prompt="",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        inference_mode=False,  # ensure generation prompt is added for correct tokenisation
    )
    # Convert BatchCollateOutput to dict
    full_enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in vars(full_enc).items()}
    
    # Convert images to bfloat16 if present
    if full_enc.get("images") is not None and isinstance(full_enc["images"], torch.Tensor):
        full_enc["images"] = full_enc["images"].to(dtype=torch.bfloat16)

    # Find where the assistant's response starts by searching for the assistant token
    assistant_token = "<|Assistant|>"
    assistant_token_ids = processor.tokenizer.encode(
        assistant_token, 
        add_special_tokens=False
    )
    
    # Convert input_ids to list for searching
    input_ids_list = full_enc["input_ids"][0].tolist()
    
    # Search for the assistant token sequence
    prompt_len = None
    for i in range(len(input_ids_list) - len(assistant_token_ids) + 1):
        if input_ids_list[i:i+len(assistant_token_ids)] == assistant_token_ids:
            # Position AFTER the assistant token is where completion starts
            prompt_len = i + len(assistant_token_ids)
            break
    
    if prompt_len is None:
        logger.warning("Could not find assistant token in sequence, using fallback")
        # Fallback: estimate based on image tokens (DeepSeek-VL2 uses 576 tokens per image)
        num_images = len(pil_images) if pil_images else 0
        prompt_len = 10 + (num_images * 576) + 30

    labels = full_enc["input_ids"].clone()
    labels[0, :prompt_len] = -100   # mask prompt tokens from the loss

    return full_enc, labels


# ── Log-probability (sum over completion tokens) ──────────────────────────────
def completion_log_prob(
    model,
    enc: dict,
    labels: torch.Tensor,
    use_autocast: bool = True,
) -> torch.Tensor:
    """
    Return the *sum* of log-probs over non-masked completion tokens.

    Using sum rather than mean avoids length bias: with mean log-prob,
    DPO would implicitly favour shorter completions.
    """
    ctx = autocast(dtype=torch.bfloat16) if use_autocast else contextmanager(lambda: iter([None]))()
    with ctx:
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            images=enc.get("images"),
            images_seq_mask=enc.get("images_seq_mask"),
            images_spatial_crop=enc.get("images_spatial_crop"),
            return_dict=True,
        ).logits  # (1, seq_len, vocab)

    # Causal shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()    # (1, L-1, V)
    shift_labels = labels[:, 1:].contiguous()         # (1, L-1)

    log_probs  = F.log_softmax(shift_logits.float(), dim=-1)
    token_lp   = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    comp_mask  = (shift_labels != -100).float()

    return (token_lp * comp_mask).sum()   # scalar


# ── Reference model context ────────────────────────────────────────────────────
@contextmanager
def reference_mode(model):
    """
    Disable LoRA adapters so the frozen base weights act as π_ref.
    Eliminates the need for a second model copy in memory.
    """
    with torch.no_grad():
        with model.disable_adapter():
            yield


# ── DPO loss ───────────────────────────────────────────────────────────────────
def dpo_step(
    model,
    enc_chosen,   lbl_chosen,
    enc_rejected, lbl_rejected,
    beta: float,
    acc_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient DPO step: only ONE policy computation graph lives in
    memory at a time, achieved via manual gradient accumulation.

    Returns:
        loss_val  — detached scalar for logging
        log_ratio — detached scalar for accuracy logging
    """
    # ── 1. Reference log-probs (no grad, freed immediately) ───────────────
    with reference_mode(model):
        with torch.no_grad():
            log_ref_c = completion_log_prob(
                model, enc_chosen,   lbl_chosen,   use_autocast=False).detach()
            torch.cuda.empty_cache()
            log_ref_r = completion_log_prob(
                model, enc_rejected, lbl_rejected, use_autocast=False).detach()
            torch.cuda.empty_cache()

    # ── 2. Peek at rejected value (no grad) to compute chosen's gradient ──
    with torch.no_grad():
        log_pi_r_val = completion_log_prob(
            model, enc_rejected, lbl_rejected).detach()
    torch.cuda.empty_cache()

    # ── 3. Chosen forward (graph lives here) ──────────────────────────────
    log_pi_c = completion_log_prob(model, enc_chosen, lbl_chosen)

    # Compute log_ratio and loss value (all detached — just for logging/grad)
    log_ratio = ((log_pi_c.detach() - log_ref_c) -
                 (log_pi_r_val        - log_ref_r))
    loss_val  = -F.logsigmoid(beta * log_ratio)

    # Analytic gradient:  ∂loss/∂log_pi_c = β·(σ(β·r) - 1)  =  -β·σ(-β·r)
    grad_c = beta * (torch.sigmoid(beta * log_ratio) - 1) / acc_steps
    log_pi_c.backward(grad_c)
    del log_pi_c
    torch.cuda.empty_cache()

    # ── 4. Rejected forward (graph lives here, chosen's is already freed) ─
    log_pi_r = completion_log_prob(model, enc_rejected, lbl_rejected)

    # Analytic gradient:  ∂loss/∂log_pi_r = β·(1 - σ(β·r))  =  β·σ(-β·r)
    grad_r = beta * (1 - torch.sigmoid(beta * log_ratio)) / acc_steps
    log_pi_r.backward(grad_r)
    del log_pi_r
    torch.cuda.empty_cache()

    return loss_val, log_ratio.detach()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def save_checkpoint(model, processor, optimizer, scheduler, scaler, step: int, out_dir: str):
    ckpt_dir = Path(out_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    state_dict = {
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "global_step": step,
    }
    if scaler is not None:
        state_dict["scaler"] = scaler.state_dict()
    torch.save(state_dict, ckpt_dir / "optimizer.pt")
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def find_latest_checkpoint(out_dir: str) -> Optional[Path]:
    ckpts = sorted(
        Path(out_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return ckpts[-1] if ckpts else None


def load_checkpoint(optimizer, scheduler, scaler, ckpt_dir: Path) -> int:
    opt_path = ckpt_dir / "optimizer.pt"
    if not opt_path.exists():
        logger.warning(f"No optimizer state in {ckpt_dir} — starting fresh.")
        return 0
    state = torch.load(opt_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    logger.info(f"Resumed from step {state['global_step']}")
    return state["global_step"]


# ── Best-model tracking ────────────────────────────────────────────────────────
class BestModelTracker:
    def __init__(self, out_dir: str, metric: str = "eval/loss", mode: str = "min"):
        self.save_dir   = Path(out_dir) / "best_model"
        self.metric     = metric
        self.best       = float("inf") if mode == "min" else float("-inf")
        self.is_better  = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

    def update(self, metrics: dict, model, processor) -> bool:
        val = metrics.get(self.metric)
        if val is None or not self.is_better(val, self.best):
            return False
        self.best = val
        self.save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.save_dir)
        processor.save_pretrained(self.save_dir)
        logger.info(f"★ New best {self.metric}={val:.4f} → {self.save_dir}")
        return True


@torch.no_grad()
def dpo_loss_eval(
    model,
    enc_chosen,   lbl_chosen,
    enc_rejected, lbl_rejected,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eval-only DPO loss. All four passes are no-grad, so run them
    sequentially with cache clears to keep peak memory low.
    """
    log_pi_c  = completion_log_prob(model, enc_chosen,   lbl_chosen)
    torch.cuda.empty_cache()
    log_pi_r  = completion_log_prob(model, enc_rejected, lbl_rejected)
    torch.cuda.empty_cache()

    with reference_mode(model):
        log_ref_c = completion_log_prob(model, enc_chosen,   lbl_chosen,   use_autocast=False)
        torch.cuda.empty_cache()
        log_ref_r = completion_log_prob(model, enc_rejected, lbl_rejected, use_autocast=False)
        torch.cuda.empty_cache()

    log_ratio = (log_pi_c - log_ref_c) - (log_pi_r - log_ref_r)
    loss      = -F.logsigmoid(beta * log_ratio)
    return loss, log_ratio

# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model, processor, eval_records: list, beta: float,
    max_seq_length: int, device: torch.device, max_samples: int,
) -> dict:
    model.eval()
    records = eval_records[:max_samples]
    total_loss = total_acc = total_ratio = 0.0

    for rec in tqdm(records, desc="Eval", leave=False):
        enc_c, lbl_c = encode_single(processor, rec["prompt"], rec["chosen"],
                                     rec["images"], max_seq_length, device)
        enc_r, lbl_r = encode_single(processor, rec["prompt"], rec["rejected"],
                                     rec["images"], max_seq_length, device)

        loss, log_ratio = dpo_loss_eval(model, enc_c, lbl_c, enc_r, lbl_r, beta)

        total_loss  += loss.item()
        total_acc   += float(log_ratio.item() > 0)
        total_ratio += log_ratio.item()

    n = len(records)
    model.train()
    return {
        "eval/loss":      total_loss  / n,
        "eval/accuracy":  total_acc   / n,
        "eval/log_ratio": total_ratio / n,
    }


# ── W&B reporter ───────────────────────────────────────────────────────────────
def setup_reporter(cfg: TrainingConfig):
    import wandb
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run,
        config={
            "model":          "deepseek-ai/deepseek-vl2-small",
            "method":         "DPO",
            "lora_r":         cfg.lora_r,
            "lora_alpha":     cfg.lora_alpha,
            "learning_rate":  cfg.learning_rate,
            "beta":           cfg.beta,
            "epochs":         cfg.num_train_epochs,
            "max_seq_length": cfg.max_seq_length,
        },
    )
    return lambda metrics, step: wandb.log(metrics, step=step)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_metric   = setup_reporter(cfg)
    best_tracker = BestModelTracker(cfg.output_dir)

    # Model & processor
    model, processor, tokenizer = setup_model(cfg.model_path, cfg.base_model_path)
    processor.tokenizer.padding_side = "right"

    # Data
    logger.info("Loading preference dataset ...")
    preference_data = load_preference_dataset()
    dataset_wrapper = RadVLMDPODataset(
        preference_data, processor, tokenizer,
        max_seq_length=cfg.max_seq_length,
    )
    train_data = dataset_wrapper.get_split_data("train")
    val_data   = dataset_wrapper.get_split_data("validate")
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # Optimiser & scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    acc             = cfg.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_data) / acc)
    total_steps     = steps_per_epoch * cfg.num_train_epochs
    warmup_steps    = int(total_steps * cfg.warmup_ratio)
    scheduler       = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # Note: No GradScaler needed for bfloat16 (only needed for float16)

    # Resume from checkpoint if one exists
    global_step = 0
    resume_ckpt = find_latest_checkpoint(cfg.output_dir)
    if resume_ckpt:
        global_step = load_checkpoint(optimizer, scheduler, None, resume_ckpt)
    samples_to_skip = global_step * acc   # re-skip already-trained samples

    # Training loop
    model.train()
    optimizer.zero_grad()
    sample_idx = 0
    accum_loss = accum_acc = 0.0

    logger.info("Starting DPO training ...")
    for epoch in range(1, cfg.num_train_epochs + 1):
        random.shuffle(train_data)

        for rec in tqdm(train_data, desc=f"Epoch {epoch}"):

            # Skip samples already covered before the checkpoint
            if samples_to_skip > 0:
                samples_to_skip -= 1
                continue

            enc_c, lbl_c = encode_single(
                processor, rec["prompt"], rec["chosen"],
                rec["images"], cfg.max_seq_length, device,
            )
            enc_r, lbl_r = encode_single(
                processor, rec["prompt"], rec["rejected"],
                rec["images"], cfg.max_seq_length, device,
            )

            loss_val, log_ratio = dpo_step(
                model, enc_c, lbl_c, enc_r, lbl_r,
                cfg.beta, acc_steps=acc,
            )

            accum_loss += loss_val.item() / acc
            accum_acc  += float(log_ratio.item() > 0) / acc
            sample_idx += 1
            
            if sample_idx % acc != 0:
                continue

            torch.cuda.empty_cache()

            # ── Optimiser step ────────────────────────────────────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % cfg.logging_steps == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"step {global_step:5d} | epoch {epoch} "
                    f"| loss {accum_loss:.4f} | acc {accum_acc:.3f} | lr {lr:.2e}"
                )
                log_metric({
                    "train/loss":     accum_loss,
                    "train/accuracy": accum_acc,
                    "train/lr":       lr,
                    "train/epoch":    epoch,
                }, global_step)

            accum_loss = accum_acc = 0.0

            if val_data and global_step % cfg.eval_steps == 0:
                eval_metrics = evaluate(
                    model, processor, val_data,
                    cfg.beta, cfg.max_seq_length, device, cfg.eval_samples,
                )
                logger.info("  eval | " + " | ".join(
                    f"{k.split('/')[-1]}: {v:.4f}" for k, v in eval_metrics.items()
                ))
                log_metric(eval_metrics, global_step)
                best_tracker.update(eval_metrics, model, processor)

            if global_step % cfg.save_steps == 0:
                save_checkpoint(model, processor, optimizer, scheduler,
                                None, global_step, cfg.output_dir)

    # Final save
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model → {final_dir}")

    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()