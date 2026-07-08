"""
DPO Fine-tuning for MedGemma — memory-efficient, variable image count per sample.

Key improvements over the original:
  1. Reference log-probs are precomputed once, cached to disk, then the ref
     model is deleted before training starts → only ONE model in GPU memory.
  2. Chosen and rejected forward passes are run sequentially with manual
     gradient accumulation (analytic gradients), so only one computation
     graph exists at a time.
  3. GradScaler removed — MedGemma already runs in bfloat16, which does not
     require loss scaling.
  4. torch.cuda.empty_cache() called between every forward pass.

Usage:
    python train_dpo_medgemma.py \
        --model_path  ../../results/pretraining/medgemma-sft-final \
        --output_dir  ../../results/dpo/medgemma-dpo-run1
"""

import os
import gc
import math
import argparse
import logging
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tqdm import tqdm

from src.radvlm.data.build_dataset import load_preference_dataset
from src.radvlm.data.medgemma_dpo_dataset import RadVLMDPODatasetMedGemma
from src.radvlm.utils.config import MEDGEMMA_BASE_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # Paths
    model_path: str = (
        "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/medgemma-1.5-mimic-cxr-poc-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-100pctdata-final"
    )
    output_dir: str = (
        "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo/medgemma-1.5-mimic-cxr-dpo-lora-r16-lr5e-5-beta0.1-model15-3pctdataset-frommodel8-1e-2lambda-discrete-processed"
    )

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
    logging_steps:   int = 50
    eval_steps:      int = 200
    save_steps:      int = 200
    eval_samples:    int = 64
    max_checkpoints: int = 3
    seed:            int = 42

    # W&B
    wandb_project: str = "medgemma-1.5-mimic-cxr-dpo"
    wandb_run:     str = (
        "medgemma-1.5-mimic-cxr-dpo-lora-r16-lr5e-5-beta0.1-model15-3pctdataset-frommodel8-1e-2lambda-discrete-processed"
    )


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    p = argparse.ArgumentParser()
    for f_name, f_val in vars(defaults).items():
        p.add_argument(
            f"--{f_name}",
            type=type(f_val) if not isinstance(f_val, list) else None,
            default=f_val,
        )
    ns = p.parse_args()
    return TrainingConfig(**{k: v for k, v in vars(ns).items()})


# ── Model loading ──────────────────────────────────────────────────────────────
def _build_base_model() -> torch.nn.Module:
    """Load the raw MedGemma weights (no LoRA)."""
    logger.info(f"Loading base model from {MEDGEMMA_BASE_MODEL_PATH} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MEDGEMMA_BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    # Freeze vision encoder
    for name, param in model.named_parameters():
        if any(k in name for k in ("vision_tower", "visual", "vision_model")):
            param.requires_grad = False
    logger.info("Vision encoder frozen.")
    return model


def setup_model(cfg: TrainingConfig) -> torch.nn.Module:
    """
    Load base MedGemma, freeze the vision encoder, then either resume an
    existing LoRA checkpoint or attach fresh LoRA adapters.
    """
    model_path   = os.path.abspath(cfg.model_path)
    base_model   = _build_base_model()
    is_lora_ckpt = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora_ckpt:
        logger.info(f"Resuming LoRA adapters from {model_path} ...")
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=cfg.lora_target_modules,
            inference_mode=False,
        )
        model = get_peft_model(base_model, lora_cfg)

    model.print_trainable_parameters()
    return model


def load_reference_model(model_path: str) -> torch.nn.Module:
    """
    Load the SFT checkpoint as a frozen reference model.
    Caller is responsible for deleting it after precomputation.
    """
    base      = _build_base_model()
    model_path = os.path.abspath(model_path)
    is_lora   = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        ref_model = PeftModel.from_pretrained(base, model_path, is_trainable=False)
    else:
        # If the checkpoint is a full model, just wrap it
        ref_model = base

    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()
    logger.info("Reference model loaded (frozen).")
    return ref_model


# ── Encoding ───────────────────────────────────────────────────────────────────
def encode_single(
    processor,
    prompt_text: str,
    completion_text: str,
    pil_images: list,
    max_length: int,
    device: torch.device,
) -> tuple[dict, torch.Tensor]:
    """
    Tokenise `prompt + completion` for one sample with any number of images.

    Prompt-masking strategy: tokenise the completion alone (no images) to get
    its token count, then derive prompt_len = total_len - completion_len.
    This avoids processing images twice.
    """
    images = pil_images if pil_images else None

    full_enc = processor(
        text=prompt_text + completion_text,
        images=images,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    full_enc = {k: v.to(device) for k, v in full_enc.items()}
    if full_enc.get("pixel_values") is not None:
        full_enc["pixel_values"] = full_enc["pixel_values"].to(dtype=torch.bfloat16)

    # Tokenise only the completion (no images) to count its tokens.
    # Subtract 1 to exclude the BOS token the processor prepends.
    completion_enc = processor(
        text=completion_text,
        images=None,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    completion_len = int(completion_enc["attention_mask"].sum().item()) - 1
    total_len      = int(full_enc["attention_mask"].sum().item())
    prompt_len     = total_len - completion_len

    labels             = full_enc["input_ids"].clone()
    labels[0, :prompt_len] = -100  # mask prompt tokens from the loss

    return full_enc, labels


# ── Log-probability (sum over completion tokens) ──────────────────────────────
def completion_log_prob(
    model,
    enc: dict,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Return the *sum* of log-probs over non-masked completion tokens.

    Using sum rather than mean avoids length bias: with mean log-prob,
    DPO would implicitly favour shorter completions.

    Always runs in bfloat16 autocast — no GradScaler needed.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=enc.get("pixel_values"),
            return_dict=True,
        ).logits  # (1, seq_len, vocab)

    # Causal shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()   # (1, L-1, V)
    shift_labels = labels[:, 1:].contiguous()         # (1, L-1)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_lp  = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    comp_mask = (shift_labels != -100).float()

    return (token_lp * comp_mask).sum()  # scalar


# ── Reference log-prob precomputation ─────────────────────────────────────────
@torch.no_grad()
def precompute_reference_logprobs(
    cfg: TrainingConfig,
    all_records: list,
    processor,
    device: torch.device,
) -> dict:
    """
    Load the SFT reference model, compute log-probs for every sample in
    all_records, cache them to disk, then delete the model.

    Returns a dict  { index -> {"log_ref_c": Tensor, "log_ref_r": Tensor} }
    with tensors on CPU.
    """
    cache_file = Path(cfg.output_dir) / "ref_logprobs.pt"
    if cache_file.exists():
        logger.info(f"Loading cached reference log-probs from {cache_file}")
        return torch.load(cache_file, map_location="cpu")

    logger.info("Loading reference model for log-prob precomputation ...")
    ref_model = load_reference_model(cfg.model_path)

    cache = {}
    for i, rec in enumerate(tqdm(all_records, desc="Precomputing ref log-probs")):
        enc_c, lbl_c = encode_single(
            processor, rec["prompt"], rec["chosen"],
            rec["images"], cfg.max_seq_length, device,
        )
        enc_r, lbl_r = encode_single(
            processor, rec["prompt"], rec["rejected"],
            rec["images"], cfg.max_seq_length, device,
        )
        log_ref_c = completion_log_prob(ref_model, enc_c, lbl_c)
        torch.cuda.empty_cache()
        log_ref_r = completion_log_prob(ref_model, enc_r, lbl_r)
        torch.cuda.empty_cache()

        cache[i] = {
            "log_ref_c": log_ref_c.cpu(),
            "log_ref_r": log_ref_r.cpu(),
        }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_file)
    logger.info(f"Reference log-probs cached → {cache_file}")

    # Free GPU memory before training starts
    del ref_model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Reference model deleted from memory.")

    return cache


# ── Memory-efficient DPO step (analytic gradients) ────────────────────────────
def dpo_step(
    model,
    enc_chosen,   lbl_chosen,
    enc_rejected, lbl_rejected,
    log_ref_c: torch.Tensor,  # CPU scalar, precomputed
    log_ref_r: torch.Tensor,  # CPU scalar, precomputed
    beta: float,
    acc_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient DPO step: only ONE policy computation graph lives in
    memory at a time, achieved via manual gradient accumulation with analytic
    gradients (no autograd through the DPO sigmoid).

    Returns:
        loss_val  — detached scalar for logging
        log_ratio — detached scalar for accuracy logging
    """
    device    = enc_chosen["input_ids"].device
    log_ref_c = log_ref_c.to(device)
    log_ref_r = log_ref_r.to(device)

    # ── 1. Peek at rejected value without building a graph ────────────────
    with torch.no_grad():
        log_pi_r_val = completion_log_prob(model, enc_rejected, lbl_rejected).detach()
    torch.cuda.empty_cache()

    # ── 2. Chosen forward (graph lives here) ──────────────────────────────
    log_pi_c  = completion_log_prob(model, enc_chosen, lbl_chosen)
    log_ratio = (log_pi_c.detach() - log_ref_c) - (log_pi_r_val - log_ref_r)
    loss_val  = -F.logsigmoid(beta * log_ratio)

    # Analytic gradient: ∂loss/∂log_pi_c = β·(σ(β·r) - 1)
    grad_c = beta * (torch.sigmoid(beta * log_ratio) - 1) / acc_steps
    log_pi_c.backward(grad_c)
    del log_pi_c
    torch.cuda.empty_cache()

    # ── 3. Rejected forward (chosen graph is already freed) ───────────────
    log_pi_r = completion_log_prob(model, enc_rejected, lbl_rejected)

    # Analytic gradient: ∂loss/∂log_pi_r = β·(1 - σ(β·r))
    grad_r = beta * (1 - torch.sigmoid(beta * log_ratio)) / acc_steps
    log_pi_r.backward(grad_r)
    del log_pi_r
    torch.cuda.empty_cache()

    return loss_val, log_ratio.detach()


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def dpo_loss_eval(
    model,
    enc_chosen,   lbl_chosen,
    enc_rejected, lbl_rejected,
    log_ref_c: torch.Tensor,
    log_ref_r: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device    = enc_chosen["input_ids"].device
    log_pi_c  = completion_log_prob(model, enc_chosen,   lbl_chosen)
    torch.cuda.empty_cache()
    log_pi_r  = completion_log_prob(model, enc_rejected, lbl_rejected)
    torch.cuda.empty_cache()

    log_ratio = (log_pi_c - log_ref_c.to(device)) - (log_pi_r - log_ref_r.to(device))
    loss      = -F.logsigmoid(beta * log_ratio)
    return loss, log_ratio


@torch.no_grad()
def evaluate(
    model,
    processor,
    eval_records: list,
    val_cache: dict,
    beta: float,
    max_seq_length: int,
    device: torch.device,
    max_samples: int,
) -> dict:
    model.eval()
    records     = eval_records[:max_samples]
    total_loss  = total_acc = total_ratio = 0.0

    for i, rec in enumerate(tqdm(records, desc="Eval", leave=False)):
        enc_c, lbl_c = encode_single(
            processor, rec["prompt"], rec["chosen"],
            rec["images"], max_seq_length, device,
        )
        enc_r, lbl_r = encode_single(
            processor, rec["prompt"], rec["rejected"],
            rec["images"], max_seq_length, device,
        )
        cached    = val_cache[i]
        loss, log_ratio = dpo_loss_eval(
            model, enc_c, lbl_c, enc_r, lbl_r,
            log_ref_c=cached["log_ref_c"],
            log_ref_r=cached["log_ref_r"],
            beta=beta,
        )
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


# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def cleanup_old_checkpoints(out_dir: str, max_checkpoints: int):
    ckpts = sorted(
        Path(out_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if len(ckpts) > max_checkpoints:
        import shutil
        for old_ckpt in ckpts[:-max_checkpoints]:
            shutil.rmtree(old_ckpt)
            logger.info(f"Removed old checkpoint: {old_ckpt}")


def save_checkpoint(
    model, processor, optimizer, scheduler,
    step: int, out_dir: str, max_checkpoints: int = None,
):
    ckpt_dir = Path(out_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(ckpt_dir)
    torch.save(
        {
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "global_step": step,
        },
        ckpt_dir / "optimizer.pt",
    )
    logger.info(f"Checkpoint saved → {ckpt_dir}")
    if max_checkpoints is not None and max_checkpoints > 0:
        cleanup_old_checkpoints(out_dir, max_checkpoints)


def find_latest_checkpoint(out_dir: str) -> Optional[Path]:
    ckpts = sorted(
        Path(out_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return ckpts[-1] if ckpts else None


def load_checkpoint(optimizer, scheduler, ckpt_dir: Path) -> int:
    opt_path = ckpt_dir / "optimizer.pt"
    if not opt_path.exists():
        logger.warning(f"No optimizer state in {ckpt_dir} — starting fresh.")
        return 0
    state = torch.load(opt_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info(f"Resumed from step {state['global_step']}")
    return state["global_step"]


# ── Best-model tracking ────────────────────────────────────────────────────────
class BestModelTracker:
    def __init__(self, out_dir: str, metric: str = "eval/loss", mode: str = "min"):
        self.save_dir  = Path(out_dir) / "best_model"
        self.metric    = metric
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.is_better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)

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


# ── W&B reporter ───────────────────────────────────────────────────────────────
def setup_reporter(cfg: TrainingConfig):
    import wandb
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run,
        config={
            "model":          MEDGEMMA_BASE_MODEL_PATH,
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

    # ── Processor (needed for precomputation too) ──────────────────────────
    processor = AutoProcessor.from_pretrained(
        MEDGEMMA_BASE_MODEL_PATH, local_files_only=True
    )
    processor.tokenizer.padding_side = "right"

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Loading preference dataset ...")
    preference_data = load_preference_dataset(
        "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/medgemma-model8-3pctdatasetreports-1e-2lambda-discrete-processed.json"
    )
    dataset_wrapper = RadVLMDPODatasetMedGemma(
        preference_data, processor, processor.tokenizer,
        max_seq_length=cfg.max_seq_length,
    )
    train_data = dataset_wrapper.get_split_data("train")
    val_data   = dataset_wrapper.get_split_data("validate")
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")
    all_data   = train_data + val_data

    # ── Precompute reference log-probs, then free the ref model ───────────
    ref_cache   = precompute_reference_logprobs(cfg, all_data, processor, device)
    train_cache = {i: ref_cache[i] for i in range(len(train_data))}
    val_cache   = {i: ref_cache[len(train_data) + i] for i in range(len(val_data))}

    # ── Policy model (loaded AFTER ref model is freed) ────────────────────
    model = setup_model(cfg)
    model.train()

    # ── Optimiser & scheduler ─────────────────────────────────────────────
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

    # ── Resume from checkpoint if available ───────────────────────────────
    global_step = 0
    resume_ckpt = find_latest_checkpoint(cfg.output_dir)
    if resume_ckpt:
        global_step = load_checkpoint(optimizer, scheduler, resume_ckpt)
    samples_to_skip = global_step * acc

    # ── Training loop ─────────────────────────────────────────────────────
    optimizer.zero_grad()
    sample_idx = 0
    accum_loss = accum_acc = 0.0

    logger.info("Starting DPO training ...")
    epoch_pbar = tqdm(
        range(1, cfg.num_train_epochs + 1), desc="Epochs", position=0
    )
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{cfg.num_train_epochs}")

        indices = list(range(len(train_data)))
        random.shuffle(indices)

        step_pbar = tqdm(
            indices,
            desc=f"Epoch {epoch} steps",
            position=1,
            leave=False,
            disable=not sys.stderr.isatty(),
        )
        for idx in step_pbar:
            rec = train_data[idx]

            # Skip samples already trained before the checkpoint
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

            cached = train_cache[idx]
            loss_val, log_ratio = dpo_step(
                model,
                enc_c, lbl_c,
                enc_r, lbl_r,
                log_ref_c=cached["log_ref_c"],
                log_ref_r=cached["log_ref_r"],
                beta=cfg.beta,
                acc_steps=acc,
            )

            accum_loss += loss_val.item() / acc
            accum_acc  += float(log_ratio.item() > 0) / acc
            sample_idx += 1

            step_pbar.set_postfix({
                "loss": f"{accum_loss:.4f}",
                "acc":  f"{accum_acc:.3f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
            })

            if sample_idx % acc != 0:
                continue

            # ── Optimiser step ────────────────────────────────────────────
            torch.cuda.empty_cache()
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
                log_metric(
                    {
                        "train/loss":     accum_loss,
                        "train/accuracy": accum_acc,
                        "train/lr":       lr,
                        "train/epoch":    epoch,
                    },
                    global_step,
                )

            accum_loss = accum_acc = 0.0

            if val_data and global_step % cfg.eval_steps == 0:
                eval_metrics = evaluate(
                    model, processor, val_data, val_cache,
                    cfg.beta, cfg.max_seq_length, device, cfg.eval_samples,
                )
                logger.info(
                    "  eval | "
                    + " | ".join(
                        f"{k.split('/')[-1]}: {v:.4f}"
                        for k, v in eval_metrics.items()
                    )
                )
                log_metric(eval_metrics, global_step)
                best_tracker.update(eval_metrics, model, processor)

            if global_step % cfg.save_steps == 0:
                save_checkpoint(
                    model, processor, optimizer, scheduler,
                    global_step, cfg.output_dir, cfg.max_checkpoints,
                )

    # ── Final save ────────────────────────────────────────────────────────
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model → {final_dir}")

    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()