import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'       # PyTorch >= 2.x recent
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # PyTorch < 2.x fallback

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
import gc

"""
Memory Optimization Strategy (OOM fixes applied 2026-05-23):

ROOT CAUSE: OOM at `logits = model(...)` in completion_log_prob() during the
chosen forward pass WITH gradients.  Storing activations for backprop roughly
doubles the GPU memory needed compared to inference, which was the tipping point.

Applied fixes (in rough order of impact):
1. ✓ max_seq_length 2048 → 1024   — halves activation memory per forward pass
2. ✓ Chunked log_softmax          — avoids materialising (seq, vocab) as float32
3. ✓ lora_r 16 → 8                — matches pretraining config, fewer trainable params
4. ✓ Gradient checkpointing       — tried at setup; skipped gracefully if unsupported
5. ✓ Aggressive GC / empty_cache  — between every major tensor release
6. ✓ dataloader_num_workers=0 / pin_memory=False (from pretraining script)

If OOM persists, try (in order):
  a) max_seq_length = 512
  b) lora_r = 4
  c) gradient_accumulation_steps > 1 (trade throughput for memory)
"""

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
    model_path: str = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/pretraining/deepseek-vl2-mimic-cxr-lora-r8-lr1e-4-3epochs-cosine-5pctwarmup-6earlystop-p10-p11-p12-p13-p15-6vision-final"
    base_model_path: str = "deepseek-ai/deepseek-vl2-small"
    output_dir: str = "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo/deepseek-vl2-mimic-cxr-dpo-lora-r8-lr5e-5-beta0.1-model16-datasetp18-25pct-lambda1e-2"

    # Training
    num_train_epochs:            int   = 3
    gradient_accumulation_steps: int   = 1
    learning_rate:               float = 5e-5
    weight_decay:                float = 0.01
    warmup_ratio:                float = 0.1
    max_grad_norm:               float = 1.0
    # *** OOM FIX #1: reduced from 2048 → 1024 (halves activation memory) ***
    max_seq_length:              int   = 1024
    beta:                        float = 0.1

    # LoRA — *** OOM FIX #3: lora_r 16 → 8 to match pretraining, fewer params ***
    lora_r:              int   = 8
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
    wandb_project: str = "deepseek-vl2-mimic-cxr-dpo"
    wandb_run:     str = "deepseek-vl2-mimic-cxr-dpo-lora-r8-lr5e-5-beta0.1-model16-datasetp18-25pct"


def parse_args() -> TrainingConfig:
    defaults = TrainingConfig()
    p = argparse.ArgumentParser()
    for f_name, f_val in vars(defaults).items():
        p.add_argument(f"--{f_name}", type=type(f_val) if not isinstance(f_val, list) else None,
                       default=f_val)
    ns = p.parse_args()
    return TrainingConfig(**{k: v for k, v in vars(ns).items()})


def patch_prepare_inputs_embeds(model):
    """
    Monkey-patch prepare_inputs_embeds to cast image features to match
    the dtype of input embeddings, fixing the masked_scatter_ dtype error.
    """
    base = model.base_model.model if hasattr(model, 'base_model') else model
    original_fn = base.prepare_inputs_embeds.__func__

    def patched_prepare_inputs_embeds(self, *args, **kwargs):
        original_masked_scatter = torch.Tensor.masked_scatter_

        def safe_masked_scatter_(self_t, mask, source):
            return original_masked_scatter(self_t, mask, source.to(dtype=self_t.dtype))

        torch.Tensor.masked_scatter_ = safe_masked_scatter_
        try:
            result = original_fn(self, *args, **kwargs)
        finally:
            torch.Tensor.masked_scatter_ = original_masked_scatter
        return result

    import types
    base.prepare_inputs_embeds = types.MethodType(patched_prepare_inputs_embeds, base)

def _enable_gradient_checkpointing(model) -> bool:
    """
    Apply torch.utils.checkpoint to each decoder layer individually.
    Avoids the DeepSeek-VL2 incompatibility with the HF Trainer flag,
    which breaks on prepare_inputs_embeds / masked_scatter_.
    Returns True if successful.
    """
    import torch.utils.checkpoint as ckpt_utils

    # PEFT requirement: inputs must require grad so checkpointing can
    # re-run the forward and recompute activations during backward.
    model.enable_input_require_grads()

    # Navigate to the transformer decoder layers
    inner = model.base_model.model if hasattr(model, "base_model") else model

    candidate_paths = [
        # DeepSeek-VL2 typical layout
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
    ]

    layers = None
    for path in candidate_paths:
        try:
            layers = path(inner)
            if layers:
                break
        except AttributeError:
            continue

    if layers is None:
        print("Could not locate decoder layers for gradient checkpointing.", flush=True)
        return False

    # Wrap each layer's forward with checkpoint
    def make_ckpt_forward(layer):
        orig_fwd = layer.__class__.forward

        def ckpt_forward(self, *args, **kwargs):
            # use_reentrant=False avoids issues with non-tensor kwargs
            def fn(*a):
                return orig_fwd(self, *a, **kwargs)
            return ckpt_utils.checkpoint(fn, *args, use_reentrant=False)

        return ckpt_forward

    for layer in layers:
        layer.__class__.forward = make_ckpt_forward(layer)

    print(f"Gradient checkpointing enabled on {len(layers)} decoder layers.", flush=True)
    return True


# ── Model loading ──────────────────────────────────────────────────────────────
def setup_model(model_path: str, base_model_path: str = "deepseek-ai/deepseek-vl2-small"):
    """Load DeepSeek-VL2 with optional LoRA adapter."""
    print("Loading base model...", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if has_adapter:
        print(f"Loading PEFT model with existing adapter from {model_path}...", flush=True)

        with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
            adapter_config = json.load(f)
            if "base_model_name_or_path" in adapter_config:
                base_model_path = adapter_config["base_model_name_or_path"]
                print(f"Base model path from adapter config: {base_model_path}", flush=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,  # from pretraining script
        )

        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
        try:
            ok = _enable_gradient_checkpointing(model)
            if not ok:
                print("Gradient checkpointing skipped, proceeding without it.", flush=True)
        except Exception as e:
            print(f"Gradient checkpointing failed ({e}), proceeding without it.", flush=True)
        print("PEFT adapter loaded successfully.", flush=True)
    else:
        print(f"Loading base model from {model_path}...", flush=True)
        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,  # from pretraining script
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,          # *** OOM FIX #3: reduced from 16 to 8 ***
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            inference_mode=False,
        )

        print("Applying new LoRA adapter...", flush=True)
        model = get_peft_model(model, lora_config)
        try:
            ok = _enable_gradient_checkpointing(model)
            if not ok:
                print("Gradient checkpointing skipped, proceeding without it.", flush=True)
        except Exception as e:
            print(f"Gradient checkpointing failed ({e}), proceeding without it.", flush=True)

    # *** OOM FIX #4: attempt gradient checkpointing on the language model ***
    # DeepSeek-VL2 may not support it globally, but try on just the LM backbone
    try:
        # Try enabling on the inner language model if accessible
        inner_lm = model.base_model.model if hasattr(model, 'base_model') else model
        if hasattr(inner_lm, 'language_model'):
            inner_lm.language_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled on language_model.", flush=True)
        elif hasattr(inner_lm, 'model'):
            inner_lm.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled on inner model.", flush=True)
        else:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled (global).", flush=True)
    except Exception as e:
        print(f"Gradient checkpointing not available ({e}), skipping.", flush=True)

    print(f"Loading processor/tokenizer from {base_model_path}...", flush=True)
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
    """Tokenise `prompt + completion` for one sample with prompt tokens masked."""
    images = pil_images if pil_images else None

    full_enc = processor(
        conversations=prompt_text + completion_text,
        images=images,
        force_batchify=True,
        system_prompt="",
        # *** FIX: pad to actual length, not max_length ***
        # With batch_size=1, "longest" == actual sequence length.
        # A 1000-token sample no longer wastes memory on 1048 padding tokens.
        padding="longest",       # was: "max_length"
        truncation=True,
        max_length=max_length,
        inference_mode=False,
    )
    full_enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in vars(full_enc).items()}

    if full_enc.get("images") is not None and isinstance(full_enc["images"], torch.Tensor):
        full_enc["images"] = full_enc["images"].to(dtype=torch.bfloat16)

    assistant_token = "<|Assistant|>"
    assistant_token_ids = processor.tokenizer.encode(assistant_token, add_special_tokens=False)
    input_ids_list = full_enc["input_ids"][0].tolist()

    prompt_len = None
    for i in range(len(input_ids_list) - len(assistant_token_ids) + 1):
        if input_ids_list[i:i+len(assistant_token_ids)] == assistant_token_ids:
            prompt_len = i + len(assistant_token_ids)
            break

    if prompt_len is None:
        logger.warning("Could not find assistant token in sequence, using fallback")
        num_images = len(pil_images) if pil_images else 0
        prompt_len = 10 + (num_images * 576) + 30

    labels = full_enc["input_ids"].clone()
    labels[0, :prompt_len] = -100

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

    *** OOM FIX #2: chunked log_softmax ***
    After the forward pass we process the logit tensor in small sequence-length
    chunks (chunk_size tokens at a time) so we never materialise the full
    (seq_len, vocab_size) float32 matrix at once.  Peak extra memory drops from
    ~2 × logits_bf16 to ~chunk_size/seq_len × logits_bf16.
    """
    images = enc.get("images")
    if images is not None and isinstance(images, torch.Tensor):
        images = images.to(dtype=torch.bfloat16)

    ctx = (autocast(dtype=torch.bfloat16)
           if use_autocast
           else contextmanager(lambda: iter([None]))())
    with ctx:
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            images=images,
            images_seq_mask=enc.get("images_seq_mask"),
            images_spatial_crop=enc.get("images_spatial_crop"),
            return_dict=True,
        ).logits  # (batch, seq_len, vocab)

    # Causal shift
    shift_logits = logits[:, :-1, :].contiguous()    # (1, L-1, V) in bfloat16
    shift_labels = labels[:, 1:].contiguous()        # (1, L-1)
    del logits
    torch.cuda.empty_cache()

    # *** Chunked log_softmax — avoids one large float32 tensor ***
    # Accumulate into a scalar so the gradient graph stays intact.
    seq_len = shift_logits.shape[1]
    chunk_size = 64  # tune down to 32 if still OOM
    total_log_prob = shift_logits.new_zeros(())  # scalar, same device/dtype
    has_grad = shift_logits.requires_grad        # True during chosen pass

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logits = shift_logits[:, start:end, :]  # (1, chunk, V) bfloat16
        chunk_labels = shift_labels[:, start:end]     # (1, chunk)

        # float32 log_softmax on a small (chunk, V) tensor — manageable
        chunk_lp = F.log_softmax(chunk_logits.float(), dim=-1)
        del chunk_logits

        token_lp = chunk_lp.gather(
            -1, chunk_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)                                  # (1, chunk)
        del chunk_lp

        mask = (chunk_labels != -100).float()
        total_log_prob = total_log_prob + (token_lp * mask).sum()
        del token_lp, mask

    del shift_logits, shift_labels
    torch.cuda.empty_cache()

    return total_log_prob


# ── Reference model context ────────────────────────────────────────────────────
@contextmanager
def reference_mode(model):
    """Disable LoRA so frozen base weights act as π_ref (no second model copy)."""
    with torch.no_grad():
        with model.disable_adapter():
            yield


# ── DPO loss ───────────────────────────────────────────────────────────────────
def to_device(enc: dict, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in enc.items()}


def dpo_step(
    model,
    enc_chosen,   lbl_chosen,
    enc_rejected, lbl_rejected,
    log_ref_c: torch.Tensor,
    log_ref_r: torch.Tensor,
    beta: float,
    acc_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    log_ref_c = log_ref_c.to(device)
    log_ref_r = log_ref_r.to(device)

    # ── Pass 1: rejected no_grad peek ────────────────────────────────────────
    # Offload chosen to CPU while we peek at rejected
    enc_chosen_cpu  = to_device(enc_chosen,  "cpu")
    lbl_chosen_cpu  = lbl_chosen.cpu()
    del enc_chosen, lbl_chosen
    torch.cuda.empty_cache()
    gc.collect()  # *** OOM FIX #5: gc after every offload ***

    with torch.no_grad():
        log_pi_r_val = completion_log_prob(
            model, enc_rejected, lbl_rejected, use_autocast=True
        ).detach()
        torch.cuda.empty_cache()
        gc.collect()

    # Offload rejected; bring chosen back
    enc_rejected_cpu  = to_device(enc_rejected,  "cpu")
    lbl_rejected_cpu  = lbl_rejected.cpu()
    del enc_rejected, lbl_rejected
    torch.cuda.empty_cache()
    gc.collect()

    enc_chosen = to_device(enc_chosen_cpu, device)
    lbl_chosen = lbl_chosen_cpu.to(device)
    del enc_chosen_cpu, lbl_chosen_cpu
    torch.cuda.empty_cache()
    gc.collect()

    # ── Pass 2: chosen forward + backward ────────────────────────────────────
    log_pi_c = completion_log_prob(model, enc_chosen, lbl_chosen)
    del enc_chosen, lbl_chosen
    torch.cuda.empty_cache()
    gc.collect()

    log_ratio = (log_pi_c.detach() - log_ref_c) - (log_pi_r_val - log_ref_r)
    loss_val  = -F.logsigmoid(beta * log_ratio)

    grad_c = beta * (torch.sigmoid(beta * log_ratio) - 1) / acc_steps
    log_pi_c.backward(grad_c)
    del log_pi_c, grad_c
    torch.cuda.empty_cache()
    gc.collect()

    # ── Pass 3: rejected forward + backward ──────────────────────────────────
    enc_rejected = to_device(enc_rejected_cpu, device)
    lbl_rejected = lbl_rejected_cpu.to(device)
    del enc_rejected_cpu, lbl_rejected_cpu
    torch.cuda.empty_cache()
    gc.collect()

    log_pi_r = completion_log_prob(model, enc_rejected, lbl_rejected)
    del enc_rejected, lbl_rejected
    torch.cuda.empty_cache()
    gc.collect()

    grad_r = beta * (1 - torch.sigmoid(beta * log_ratio)) / acc_steps
    log_pi_r.backward(grad_r)
    del log_pi_r, grad_r
    torch.cuda.empty_cache()
    gc.collect()

    return loss_val, log_ratio.detach()


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


def save_checkpoint(model, processor, optimizer, scheduler, scaler, step: int,
                    out_dir: str, max_checkpoints: int = None):
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
    if max_checkpoints is not None and max_checkpoints > 0:
        cleanup_old_checkpoints(out_dir, max_checkpoints)


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
    state = torch.load(opt_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
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


# ── Reference log-prob precomputation ─────────────────────────────────────────
@torch.no_grad()
def precompute_reference_logprobs(
    ref_model_path: str,
    base_model_path: str,
    all_records: list,
    processor,
    max_seq_length: int,
    device: torch.device,
    cache_path: str,
) -> dict:
    cache_file = Path(cache_path) / "ref_logprobs.pt"
    if cache_file.exists():
        logger.info(f"Loading cached reference log-probs from {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    logger.info("Loading reference model for log-prob precomputation...")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    ref_model = PeftModel.from_pretrained(base, ref_model_path, is_trainable=False)
    patch_prepare_inputs_embeds(ref_model)
    ref_model.eval()

    cache = {}
    for i, rec in enumerate(tqdm(all_records, desc="Precomputing ref log-probs")):
        try:
            enc_c, lbl_c = encode_single(
                processor, rec["prompt"], rec["chosen"],
                rec["images"], max_seq_length, device,
            )
            log_ref_c = completion_log_prob(ref_model, enc_c, lbl_c, use_autocast=False)
            torch.cuda.empty_cache()
            gc.collect()
            del enc_c, lbl_c

            enc_r, lbl_r = encode_single(
                processor, rec["prompt"], rec["rejected"],
                rec["images"], max_seq_length, device,
            )
            log_ref_r = completion_log_prob(ref_model, enc_r, lbl_r, use_autocast=False)
            torch.cuda.empty_cache()
            gc.collect()
            del enc_r, lbl_r

            cache[i] = {
                "log_ref_c": log_ref_c.cpu(),
                "log_ref_r": log_ref_r.cpu(),
            }
            del log_ref_c, log_ref_r
            if i % 10 == 0:
                gc.collect()

        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_file)
    logger.info(f"Reference log-probs cached → {cache_file}")

    del ref_model, base
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Reference model deleted from memory.")

    return cache


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def dpo_loss_eval(
    model,
    enc_chosen, lbl_chosen,
    enc_rejected, lbl_rejected,
    log_ref_c, log_ref_r,
    beta,
    device,
):
    enc_rejected_cpu = to_device(enc_rejected, "cpu")
    lbl_rejected_cpu = lbl_rejected.cpu()

    log_pi_c = completion_log_prob(model, enc_chosen, lbl_chosen)
    torch.cuda.empty_cache()

    enc_rejected = to_device(enc_rejected_cpu, device)
    lbl_rejected = lbl_rejected_cpu.to(device)

    log_pi_r = completion_log_prob(model, enc_rejected, lbl_rejected)
    torch.cuda.empty_cache()

    log_ratio = (log_pi_c - log_ref_c.to(device)) - (log_pi_r - log_ref_r.to(device))
    loss = -F.logsigmoid(beta * log_ratio)
    return loss, log_ratio


@torch.no_grad()
def evaluate(
    model, processor, eval_records, val_cache, beta,
    max_seq_length, device, max_samples,
):
    model.eval()
    records = eval_records[:max_samples]
    total_loss = total_acc = total_ratio = 0.0

    for i, rec in enumerate(tqdm(records, desc="Eval", leave=False)):
        enc_c, lbl_c = encode_single(processor, rec["prompt"], rec["chosen"],
                                     rec["images"], max_seq_length, device)
        enc_r, lbl_r = encode_single(processor, rec["prompt"], rec["rejected"],
                                     rec["images"], max_seq_length, device)

        cached = val_cache[i]
        loss, log_ratio = dpo_loss_eval(
            model, enc_c, lbl_c, enc_r, lbl_r,
            log_ref_c=cached["log_ref_c"],
            log_ref_r=cached["log_ref_r"],
            beta=beta,
            device=device,
        )

        total_loss  += loss.item()
        total_acc   += float(log_ratio.item() > 0)
        total_ratio += log_ratio.item()

        del enc_c, lbl_c, enc_r, lbl_r, loss, log_ratio
        torch.cuda.empty_cache()
        gc.collect()

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
    patch_prepare_inputs_embeds(model)
    processor.tokenizer.padding_side = "right"

    # Data
    logger.info("Loading preference dataset ...")
    preference_data = load_preference_dataset(
        "/pfss/mlde/workspaces/mlde_wsp_RWTH_MedReport/ag88juba/RadVLM/results/dpo_dataset/model16-p18reports-1e-2lambda-with-study-ids.json"
    )
    print(f"Loaded {len(preference_data)} raw data points.")
    dataset_wrapper = RadVLMDPODataset(
        preference_data, processor, tokenizer,
        max_seq_length=cfg.max_seq_length,
    )
    train_data = dataset_wrapper.get_split_data("train")
    val_data   = dataset_wrapper.get_split_data("validate")
    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")
    all_data = train_data + val_data

    ref_cache = precompute_reference_logprobs(
        ref_model_path=cfg.model_path,
        base_model_path=cfg.base_model_path,
        all_records=all_data,
        processor=processor,
        max_seq_length=cfg.max_seq_length,
        device=device,
        cache_path=cfg.output_dir,
    )

    train_cache = {i: ref_cache[i] for i in range(len(train_data))}
    val_cache   = {i: ref_cache[len(train_data) + i] for i in range(len(val_data))}

    # Optimiser & scheduler
    # At the top of main(), replace the AdamW import/init:
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        logger.info("Using 8-bit AdamW (bitsandbytes).")
    except ImportError:
        logger.warning("bitsandbytes not available, falling back to 32-bit AdamW.")
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

    # Resume from checkpoint
    global_step = 0
    resume_ckpt = find_latest_checkpoint(cfg.output_dir)
    if resume_ckpt:
        global_step = load_checkpoint(optimizer, scheduler, None, resume_ckpt)
    samples_to_skip = global_step * acc

    # Training loop
    model.train()
    optimizer.zero_grad()
    sample_idx = 0
    accum_loss = accum_acc = 0.0

    logger.info("Starting DPO training ...")
    epoch_pbar = tqdm(range(1, cfg.num_train_epochs + 1), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{cfg.num_train_epochs}")

        indices = list(range(len(train_data)))
        random.shuffle(indices)

        step_pbar = tqdm(
            indices, desc=f"Epoch {epoch} steps", position=1, leave=False,
            disable=not sys.stderr.isatty()
        )
        for idx in step_pbar:
            rec = train_data[idx]

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
                model, enc_c, lbl_c, enc_r, lbl_r,
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

            torch.cuda.empty_cache()
            gc.collect()

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
                    model, processor, val_data, val_cache,
                    cfg.beta, cfg.max_seq_length, device, cfg.eval_samples,
                )
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("  eval | " + " | ".join(
                    f"{k.split('/')[-1]}: {v:.4f}" for k, v in eval_metrics.items()
                ))
                log_metric(eval_metrics, global_step)
                best_tracker.update(eval_metrics, model, processor)

            if global_step % cfg.save_steps == 0:
                save_checkpoint(
                    model, processor, optimizer, scheduler,
                    None, global_step, cfg.output_dir, cfg.max_checkpoints
                )

    # Final save
    final_dir = Path(cfg.output_dir) / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model → {final_dir}")

    import wandb
    wandb.finish()


if __name__ == "__main__":
    main()