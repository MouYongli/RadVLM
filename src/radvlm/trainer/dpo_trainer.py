# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import random
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

try:
    from transformers.utils import is_peft_available
except ImportError:
    def is_peft_available():
        try:
            import peft
            return True
        except ImportError:
            return False

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


@dataclass
class PreferenceCollator(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.
        image_processor (`AutoImageProcessor`, *optional*):
            Image processor for vision inputs.
    """

    pad_token_id: int
    return_tensors: str = "pt"
    image_processor: Optional[Any] = None

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Check if data is already batched from custom collate_fn
        if len(examples) == 1 and "chosen_input_ids" in examples[0] and isinstance(examples[0]["chosen_input_ids"], torch.Tensor):
            # Already batched by custom collate_fn, just pass through
            batch = examples[0]
            output = {
                "prompt_input_ids": batch.get("prompt_input_ids"),
                "prompt_attention_mask": batch.get("prompt_attention_mask"),
                "chosen_input_ids": batch["chosen_input_ids"],
                "chosen_attention_mask": batch.get("attention_mask_chosen"),
                "rejected_input_ids": batch["rejected_input_ids"],
                "rejected_attention_mask": batch.get("attention_mask_rejected"),
            }
            # Handle pixel values
            if "pixel_values_chosen" in batch:
                output["pixel_values_chosen"] = batch["pixel_values_chosen"]
            if "pixel_values_rejected" in batch:
                output["pixel_values_rejected"] = batch["pixel_values_rejected"]
            return output
        
        # Standard tokenized examples - convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]

        # Pad
        output = {}
        output["prompt_input_ids"] = self.pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = self.pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = self.pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = self.pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = self.pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = self.pad(rejected_attention_mask, padding_value=0)

        # Handle images if present
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
            output["pixel_values"] = torch.stack(pixel_values)

        return output

    @staticmethod
    def pad(tensors, padding_value, padding_side="right"):
        """Pad a list of tensors to the same length."""
        max_len = max(t.size(0) for t in tensors)
        padded = []
        for t in tensors:
            if padding_side == "left":
                padding = torch.full((max_len - t.size(0),) + t.shape[1:], padding_value, dtype=t.dtype)
                padded.append(torch.cat([padding, t], dim=0))
            else:
                padding = torch.full((max_len - t.size(0),) + t.shape[1:], padding_value, dtype=t.dtype)
                padded.append(torch.cat([t, padding], dim=0))
        return torch.stack(padded)


def pad_to_length(tensor, length, pad_value):
    """Pad tensor to specified length."""
    if tensor.size(1) >= length:
        return tensor[:, :length]
    return F.pad(tensor, (0, length - tensor.size(1)), value=pad_value)


def disable_dropout_in_model(model):
    """Disable dropout in model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0


def create_reference_model(model):
    """Create a reference model by deep copying the model."""
    ref_model = deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model


class DPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`PreTrainedModel`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        args (`transformers.TrainingArguments`):
            The training arguments to use for training.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        label_smoothing (`float`, defaults to 0):
            The label smoothing factor.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of DPO loss to use. Either "sigmoid", "hinge", "ipo", "bco_pair", etc.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training.
        label_pad_token_id (`int`, defaults to -100):
            The label pad token id.
        padding_value (`int`, *optional*):
            The padding value to use. If None, will use the tokenizer's pad token id.
        truncation_mode (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to use for training.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, *optional*):
            The maximum length of the sequences in the batch.
        max_prompt_length (`int`, *optional*):
            The maximum length of the prompt.
        max_target_length (`int`, *optional*):
            The maximum length of the target.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training.
        is_encoder_decoder (`bool`, *optional*):
            If no model is provided, we need to know if the model to be created is an encoder decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to generate during evaluation.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Whether to precompute reference model log probabilities for the dataset.
        model_init_kwargs (`Dict`, *optional*):
            Dict of keyword arguments to pass to `AutoModelForCausalLM.from_pretrained`.
        ref_model_init_kwargs (`Dict`, *optional*):
            Dict of keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` for the reference model.
        reference_free (`bool`, defaults to `False`):
            If True, we ignore the reference model and implicitly use a reference model that assigns equal probability to all responses.
        use_separate_forward (`bool`, defaults to `True`):
            If True, process chosen and rejected examples separately instead of concatenating them. This reduces memory usage at the cost of slightly more computation time.
    """

    _tag_names = ["trl", "dpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: str = "sigmoid",
        args = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        image_processor: Optional[Any] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        reference_free: bool = False,
        use_separate_forward: bool = True,
    ):
        if not isinstance(model, str) and ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you must pass a copy of it, or `None` if you use peft."
            )

        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_init_kwargs but your model is already instantiated.")

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError("You passed ref_model_init_kwargs but your ref_model is already instantiated.")

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(args, "gradient_checkpointing_kwargs") and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                for name, module in model.named_modules():
                    if "norm" in name:
                        module = module.to(torch.bfloat16)
                self._peft_has_been_casted_to_bf16 = True

        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.reference_free = reference_free
        self.precompute_ref_log_probs = precompute_ref_log_probs

        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is None:
            max_length = 512
        if max_prompt_length is None:
            max_prompt_length = 128
        if max_target_length is None:
            max_target_length = 128

        if padding_value is None:
            padding_value = tokenizer.pad_token_id

        self.image_processor = image_processor

        if data_collator is None:
            data_collator = PreferenceCollator(pad_token_id=padding_value, image_processor=image_processor)

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_completion_length = max_target_length

        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.use_separate_forward = use_separate_forward

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Tokenize datasets
        if train_dataset is not None:
            train_dataset = train_dataset.map(
                lambda x: self.tokenize_row(x, tokenizer, max_prompt_length, max_target_length, image_processor),
                desc="Tokenizing train dataset",
            )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                lambda x: self.tokenize_row(x, tokenizer, max_prompt_length, max_target_length, image_processor),
                desc="Tokenizing eval dataset",
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    @staticmethod
    def tokenize_row(features, tokenizer, max_prompt_length, max_completion_length, image_processor=None):
        """Tokenize a row of the dataset. Note: Image processing should be handled by collate_fn."""
        # If using custom collate_fn, it handles tokenization and image processing
        # This method is only used when using standard HF datasets without custom collate_fn
        
        # Check if already tokenized by collate_fn
        if "chosen_input_ids" in features and isinstance(features["chosen_input_ids"], list):
            return features
        
        # Standard tokenization for text-only or when not using custom collate_fn
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        result = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        # Process images if present and image_processor is available
        # (Only used when not using custom collate_fn)
        if "image" in features and image_processor is not None:
            from PIL import Image
            image = features["image"]
            # Handle different image formats
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")
            
            # Process image
            processed = image_processor(images=image, return_tensors="pt")
            result["pixel_values"] = processed["pixel_values"][0].numpy()

        return result

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids", "chosen_input_ids", "rejected_input_ids", 
                "pixel_values", "pixel_values_chosen", "pixel_values_rejected",
                "attention_mask_chosen", "attention_mask_rejected", "images", "image_paths"
            ]

    def get_train_dataloader(self) -> DataLoader:
        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

                torch.cuda.empty_cache()
                self.accelerator.free_memory()

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            self.train_dataset = self.train_dataset.add_column(
                name="ref_rejected_logps", column=all_ref_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            eval_dataset = eval_dataset.add_column(name="ref_rejected_logps", column=all_ref_rejected_logps)

            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(self.model).disable_adapter() if self.is_peft_model else nullcontext():
            yield

    def compute_ref_log_probs(self, batch: Dict[str, torch.LongTensor]) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]

    @staticmethod
    def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]], padding_value: int) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs from the batch."""
        output = {}

        # Handle prompt if present
        if "prompt_input_ids" in batch and batch["prompt_input_ids"] is not None:
            output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
            output["prompt_attention_mask"] = torch.cat(
                [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
            )

        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        # Handle pixel_values - can be same for both or separate
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            # Same image for chosen and rejected
            output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)
        elif "pixel_values_chosen" in batch and batch["pixel_values_chosen"] is not None:
            # Separate pixel_values for chosen and rejected
            output["pixel_values"] = torch.cat([batch["pixel_values_chosen"], batch["pixel_values_rejected"]], dim=0)

        return output

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities."""
        device = self.accelerator.device

        chosen_logratios = chosen_logps - (not self.reference_free) * ref_chosen_logps
        rejected_logratios = rejected_logps - (not self.reference_free) * ref_rejected_logps

        logratios = chosen_logps - rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
        else:
            ref_logratios = ref_chosen_logps - ref_rejected_logps

        logratios = logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo']")

        chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
        rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _separate_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], num_examples: int):
        """Process chosen and rejected examples separately to reduce memory usage."""
        def process_single_batch(prompt_ids, prompt_mask, completion_ids, completion_mask, pixel_values=None):
            if self.is_encoder_decoder:
                labels = completion_ids.clone()
                labels[completion_mask == 0] = self.label_pad_token_id
                outputs = model(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    labels=labels,
                )
                logits = outputs.logits
                loss_mask = completion_mask.bool()
            else:
                input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
                attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
                loss_mask = torch.cat(
                    (torch.zeros_like(prompt_mask), completion_mask),
                    dim=1,
                )

                # Flush left to reduce memory usage
                for i in range(attention_mask.size(0)):
                    first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                    input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                    attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                    loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

                # Remove empty columns
                empty_cols = torch.sum(attention_mask, dim=0) == 0
                first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1
                input_ids = input_ids[:, : first_empty_col - 1]
                attention_mask = attention_mask[:, : first_empty_col - 1]
                loss_mask = loss_mask[:, : first_empty_col - 1]

                # Truncate right
                if self.max_length is not None:
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]

                # Prepare model inputs
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                
                # Add pixel_values if present
                if pixel_values is not None:
                    model_inputs["pixel_values"] = pixel_values

                outputs = model(**model_inputs)

                logits = outputs.logits[:, :-1, :]
                labels = input_ids[:, 1:].clone()
                loss_mask = loss_mask[:, 1:].bool()

            # Compute log probabilities
            labels[~loss_mask] = 0
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            per_token_logps[~loss_mask] = 0
            all_logps = per_token_logps.sum(-1)
            
            if self.loss_type == "ipo":
                all_logps = all_logps / loss_mask.sum(-1)
                
            mean_logits = logits[loss_mask].mean()
            
            return all_logps, mean_logits

        # Process chosen examples
        chosen_pixel_values = None
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            chosen_pixel_values = batch["pixel_values"]
        elif "pixel_values_chosen" in batch and batch["pixel_values_chosen"] is not None:
            chosen_pixel_values = batch["pixel_values_chosen"]
            
        chosen_logps, chosen_logits = process_single_batch(
            batch["prompt_input_ids"],
            batch["prompt_attention_mask"],
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            chosen_pixel_values
        )
        
        # Clear cache between forward passes
        torch.cuda.empty_cache()
        
        # Process rejected examples
        rejected_pixel_values = None
        if "pixel_values" in batch and batch["pixel_values"] is not None:
            rejected_pixel_values = batch["pixel_values"]
        elif "pixel_values_rejected" in batch and batch["pixel_values_rejected"] is not None:
            rejected_pixel_values = batch["pixel_values_rejected"]
            
        rejected_logps, rejected_logits = process_single_batch(
            batch["prompt_input_ids"],
            batch["prompt_attention_mask"],
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            rejected_pixel_values
        )
        
        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logits,
            "mean_rejected_logits": rejected_logits,
        }

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together."""
        num_examples = batch["prompt_input_ids"].shape[0]

        # Process chosen and rejected separately to save memory
        # Instead of concatenating and processing together
        use_separate_forward = getattr(self, "use_separate_forward", True)
        
        if use_separate_forward:
            return self._separate_forward(model, batch, num_examples)

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce memory usage
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            # Remove empty columns
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1) + 1
            input_ids = input_ids[:, : first_empty_col - 1]
            attention_mask = attention_mask[:, : first_empty_col - 1]
            loss_mask = loss_mask[:, : first_empty_col - 1]

            # Truncate right
            if self.max_length is not None:
                input_ids = input_ids[:, : self.max_length]
                attention_mask = attention_mask[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]

            # Prepare model inputs
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
            # Add pixel_values if present
            if "pixel_values" in concatenated_batch:
                model_inputs["pixel_values"] = concatenated_batch["pixel_values"]

            outputs = model(**model_inputs)

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

        # Compute log probabilities
        labels[~loss_mask] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~loss_mask] = 0
        all_logps = per_token_logps.sum(-1)

        output = {}

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        compute_loss_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = amp.autocast("cuda") if self._peft_has_been_casted_to_bf16 else nullcontext()

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        """Log `logs` on the various objects watching training, including stored metrics."""
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)