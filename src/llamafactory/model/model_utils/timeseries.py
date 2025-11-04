# Copyright 2025 the LlamaFactory team.
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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch.nn as nn
from torch.optim import Optimizer

from ...extras import logging

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments
    from torch.optim import Optimizer

logger = logging.get_logger(__name__)


@dataclass
class TimeSeriesModel:
    model_type: str
    encoder_key: str
    lora_target_prefixes: list[str]
    modules_to_save: list[str]

    def resolve_encoder(self, model: "PreTrainedModel") -> Optional[object]:
        module: object = model
        for key in self.encoder_key.split("."):
            if not hasattr(module, key):
                return None
            module = getattr(module, key)
        return module


TIMESERIES_MODELS: dict[str, TimeSeriesModel] = {}


def _register_timeseries_model(
    model_type: str,
    encoder_key: Optional[str] = None,
    lora_target_prefixes: Optional[list[str]] = None,
    modules_to_save: Optional[list[str]] = None,
) -> None:
    TIMESERIES_MODELS[model_type] = TimeSeriesModel(
        model_type=model_type,
        encoder_key=encoder_key or "ts_encoder",
        lora_target_prefixes=lora_target_prefixes or [],
        modules_to_save=modules_to_save or [],
    )


_register_timeseries_model(
    model_type="chatts",
    encoder_key="ts_encoder",
    lora_target_prefixes=["ts_encoder.mlp"],
    modules_to_save=[],
)

_register_timeseries_model(
    model_type="qwen3ts",
    encoder_key="ts_encoder",
    lora_target_prefixes=["ts_encoder.mlp"],
    modules_to_save=[],
)


def _collect_ts_target_modules(model: "PreTrainedModel", prefixes: list[str]) -> list[str]:
    if not prefixes:
        return []

    collected: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        for prefix in prefixes:
            if name == prefix or name.startswith(f"{prefix}."):
                collected.append(name)
                break

    # preserve order but remove duplicates
    return list(dict.fromkeys(collected))


def maybe_disable_timeseries_gradients(model: "PreTrainedModel", finetuning_args: "FinetuningArguments") -> None:
    if getattr(finetuning_args, "train_timeseries_modules", True):
        return

    model_type = getattr(model.config, "model_type", None)
    if model_type not in TIMESERIES_MODELS:
        return

    prefix = TIMESERIES_MODELS[model_type].encoder_key
    disabled = []
    for name, param in model.named_parameters():
        if name.startswith(prefix):
            if param.requires_grad:
                param.requires_grad_(False)
                disabled.append(name)

    if disabled:
        logger.info_rank0(
            "Disabled gradients for time-series modules: {}.".format(",".join(disabled))
        )


def get_timeseries_lora_settings(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments"
) -> tuple[list[str], list[str]]:
    if not getattr(finetuning_args, "train_timeseries_modules", True):
        logger.info_rank0("Skip time-series specific LoRA patching as requested.")
        return [], []

    model_type = getattr(model.config, "model_type", None)
    if model_type not in TIMESERIES_MODELS:
        return [], []

    ts_model = TIMESERIES_MODELS[model_type]
    encoder = ts_model.resolve_encoder(model)
    if encoder is None:
        logger.warning_rank0(
            "Time-series encoder `%s` not found on the current model, skip TS-specific LoRA patching.",
            ts_model.encoder_key,
        )
        return [], []

    modules_to_save = list(ts_model.modules_to_save)

    target_modules = _collect_ts_target_modules(model, ts_model.lora_target_prefixes)
    if ts_model.lora_target_prefixes and not target_modules:
        logger.warning_rank0(
            "LoRA target prefixes %s resolved to no modules, please check model structure.",
            ",".join(ts_model.lora_target_prefixes),
        )

    if getattr(finetuning_args, "additional_target", None):
        modules_to_save = [
            module for module in modules_to_save if module not in finetuning_args.additional_target
        ]

    return modules_to_save, target_modules


def _attach_parameter_names(model: "PreTrainedModel") -> dict[int, str]:
    name_map: dict[int, str] = {}
    for name, param in model.named_parameters():
        if id(param) not in name_map:
            name_map[id(param)] = name
            setattr(param, "_llamafactory_param_name", name)
    return name_map


def _summarize_param_groups(optimizer: Optimizer) -> str:
    segments = []
    for idx, group in enumerate(optimizer.param_groups):
        lr = float(group.get("lr", 0.0) or 0.0)
        tag = group.get("llamafactory_group", "default")
        params = group.get("params", [])
        named_params = [getattr(param, "_llamafactory_param_name", None) for param in params]
        filtered_names = [name for name in named_params if name]
        if filtered_names:
            preview = ",".join(filtered_names[:5])
        else:
            preview = "n/a"
        segments.append(
            f"group={idx} tag={tag} lr={lr:.8f} params={len(params)} modules=[{preview}]"
        )
    return "; ".join(segments) if segments else "<no-param-groups>"


def maybe_apply_timeseries_sft_lr(
    optimizer: Optimizer, model: "PreTrainedModel", finetuning_args: "FinetuningArguments"
) -> bool:
    timeseries_lr = getattr(finetuning_args, "timeseries_sft_lr", None)
    if timeseries_lr is None:
        return False

    if getattr(optimizer, "_timeseries_lr_overridden", False):
        return False

    if not getattr(finetuning_args, "train_timeseries_modules", True):
        logger.warning_rank0(
            "`timeseries_sft_lr` is set but `train_timeseries_modules` is False; skip applying the override."
        )
        setattr(optimizer, "_timeseries_lr_overridden", True)
        return False

    model_type = getattr(model.config, "model_type", None)
    if model_type not in TIMESERIES_MODELS:
        logger.warning_rank0(
            "`timeseries_sft_lr` is set but no time-series model definition is registered for `%s`.", model_type
        )
        setattr(optimizer, "_timeseries_lr_overridden", True)
        return False

    encoder = TIMESERIES_MODELS[model_type].resolve_encoder(model)
    if encoder is None:
        logger.warning_rank0(
            "Time-series encoder `%s` not found on the current model, cannot override its learning rate.",
            TIMESERIES_MODELS[model_type].encoder_key,
        )
        setattr(optimizer, "_timeseries_lr_overridden", True)
        return False

    _attach_parameter_names(model)
    for group in optimizer.param_groups:
        group.setdefault("llamafactory_group", "default")
    ts_params = [param for param in encoder.parameters() if param.requires_grad]
    if not ts_params:
        logger.info_rank0("No trainable parameters found in the time-series encoder, skip LR override.")
        setattr(optimizer, "_timeseries_lr_overridden", True)
        return False

    ts_param_ids = {id(param) for param in ts_params}
    applied = 0
    for group in list(optimizer.param_groups):  # copy to avoid iterating newly added groups
        ts_group_params = [param for param in group["params"] if id(param) in ts_param_ids]
        if not ts_group_params:
            continue

        remaining_params = [param for param in group["params"] if id(param) not in ts_param_ids]
        group_options = {key: value for key, value in group.items() if key != "params"}

        if remaining_params:
            group["params"] = remaining_params
        else:
            optimizer.param_groups.remove(group)

        new_group = dict(group_options)
        if "initial_lr" in new_group:
            new_group["initial_lr"] = timeseries_lr
        new_group["lr"] = timeseries_lr
        new_group["params"] = ts_group_params
        new_group["llamafactory_group"] = "timeseries"
        optimizer.add_param_group(new_group)
        applied += len(ts_group_params)

    if applied == 0:
        logger.info_rank0("No optimizer parameters matched the time-series encoder, skip LR override.")
        setattr(optimizer, "_timeseries_lr_overridden", True)
        return False

    setattr(optimizer, "_timeseries_lr_overridden", True)
    logger.info_rank0(
        "Applied SFT learning rate %.6f to %d time-series encoder parameter(s).",
        timeseries_lr,
        applied,
    )
    logger.info_rank0(
        "[OPTIMIZER] Optimizer param-group layout after TS LR override: %s",
        _summarize_param_groups(optimizer),
    )
    return True

def get_timeseries_learning_rate(optimizer: Optional[Optimizer]) -> Optional[float]:
    if optimizer is None:
        return None
    for group in optimizer.param_groups:
        if group.get("llamafactory_group") == "timeseries":
            lr = group.get("lr")
            if lr is None:
                return None
            return float(lr)
    return None


def patch_timeseries_modules_for_lora(
    model: "PreTrainedModel", finetuning_args: "FinetuningArguments", target_modules: list[str]
) -> list[str]:
    modules_to_save, ts_target_modules = get_timeseries_lora_settings(model, finetuning_args)

    model_type = getattr(model.config, "model_type", None)
    ts_prefixes = (
        TIMESERIES_MODELS[model_type].lora_target_prefixes if model_type in TIMESERIES_MODELS else []
    )

    if ts_prefixes:
        filtered_target_modules = [module for module in target_modules if module not in ts_prefixes]
        if len(filtered_target_modules) != len(target_modules):
            removed = [module for module in target_modules if module not in filtered_target_modules]
            logger.info_rank0(
                "Remove unsupported time-series prefixes from LoRA targets: {}.".format(",".join(removed))
            )
        target_modules = filtered_target_modules

    if ts_target_modules:
        merged = list(dict.fromkeys(target_modules + ts_target_modules))
        added = [module for module in merged if module not in target_modules]
        if added:
            logger.info_rank0(
                "Extend LoRA target modules with time-series components: {}.".format(",".join(added))
            )
        target_modules = merged

    if modules_to_save:
        existing_modules_to_save = finetuning_args.additional_target or []
        merged_modules_to_save = list(dict.fromkeys(existing_modules_to_save + modules_to_save))
        added = [module for module in merged_modules_to_save if module not in existing_modules_to_save]
        if added:
            logger.info_rank0(
                "Saving time-series modules alongside LoRA adapters: {}.".format(",".join(added))
            )
        finetuning_args.additional_target = merged_modules_to_save

    return target_modules
