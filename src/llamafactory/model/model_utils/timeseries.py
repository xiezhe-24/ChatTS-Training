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

from ...extras import logging

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import FinetuningArguments

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
