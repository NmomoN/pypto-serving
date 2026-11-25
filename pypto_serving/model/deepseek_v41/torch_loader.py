# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Metadata-only loader for native vLLM V4.1 A3 W8A8 execution."""

import json
from pathlib import Path

import torch

from pypto_serving.config.types import LoadedModel, ModelConfig, RuntimeConfig, RuntimeModel
from pypto_serving.model.model_loader import SafetensorsDirectoryLoader, _build_layer_specs
from pypto_serving.model.model_family import read_model_config
from pypto_serving.model.tokenizer import load_tokenizer

from .torch_config import validate_checkpoint_config


class DeepSeekV41TorchDirectoryLoader(SafetensorsDirectoryLoader):
    """Payload loading and quantized tensor validation remain in vLLM Ascend."""

    format_names = ("deepseek_v41_torch",)

    def can_load(self, model_path):
        # Explicit format selection only; preserve the default A5/Lib loader.
        return False

    def _recognises(self, model_path):
        try:
            validate_checkpoint_config(read_model_config(model_path))
            return True
        except ValueError:
            return False

    def load(self, request):
        directory = Path(request.model_dir).resolve()
        raw = read_model_config(directory)
        text = validate_checkpoint_config(raw)
        index = json.loads((directory / "quant_model_weights.safetensors.index.json").read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("V4.1 requires a nonempty quant_model_weights safetensors index")
        for shard in set(weight_map.values()):
            if (not isinstance(shard, str) or Path(shard).name != shard
                    or not shard.endswith(".safetensors")
                    or not (directory / shard).resolve().is_relative_to(directory)
                    or not (directory / shard).is_file()):
                raise ValueError(f"Missing or unsafe V4.1 shard: {shard!r}")
        required = {"embed.weight", "norm.weight", "head.weight"}
        for layer in range(text["num_hidden_layers"]):
            required.update(f"layers.{layer}.attn.wq_a.{suffix}"
                            for suffix in ("weight", "weight_scale", "weight_offset"))
        missing = sorted(required.difference(weight_map))
        if missing:
            raise ValueError(f"Missing V4.1 checkpoint tensors: {', '.join(missing[:12])}")
        tokenizer = load_tokenizer(directory)
        runtime = request.runtime_config or RuntimeConfig()
        config = ModelConfig(
            model_id=request.model_id, architecture="DeepseekV41ForCausalLM",
            vocab_size=text["vocab_size"], hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"], num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=int(text.get("num_key_value_heads", 1)), head_dim=text["head_dim"],
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(text["rope_theta"]), torch_dtype="bfloat16",
            bos_token_id=raw.get("bos_token_id", tokenizer.bos_token_id),
            eos_token_id=raw.get("eos_token_id", tokenizer.eos_token_id),
            pad_token_id=raw.get("pad_token_id", tokenizer.pad_token_id),
        )
        empty = torch.empty((0, config.hidden_size), dtype=torch.bfloat16)
        model = RuntimeModel(
            config=config, runtime=runtime, embed_tokens=empty,
            final_norm_weight=torch.empty(0, dtype=torch.bfloat16), lm_head=empty,
            extra=dict(family="deepseek_v41", model_dir=str(directory),
                       checkpoint_format="ascend-w8a8-dynamic", config_data=raw),
        )
        return LoadedModel(model_id=request.model_id, model_dir=str(directory), config=config,
                           tokenizer=tokenizer, layer_specs=_build_layer_specs(config), runtime_model=model)
