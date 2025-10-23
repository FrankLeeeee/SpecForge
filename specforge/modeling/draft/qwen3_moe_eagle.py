import logging

from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from .llama3_eagle import LlamaForCausalLMEagle3, prepare_decoder_attention_mask
import torch
from specforge.utils import print_with_rank
from transformers.cache_utils import Cache

from typing import Optional

logger = logging.getLogger(__name__)




class Qwen3MoEForCausalLMEagle3(LlamaForCausalLMEagle3):

    config_class = Qwen3MoeConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config, attention_backend=attention_backend)
        self.midlayer.mlp = Qwen3MoeSparseMoeBlock(config)
    

__all__ = ["Qwen3MoEForCausalLMEagle3"]