import logging
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

from .llama3_eagle import LlamaForCausalLMEagle3, prepare_decoder_attention_mask
import torch
from typing import Optional
from transformers.cache_utils import Cache
from specforge.utils import print_with_rank

logger = logging.getLogger(__name__)


class OlMoEForCausalLMEagle3(LlamaForCausalLMEagle3):

    config_class = OlmoeConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config, attention_backend=attention_backend)
        self.midlayer.mlp = OlmoeSparseMoeBlock(config)

__all__ = ["OlMoEForCausalLMEagle3"]