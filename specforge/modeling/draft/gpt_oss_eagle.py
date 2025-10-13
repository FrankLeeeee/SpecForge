import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import GptOssConfig
from transformers.cache_utils import Cache
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from .base import Eagle3DraftModel
from .llama3_eagle import LlamaForCausalLMEagle3

logger = logging.getLogger(__name__)


class Wrapper(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, *args, **kwargs):
        hidden_states, _ = self.mlp(*args, **kwargs)
        return hidden_states


class GptOssForCausalLMEagle3(LlamaForCausalLMEagle3):

    config_class = GptOssConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        print(f"{config=}")
        super().__init__(config, attention_backend=attention_backend)
        self.midlayer.mlp = Wrapper(GptOssMLP(config))


__all__ = ["GptOssForCausalLMEagle3"]
