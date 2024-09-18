from typing import Optional, Union, List
from config import LazyLlamaConfig
from transformers import PreTrainedModel, Cache
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, LlamaRotaryEmbedding, _prepare_4d_causal_attention_mask_with_cache_position
)
import torch.nn as nn
import torch
from decoder_layer import DecoderLayer
from caches import KVCache, AuxCache
from context import Context

class LazyLlamaModel(PreTrainedModel):
    def __init__(self, config: LazyLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
    
    def forward(
        self,
        # Those three were added in the LazyLlamaModel, and are not present in the original code
        kv_cache: KVCache,
        aux_cache: AuxCache,
        tokens_positions_idxs: torch.LongTensor,
        # Original inputs
        cache_position: torch.LongTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        # Those are not supported, either because they are not needed or because they are not implemented
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if (
            past_key_values is not None or
            use_cache is not None or
            output_hidden_states is not None or
            return_dict is not None
        ):
            raise NotImplementedError("The LazyLlamaModel does not support past_key_values, use_cache, output_hidden_states or return_dict")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        dtype, device = inputs_embeds.dtype, inputs_embeds.device 
        batch_size = inputs_embeds.shape[0]

        # The cache_position tensor stores positions of hidden states in the sequence,
        # so the sequence length is the position of the last hidden state + 1  
        sequence_length = cache_position[-1].item() + 1

        tokens_positions_idxs.index_copy_(dim=1, index=cache_position, source=position_ids)

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length,
            sequence_length,
            dtype,
            device,
            torch.finfo(dtype).min,
            cache_position,
            batch_size,
        )

        context = Context(
            inputs_embeds,
            kv_cache,
            aux_cache,
            tokens_positions_idxs,
            cache_position,
            sequence_length,
        )

        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                context,
                causal_mask,
                self.rotary_emb,
                output_attentions,
            )
            context = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        context.hidden_states = self.norm(context.hidden_states)

        return context.hidden_states, all_self_attns