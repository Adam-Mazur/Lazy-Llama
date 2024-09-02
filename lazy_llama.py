from typing import Optional, Union, List
from config import LazyLlamaConfig
from transformers import PreTrainedModel, DynamicCache, Cache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, _prepare_4d_causal_attention_mask_with_cache_position
)
import torch.nn as nn
import torch


class KVCache:
    def __init__(self, n_layers, batch_size, num_heads, sequence_length, embed_size_per_head):
        """A static storage for the KV caches of all layers, preserving the positions of the caches in the sequence"""
        self.size = (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.n_layers = n_layers
        self.key_cache = tuple(torch.zeros(self.size) for _ in range(n_layers))
        self.value_cache = tuple(torch.zeros(self.size) for _ in range(n_layers))
        self.cache_status_bit_array = torch.zeros((n_layers, sequence_length), dtype=torch.bool)

class AuxCache:
    def __init__(self, n_layers, batch_size, sequence_length, hidden_size):
        """The Aux Cache is storing hidden states of pruned tokens that are not present in the subsequent layers' KV caches"""
        self.size = (batch_size, sequence_length, hidden_size)
        self.n_layers = n_layers-1
        self.cache = tuple(torch.zeros(self.size) for _ in range(n_layers-1))
        self.cache_status_bit_array = torch.zeros((n_layers-1, sequence_length), dtype=torch.bool)


class Context:
    def __init__(
            self, 
            hidden_states: torch.FloatTensor, 
            kv_cache: KVCache,
            aux_cache: AuxCache,
            tokens_positions_idxs: torch.LongTensor, 
            hidden_states_idxs: torch.LongTensor,
            sequence_length: int 
        ):
        assert hidden_states.shape[1] == 1 or hidden_states.shape[1] == sequence_length, "The sequence length must either match the hidden states or there should be only one token in the hidden states"
        
        self.kv_cache = kv_cache
        self.aux_cache = aux_cache

        self.sequence_length = sequence_length
        self.device = hidden_states.device

        self.hidden_states = hidden_states

        # Similar to position_ids in the original code, but it's storing the positions of all tokens in the sequence, 
        # not just the ones in the hidden states
        self.tokens_positions_idxs = tokens_positions_idxs

        # Equivalent to the cache_position in the original code
        self.hidden_states_idxs = hidden_states_idxs
        
        self.selected_tokens_bit_array = torch.ones(sequence_length, device=self.device, dtype=torch.bool)
        self.in_kv_cache_idxs = None

    @property
    def keys_idxs_to_tokens_idxs(self):
        """A mapping from the attention keys indexes to the tokens indexes"""
        return torch.cat([self.in_kv_cache_idxs, self.hidden_states_idxs], dim=0)

    @property
    def tkns_idxs_to_hidden_states_idxs(self):
        """Mapping from the tokens indexes (positions in the sequence) to the hidden states indexes"""
        mapping = torch.empty(self.sequence_length, device=self.device, dtype=torch.long)
        mapping[self.hidden_states_idxs] = torch.arange(self.hidden_states_idxs.shape[0], device=self.device, dtype=torch.long)
        return mapping
    
    @property
    def hidden_states_bit_array(self):
        bit_array = torch.zeros_like(self.selected_tokens_bit_array)
        bit_array[self.hidden_states_idxs] = True
        return bit_array

    def get_kv_cache(self, layer_idx: int):
        local_kv_cache = DynamicCache()

        in_kv_cache_bit_array = torch.logical_and(self.kv_cache.cache_status_bit_array[layer_idx], self.selected_tokens_bit_array)
        in_kv_cache_idxs = torch.nonzero(in_kv_cache_bit_array).view(-1)

        self.in_kv_cache_idxs = in_kv_cache_idxs
        self.in_kv_cache_bit_array = in_kv_cache_bit_array

        for position_idx in in_kv_cache_idxs:
            local_kv_cache.update(
                self.kv_cache.key_cache[layer_idx][:, :, position_idx, :], 
                self.kv_cache.value_cache[layer_idx][:, :, position_idx, :],
                0,
            )

        return local_kv_cache

    def get_aux_cache(self, layer_idx: int):
        in_aux_cache_bit_array = torch.logical_and(
            torch.logical_and(self.aux_cache.cache_status_bit_array[layer_idx-1], self.selected_tokens_bit_array), 
            # Removing those tokens that are in KV Cache
            torch.logical_not(self.in_kv_cache_bit_array)
        )
        in_aux_cache_idxs = torch.nonzero(in_aux_cache_bit_array).view(-1)

        states_to_concatenate = [None] * (len(in_aux_cache_idxs) + 1)
        states_to_concatenate[0] = self.hidden_states

        for position_idx in in_aux_cache_idxs:
            states_to_concatenate[position_idx] = self.aux_cache.cache[layer_idx-1][:, position_idx, :]

        self.hidden_states = torch.cat(states_to_concatenate, dim=1)

        self.hidden_states_idxs = torch.cat([self.hidden_states_idxs, in_aux_cache_idxs], dim=0)

    @property
    def hidden_states_positions(self):
        return self.tokens_positions_idxs[:, self.hidden_states_idxs]

    def update_kv_cache(self, local_kv_cache: DynamicCache, layer_idx: int):
        in_hidden_states_bit_array = torch.logical_and(
            self.selected_tokens_bit_array, 
            torch.logical_not(self.kv_cache.cache_status_bit_array[layer_idx])
        )
        torch.logical_or(
            self.kv_cache.cache_status_bit_array[layer_idx],
            in_hidden_states_bit_array,
            out=self.kv_cache.cache_status_bit_array[layer_idx],
        )

        new_kv_cache_idxs = torch.nonzero(in_hidden_states_bit_array).view(-1)

        # Mapping the new KV Caches from the DynamicCache, to the KVCache. The new caches in DynamicCache are subsequent to the old ones,
        # therefore, [:, :, self.in_kv_cache_idxs.shape[0]:, :] is used to index them.
        self.kv_cache.key_cache[layer_idx][:, :, new_kv_cache_idxs, :] = local_kv_cache.key_cache[0][:, :, self.in_kv_cache_idxs.shape[0]:, :]
        self.kv_cache.value_cache[layer_idx][:, :, new_kv_cache_idxs, :] = local_kv_cache.value_cache[0][:, :, self.in_kv_cache_idxs.shape[0]:, :]

    def update_aux_cache(self, to_prune_idxs: torch.LongTensor, layer_idx: int):
        in_next_layer_kv_bit_array = self.kv_cache.cache_status_bit_array[layer_idx+1]

        pruned_tokens_bit_array = torch.zeros_like(self.selected_tokens_bit_array)
        pruned_tokens_bit_array[to_prune_idxs] = True
        
        in_aux_cache_bit_array = self.aux_cache.cache_status_bit_array[layer_idx]

        to_add_to_aux_bit_array = torch.logical_and(
            pruned_tokens_bit_array, 
            torch.logical_and(
                torch.logical_not(in_next_layer_kv_bit_array), 
                torch.logical_not(in_aux_cache_bit_array)
            )
        ) 
        torch.logical_or(
            self.aux_cache.cache_status_bit_array[layer_idx],
            to_add_to_aux_bit_array,
            out=self.aux_cache.cache_status_bit_array[layer_idx],
        )
        
        to_add_to_aux_idxs = torch.nonzero(to_add_to_aux_bit_array).view(-1)

        # Hidden states are stored in random order, so the tkns_idxs_to_hidden_states_idxs mapping is needed. 
        self.aux_cache.cache[layer_idx][:, to_add_to_aux_idxs, :] = self.hidden_states[:, self.tkns_idxs_to_hidden_states_idxs[to_add_to_aux_idxs], :]

    def prune(self, to_prune_idxs: torch.LongTensor):
        self.selected_tokens_bit_array[to_prune_idxs] = False

        hidden_states_to_keep_bit_array = self.hidden_states_bit_array
        hidden_states_to_keep_bit_array[to_prune_idxs] = False
        hidden_states_to_keep_idxs = torch.nonzero(hidden_states_to_keep_bit_array).view(-1)

        # Hidden states are stored in random order, so the tkns_idxs_to_hidden_states_idxs mapping is needed.
        self.hidden_states = self.hidden_states[:, self.tkns_idxs_to_hidden_states_idxs[hidden_states_to_keep_idxs], :]

        self.hidden_states_idxs = hidden_states_to_keep_idxs


class DecoderLayer(nn.Module):
    def __init__(self, config: LazyLlamaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.decoder = LlamaDecoderLayer(config, 0)

    def forward(
        self,
        context: Context,
        causal_mask: torch.Tensor,
        rotary_emb: LlamaRotaryEmbedding,
        output_attentions: bool,
    ):
        local_kv_cache = context.get_kv_cache(self.layer_idx)

        if self.layer_idx > 0:
            context.get_aux_cache(self.layer_idx)
  
        # Removing the columns corresponding to the tokens that were pruned
        causal_mask = causal_mask[:, :, :, context.keys_idxs_to_tokens_idxs]

        # Modifying the causal mask's rows to only include the tokens in hidden states, in correct order 
        causal_mask = causal_mask[:, :, context.hidden_states_idxs, :]

        position_embeddings = rotary_emb(context.hidden_states, context.hidden_states_positions)

        new_hidden_states, attention_weights, new_local_kv_cache = self.decoder(
            context.hidden_states,
            attention_mask=causal_mask,
            past_key_value=local_kv_cache,
            output_attentions=True,
            use_cache=True,
            position_embeddings=position_embeddings,
        )

        context.hidden_states = new_hidden_states

        context.update_kv_cache(new_local_kv_cache, self.layer_idx)

        # The last token's key index will be the index of the last token in the hidden states, plus the number of tokens in the KV Cache.
        # This is because the KV Cache always comes before the hidden states in the attention mechanism.
        last_token_key_idx = context.in_kv_cache_idxs.shape[0] + context.tkns_idxs_to_hidden_states_idxs[-1]

        attn_weights_to_last_tkn = attention_weights[:, :, last_token_key_idx, :]
        importance_scores_list = torch.sum(attn_weights_to_last_tkn, dim=(0,1)) / (attention_weights.shape[0] * attention_weights.shape[1])

        pruning_rate = self.config.pruning_rates[self.layer_idx]

        if importance_scores_list.shape[0] > 1:
            # Removing the last token's key from the importance scores list, because we don't want to prune it
            importance_scores_list = torch.cat([importance_scores_list[:last_token_key_idx], importance_scores_list[last_token_key_idx+1:]])
            _, to_prune_list_idxs = torch.topk(importance_scores_list, int(pruning_rate * importance_scores_list.shape[0]), largest=False)
        else:
            to_prune_list_idxs = torch.tensor([], dtype=torch.long)

        to_prune_list_idxs = to_prune_list_idxs.to(context.device)
        to_prune_idxs = context.keys_idxs_to_tokens_idxs[to_prune_list_idxs]

        if self.layer_idx < self.config.num_hidden_layers - 1:
            context.update_aux_cache(to_prune_idxs, self.layer_idx)

        context.prune(to_prune_idxs)
        
        outputs = (context,)
        
        if output_attentions:
            outputs += (attention_weights,)

        return outputs


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

        tokens_positions_idxs = tokens_positions_idxs.to(device)
        tokens_positions_idxs = torch.cat([tokens_positions_idxs, position_ids], dim=1)

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