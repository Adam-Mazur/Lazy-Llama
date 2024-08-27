from config import LazyLlamaConfig
from transformers import PreTrainedModel, DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, _prepare_4d_causal_attention_mask_with_cache_position
import torch.nn as nn
import torch

class KVCache:
    def __init__(self, n_layers, batch_size, num_heads, sequence_length, embed_size_per_head):
        self.size = (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.n_layers = n_layers
        self.key_cache = tuple(torch.zeros(self.size) for _ in range(n_layers))
        self.value_cache = tuple(torch.zeros(self.size) for _ in range(n_layers))
        self.cache_status_bit_array = torch.zeros((n_layers, sequence_length), dtype=torch.long)

class AuxCache:
    def __init__(self, n_layers, batch_size, sequence_length, hidden_size):
        self.size = (batch_size, sequence_length, hidden_size)
        self.n_layers = n_layers-1
        self.cache = tuple(torch.zeros(self.size) for _ in range(n_layers-1))
        self.cache_status_bit_array = torch.zeros((n_layers-1, sequence_length), dtype=torch.long)


class DecoderLayer(nn.Module):
    def __init__(self, config: LazyLlamaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.decoder = LlamaDecoderLayer(config, 0)

    def forward(
        self,
        hidden_states,
        # A torch tensor 
        hidden_states_idxs,
        # Selecte tokens table should include all non-pruned tokens, including the ones in hidden states
        selected_tokens_bit_array,
        # Causal mask should be of size (batch_size, 1, query_sequence_length, key_sequence_length)
        causal_mask,
        kv_cache,
        aux_cache,
        rotary_emb,
        cache_position,
        position_ids,
        output_attentions,
    ):
        # Retrieving from KV Cache
        local_kv_cache = DynamicCache()

        in_kv_cache_bit_array = kv_cache.cache_status_bit_array[self.layer_idx] * selected_tokens_bit_array
        in_kv_cache_idxs = torch.nonzero(in_kv_cache_bit_array).view(-1)

        for position_idx in in_kv_cache_idxs:
            local_kv_cache.update(
                kv_cache.key_cache[self.layer_idx][:, :, position_idx, :], 
                kv_cache.value_cache[self.layer_idx][:, :, position_idx, :],
                0,
            )

        # Retrieving from Aux Cache
        in_aux_cache_idxs = None
        if self.layer_idx > 0:
            in_aux_cache_bit_array = aux_cache.cache_status_bit_array[self.layer_idx-1] * selected_tokens_bit_array
            
            # Removing those tokens that are in KV Cache
            in_aux_cache_bit_array = in_aux_cache_bit_array * (1 - in_kv_cache_bit_array)
            in_aux_cache_idxs = torch.nonzero(in_aux_cache_bit_array).view(-1)

            states_to_concatenate = [None] * (len(in_aux_cache_idxs) + 1)
            states_to_concatenate[0] = hidden_states

            for position_idx in in_aux_cache_idxs:
                states_to_concatenate[position_idx] = aux_cache.cache[self.layer_idx-1][:, position_idx, :]

            hidden_states = torch.cat(states_to_concatenate, dim=1)

        # Modifying the causal mask
        if in_aux_cache_idxs is not None:
            hidden_states_idxs = torch.cat([hidden_states_idxs, in_aux_cache_idxs], dim=0)
        
        # Removing the columns corresponding to the tokens that were pruned
        selected_tokens_idxs = torch.nonzero(selected_tokens_bit_array).view(-1)
        causal_mask = causal_mask[:, :, :, selected_tokens_idxs]

        # Modifying the causal mask's columns to only include the tokens that were not pruned, and in correct order 
        causal_mask = causal_mask[:, :, hidden_states_idxs, :]

        # Generating the position embeddings 
        position_embeddings = rotary_emb(hidden_states, position_ids)

        # Calling the decoder
        new_hidden_states, attention_weights, new_local_kv_cache = self.decoder(
            hidden_states,
            attention_mask=causal_mask,
            # It's not even using position_ids is the position_embeddings are provided
            position_ids=hidden_states_idxs,
            past_key_value=local_kv_cache,
            output_attentions=True,
            use_cache=True,
            # This is not used either, if the DynamicCache is used
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # Updating the KV Cache
        kv_cache.cache_status_bit_array[self.layer_idx] += selected_tokens_bit_array - in_kv_cache_bit_array
        new_kv_cache_idxs = torch.nonzero(selected_tokens_bit_array - in_kv_cache_bit_array).view(-1)
        kv_cache.key_cache[self.layer_idx][:, :, new_kv_cache_idxs, :] = new_local_kv_cache.key_cache[0]
        kv_cache.value_cache[self.layer_idx][:, :, new_kv_cache_idxs, :] = new_local_kv_cache.value_cache[0]
        
        # Computing the importance scores
        attn_weights_to_last_tkn = attention_weights[:, :, -1, :]
        # TODO: Solve the batch problem. For now, summing over the batch dimension
        importance_scores_list = torch.sum(attn_weights_to_last_tkn, dim=(0,1)) / (attention_weights.shape[0] * attention_weights.shape[1])
        
        # Pruning the tokens
        pruning_rate = self.config.pruning_rates[self.layer_idx]

        _, to_prune_list_idxs = torch.topk(importance_scores_list, int(pruning_rate * importance_scores_list.shape[0]), largest=False)
        new_selected_tokens_bit_array = selected_tokens_bit_array.clone()
        new_selected_tokens_bit_array[selected_tokens_idxs[to_prune_list_idxs]] = 0
        
        # Updating the Aux Cache
        if self.layer_idx < self.config.num_hidden_layers - 1:
            in_next_layer_kv_bit_array = kv_cache.cache_status_bit_array[self.layer_idx+1]
            pruned_tokens_bit_array = selected_tokens_bit_array - new_selected_tokens_bit_array
            in_aux_cache_bit_array = aux_cache.cache_status_bit_array[self.layer_idx]

            to_add_to_aux_bit_array = pruned_tokens_bit_array * (1-in_next_layer_kv_bit_array) * (1-in_aux_cache_bit_array) 
            
            aux_cache.cache_status_bit_array[self.layer_idx] += to_add_to_aux_bit_array
            to_add_to_aux_idxs = torch.nonzero(to_add_to_aux_bit_array).view(-1)

            hidden_states_idxs_map = torch.empty_like(selected_tokens_bit_array)
            hidden_states_idxs_map[hidden_states_idxs] = torch.arange(hidden_states_idxs.shape[0], device=hidden_states_idxs_map.device)

            aux_cache.cache[self.layer_idx][:, to_add_to_aux_idxs, :] = new_hidden_states[:, hidden_states_idxs_map[to_add_to_aux_idxs], :]

        # Removing the hidden states corresponding to the pruned tokens
        to_keep = torch.zeros_like(selected_tokens_bit_array)
        to_keep[hidden_states_idxs] = 1
        to_keep[selected_tokens_idxs[to_prune_list_idxs]] = 0
        to_keep_idxs = torch.nonzero(to_keep).view(-1)
        # TODO: move hidden_states_idxs_map from the if statement
        new_hidden_states = new_hidden_states[:, hidden_states_idxs_map[to_keep_idxs], :]
        
        outputs = (new_hidden_states, new_selected_tokens_bit_array, to_keep_idxs)
        
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
        input_ids,
        position_ids,
        attention_mask,
        kv_cache,
        aux_cache,
        output_attentions,
        cache_position,
        sequence_length,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        inputs_embeds = self.embed_tokens(input_ids)

        dtype, device = inputs_embeds.dtype, inputs_embeds.device 
        batch_size = inputs_embeds.shape[0]

        # TODO: Make sure the SDPA is always used 
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

        selected_tokens_bit_array = torch.ones(sequence_length, device=device, dtype=torch.long)
        
        hidden_states = inputs_embeds
        hidden_states_idxs = cache_position
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                hidden_states_idxs,
                selected_tokens_bit_array,
                causal_mask,
                kv_cache,
                aux_cache,
                self.rotary_emb,
                cache_position,
                position_ids,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            selected_tokens_bit_array = layer_outputs[1]
            hidden_states_idxs = layer_outputs[2]

            if output_attentions:
                all_self_attns += (layer_outputs[3],)

        hidden_states = self.norm(hidden_states)

        return hidden_states, all_self_attns