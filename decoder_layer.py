from config import LazyLlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
import torch
from context import Context
import torch.nn as nn
from typing import Tuple, Optional

class DecoderLayer(nn.Module):
    """
    A custom decoder layer that builds upon the LlamaDecoderLayer and implements dynamic token pruning.

    This layer utilizes KV cache and Aux Cache to speed up the "time-to-first-token" (TTFT) of Hugging 
    Face's LLaMa 2 implementation. Dynamic token pruning is used in each forward pass, based on attention 
    importance scores and pruning rates defined in the configuration. 
    """
    def __init__(self, config: LazyLlamaConfig, layer_idx: int):
        """
        Initializes the decoder layer.

        Args:
            config (LazyLlamaConfig): Configuration object containing model hyperparameters.
            layer_idx (int): The index of the current decoder layer.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # The LlamaDecoderLayer needs the layer index to retrieve the correct KV Cache, however we only pass it the 
        # KV Cache of the current layer. Therefore the layer index needs to be 0 for all LlamaDecoderLayers.
        self.decoder = LlamaDecoderLayer(config, 0)

    def forward(
            self,
            context: Context,
            causal_mask: torch.FloatTensor,
            rotary_emb: LlamaRotaryEmbedding,
            output_attentions: bool,
        ) -> Tuple[Context, Optional[torch.Tensor]]:
        """
        Executes the forward pass for the decoder layer, updating the hidden states and caches, 
        and optionally returning attention weights.

        Args:
            context (Context): The context object containing hidden states, KV Cache, Aux Cache, etc.
            causal_mask (torch.Tensor): The 4D causal mask for the attention mechanism. 
            rotary_emb (LlamaRotaryEmbedding): The rotary embedding layer. This must be passed to the decoder since it 
                recomputes the embeddings for tokens from the Aux Cache.
            output_attentions (bool): Whether to return attention weights.
        """
        local_kv_cache = context.get_kv_cache(self.layer_idx)

        if self.layer_idx > 0:
            context.get_aux_cache(self.layer_idx)
  
        # Removing the columns corresponding to the tokens that were pruned
        causal_mask = torch.index_select(causal_mask, 3, context.keys_idxs_to_tokens_idxs)

        # Modifying the causal mask's rows to only include the tokens in hidden states, in correct order
        causal_mask = torch.index_select(causal_mask, 2, context.hidden_states_idxs) 

        position_embeddings = rotary_emb(context.hidden_states, context.hidden_states_positions)

        new_hidden_states, attention_weights, new_local_kv_cache = self.decoder(
            context.hidden_states,
            attention_mask=causal_mask,
            past_key_value=local_kv_cache,
            output_attentions=True,
            use_cache=True,
            position_embeddings=position_embeddings,
            # The cache position is used to insert new keys and values into the cache. Since I just want them to be 
            # appended to the end of the cache, I need to make sure they get inserted after the last token from KV cache.
            cache_position=torch.arange(
                context.in_kv_cache_idxs.shape[0],
                context.hidden_states.shape[1] + context.in_kv_cache_idxs.shape[0], 
                device=context.device
            ),
        )

        context.hidden_states = new_hidden_states

        context.update_kv_cache(new_local_kv_cache, self.layer_idx)

        last_token_query_idx = context.tkns_idxs_to_hidden_states_idxs[-1]
        # The last token key's index will be the index of the last token in the hidden states, plus the number of tokens in the KV Cache.
        # This is because the KV Cache always comes before the hidden states in the attention mechanism.
        last_token_key_idx = context.in_kv_cache_idxs.shape[0] + last_token_query_idx

        attn_weights_to_last_tkn = attention_weights[:, :, last_token_query_idx, :]
        importance_scores_list = torch.sum(attn_weights_to_last_tkn, dim=(0,1)) / (attention_weights.shape[0] * attention_weights.shape[1])

        pruning_rate = self.config.pruning_rates[self.layer_idx]

        if importance_scores_list.shape[0] > 1:
            # Setting the last token's importance to infinity, because we don't want to prune it
            importance_scores_list[last_token_key_idx] = float("inf")
            _, to_prune_list_idxs = torch.topk(importance_scores_list, int(pruning_rate * importance_scores_list.shape[0]), largest=False)
        else:
            to_prune_list_idxs = torch.tensor([], dtype=torch.long, device=context.device)

        to_prune_idxs = context.keys_idxs_to_tokens_idxs[to_prune_list_idxs]

        if self.layer_idx < self.config.num_hidden_layers - 1:
            context.update_aux_cache(to_prune_idxs, self.layer_idx)

        context.prune(to_prune_idxs)
        
        outputs = (context,)
        
        if output_attentions:
            outputs += (attention_weights,)

        return outputs