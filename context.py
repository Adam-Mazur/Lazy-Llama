import torch
from caches import KVCache, AuxCache, HFCache

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
        
        max_sequence_length = kv_cache.size[2]

        self.selected_tokens_bit_array = torch.zeros(max_sequence_length, device=self.device, dtype=torch.bool)
        self.selected_tokens_bit_array[torch.arange(sequence_length, device=self.device)] = True

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
        in_kv_cache_bit_array = torch.logical_and(self.kv_cache.cache_status_bit_array[layer_idx], self.selected_tokens_bit_array)
        in_kv_cache_idxs = torch.nonzero(in_kv_cache_bit_array).view(-1)

        self.in_kv_cache_idxs = in_kv_cache_idxs
        self.in_kv_cache_bit_array = in_kv_cache_bit_array

        if in_kv_cache_idxs.shape[0] == 0:
            cache = None
        else:
            cache = (
                torch.index_select(self.kv_cache.key_cache[layer_idx], 2, in_kv_cache_idxs),
                torch.index_select(self.kv_cache.value_cache[layer_idx], 2, in_kv_cache_idxs),
            )

        local_kv_cache = HFCache(
            shape = (
                self.kv_cache.size[0], 
                self.kv_cache.size[1], 
                # The cache size must be equal to the number of selected tokens, because otherwise the attention mechanism will break
                torch.nonzero(self.selected_tokens_bit_array).view(-1).shape[0], 
                self.kv_cache.size[3]
            ), 
            cache=cache,
            device=self.device
        )

        return local_kv_cache

    def get_aux_cache(self, layer_idx: int):
        in_aux_cache_bit_array = torch.logical_and(
            torch.logical_and(self.aux_cache.cache_status_bit_array[layer_idx-1], self.selected_tokens_bit_array), 
            # Removing those tokens that are in KV Cache
            torch.logical_not(self.in_kv_cache_bit_array)
        )
        in_aux_cache_idxs = torch.nonzero(in_aux_cache_bit_array).view(-1)

        states_to_concatenate = [self.hidden_states]

        for position_idx in in_aux_cache_idxs:
            states_to_concatenate.append(self.aux_cache.cache[layer_idx-1][:, [position_idx], :])

        self.hidden_states = torch.cat(states_to_concatenate, dim=1)

        self.hidden_states_idxs = torch.cat([self.hidden_states_idxs, in_aux_cache_idxs], dim=0)

    @property
    def hidden_states_positions(self):
        return self.tokens_positions_idxs[:, self.hidden_states_idxs]

    def update_kv_cache(self, local_kv_cache: HFCache, layer_idx: int):
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

        # Mapping the new KV Caches from the HFCache, to the KVCache. The new caches in HFCache are subsequent to the old ones,
        # therefore, we need to slice them starting from "self.in_kv_cache_idxs.shape[0]" along the second dimension.
        self.kv_cache.key_cache[layer_idx].index_copy_(
            2, 
            new_kv_cache_idxs, 
            torch.narrow(local_kv_cache.key_cache[0], 2, self.in_kv_cache_idxs.shape[0], new_kv_cache_idxs.shape[0])
        )
        self.kv_cache.value_cache[layer_idx].index_copy_(
            2,
            new_kv_cache_idxs,
            torch.narrow(local_kv_cache.value_cache[0], 2, self.in_kv_cache_idxs.shape[0], new_kv_cache_idxs.shape[0])
        )

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
        self.aux_cache.cache[layer_idx].index_copy_(
            1, 
            to_add_to_aux_idxs, 
            torch.index_select(self.hidden_states, 1, self.tkns_idxs_to_hidden_states_idxs[to_add_to_aux_idxs])
        )

    def prune(self, to_prune_idxs: torch.LongTensor):
        self.selected_tokens_bit_array[to_prune_idxs] = False

        hidden_states_to_keep_bit_array = self.hidden_states_bit_array
        hidden_states_to_keep_bit_array[to_prune_idxs] = False
        hidden_states_to_keep_idxs = torch.nonzero(hidden_states_to_keep_bit_array).view(-1)

        # Hidden states are stored in random order, so the tkns_idxs_to_hidden_states_idxs mapping is needed.
        self.hidden_states = self.hidden_states[:, self.tkns_idxs_to_hidden_states_idxs[hidden_states_to_keep_idxs], :]

        self.hidden_states_idxs = hidden_states_to_keep_idxs