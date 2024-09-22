import torch
from caches import KVCache, AuxCache, HFCache

class Context:
    """
    This class provides a way to manage the state of tokens inbetween the layers of the transformer model.
    It keeps track of which tokens are in the hidden states, KV Cache, and Aux Cache, and provides methods to 
    update the caches and hidden states, and prune the tokens.
    """
    def __init__(
            self, 
            hidden_states: torch.FloatTensor, 
            kv_cache: KVCache,
            aux_cache: AuxCache,
            tokens_positions_idxs: torch.LongTensor, 
            hidden_states_idxs: torch.LongTensor,
            sequence_length: int 
        ):
        """
        Initializes the Context with the hidden states, KV Cache, Aux Cache, token positions, and hidden states indexes.
        
        Args:
            hidden_states (torch.FloatTensor): The hidden states of the transformer model. 
                The shape is (batch_size, sequence_length, hidden_size).
            kv_cache (KVCache): The KV Cache for the transformer model.
            aux_cache (AuxCache): The Aux Cache for the transformer model.
            tokens_positions_idxs (torch.LongTensor): The positions of the tokens in the sequence. 
                Serves a similar purpose to `position_ids` in the original code, but it's storing the positions of all tokens 
                in the sequence, not just the ones in the hidden states. The shape is (batch_size, sequence_length).
            hidden_states_idxs (torch.LongTensor): The indexes of the hidden states in the sequence. Equivalent to `cache_position`
                in the original code. The shape is (sequence_length).
            sequence_length (int): The current length of the sequence.
        """
        assert hidden_states.shape[1] == 1 or hidden_states.shape[1] == sequence_length, \
            "The sequence length must either match the hidden states or there should be only one token in the hidden states"
        
        self.kv_cache = kv_cache
        self.aux_cache = aux_cache

        self.sequence_length = sequence_length
        self.device = hidden_states.device

        self.hidden_states = hidden_states

        self.tokens_positions_idxs = tokens_positions_idxs

        self.hidden_states_idxs = hidden_states_idxs
        
        max_sequence_length = kv_cache.size[2]

        self.selected_tokens_bit_array = torch.zeros(max_sequence_length, device=self.device, dtype=torch.bool)
        self.selected_tokens_bit_array[torch.arange(sequence_length, device=self.device)] = True

        self.in_kv_cache_idxs = None

        self._update_keys_idxs_to_tokens_idxs = True
        self._update_tkns_idxs_to_hidden_states_idxs = True

    @property
    def keys_idxs_to_tokens_idxs(self):
        """A mapping from the key's indexes to the token's indexes"""
        if self._update_keys_idxs_to_tokens_idxs:
            self._keys_idxs_to_tokens_idxs = torch.cat([self.in_kv_cache_idxs, self.hidden_states_idxs], dim=0)
            self._update_keys_idxs_to_tokens_idxs = False
        return self._keys_idxs_to_tokens_idxs

    @property
    def tkns_idxs_to_hidden_states_idxs(self):
        """A mapping from the token's indexes (positions in the sequence) to the hidden state's indexes"""
        if self._update_tkns_idxs_to_hidden_states_idxs:
            self._tkns_idxs_to_hidden_states_idxs = torch.empty(self.sequence_length, device=self.device, dtype=torch.long)
            self._tkns_idxs_to_hidden_states_idxs[self.hidden_states_idxs] = \
                torch.arange(self.hidden_states_idxs.shape[0], device=self.device, dtype=torch.long)
            self._update_tkns_idxs_to_hidden_states_idxs = False
        return self._tkns_idxs_to_hidden_states_idxs
    
    @property
    def hidden_states_bit_array(self):
        bit_array = torch.zeros_like(self.selected_tokens_bit_array)
        bit_array[self.hidden_states_idxs] = True
        return bit_array

    def get_kv_cache(self, layer_idx: int) -> HFCache:
        """Returns the KV cache for the given layer"""
        in_kv_cache_bit_array = torch.logical_and(self.kv_cache.cache_status_bit_array[layer_idx], self.selected_tokens_bit_array)
        in_kv_cache_idxs = torch.nonzero(in_kv_cache_bit_array).view(-1)

        self.in_kv_cache_idxs = in_kv_cache_idxs
        self.in_kv_cache_bit_array = in_kv_cache_bit_array

        self._update_keys_idxs_to_tokens_idxs = True

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
        """Updates the hidden states with the tokens from the Aux Cache"""
        in_aux_cache_bit_array = torch.logical_and(
            torch.logical_and(self.aux_cache.cache_status_bit_array[layer_idx-1], self.selected_tokens_bit_array), 
            # Removing those tokens that are in KV Cache
            torch.logical_not(self.in_kv_cache_bit_array)
        )
        in_aux_cache_idxs = torch.nonzero(in_aux_cache_bit_array).view(-1)

        self.hidden_states = torch.cat([
            self.hidden_states, 
            torch.index_select(self.aux_cache.cache[layer_idx-1], 1, in_aux_cache_idxs)
        ], dim=1)

        self.hidden_states_idxs = torch.cat([self.hidden_states_idxs, in_aux_cache_idxs], dim=0)

        self._update_keys_idxs_to_tokens_idxs = True
        self._update_tkns_idxs_to_hidden_states_idxs = True

    @property
    def hidden_states_positions(self):
        return self.tokens_positions_idxs[:, self.hidden_states_idxs]

    def update_kv_cache(self, local_kv_cache: HFCache, layer_idx: int):
        """Updates the KV Cache with the new keys and values"""
        in_hidden_states_bit_array = torch.logical_and(
            self.selected_tokens_bit_array, 
            torch.logical_not(self.kv_cache.cache_status_bit_array[layer_idx])
        )

        self.kv_cache.cache_status_bit_array[layer_idx].logical_or_(in_hidden_states_bit_array)

        new_kv_cache_idxs = torch.nonzero(in_hidden_states_bit_array).view(-1)

        # Mapping the new KV Caches from the HFCache, to the KVCache. The new caches in HFCache are subsequent to the old ones,
        # therefore, we need to slice HFCache starting from "self.in_kv_cache_idxs.shape[0]" along the second dimension.
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
        """Updates the Aux Cache with the hidden states of the pruned tokens"""
        in_next_layer_kv_bit_array = self.kv_cache.cache_status_bit_array[layer_idx+1]

        pruned_tokens_bit_array = torch.zeros_like(self.selected_tokens_bit_array)
        pruned_tokens_bit_array[to_prune_idxs] = True
        
        in_aux_cache_bit_array = self.aux_cache.cache_status_bit_array[layer_idx]

        to_add_to_aux_bit_array = torch.logical_and(
            pruned_tokens_bit_array, 
            # Removing those tokens that are in the next layer's KV Cache and those that are already in the Aux Cache
            torch.logical_not(torch.logical_or(in_next_layer_kv_bit_array, in_aux_cache_bit_array))
        ) 

        self.aux_cache.cache_status_bit_array[layer_idx].logical_or_(to_add_to_aux_bit_array)

        to_add_to_aux_idxs = torch.nonzero(to_add_to_aux_bit_array).view(-1)

        # Hidden states are stored in random order, so the tkns_idxs_to_hidden_states_idxs mapping is needed. 
        self.aux_cache.cache[layer_idx].index_copy_(
            1, 
            to_add_to_aux_idxs, 
            torch.index_select(self.hidden_states, 1, self.tkns_idxs_to_hidden_states_idxs[to_add_to_aux_idxs])
        )

    def prune(self, to_prune_idxs: torch.LongTensor):
        """Prunes the tokens from the hidden states"""
        self.selected_tokens_bit_array[to_prune_idxs] = False

        hidden_states_to_keep_bit_array = self.hidden_states_bit_array
        hidden_states_to_keep_bit_array[to_prune_idxs] = False
        hidden_states_to_keep_idxs = torch.nonzero(hidden_states_to_keep_bit_array).view(-1)

        # Hidden states are stored in random order, so the tkns_idxs_to_hidden_states_idxs mapping is needed.
        self.hidden_states = torch.index_select(self.hidden_states, 1, self.tkns_idxs_to_hidden_states_idxs[hidden_states_to_keep_idxs])

        self.hidden_states_idxs = hidden_states_to_keep_idxs

        self._update_keys_idxs_to_tokens_idxs = True
        self._update_tkns_idxs_to_hidden_states_idxs = True