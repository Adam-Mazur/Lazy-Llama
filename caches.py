import torch
from transformers import StaticCache

class KVCache:
    def __init__(self, n_layers, batch_size, num_heads, sequence_length, embed_size_per_head, device):
        """A static storage for the KV caches of all layers, preserving the positions of the caches in the sequence"""
        self.size = (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.n_layers = n_layers
        self.key_cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers))
        self.value_cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers))
        self.cache_status_bit_array = torch.zeros((n_layers, sequence_length), dtype=torch.bool, device=device)

class AuxCache:
    def __init__(self, n_layers, batch_size, sequence_length, hidden_size, device):
        """The Aux Cache is storing hidden states of pruned tokens that are not present in the subsequent layers' KV caches"""
        self.size = (batch_size, sequence_length, hidden_size)
        self.n_layers = n_layers-1
        self.cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers-1))
        self.cache_status_bit_array = torch.zeros((n_layers-1, sequence_length), dtype=torch.bool, device=device)

class HFCache(StaticCache):
    def __init__(self, shape, device, cache=None):
        self.shape = shape
        self.max_cache_len = shape[2]
        if cache is None:
            self.key_cache = [torch.zeros(shape, device=device)]
            self.value_cache = [torch.zeros(shape, device=device)]
        else:
            new_shape = (shape[0], shape[1], shape[2]-cache[0].shape[2], shape[3])
            self.key_cache = [torch.cat([cache[0], torch.zeros(new_shape, device=device)], dim=2)]
            self.value_cache = [torch.cat([cache[1], torch.zeros(new_shape, device=device)], dim=2)]