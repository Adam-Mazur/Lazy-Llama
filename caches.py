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
