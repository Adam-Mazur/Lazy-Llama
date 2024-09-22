import torch
from transformers import StaticCache
from typing import Tuple, Optional

class KVCache:
    """A static storage for the KV caches of all layers, preserving the positions of the caches in the sequence."""
    def __init__(
            self, 
            n_layers: int, 
            batch_size: int, 
            num_heads: int, 
            sequence_length: int, 
            embed_size_per_head: int, 
            device: torch.device
        ):
        """
        Initializes a KVCache to store key and value caches for all layers.

        Args:
            n_layers (int): Number of layers in the transformer model.
            batch_size (int): Number of batches.
            num_heads (int): Number of attention heads.
            sequence_length (int): The (maximal) length of the input sequence.
            embed_size_per_head (int): Embedding size per attention head.
            device (torch.device): Device to store the tensors (e.g., 'cpu' or 'cuda').
        """
        self.size = (batch_size, num_heads, sequence_length, embed_size_per_head)
        self.n_layers = n_layers
        self.key_cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers))
        self.value_cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers))
        self.cache_status_bit_array = torch.zeros((n_layers, sequence_length), dtype=torch.bool, device=device)

class AuxCache:
    """The Aux Cache stores hidden states of pruned tokens that are not present in the subsequent layers' KV caches."""
    def __init__(
            self, 
            n_layers: int, 
            batch_size: int, 
            sequence_length: int, 
            hidden_size: int, 
            device: torch.device
        ):
        """
        Initializes an AuxCache to store hidden states of pruned tokens.
        
        Args:
            n_layers (int): Number of layers in the transformer model.
            batch_size (int): Number of batches.
            sequence_length (int): The (maximal) length of the input sequence.
            hidden_size (int): Size of the hidden state vectors.
            device (torch.device): Device to store the tensors (e.g., 'cpu' or 'cuda').
        """
        self.size = (batch_size, sequence_length, hidden_size)
        self.n_layers = n_layers-1
        self.cache = tuple(torch.zeros(self.size, device=device) for _ in range(n_layers-1))
        self.cache_status_bit_array = torch.zeros((n_layers-1, sequence_length), dtype=torch.bool, device=device)

class HFCache(StaticCache):
    """
    A cache class that stores key-value (KV) caches compatible with Hugging Face's transformers.
    This class is used for passing KV caches into the decoder layer.
    """
    def __init__(
            self, 
            shape: Tuple[int], 
            device: torch.device, 
            cache: Optional[Tuple[torch.FloatTensor]] = None
        ):
        """
        Initializes an HFCache object.
        
        Args:
            shape (Tuple[int]): Shape of the KV cache tensors in the format: (batch_size, num_heads, sequence_length, embed_size_per_head).
            device (torch.device): Device to store the tensors (e.g., 'cpu' or 'cuda').
            cache (Optional[Tuple[torch.FloatTensor]]): A tuple of two tensors (key_cache, value_cache) to initialize the cache.
        """
        self.shape = shape
        self.max_cache_len = shape[2]
        if cache is None:
            self.key_cache = [torch.zeros(shape, device=device)]
            self.value_cache = [torch.zeros(shape, device=device)]
        else:
            new_shape = (shape[0], shape[1], shape[2]-cache[0].shape[2], shape[3])
            self.key_cache = [torch.cat([cache[0], torch.zeros(new_shape, device=device)], dim=2)]
            self.value_cache = [torch.cat([cache[1], torch.zeros(new_shape, device=device)], dim=2)]