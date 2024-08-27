from lazy_llama import KVCache, AuxCache, LazyLlamaModel
from config import LazyLlamaConfig
from transformers import LlamaConfig
import torch


config_llama = LlamaConfig()
config_llama.hidden_size = 128
config_llama.num_attention_heads = 4
config_llama.num_hidden_layers = 4
config_llama.num_key_value_heads = 4 
config_llama.pad_token_id = 0

config = LazyLlamaConfig.from_llama_config(
    pruning_rates={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
    config=config_llama,
)

model = LazyLlamaModel(config)

input_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 6, 7]], dtype=torch.long)
position_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 2]], dtype=torch.long)
attention_mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 1, 1]])
kv_cache = KVCache(4, 2, 4, 5, 32)
aux_cache = AuxCache(4, 2, 5, 128)
output_attentions = False
cache_position = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
sequence_length = 5

output = model(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
    kv_cache=kv_cache,
    aux_cache=aux_cache,
    output_attentions=output_attentions,
    cache_position=cache_position,
    sequence_length=sequence_length,
)

torch.save(output[0], "output.pt")
print(output)