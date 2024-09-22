# LazyLlama

LazyLlama is an implementation of dynamic token prunning from this [paper](https://arxiv.org/abs/2407.14057) using LLaMa 2 family of models as a base.

Dynamic token pruning is a technique that helps speed up the generation of long prompts. The LazyLlama model focuses on calculating keys and values only for the tokens that are most important for predicting the next token. By using attention maps from the previous layer, it prunes tokens that are less relevant to the last token in the sequence. This approach uses a KV cache to store the previously calculated keys and values, ensuring that only the keys and values for the pruned tokens are recalculated when needed.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Introduction

LazyLlama is an implementation of [LazyLLM](https://arxiv.org/abs/2407.14057) token prunning for the LLaMa 2 family of models from Hugging Face. The API is similar to the original LLaMa 2 implementation and the weights from the Hugging Face Model Hub can be easily loaded into this model. To match the original API, two models are provided in this repository: `LazyLlamaModel` and `LazyLlamaForCausalLM`. The `LazyLlamaModel` model is a base model that can be used for various tasks, while the `LazyLlamaForCausalLM` model is specifically designed for causal LM.

## Dependencies

To use LazyLlama, you will need to have the following dependencies:

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)
- [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) family of models from Hugging Face

## Usage

To load the weight of the original LLaMa 2 model from the Hugging Face Model Hub, you can use the following code:

```python
from transformers import LlamaForCausalLM, LlamaTokenizer
from models import *
import torch

# Setting the device to cuda if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The model name from the Hugging Face Model Hub. You need to have access to llama-2-7b-chat-hf to run this code. 
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Loading the original model
llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Mapping the original model to LazyLlamaForCausalLM
lazy_llama_model = LazyLlamaForCausalLM.from_llama_state_dict(
    llama_model.state_dict(), 
    llama_model.config,
    pruning_rates={i: 0.1 for i in range(32)}, 
).to(device)
```
To run the model, you can use the following code:

```python
# Getting the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Using the tokenizer to encode the input sequence
prompt = 'Write a delicious recipe for french fries.\n\n'
input_sequence = tokenizer([prompt], return_tensors="pt")

# Generating the output sequence with the Lazy LLaMa 2 model
output_ids = lazy_llama_model.generate(
    input_sequence["input_ids"].to(device), 
    attention_mask=input_sequence["attention_mask"].to(device), 
    max_length=250, 
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
)

# Decoding the output sequence
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Lazy LLaMa 2 model output:\n{output_text}")
```

Refer to the `example.py` file for a complete example.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.