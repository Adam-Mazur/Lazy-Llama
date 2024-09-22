from transformers import LlamaForCausalLM, LlamaTokenizer
from models import *
import torch

# Setting the device to cuda if available, otherwise cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The model name from the Hugging Face Model Hub. You need to have access to llama-2-7b-chat-hf to run this code. 
model_id = "meta-llama/Llama-2-7b-chat-hf"

# Getting the tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Loading the original model
llama_model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Mapping the original model to LazyLlamaForCausalLM
lazy_llama_model = LazyLlamaForCausalLM.from_llama_state_dict(
    llama_model.state_dict(), 
    llama_model.config,
    pruning_rates={i: 0.1 for i in range(32)}, 
).to(device)

# Using the tokenizer to encode the input sequence
prompt = 'Write a delicious recipe for french fries.\n\n'
input_sequence = tokenizer([prompt], return_tensors="pt")

# Generating the output sequence with the original LLaMa 2 model
output_ids = llama_model.generate(input_sequence["input_ids"].to(device), max_length=250, num_return_sequences=1)

# Decoding the output sequence
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Original LLaMa 2 model output:\n{output_text}")

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