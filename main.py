from typing import Optional
from config import LazyLlamaConfig
from transformers import PreTrainedModel, LogitsProcessorList
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, LlamaRotaryEmbedding, _prepare_4d_causal_attention_mask_with_cache_position
)
import torch.nn as nn
import torch
from decoder_layer import DecoderLayer
from caches import KVCache, AuxCache
from context import Context

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
        # Those three were added in the LazyLlamaModel, and are not present in the original code
        kv_cache: KVCache,
        aux_cache: AuxCache,
        # Original inputs
        cache_position: torch.LongTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        dtype, device = inputs_embeds.dtype, inputs_embeds.device 
        batch_size = inputs_embeds.shape[0]

        # The cache_position tensor stores positions of hidden states in the sequence,
        # so the sequence length is the position of the last hidden state + 1  
        sequence_length = cache_position[-1].item() + 1

        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length,
            sequence_length,
            dtype,
            device,
            torch.finfo(dtype).min,
            # The cache_position tensor only includes the positions of current hidden states, but
            # we need the positions of all tokens in the sequence
            torch.arange(sequence_length, device=device),
            batch_size,
        )

        context = Context(
            inputs_embeds,
            kv_cache,
            aux_cache,
            position_ids,
            cache_position,
            sequence_length,
        )

        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                context,
                causal_mask,
                self.rotary_emb,
                output_attentions,
            )
            context = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        context.hidden_states = self.norm(context.hidden_states)

        return context.hidden_states, all_self_attns
    
class LazyLlamaForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LazyLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        kv_cache: KVCache,
        aux_cache: AuxCache,
        cache_position: torch.LongTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        outputs = self.model(
            kv_cache=kv_cache,
            aux_cache=aux_cache,
            cache_position=cache_position,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        hidden_states = outputs[0] 
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits, outputs[1] if output_attentions else None
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        max_length: int,
        eos_token_id: int,
        pad_token_id: int,
        output_attentions: bool = False, 
        logits_processor: Optional[LogitsProcessorList] = None,
        do_sample: bool = False,
    ):
        output_sequence = input_ids

        batch_size = input_ids.shape[0]
        embed_size_per_head = self.config.hidden_size // self.config.num_attention_heads
        
        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        kv_cache = KVCache(
            self.config.num_hidden_layers,
            batch_size,
            self.config.num_attention_heads,
            max_length,
            embed_size_per_head,
        )

        aux_cache = AuxCache(
            self.config.num_hidden_layers,
            batch_size,
            max_length,
            self.config.hidden_size,
        )

        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

        # Creating position_ids on the fly. The default value (for padding tokens) is 1
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "kv_cache": kv_cache,
            "aux_cache": aux_cache,
            "cache_position": cache_position,
            "position_ids": position_ids,
            "output_attentions": output_attentions, 
        }

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        while cache_position[-1].item() < max_length and not torch.all(input_ids[:, -1] == eos_token_id):
            outputs = self(**model_inputs)

            next_token_logits = outputs[0][:, -1, :].clone()
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # Finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + (1 - unfinished_sequences) * pad_token_id

            unfinished_sequences.mul_(next_tokens != eos_token_id)

            # Updating model inputs for the next generation step
            input_ids = next_tokens.view(-1, 1) 
            output_sequence = torch.cat([output_sequence, input_ids], dim=-1)             
            cache_position = torch.tensor([cache_position[-1] + 1], device=cache_position.device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], 
                dim=-1
            )
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            model_inputs.update({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "cache_position": cache_position,
            })
            
        return output_sequence