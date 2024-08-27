from transformers import LlamaConfig

class LazyLlamaConfig(LlamaConfig):
    def __init__(
            self,
            pruning_rates: dict, 
            **kwargs
        ):
        self.pruning_rates = pruning_rates
        super().__init__(**kwargs)

    def from_llama_config(pruning_rates: dict, config: LlamaConfig):
        return LazyLlamaConfig(
            pruning_rates=pruning_rates,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_act=config.hidden_act,
            max_position_embeddings=config.max_position_embeddings,
            initializer_range=config.initializer_range,
            rms_norm_eps=config.rms_norm_eps,
            use_cache=config.use_cache,
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pretraining_tp=config.pretraining_tp,
            tie_word_embeddings=config.tie_word_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            mlp_bias=config.mlp_bias,
        )