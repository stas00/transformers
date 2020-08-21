# WIP

## Converting weights

### Missing

1. Missing

"final_logits_bias"

2. Missing

"model.encoder_embed_tokens.weight"
"model.decoder_embed_tokens.weight"

3. Missing:

"model.decoder.embed_positions.weight"
"model.encoder.embed_positions.weight"

4.  Instead of these:

"model.encoder.layers.0.final_layer_norm.weight"
"model.encoder.layers.0.final_layer_norm.bias"
"model.encoder.layernorm_embedding.weight"
"model.encoder.layernorm_embedding.bias"
"model.decoder.layernorm_embedding.weight"
"model.decoder.layernorm_embedding.bias".


Have these:

"model.encoder.layers.0.layer_norms.0.weight"
"model.encoder.layers.0.layer_norms.0.bias"
"model.encoder.layers.0.layer_norms.1.weight"
"model.encoder.layers.0.layer_norms.1.bias"

decoder doesn't seem to have layernorm at all


5. Instead of these:

"model.encoder.layers.0.self_attn.k_proj.weight"
"model.encoder.layers.0.self_attn.k_proj.bias"
"model.encoder.layers.0.self_attn.v_proj.weight"
"model.encoder.layers.0.self_attn.v_proj.bias"
"model.encoder.layers.0.self_attn.q_proj.weight"
"model.encoder.layers.0.self_attn.q_proj.bias"
"model.encoder.layers.0.self_attn_layer_norm.weight"
"model.encoder.layers.0.self_attn_layer_norm.bias"
"model.decoder.layers.0.self_attn.k_proj.weight"
"model.decoder.layers.0.self_attn.k_proj.bias"
"model.decoder.layers.0.self_attn.v_proj.weight"
"model.decoder.layers.0.self_attn.v_proj.bias"
"model.decoder.layers.0.self_attn.q_proj.weight"
"model.decoder.layers.0.self_attn.q_proj.bias"
"model.decoder.layers.0.encoder_attn.k_proj.weight"
"model.decoder.layers.0.encoder_attn.k_proj.bias"
"model.decoder.layers.0.encoder_attn.v_proj.weight"
"model.decoder.layers.0.encoder_attn.v_proj.bias"
"model.decoder.layers.0.encoder_attn.q_proj.weight"
"model.decoder.layers.0.encoder_attn.q_proj.bias"

Have these:


"model.encoder.layers.0.self_attn.in_proj_weight"
"model.encoder.layers.0.self_attn.in_proj_bias"
"model.decoder.layers.0.self_attn.in_proj_weight"
"model.decoder.layers.0.self_attn.in_proj_bias"
"model.decoder.layers.0.encoder_attn.in_proj_weight"
"model.decoder.layers.0.encoder_attn.in_proj_bias"

Sam:
- I'd probably copy modeling_bart.py and then modify the attention if I were you. We can try to share code at the end when things are working
- replace the SelfAttention class with whatever fairseq is doing
