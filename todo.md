# WIP

- add model cards

- I think merges file should actually still have the frequency counters? check the other models (I remove those now in convert)

- config.num_beams=5?

- share_decoder_input_output_embed is True in fairseq!


- beam diversion:


fs: sequence_generator.py:289 : tokens

cand_indices - is where candidates of the current step are placed

tf: generation_utils.py:726 : input_ids

beam_tokens - is where candidates of the current step are placed

fairseq      | tf                              | bookmark
--------------------------------  -------------|---------
lprobes      | scores (same var)               | 5
cand_indices | next_tokens (beam_tokens)       |
tokens       | input_ids                       |
