# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert OpenAI GPT checkpoint."""

# exec:
# cd /code/huggingface/transformers-fair-wmt
# PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.ru-en.ensemble --pytorch_dump_folder_path data/wmt19-ru-en

import argparse
import json
import logging
import os
import re

import numpy
import torch
from pathlib import Path

from transformers import CONFIG_NAME, WEIGHTS_NAME
from transformers.tokenization_fsmt import FSMTTokenizer, VOCAB_FILES_NAMES
from transformers.configuration_fsmt import FSMTConfig

from fairseq.data.dictionary import Dictionary

logging.basicConfig(level=logging.INFO)

# VOCAB_FILES_NAMES = {
#     "vocab_file": "vocab.json",
#     "merges_file": "merges.txt",
# }


DEBUG = 1

json_indent = 2 if DEBUG else None

def rewrite_dict_keys(d):
    # (1) remove word breaking symbol, (2) add word ending symbol where the word is not broken up
    # d = {'le@@': 5, 'tt@@': 6, 'er': 7} => {'le': 5, 'tt': 6, 'er</w>': 7}
    d2=dict((re.sub(r'@@$', '', k), v) if k.endswith('@@') else (re.sub(r'$', '</w>', k), v) for k, v in d.items())
    keep_keys = "<s> <pad> </s> <unk>".split()
    # restore the special tokens
    for k in keep_keys:
        del d2[f"{k}</w>"]
        d2[k] = d[k] # restore
    return d2
    #return dict((re.sub(r'@@', '</w>', k, 0, re.M), v) if k.endswith('@@') else (k, v) for k, v in d.items())

def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):

    #vocab_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["vocab_file"])
    merge_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])

    # prep
    assert os.path.exists(fsmt_checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    print(f"Writing results to {pytorch_dump_folder_path}")


    # XXX: what about 2,3,4? need to merge the ensemble
    fsmt_checkpoint = os.path.join(fsmt_checkpoint_path, "model1.pt")
    chkpt = torch.load(fsmt_checkpoint, map_location="cpu")
    conf_orig = vars(chkpt["args"])
    #print(dir(chkpt))

    src_lang = conf_orig["source_lang"]
    tgt_lang = conf_orig["target_lang"]

    # done:
    # convert dicts
    src_dict_file = os.path.join(fsmt_checkpoint_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fsmt_checkpoint_path, f"dict.{tgt_lang}.txt")

    # XXX: adjust json.dumps to not have new lines with indent=None
    src_dict = Dictionary.load(src_dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    src_vocab_size = len(src_vocab)
    pytorch_vocab_file_src = os.path.join(pytorch_dump_folder_path, f"vocab-{src_lang}.json")
    print(f"Generating {pytorch_vocab_file_src}")
    with open(pytorch_vocab_file_src, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=json_indent))

    tgt_dict = Dictionary.load(tgt_dict_file)
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    tgt_vocab_size = len(tgt_vocab)
    pytorch_vocab_file_tgt = os.path.join(pytorch_dump_folder_path, f"vocab-{tgt_lang}.json")
    print(f"Generating {pytorch_vocab_file_tgt}")
    with open(pytorch_vocab_file_tgt, "w", encoding="utf-8") as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=json_indent))

    # done:
    # convert merge_file (bpecodes)
    fairseq_merge_file = os.path.join(fsmt_checkpoint_path, "bpecodes")
    with open(fairseq_merge_file, encoding="utf-8") as fin:
        merges = fin.read()
    # not needed, already uses </w>
    # re.sub(r'@@', '</w>', merges, 0, re.M) # @@ => </w>
    merges = re.sub(r' \d+$', '', merges, 0, re.M)  # remove frequency number
    print(f"Generating {merge_file}")
    with open(merge_file, "w", encoding="utf-8") as fout:
       fout.write(merges)
    # reversed:
    # with open(fairseq_merge_file, encoding="utf-8") as fin:
    #     merges = fin.read().split("\n")
    # merges = [re.sub(r'@@$', '', m) if m.endswith('@@') else re.sub(r'$', '</w>', m) for m in merges]
    # merges = [re.sub(r' \d+$', '', m) for m in merges]
    # with open(merge_file, "w", encoding="utf-8") as fout:
    #     fout.write("\n".join(merges) + "\n")


    # config + model


    # config - XXX: what really should go there, other than "do_lower_case": False,
    fairseq_config_file = os.path.join(pytorch_dump_folder_path, "config.json")

    #conf = {"do_lower_case": False, "model_max_length": 1024}
    #attrs_to_keep = "".split()
    #conf = (x:conf_orig[x] for x in attrs_to_keep)


    conf = {
        "activation_dropout": 0.0,
        "activation_function": "relu",
        "add_bias_logits": False,
        "add_final_layer_norm": False,
        "architectures": [
          "FSMTForConditionalGeneration"
        ],
        "attention_dropout": conf_orig["attention_dropout"],
        "classif_dropout": 0.0,
        "d_model": conf_orig["decoder_embed_dim"],
        "dropout": conf_orig["dropout"],

        "encoder_attention_heads": conf_orig["encoder_attention_heads"],
        "encoder_ffn_dim": conf_orig["encoder_ffn_embed_dim"],
        "encoder_layerdrop": 0.0,
        "encoder_layers":  conf_orig["encoder_layers"],
        "encoder_emd_tok_dim": src_vocab_size,

        "decoder_attention_heads": conf_orig["decoder_attention_heads"],
        "decoder_ffn_dim": conf_orig["decoder_ffn_embed_dim"],
        "decoder_layerdrop": 0.0,
        "decoder_layers": conf_orig["decoder_layers"],
        "decoder_emd_tok_dim": tgt_vocab_size,

        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,

        "id2label": {
          "0": "LABEL_0",
          "1": "LABEL_1",
          "2": "LABEL_2"
        },
        "init_std": 0.02,
        "is_encoder_decoder": True,
        "label2id": {
          "LABEL_0": 0,
          "LABEL_1": 1,
          "LABEL_2": 2
        },
        "max_position_embeddings": 1024,
        "model_type": "bart",
        "normalize_before": False,
        "normalize_embedding": True,
        "num_hidden_layers": 6,

        "scale_embedding": False,
        "static_position_embeddings": True,
#        "vocab_size": 31640, #src_vocab_size,
    }




    print(f"Generating {fairseq_config_file}")
    with open(fairseq_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(conf, ensure_ascii=False, indent=json_indent))

    # todo:
    # model

    #model_orig = chkpt["model"]
    #state_dict = model_orig.state_dict()



    #state_dict = chkpt["model"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fsmt_checkpoint_path", default=None, type=str, required=True, help="Path to the official PyTorch dump dir."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_fsmt_checkpoint_to_pytorch(args.fsmt_checkpoint_path, args.pytorch_dump_folder_path)
