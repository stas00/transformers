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
# PYTHONPATH="src" python src/transformers/convert_fs_translator_original_pytorch_checkpoint_to_pytorch.py --fairseq_transformer_checkpoint_path data/wmt19.ru-en.ensemble --pytorch_dump_folder_path data/wmt19-ru-en

import argparse
import json
import logging
import os
import re

import numpy
import torch
from pathlib import Path

from transformers import CONFIG_NAME, WEIGHTS_NAME
from transformers.tokenization_fs_translator import FairseqBPETokenizer, VOCAB_FILES_NAMES

from fairseq.data.dictionary import Dictionary

logging.basicConfig(level=logging.INFO)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

def rewrite_dict_keys(d):
    # (1) remove word breaking symbol, (2) add word ending symbol where the word is not broken up
    # d = {'le@@': 5, 'tt@@': 6, 'er': 7} => {'le': 5, 'tt': 6, 'er</w>': 7}
    return dict((re.sub(r'@@$', '', k), v) if k.endswith('@@') else (re.sub(r'$', '</w>', k), v) for k, v in d.items())
    #return dict((re.sub(r'@@', '</w>', k, 0, re.M), v) if k.endswith('@@') else (k, v) for k, v in d.items())

def convert_fairseq_transformer_checkpoint_to_pytorch(fairseq_transformer_checkpoint_path, pytorch_dump_folder_path):

    src_lang = 'ru'
    tgt_lang = 'en'

    vocab_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["vocab_file"])
    merge_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])

    # prep
    assert os.path.exists(fairseq_transformer_checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)

    # done:
    # convert dicts
    src_dict_file = os.path.join(fairseq_transformer_checkpoint_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fairseq_transformer_checkpoint_path, f"dict.{tgt_lang}.txt")

    # XXX: adjust json.dumps to not have new lines with indent=None
    src_dict = Dictionary.load(src_dict_file)
    src_vocab = rewrite_dict_keys(src_dict.indices)
    pytorch_vocab_file_src = os.path.join(pytorch_dump_folder_path, f"vocab-{src_lang}.json")
    with open(pytorch_vocab_file_src, "w", encoding="utf-8") as f:
        f.write(json.dumps(src_vocab, ensure_ascii=False, indent=2) + "\n")

    tgt_dict = Dictionary.load(tgt_dict_file)
    tgt_vocab = rewrite_dict_keys(tgt_dict.indices)
    pytorch_vocab_file_tgt = os.path.join(pytorch_dump_folder_path, f"vocab-{tgt_lang}.json")
    with open(pytorch_vocab_file_tgt, "w", encoding="utf-8") as f:
        f.write(json.dumps(tgt_vocab, ensure_ascii=False, indent=2) + "\n")

    # done:
    # convert merge_file (bpecodes)
    fairseq_merge_file = os.path.join(fairseq_transformer_checkpoint_path, "bpecodes")
    with open(fairseq_merge_file, encoding="utf-8") as fin:
        merges = fin.read()
    # not needed, already uses </w>
    # re.sub(r'@@', '</w>', merges, 0, re.M) # @@ => </w>
    merges = re.sub(r' \d+$', '', merges, 0, re.M)  # remove frequency number
    with open(merge_file, "w", encoding="utf-8") as fout:
       fout.write(merges)
    # reversed:
    # with open(fairseq_merge_file, encoding="utf-8") as fin:
    #     merges = fin.read().split("\n")
    # merges = [re.sub(r'@@$', '', m) if m.endswith('@@') else re.sub(r'$', '</w>', m) for m in merges]
    # merges = [re.sub(r' \d+$', '', m) for m in merges]
    # with open(merge_file, "w", encoding="utf-8") as fout:
    #     fout.write("\n".join(merges) + "\n")


    # todo:
    # model

    # XXX: what about 2,3,4? need to merge the ensemble
    #fairseq_transformer_checkpoint = os.path.join(fairseq_transformer_checkpoint_path, "model1.pt")
    #chkpt = torch.load(fairseq_transformer_checkpoint, map_location="cpu")
    #print(dir(chkpt))

    #state_dict = chkpt["model"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--fairseq_transformer_checkpoint_path", default=None, type=str, required=True, help="Path to the official PyTorch dump dir."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_fairseq_transformer_checkpoint_to_pytorch(args.fairseq_transformer_checkpoint_path, args.pytorch_dump_folder_path)
