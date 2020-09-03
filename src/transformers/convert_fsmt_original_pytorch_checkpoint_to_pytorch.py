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
"""Convert fairseq transform wmt19 checkpoint."""

"""
To convert run:
assuming the fairseq data is under data/wmt19.ru-en.ensemble, data/wmt19.en-ru.ensemble, etc

export ROOT=/code/huggingface/transformers-fair-wmt
cd $ROOT
mkdir data

# get data (run once)
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz
tar -xvzf wmt19.en-de.joined-dict.ensemble.tar.gz
tar -xvzf wmt19.de-en.joined-dict.ensemble.tar.gz
tar -xvzf wmt19.en-ru.ensemble.tar.gz
tar -xvzf wmt19.ru-en.ensemble.tar.gz


# run conversions and uploads

export PAIR=ru-en
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR

export PAIR=en-ru
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR

export PAIR=de-en
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.joined-dict.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR

export PAIR=en-de
PYTHONPATH="src" python src/transformers/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py --fsmt_checkpoint_path data/wmt19.$PAIR.joined-dict.ensemble --pytorch_dump_folder_path data/fsmt-wmt19-$PAIR


# upload
cd data
transformers-cli upload fsmt-wmt19-ru-en
transformers-cli upload fsmt-wmt19-en-ru
transformers-cli upload fsmt-wmt19-de-en
transformers-cli upload fsmt-wmt19-en-de
cd -

# happy translations

"""

import argparse
import json
import logging
import os
import re
from collections import OrderedDict

import fairseq
import torch
from fairseq import hub_utils
from fairseq.data.dictionary import Dictionary

from transformers import WEIGHTS_NAME
from transformers.configuration_fsmt import FSMTConfig
from transformers.modeling_fsmt import FSMTForConditionalGeneration
from transformers.tokenization_fsmt import VOCAB_FILES_NAMES


logging.basicConfig(level=logging.INFO)

DEBUG = 1

json_indent = 2 if DEBUG else None


def rewrite_dict_keys(d):
    # (1) remove word breaking symbol, (2) add word ending symbol where the word is not broken up,
    # e.g.: d = {'le@@': 5, 'tt@@': 6, 'er': 7} => {'le': 5, 'tt': 6, 'er</w>': 7}
    d2 = dict((re.sub(r"@@$", "", k), v) if k.endswith("@@") else (re.sub(r"$", "</w>", k), v) for k, v in d.items())
    keep_keys = "<s> <pad> </s> <unk>".split()
    # restore the special tokens
    for k in keep_keys:
        del d2[f"{k}</w>"]
        d2[k] = d[k]  # restore
    return d2


def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):

    # prep
    assert os.path.exists(fsmt_checkpoint_path)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    print(f"Writing results to {pytorch_dump_folder_path}")

    # XXX: Need to work out the ensemble as fairseq does, for now using just one chkpt
    # checkpoint_file = 'model1.pt:model2.pt:model3.pt:model4.pt'
    checkpoint_file = "model1.pt"
    # model_name_or_path = 'transformer.wmt19.ru-en'
    data_name_or_path = "."
    cls = fairseq.model_parallel.models.transformer.ModelParallelTransformerModel
    models = cls.hub_models()
    kwargs = {"bpe": "fastbpe", "tokenizer": "moses"}

    # note: there is some magic happening here, so can't use torch.load() directly on the model file
    chkpt = hub_utils.from_pretrained(
        fsmt_checkpoint_path, checkpoint_file, data_name_or_path, archive_map=models, **kwargs
    )

    args = dict(vars(chkpt["args"]))

    src_lang = args["source_lang"]
    tgt_lang = args["target_lang"]

    # dicts
    src_dict_file = os.path.join(fsmt_checkpoint_path, f"dict.{src_lang}.txt")
    tgt_dict_file = os.path.join(fsmt_checkpoint_path, f"dict.{tgt_lang}.txt")

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

    # merge_file (bpecodes)
    merge_file = os.path.join(pytorch_dump_folder_path, VOCAB_FILES_NAMES["merges_file"])
    fairseq_merge_file = os.path.join(fsmt_checkpoint_path, "bpecodes")
    with open(fairseq_merge_file, encoding="utf-8") as fin:
        merges = fin.read()
    merges = re.sub(r" \d+$", "", merges, 0, re.M)  # remove frequency number
    print(f"Generating {merge_file}")
    with open(merge_file, "w", encoding="utf-8") as fout:
        fout.write(merges)

    # config
    fairseq_config_file = os.path.join(pytorch_dump_folder_path, "config.json")

    # XXX: need to compare with the other pre-trained models of this type and
    # only set here what's different between them - the common settings go into
    # config_fsmt
    conf = {
        "architectures": ["FSMTForConditionalGeneration"],
        "model_type": "fsmt",
        "activation_dropout": 0.0,
        "activation_function": "relu",
        "attention_dropout": args["attention_dropout"],
        "d_model": args["decoder_embed_dim"],
        "dropout": args["dropout"],
        "init_std": 0.02,
        "max_position_embeddings": 1024,  # XXX: look up?
        "num_hidden_layers": 6,  # XXX: look up?
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "langs": [src_lang, tgt_lang],
        "encoder_attention_heads": args["encoder_attention_heads"],
        "encoder_ffn_dim": args["encoder_ffn_embed_dim"],
        "encoder_layerdrop": 0.0,
        "encoder_layers": args["encoder_layers"],
        "decoder_attention_heads": args["decoder_attention_heads"],
        "decoder_ffn_dim": args["decoder_ffn_embed_dim"],
        "decoder_layerdrop": 0.0,
        "decoder_layers": args["decoder_layers"],
        "bos_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},  # not needed?
        "label2id": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},  # not needed?
        "add_bias_logits": False,
        "add_final_layer_norm": False,
        "is_encoder_decoder": True,
        "normalize_before": False,
        "normalize_embedding": False,
        "scale_embedding": True,
        "static_position_embeddings": True,
        "tie_word_embeddings": False,
    }

    print(f"Generating {fairseq_config_file}")
    with open(fairseq_config_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(conf, ensure_ascii=False, indent=json_indent))

    # model
    model = chkpt["models"][0]
    model_state_dict = model.state_dict()

    # rename keys to start with 'model.'
    model_state_dict = OrderedDict(("model." + k, v) for k, v in model_state_dict.items())

    # remove unneeded keys
    ignore_keys = [
        "model.model",
        "model.encoder.version",
        "model.decoder.version",
        "model.encoder_embed_tokens.weight",
        "model.decoder_embed_tokens.weight",
    ]
    for k in ignore_keys:
        model_state_dict.pop(k, None)

    # hf_checkpoint_name = "fsmt-wmt19-ru-en"
    config = FSMTConfig.from_pretrained(pytorch_dump_folder_path)
    model_new = FSMTForConditionalGeneration(config)

    # check that it loads ok
    model_new.load_state_dict(model_state_dict)

    # save
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
    print(f"Generating {pytorch_weights_dump_path}")
    torch.save(model_state_dict, pytorch_weights_dump_path)

    # test that it's the same
    test_state_dict = torch.load(pytorch_weights_dump_path)
    # print(test_state_dict)

    def compare_state_dicts(d1, d2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(d1.items(), d2.items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    print("Mismatch found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print("Models match perfectly! :)")

    compare_state_dicts(model_state_dict, test_state_dict)


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
