# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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


import json
import os
import unittest

from transformers.testing_utils import slow
from transformers.tokenization_fsmt import VOCAB_FILES_NAMES, FSMTTokenizer

from .test_tokenization_common import TokenizerTesterMixin


class FSMTTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    __test__ = True

    tokenizer_class = FSMTTokenizer

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "w</w>",
            "r</w>",
            "t</w>",
            "lo",
            "low",
            "er</w>",
            "low</w>",
            "lowest</w>",
            "newer</w>",
            "wider</w>",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["l o 123", "lo w 1456", "e r</w> 1789", ""]

        self.langs = ["en", "ru"]
        config = {
            "langs": self.langs,
        }

        self.src_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["src_vocab_file"])
        self.tgt_vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["tgt_vocab_file"])
        config_file = os.path.join(self.tmpdirname, "tokenizer_config.json")
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.src_vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        # XXX: ru content
        with open(self.tgt_vocab_file, "w") as fp:
            fp.write(json.dumps(vocab_tokens))
        with open(self.merges_file, "w") as fp:
            fp.write("\n".join(merges))
        with open(config_file, "w") as fp:
            fp.write(json.dumps(config))

    def test_full_tokenizer(self):
        """ Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt """
        tokenizer = FSMTTokenizer(self.langs, self.src_vocab_file, self.tgt_vocab_file, self.merges_file)

        text = "lower"
        bpe_tokens = ["low", "er</w>"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + ["<unk>"]
        input_bpe_tokens = [14, 15, 20]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    @slow
    def test_sequence_builders(self):
        tokenizer = FSMTTokenizer.from_pretrained("stas/fsmt-wmt19-ru-en")

        text = tokenizer.encode("sequence builders", add_special_tokens=False)
        text_2 = tokenizer.encode("multi-sequence build", add_special_tokens=False)

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == text + [2]
        assert encoded_pair == text + [2] + text_2 + [2]
