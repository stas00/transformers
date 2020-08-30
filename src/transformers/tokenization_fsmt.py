# coding=utf-8
# Copyright 2019 The Open AI Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for XLM."""


import json
import logging
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional

import sacremoses as sm

from .tokenization_utils import PreTrainedTokenizer


logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "src_vocab_file": "vocab-src.json",
    "tgt_vocab_file": "vocab-tgt.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "src_vocab_file": {
        "stas/fsmt-wmt19-ru-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-ru-en/vocab-ru.json",
        "stas/fsmt-wmt19-en-ru": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-ru/vocab-en.json",
        "stas/fsmt-wmt19-de-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-de-en/vocab-de.json",
        "stas/fsmt-wmt19-en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-de/vocab-en.json",
    },
    "tgt_vocab_file": {
        "stas/fsmt-wmt19-ru-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-ru-en/vocab-en.json",
        "stas/fsmt-wmt19-en-ru": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-ru/vocab-ru.json",
        "stas/fsmt-wmt19-de-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-de-en/vocab-en.json",
        "stas/fsmt-wmt19-en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-de/vocab-de.json",
    },
    "merges_file": {
        "stas/fsmt-wmt19-ru-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-ru-en/merges.txt",
        "stas/fsmt-wmt19-en-ru": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-ru/merges.txt",
        "stas/fsmt-wmt19-de-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-de-en/merges.txt",
        "stas/fsmt-wmt19-en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-de/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "stas/fsmt-wmt19-ru-en": 1024,
    "stas/fsmt-wmt19-en-ru": 1024,
    "stas/fsmt-wmt19-de-en": 1024,
    "stas/fsmt-wmt19-en-de": 1024,
}

PRETRAINED_INIT_CONFIGURATION = {
    "stas/fsmt-wmt19-ru-en": {
        "langs": ["ru", "en"],
    },
    "stas/fsmt-wmt19-en-ru": {
        "langs": ["en", "ru"],
    },
    "stas/fsmt-wmt19-de-en": {
        "langs": ["de", "en"],
    },
    "stas/fsmt-wmt19-en-de": {
        "langs": ["en", "de"],
    },
}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def replace_unicode_punct(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    """
    text = text.replace("，", ",")
    text = re.sub(r"。\s*", ". ", text)
    text = text.replace("、", ",")
    text = text.replace("”", '"')
    text = text.replace("“", '"')
    text = text.replace("∶", ":")
    text = text.replace("：", ":")
    text = text.replace("？", "?")
    text = text.replace("《", '"')
    text = text.replace("》", '"')
    text = text.replace("）", ")")
    text = text.replace("！", "!")
    text = text.replace("（", "(")
    text = text.replace("；", ";")
    text = text.replace("１", "1")
    text = text.replace("」", '"')
    text = text.replace("「", '"')
    text = text.replace("０", "0")
    text = text.replace("３", "3")
    text = text.replace("２", "2")
    text = text.replace("５", "5")
    text = text.replace("６", "6")
    text = text.replace("９", "9")
    text = text.replace("７", "7")
    text = text.replace("８", "8")
    text = text.replace("４", "4")
    text = re.sub(r"．\s*", ". ", text)
    text = text.replace("～", "~")
    text = text.replace("’", "'")
    text = text.replace("…", "...")
    text = text.replace("━", "-")
    text = text.replace("〈", "<")
    text = text.replace("〉", ">")
    text = text.replace("【", "[")
    text = text.replace("】", "]")
    text = text.replace("％", "%")
    return text


def remove_non_printing_char(text):
    """
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    """
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            continue
        output.append(char)
    return "".join(output)


class FSMTTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer

    notes:
    removed
    - vocab_file
    - do_lowercase_and_remove_accent
    added:
    - src_vocab_file,
    - tgt_vocab_file,
    - langs,
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(
        self,
        langs=None,
        src_vocab_file=None,
        tgt_vocab_file=None,
        merges_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="</s>",
        mask_token="<special1>",
        additional_special_tokens=[
            "<special0>",
            "<special1>",
            "<special2>",
            "<special3>",
            "<special4>",
            "<special5>",
            "<special6>",
            "<special7>",
            "<special8>",
            "<special9>",
        ],
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.src_vocab_file = src_vocab_file
        self.tgt_vocab_file = tgt_vocab_file
        self.merges_file = merges_file

        # cache of sm.MosesPunctNormalizer instance
        self.cache_moses_punct_normalizer = dict()
        # cache of sm.MosesTokenizer instance
        self.cache_moses_tokenizer = dict()
        self.cache_moses_detokenizer = dict()

        if len(langs) != 2:
            raise ValueError(f"langs arg needs to be a list of 2 langs, e.g. ['en', 'ru'], but got f{langs}")
        self.src_lang, self.tgt_lang = langs[0], langs[1]

        with open(src_vocab_file, encoding="utf-8") as src_vocab_handle:
            self.encoder = json.load(src_vocab_handle)
        with open(tgt_vocab_file, encoding="utf-8") as tgt_vocab_handle:
            tgt_vocab = json.load(tgt_vocab_handle)
            self.decoder = {v: k for k, v in tgt_vocab.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            merges = merges_handle.read().split("\n")[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    # hack override
    def get_vocab(self) -> Dict[str, int]:
        return self.get_src_vocab()

    # hack override
    @property
    def vocab_size(self) -> int:
        return self.src_vocab_size

    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    def moses_detokenize(self, tokens, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_detokenizer = sm.MosesDetokenizer(lang=self.tgt_lang)
            self.cache_moses_detokenizer[lang] = moses_detokenizer
        else:
            moses_detokenizer = self.cache_moses_detokenizer[lang]
        return moses_detokenizer.detokenize(tokens)

    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    @property
    def src_vocab_size(self):
        return len(self.encoder)

    @property
    def tgt_vocab_size(self):
        return len(self.decoder)

    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def get_tgt_vocab(self):
        return dict(self.decoder, **self.added_tokens_decoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def _tokenize(self, text, lang="en", bypass_tokenizer=False):
        """
        Tokenize a string given language code using Moses.

        Details of tokenization:
        - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:
            - lang: ISO language code (default = 'en') (string). Languages should belong of the model supported languages. However, we don't enforce it.
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False) (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
        # ignore `lang` which is currently isn't explicitly passed in tokenization_utils.py and always results in lang=en
        # if lang != self.src_lang:
        #     raise ValueError(f"Expected lang={self.src_lang}, but got {lang}")
        lang = self.src_lang

        if bypass_tokenizer:
            text = text.split()
        else:
            text = self.moses_pipeline(text, lang=lang)
            text = self.moses_tokenize(text, lang=lang)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """

        # remove BPE
        tokens = [t.replace(" ", "").replace("</w>", " ") for t in tokens]
        tokens = "".join(tokens).split()
        # detokenize
        text = self.moses_detokenize(tokens, self.tgt_lang)
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A FAIRSEQ_TRANSFORMER sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.

        """
        bos = [self.bos_token_id]
        sep = [self.sep_token_id]

        # no bos used in fairseq
        if token_ids_1 is None:
            return token_ids_0 + sep
        return token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` methods.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0,
                )
            )
        # no bos used in fairseq
        if token_ids_1 is not None:
            return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        An FAIRSEQ_TRANSFORMER sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]

        # no bos used in fairseq
        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return

        src_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["src_vocab_file"])
        tgt_vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["tgt_vocab_file"])
        merges_file = os.path.join(save_directory, VOCAB_FILES_NAMES["merges_file"])

        with open(src_vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        with open(tgt_vocab_file, "w", encoding="utf-8") as f:
            tgt_vocab = {v: k for k, v in self.decoder.items()}
            f.write(json.dumps(tgt_vocab, ensure_ascii=False))

        index = 0
        with open(merges_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!".format(merges_file)
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return src_vocab_file, tgt_vocab_file, merges_file