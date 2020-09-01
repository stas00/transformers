# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
""" XLM configuration """


import logging

from .configuration_utils import PretrainedConfig


logger = logging.getLogger(__name__)

FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "stas/fsmt-wmt19-ru-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-ru-en/config.json",
    "stas/fsmt-wmt19-en-ru": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-ru/config.json",
    "stas/fsmt-wmt19-de-en": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-de-en/config.json",
    "stas/fsmt-wmt19-en-de": "https://s3.amazonaws.com/models.huggingface.co/bert/stas/fsmt-wmt19-en-de/config.json",
}


# Porting notes:
# this one is modeled after BartConfig
#
#    Differences with BART:
#    - src/tgt vocabs aren't shared
#    - token embeddings aren't shared
#    - needs a language pair
#
class FSMTConfig(PretrainedConfig):
    r"""
        Configuration class for FSMT. Parameters are renamed from the fairseq implementation

    Differences with BART:
    - src/tgt vocabs aren't shared - token embeddings aren't shared

    """
    model_type = "fairseq"

    # update the defaults from config file
    def __init__(
        self,
        src_vocab_size=None,  # new
        tgt_vocab_size=None,  # new
        langs=None,  # new
        activation_function="relu",  # changed
        num_labels=3,
        d_model=1024,
        max_length=256,
        max_position_embeddings=1024,
        extra_pos_embeddings=2,  # FIXME(@sshleifer): delete? # XXX: sshleifer said might need to turn it off?
        encoder_ffn_dim=4096,
        encoder_layers=12,
        encoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_attention_heads=16,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        add_bias_logits=False,
        add_final_layer_norm=False,
        is_encoder_decoder=True,
        normalize_before=False,
        normalize_embedding=False,
        scale_embedding=True,
        static_position_embeddings=True,
        tie_word_embeddings=False,
        **common_kwargs
    ):
        r"""
        :class:`~transformers.FSMTConfig` is the configuration class for `FSMTModel`.

        Examples::

            >>> from transformers import FSMTConfig, FSMTModel

            >>> config = FSMTConfig.from_pretrained('stas/fsmt-wmt19-en-ru')
            >>> model = FSMTModel(config)

        """
        if "hidden_size" in common_kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **common_kwargs,
        )
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.max_length = max_length
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        # XXX: needed in generation_utils.py:382
        # alternatively need to setup config.decoder object
        self.decoder_start_token_id = eos_token_id

        # Params introduced for Mbart
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.normalize_embedding = normalize_embedding  # True for mbart, False otherwise
        self.normalize_before = normalize_before  # combo of fairseq's encoder_ and decoder_normalize_before
        self.add_final_layer_norm = add_final_layer_norm

        # Params introduced for Marian
        self.add_bias_logits = add_bias_logits
        self.static_position_embeddings = static_position_embeddings

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        # pos embedding offset
        self.extra_pos_embeddings = self.pad_token_id + 1

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model

    def is_valid_mbart(self) -> bool:
        """Is the configuration aligned with the MBART paper."""
        if self.normalize_before and self.add_final_layer_norm and self.scale_embedding:
            return True
        if self.normalize_before or self.add_final_layer_norm or self.scale_embedding:
            logger.info("This configuration is a mixture of MBART and BART settings")
        return False
