
from fairseq.models.transformer import TransformerModel, TransformerEncoder
from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional


class CollapsingEncoder(TransformerEncoder):
    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        embedded_tokens = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * embedded_tokens.sum(dim=2)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens[:, :, 0])
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens[:, :, 0].eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )


@register_model("morpho-transformer")
class MorphoTransformer(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return CollapsingEncoder(args, src_dict, embed_tokens)


@register_model_architecture("morpho-transformer", "morpho-transformer")
def base_architecture(args):
    from fairseq.models.transformer import base_architecture as transformer_base
    transformer_base(args)
