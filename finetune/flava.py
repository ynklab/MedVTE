from typing import Optional

from transformers import FlavaPreTrainedModel, FlavaModel
from transformers.modeling_outputs import ModelOutput
import torch
from torch import nn
from torchmultimodal.modules.layers.mlp import MLP


class FlavaForImagesAndTextClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class FlavaForImagesAndTextClassification(FlavaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.flava = FlavaModel(config)

        self.classifier = MLP(
            in_dim=config.hidden_size,
            out_dim=config.num_labels,
            hidden_dims=1536,
            dropout=0.0,
            activation=nn.ReLU,
            normalization=None,
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_attention_mask: Optional[torch.Tensor] = None,
        skip_multimodal_encoder: Optional[bool] = None,
        # head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: bool = True,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # forward image through the model
        outputs = self.flava(
            input_ids,
            pixel_values=pixel_values if pixel_values is not None else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bool_masked_pos=bool_masked_pos,
            position_ids=position_ids,
            image_attention_mask=image_attention_mask,
            skip_multimodal_encoder=skip_multimodal_encoder,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooler_output = outputs.multimodal_output.pooler_output

        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return FlavaForImagesAndTextClassificationOutput(
            loss=loss,
            logits=logits,
        )
