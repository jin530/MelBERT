import numpy as np
import torch
import torch.nn as nn

from utils import Config
from transformers import AutoTokenizer, AutoModel


class AutoModelForSequenceClassification(nn.Module):
    """Base model for sequence classification"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        target_mask=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target wor. 1 for target word and 0 otherwise.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1].
                It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch.
                It's the mask that we typically use for attention when a batch has varying length sentences.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForTokenClassification(nn.Module):
    """Base model for token classification"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        target_mask,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target wor. 1 for target word and 0 otherwise.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1].
                It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch.
                It's the mask that we typically use for attention when a batch has varying length sentences.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)
        target_output = target_output.sum(1) / target_mask.sum()  # [batch, hideen]

        logits = self.classifier(target_output)
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_SPV(nn.Module):
    """MelBERT with only SPV"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_SPV, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        target_mask,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target wor. 1 for target word and 0 otherwise.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1].
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)  # [batch, hidden]

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        # Get mean value of target output if the target output consistst of more than one token
        target_output = target_output.mean(1)

        logits = self.classifier(torch.cat([target_output, pooled_output], dim=1))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_MIP(nn.Module):
    """MelBERT with only MIP"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_MIP, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the second input token indicies
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the second input.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """
        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)
        target_output = self.dropout(target_output)
        target_output = target_output.sum(1) / target_mask.sum()  # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.sum(1) / target_mask_2.sum()

        logits = self.classifier(torch.cat([target_output_2, target_output], dim=1))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits


class AutoModelForSequenceClassification_SPV_MIP(nn.Module):
    """MelBERT"""

    def __init__(self, args, Model, config, num_labels=2):
        """Initialize the model"""
        super(AutoModelForSequenceClassification_SPV_MIP, self).__init__()
        self.num_labels = num_labels
        self.encoder = Model
        self.config = config
        self.dropout = nn.Dropout(args.drop_ratio)
        self.args = args

        self.SPV_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.MIP_linear = nn.Linear(config.hidden_size * 2, args.classifier_hidden)
        self.classifier = nn.Linear(args.classifier_hidden * 2, num_labels)
        self._init_weights(self.SPV_linear)
        self._init_weights(self.MIP_linear)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids,
        input_ids_2,
        target_mask,
        target_mask_2,
        attention_mask_2,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        head_mask=None,
    ):
        """
        Inputs:
            `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the first input token indices in the vocabulary
            `input_ids_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the second input token indicies
            `target_mask`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the first input. 1 for target word and 0 otherwise.
            `target_mask_2`: a torch.LongTensor of shape [batch_size, sequence_length] with the mask for target word in the second input. 1 for target word and 0 otherwise.
            `attention_mask_2`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the second input.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices
                selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
            `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1] for the first input.
            `labels`: optional labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
                with indices selected in [0, ..., num_labels].
            `head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
                It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
        """

        # First encoder for full sentence
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]  # [batch, max_len, hidden]
        pooled_output = outputs[1]  # [batch, hidden]

        # Get target ouput with target mask
        target_output = sequence_output * target_mask.unsqueeze(2)

        # dropout
        target_output = self.dropout(target_output)
        pooled_output = self.dropout(pooled_output)

        target_output = target_output.mean(1)  # [batch, hidden]

        # Second encoder for only the target word
        outputs_2 = self.encoder(input_ids_2, attention_mask=attention_mask_2, head_mask=head_mask)
        sequence_output_2 = outputs_2[0]  # [batch, max_len, hidden]

        # Get target ouput with target mask
        target_output_2 = sequence_output_2 * target_mask_2.unsqueeze(2)
        target_output_2 = self.dropout(target_output_2)
        target_output_2 = target_output_2.mean(1)

        # Get hidden vectors each from SPV and MIP linear layers
        SPV_hidden = self.SPV_linear(torch.cat([pooled_output, target_output], dim=1))
        MIP_hidden = self.MIP_linear(torch.cat([target_output_2, target_output], dim=1))

        logits = self.classifier(self.dropout(torch.cat([SPV_hidden, MIP_hidden], dim=1)))
        logits = self.logsoftmax(logits)

        if labels is not None:
            loss_fct = nn.NLLLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        return logits
