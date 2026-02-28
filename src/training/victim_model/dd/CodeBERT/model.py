# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Get dropout probability from config, with fallback to 0.1 if not present
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)

        # Classification head following standard BERT/RoBERTa architecture
        # First dropout layer for regularization
        self.dropout = nn.Dropout(dropout_prob)

        # Dense layer: transforms CLS representation (keeps same dimension)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        # Activation function: tanh for non-linearity
        self.activation = nn.Tanh()

        # Output projection layer: reduces to single logit for binary classification
        self.out_proj = nn.Linear(config.hidden_size, 1)

        # Loss function: BCEWithLogitsLoss combines sigmoid and BCE for numerical stability
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input_ids=None, labels=None, each_loss=False):
        # Get encoder outputs: [batch_size, sequence_length, hidden_size]
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]

        # Extract CLS token representation (first token at position 0)
        # Shape: [batch_size, hidden_size]
        # The CLS token is trained to aggregate the semantic meaning of the entire sequence
        cls_output = outputs[:, 0, :]

        # Pass through classification head
        # Step 1: Apply dropout for regularization
        cls_output = self.dropout(cls_output)

        # Step 2: Apply dense layer for transformation
        cls_output = self.dense(cls_output)

        # Step 3: Apply tanh activation for non-linearity
        cls_output = self.activation(cls_output)

        # Step 4: Apply dropout again (optional, for additional regularization)
        cls_output = self.dropout(cls_output)

        # Step 5: Project to final logit
        # Shape: [batch_size, 1]
        logits = self.out_proj(cls_output)

        # Apply sigmoid to get probability
        # Shape: [batch_size, 1]
        prob = torch.sigmoid(logits)

        # Calculate loss if labels are provided
        if labels is not None:
            # Ensure labels have the correct shape [batch_size, 1]
            labels = labels.float().view(-1, 1)

            # Compute loss using logits (more numerically stable than using prob)
            # BCEWithLogitsLoss internally applies sigmoid
            loss = self.loss_fct(logits, labels)

            # Aggregate loss based on each_loss flag
            if not each_loss:
                # Return mean loss across batch (scalar)
                loss = loss.mean()
            else:
                # Return loss for each sample (vector)
                # Squeeze to remove the extra dimension: [batch_size, 1] -> [batch_size]
                loss = loss.squeeze()

            return loss, prob
        else:
            # Inference mode: only return probabilities
            return prob