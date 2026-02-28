import torch.nn as nn
import torch.nn.functional as F


class Seq2seq(nn.Module):
    def __init__(
        self, encoder, decoder, decode_function=F.log_softmax, task_type="generate"
    ):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.task_type = task_type

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        try:
            self.decoder.rnn.flatten_parameters()
        except:
            pass

    def forward(
        self,
        input_variable,
        input_lengths=None,
        target_variable=None,
        teacher_forcing_ratio=0,
        embedded=None,
        already_one_hot=False,
        get_reps=False,
    ):
        # (batch_size, seq_len, hidden_size)
        encoder_outputs, encoder_hidden = self.encoder(
            input_variable,
            input_lengths,
            embedded=embedded,
            already_one_hot=already_one_hot,
        )
        if self.task_type == "classify":
            result = self.decoder(encoder_outputs[:, 0, :].squeeze(1))
        else:
            result = self.decoder(
                inputs=target_variable,
                encoder_hidden=encoder_hidden,
                encoder_outputs=encoder_outputs,
                function=self.decode_function,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

        if get_reps:
            return result, (encoder_outputs, encoder_hidden)
        return result
