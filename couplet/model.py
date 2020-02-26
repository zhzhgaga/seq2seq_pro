import tensorflow as tf
from tensorflow.contrib import rnn

from couplet.Config import Config


# class ModelSeq2Seq(Config):

def get_layered_cell(layer_size, num_units, input_keep_prob, output_keep_prob=1.0):
    return rnn.MultiRNNCell([rnn.DropoutWrapper(
        tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=num_units), input_keep_prob, output_keep_prob) for
        i in range(layer_size)])


def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
    bi_layer_size = int(layer_size / 2)
    fw_encoder_cell = get_layered_cell(bi_layer_size, num_units, input_keep_prob)
    bw_encoder_cell = get_layered_cell(bi_layer_size, num_units, input_keep_prob)
    bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell,
                                    cell_bw=bw_encoder_cell,
                                    inputs=embed_input,
                                    sequence_length=in_seq_len,
                                    dtype=embed_input.dtype,
                                    time_major=False)


bi_encoder(embed_input=[1,2], in_seq_len=200, num_units=4, layer_size=10, input_keep_prob=0.3)
