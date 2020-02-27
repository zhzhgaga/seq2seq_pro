import tensorflow as tf
from tensorflow.contrib import rnn


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
    encoder_output = tf.concat(bi_encoder_output, -1)
    encoder_state = []
    for layer_id in range(layer_size):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encoder_output, encoder_state


def attention_decoder_cell(encoder_input, in_sel_len, unm_units, layer_size, input_keep_prob):

