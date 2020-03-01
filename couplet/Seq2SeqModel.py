import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core

from couplet.Config import CoupletConfig


class Seq2SeqModel():
    config = CoupletConfig()

    def get_layered_cell(self, layer_size, num_units, input_keep_prob, output_keep_prob=1.0):
        return rnn.MultiRNNCell([rnn.DropoutWrapper(
            tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=num_units), input_keep_prob, output_keep_prob) for
            i in range(layer_size)])

    def bi_encoder(self, embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
        bi_layer_size = int(layer_size / 2)
        fw_encoder_cell = self.get_layered_cell(bi_layer_size, num_units, input_keep_prob)
        bw_encoder_cell = self.get_layered_cell(bi_layer_size, num_units, input_keep_prob)
        bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_encoder_cell,
                                                                              cell_bw=bw_encoder_cell,
                                                                              inputs=embed_input,
                                                                              sequence_length=in_seq_len,
                                                                              dtype=embed_input.dtype,
                                                                              time_major=False)
        encoder_output = tf.concat(bi_encoder_output, -1)
        encoder_state = []
        for layer_id in range(bi_layer_size):
            encoder_state.append(bi_encoder_state[0][layer_id])
            encoder_state.append(bi_encoder_state[1][layer_id])
        encoder_state = tuple(encoder_state)
        return encoder_output, encoder_state

    def attention_decoder_cell(self, encoder_output, in_seq_len, num_units, layer_size, input_keep_prob):
        attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(num_units, encoder_output, in_seq_len, normalize=True)
        cell = self.get_layered_cell(layer_size, num_units, input_keep_prob)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim, attention_layer_size=num_units)
        return cell

    def decoder_projection(self, output, output_size):
        return tf.layers.dense(output, output_size, activation=True, use_bias=False, name='output_mlp')

    def train_decoder(self, encoder_output, encoder_state, in_seq_leq, target_len, target_seq_len, num_units,
                      layer_size,
                      embedding, output_size, input_keep_pro, projection_layer):
        decoder_cell = self.attention_decoder_cell(encoder_output, in_seq_leq, num_units, layer_size, input_keep_pro)

        batch_size = tf.shape(in_seq_leq)[0]

        init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

        train_helper = tf.contrib.seq2seq.TrainingHelper(target_len, target_seq_len, time_major=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, init_state, output_layer=projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=100)

        return outputs.rnn_output

    def infer_decoder(self, encoder_state, encoder_output, in_seq_leq, num_units, layer_size, input_keep_prob,
                      embedding,
                      projection_layer):
        decoder_cell = self.attention_decoder_cell(encoder_output, in_seq_leq, num_units, layer_size, input_keep_prob)
        batch_size = tf.shape(in_seq_leq)[0]
        init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=embedding,
            start_tokens=tf.fill([batch_size], 0),
            end_token=1,
            initial_state=init_state,
            beam_width=10,
            output_layer=projection_layer,
            length_penalty_weight=1.0)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=100)
        return outputs.sample_id

    def seq2seq_model(self, in_seq, in_seq_len, target_seq, target_seq_len, vocab_size, num_units, layer_size, dropout):
        in_shape = tf.shape(in_seq)
        batch_size = in_shape[0]

        if target_seq is not None:
            input_keep_prob = 1 - dropout
        else:
            input_keep_prob = 1

        projection_layer = layers_core.Dense(vocab_size, use_bias=False)

        with tf.device('/gpu:0'):
            embedding = tf.get_variable(name='embedding', shape=[vocab_size, num_units])

            embedding_input = tf.nn.embedding_lookup(embedding, in_seq, name='embedding_input')

            encoder_output, encoder_state = self.bi_encoder(embedding_input, in_seq_len, num_units, layer_size,
                                                            input_keep_prob)

            decoder_cell = self.attention_decoder_cell(encoder_output, in_seq_len, num_units, layer_size,
                                                       input_keep_prob)

            batch_size = tf.shape(in_seq_len)[0]

            init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

            if target_seq is not None:
                embedding_target = tf.nn.embedding_lookup(embedding, target_seq, name='embedding_target')
                helper = tf.contrib.seq2seq.TrainingHelper(embedding_target, target_seq_len, time_major=False)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, tf.fill([batch_size], 0), 1)

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, init_state, output_layer=projection_layer)

            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=100)

            if target_seq is not None:
                return outputs.rnn_output
            else:
                return outputs.sample_id

    def seq_loss(self, output, target, seq_len):
        target = target[:, 1:]
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=target)
        batch_size = tf.shape(target)[0]
        loss_mark = tf.sequence_mask(seq_len, tf.shape(output)[1])
        cost = cost * tf.to_float(loss_mark)
        return tf.reduce_sum(cost) / tf.to_float(batch_size)
