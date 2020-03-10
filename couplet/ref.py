from queue import Queue

'''
序列截断与补齐，保持一样的长度
'''


def padding_seq(seq):
    results = []
    max_len = 0
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for j in range(l)])
    return results


'''
把文本序列映射为下标id序列
'''


def encode_text(words, vocab_indices):
    return [vocab_indices[word] for word in words if word in vocab_indices]


'''
把输出的下标id序列映射回文本序列
'''


def decode_text(labels, vocabs, end_token='</s>'):
    results = []
    for idx in labels:
        word = vocabs[idx]
        if word == end_token:
            return ' '.join(results)
        results.append(word)
    return ' '.join(results)


'''
加载词表
'''


def read_vocab(vocab_file):
    f = open(vocab_file, 'rb')
    vocabs = [line.decode('utf8')[:-1] for line in f]
    f.close()
    return vocabs


'''
数据读取器
'''


class SeqReader():
    def __init__(self, input_file, target_file, vocab_file, batch_size,
                 queue_size=2048, worker_size=2, end_token='</s>',
                 padding=True, max_len=50):
        self.input_file = input_file
        self.target_file = target_file
        self.end_token = end_token
        self.batch_size = batch_size
        self.padding = padding
        self.max_len = max_len
        # 读取词汇表
        self.vocabs = read_vocab(vocab_file)
        # 构建词汇与下标对应的字典
        self.vocab_indices = dict((c, i) for i, c in enumerate(self.vocabs))
        self.data_queue = Queue(queue_size)
        self.worker_size = worker_size
        # 计算全量数据有多少个batch
        with open(self.input_file, 'rb') as f:
            for i, l in enumerate(f):
                pass
            f.close()
            self.single_lines = i + 1
        self.data_size = int(self.single_lines / batch_size)
        self.data_pos = 0
        self._init_reader()

    def start(self):
        return

    '''
        for i in range(self.worker_size):
            t = Thread(target=self._init_reader())
            t.daemon = True
            t.start()
    '''

    # 读取一个batch的数据
    def read_single_data(self):
        if self.data_pos >= len(self.data):
            random.shuffle(self.data)
            self.data_pos = 0
        result = self.data[self.data_pos]
        self.data_pos += 1
        return result

    # 读取数据到batch字典中
    def read(self):
        while True:
            batch = {'in_seq': [],
                     'in_seq_len': [],
                     'target_seq': [],
                     'target_seq_len': []}
            for i in range(0, self.batch_size):
                item = self.read_single_data()
                batch['in_seq'].append(item['in_seq'])
                batch['in_seq_len'].append(item['in_seq_len'])
                batch['target_seq'].append(item['target_seq'])
                batch['target_seq_len'].append(item['target_seq_len'])
            if self.padding:
                batch['in_seq'] = padding_seq(batch['in_seq'])
                batch['target_seq'] = padding_seq(batch['target_seq'])
            yield batch

    # 读取文件，准备成序列对
    def _init_reader(self):
        self.data = []
        input_f = open(self.input_file, 'rb')
        target_f = open(self.target_file, 'rb')
        for input_line in input_f:
            input_line = input_line.decode('utf-8')[:-1]
            target_line = target_f.readline().decode('utf-8')[:-1]
            input_words = [x for x in input_line.split(' ') if x != '']
            if len(input_words) >= self.max_len:
                input_words = input_words[:self.max_len - 1]
            input_words.append(self.end_token)

            target_words = [x for x in target_line.split(' ') if x != '']
            if len(target_words) >= self.max_len:
                target_words = target_words[:self.max_len - 1]
            target_words = ['<s>', ] + target_words

            target_words.append(self.end_token)
            in_seq = encode_text(input_words, self.vocab_indices)
            target_seq = encode_text(target_words, self.vocab_indices)

            self.data.append({
                'in_seq': in_seq,
                'in_seq_len': len(in_seq),
                'target_seq': target_seq,
                'target_seq_len': len(target_seq) - 1
            })
        input_f.close()
        target_f.close()
        self.data_pos = len(self.data)


# Copyright 2017 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core


# 设定LSTM的cell类型
def getLayeredCell(layer_size, num_units, input_keep_prob,
                   output_keep_prob=1.0):
    return rnn.MultiRNNCell([rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=num_units),
                                                input_keep_prob, output_keep_prob) for i in range(layer_size)])


# 双向RNN
def bi_encoder(embed_input, in_seq_len, num_units, layer_size, input_keep_prob):
    # 对输入编码
    bi_layer_size = int(layer_size / 2)
    encode_cell_fw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    encode_cell_bw = getLayeredCell(bi_layer_size, num_units, input_keep_prob)
    bi_encoder_output, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encode_cell_fw,
        cell_bw=encode_cell_bw,
        inputs=embed_input,
        sequence_length=in_seq_len,
        dtype=embed_input.dtype,
        time_major=False)

    # 拼接 编码的output和state
    encoder_output = tf.concat(bi_encoder_output, -1)
    encoder_state = []
    for layer_id in range(bi_layer_size):
        encoder_state.append(bi_encoder_state[0][layer_id])
        encoder_state.append(bi_encoder_state[1][layer_id])
    encoder_state = tuple(encoder_state)
    return encoder_output, encoder_state


# 加“注意力”的解码器
def attention_decoder_cell(encoder_output, in_seq_len, num_units, layer_size,
                           input_keep_prob):
    # 可以选择不同的注意力机制
    attention_mechanim = tf.contrib.seq2seq.BahdanauAttention(num_units,
                                                              encoder_output, in_seq_len, normalize=True)
    # attention_mechanim = tf.contrib.seq2seq.LuongAttention(num_units,
    #         encoder_output, in_seq_len, scale = True)
    cell = getLayeredCell(layer_size, num_units, input_keep_prob)
    cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanim,
                                               attention_layer_size=num_units)
    return cell


# 输出端的全连接层
def decoder_projection(output, output_size):
    return tf.layers.dense(output, output_size, activation=None,
                           use_bias=False, name='output_mlp')


# 训练阶段解码器部分
def train_decoder(encoder_output, in_seq_len, target_seq, target_seq_len,
                  encoder_state, num_units, layers, embedding, output_size,
                  input_keep_prob, projection_layer):
    # 解码结构的cell
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
                                          layers, input_keep_prob)
    # batch size
    batch_size = tf.shape(in_seq_len)[0]
    # 初始状态
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
        cell_state=encoder_state)
    # 训练器
    helper = tf.contrib.seq2seq.TrainingHelper(
        target_seq, target_seq_len, time_major=False)
    # 解码器
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                              init_state, output_layer=projection_layer)
    # 解码输出
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      maximum_iterations=100)
    return outputs.rnn_output


# 预测阶段的解码过程
def infer_decoder(encoder_output, in_seq_len, encoder_state, num_units, layers,
                  embedding, output_size, input_keep_prob, projection_layer):
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
                                          layers, input_keep_prob)

    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
        cell_state=encoder_state)

    # TODO: start tokens and end tokens are hard code
    """
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, tf.fill([batch_size], 0), 1)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
            init_state, output_layer=projection_layer)
    """
    # 使用beam search解码
    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=decoder_cell,
        embedding=embedding,
        start_tokens=tf.fill([batch_size], 0),
        end_token=1,
        initial_state=init_state,
        beam_width=10,
        output_layer=projection_layer,
        length_penalty_weight=1.0)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      maximum_iterations=100)
    return outputs.sample_id


# 序列到序列模型
def seq2seq(in_seq, in_seq_len, target_seq, target_seq_len, vocab_size,
            num_units, layers, dropout):
    in_shape = tf.shape(in_seq)
    batch_size = in_shape[0]

    if target_seq != None:
        input_keep_prob = 1 - dropout
    else:
        input_keep_prob = 1

    projection_layer = layers_core.Dense(vocab_size, use_bias=False)

    # 对输入和输出序列做embedding
    with tf.device('/gpu:0'):
        embedding = tf.get_variable(
            name='embedding',
            shape=[vocab_size, num_units])
    embed_input = tf.nn.embedding_lookup(embedding, in_seq, name='embed_input')

    # 编码
    encoder_output, encoder_state = bi_encoder(embed_input, in_seq_len,
                                               num_units, layers, input_keep_prob)

    # 解码
    decoder_cell = attention_decoder_cell(encoder_output, in_seq_len, num_units,
                                          layers, input_keep_prob)
    batch_size = tf.shape(in_seq_len)[0]
    init_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
        cell_state=encoder_state)

    if target_seq != None:
        embed_target = tf.nn.embedding_lookup(embedding, target_seq, name='embed_target')
        helper = tf.contrib.seq2seq.TrainingHelper(embed_target, target_seq_len, time_major=False)
    else:
        # TODO: start tokens and end tokens are hard code
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, tf.fill([batch_size], 0), 1)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                              init_state, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      maximum_iterations=100)
    if target_seq != None:
        return outputs.rnn_output
    else:
        return outputs.sample_id


# 损失函数
def seq_loss(output, target, seq_len):
    target = target[:, 1:]
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                          labels=target)
    batch_size = tf.shape(target)[0]
    loss_mask = tf.sequence_mask(seq_len, tf.shape(output)[1])
    cost = cost * tf.to_float(loss_mask)
    return tf.reduce_sum(cost) / tf.to_float(batch_size)


import tensorflow as tf
from os import path
import random


class Model():

    def __init__(self, train_input_file, train_target_file,
                 test_input_file, test_target_file, vocab_file,
                 num_units, layers, dropout,
                 batch_size, learning_rate, output_dir,
                 save_step=100, eval_step=1000,
                 param_histogram=False, restore_model=False,
                 init_train=True, init_infer=False):
        self.num_units = num_units
        self.layers = layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_step = save_step
        self.eval_step = eval_step
        self.param_histogram = param_histogram
        self.restore_model = restore_model
        self.init_train = init_train
        self.init_infer = init_infer

        if init_train:
            self.train_reader = SeqReader(train_input_file,
                                          train_target_file, vocab_file, batch_size)
            self.train_reader.start()
            self.train_data = self.train_reader.read()
            self.eval_reader = SeqReader(test_input_file, test_target_file,
                                         vocab_file, batch_size)
            self.eval_reader.start()
            self.eval_data = self.eval_reader.read()

        self.model_file = path.join(output_dir, 'model.ckpl')
        self.log_writter = tf.summary.FileWriter(output_dir)

        if init_train:
            self._init_train()
            self._init_eval()

        if init_infer:
            self.infer_vocabs = read_vocab(vocab_file)
            self.infer_vocab_indices = dict((c, i) for i, c in
                                            enumerate(self.infer_vocabs))
            self._init_infer()
            self.reload_infer_model()

    def gpu_session_config(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        return config

    def _init_train(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.train_in_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.train_in_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_target_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.train_target_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])

            output = seq2seq(self.train_in_seq, self.train_in_seq_len,
                             self.train_target_seq, self.train_target_seq_len,
                             len(self.train_reader.vocabs),
                             self.num_units, self.layers, self.dropout)

            self.train_output = tf.argmax(tf.nn.softmax(output), 2)

            self.loss = seq_loss(output, self.train_target_seq, self.train_target_seq_len)

            params = tf.trainable_variables()

            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, 0.5)
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate
            ).apply_gradients(zip(clipped_gradients, params))
            if self.param_histogram:
                for v in tf.trainable_variables():
                    tf.summary.histogram('train_' + v.name, v)
            tf.summary.scalar('loss', self.loss)
            self.train_summary = tf.summary.merge_all()
            self.train_init = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()
        self.train_session = tf.Session(graph=self.train_graph,
                                        config=self.gpu_session_config())

    def _init_eval(self):
        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default():
            self.eval_in_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.eval_in_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.eval_output = seq2seq(self.eval_in_seq,
                                       self.eval_in_seq_len, None, None,
                                       len(self.eval_reader.vocabs),
                                       self.num_units, self.layers, self.dropout)
            if self.param_histogram:
                for v in tf.trainable_variables():
                    tf.summary.histogram('eval_' + v.name, v)
            self.eval_summary = tf.summary.merge_all()
            self.eval_saver = tf.train.Saver()
        self.eval_session = tf.Session(graph=self.eval_graph,
                                       config=self.gpu_session_config())

    def _init_infer(self):
        self.infer_graph = tf.Graph()
        with self.infer_graph.as_default():
            self.infer_in_seq = tf.placeholder(tf.int32, shape=[1, None])
            self.infer_in_seq_len = tf.placeholder(tf.int32, shape=[1])
            self.infer_output = seq2seq(self.infer_in_seq,
                                        self.infer_in_seq_len, None, None,
                                        len(self.infer_vocabs),
                                        self.num_units, self.layers, self.dropout)
            self.infer_saver = tf.train.Saver()
        self.infer_session = tf.Session(graph=self.infer_graph,
                                        config=self.gpu_session_config())

    # 训练
    def train(self, epochs, start=0):
        if not self.init_train:
            raise Exception('Train graph is not inited!')
        with self.train_graph.as_default():
            if path.isfile(self.model_file + '.meta') and self.restore_model:
                print("Reloading model file before training.")
                self.train_saver.restore(self.train_session, self.model_file)
            else:
                self.train_session.run(self.train_init)
            total_loss = 0
            for step in range(start, epochs):
                data = next(self.train_data)
                in_seq = data['in_seq']
                in_seq_len = data['in_seq_len']
                target_seq = data['target_seq']
                target_seq_len = data['target_seq_len']
                output, loss, train, summary = self.train_session.run(
                    [self.train_output, self.loss, self.train_op, self.train_summary],
                    feed_dict={
                        self.train_in_seq: in_seq,
                        self.train_in_seq_len: in_seq_len,
                        self.train_target_seq: target_seq,
                        self.train_target_seq_len: target_seq_len})
                total_loss += loss
                self.log_writter.add_summary(summary, step)
                if step % self.save_step == 0:
                    self.train_saver.save(self.train_session, self.model_file)
                    print("Saving model. Step: %d, loss: %f" % (step,
                                                                total_loss / self.save_step))
                    # print sample output
                    sid = random.randint(0, self.batch_size - 1)
                    input_text = decode_text(in_seq[sid],
                                             self.eval_reader.vocabs)
                    output_text = decode_text(output[sid],
                                              self.train_reader.vocabs)
                    target_text = decode_text(target_seq[sid],
                                              self.train_reader.vocabs).split(' ')[1:]
                    target_text = ' '.join(target_text)
                    print('******************************')
                    print('src: ' + input_text)
                    print('output: ' + output_text)
                    print('target: ' + target_text)
                if step % self.eval_step == 0:
                    bleu_score = self.eval(step)
                    print("Evaluate model. Step: %d, score: %f, loss: %f" % (
                        step, bleu_score, total_loss / self.save_step))
                    eval_summary = tf.Summary(value=[tf.Summary.Value(
                        tag='bleu', simple_value=bleu_score)])
                    self.log_writter.add_summary(eval_summary, step)
                if step % self.save_step == 0:
                    total_loss = 0

    # 评估
    def eval(self, train_step):
        with self.eval_graph.as_default():
            self.eval_saver.restore(self.eval_session, self.model_file)
            bleu_score = 0
            target_results = []
            output_results = []
            for step in range(0, self.eval_reader.data_size):
                data = next(self.eval_data)
                in_seq = data['in_seq']
                in_seq_len = data['in_seq_len']
                target_seq = data['target_seq']
                target_seq_len = data['target_seq_len']
                outputs = self.eval_session.run(
                    self.eval_output,
                    feed_dict={
                        self.eval_in_seq: in_seq,
                        self.eval_in_seq_len: in_seq_len})
                for i in range(len(outputs)):
                    output = outputs[i]
                    target = target_seq[i]
                    output_text = decode_text(output,
                                              self.eval_reader.vocabs).split(' ')
                    target_text = decode_text(target[1:],
                                              self.eval_reader.vocabs).split(' ')
                    prob = int(self.eval_reader.data_size * self.batch_size / 10)
                    target_results.append([target_text])
                    output_results.append(output_text)
                    if random.randint(1, prob) == 1:
                        print('====================')
                        input_text = decode_text(in_seq[i],
                                                 self.eval_reader.vocabs)
                        print('src:' + input_text)
                        print('output: ' + ' '.join(output_text))
                        print('target: ' + ' '.join(target_text))
            return compute_bleu(target_results, output_results)[0] * 100

    def reload_infer_model(self):
        with self.infer_graph.as_default():
            self.infer_saver.restore(self.infer_session, self.model_file)

    def infer(self, text):
        if not self.init_infer:
            raise Exception('Infer graph is not inited!')
        with self.infer_graph.as_default():
            in_seq = encode_text(text.split(' ') + ['</s>', ],
                                 self.infer_vocab_indices)
            in_seq_len = len(in_seq)
            outputs = self.infer_session.run(self.infer_output,
                                             feed_dict={
                                                 self.infer_in_seq: [in_seq],
                                                 self.infer_in_seq_len: [in_seq_len]})
            output = outputs[0]
            output_text = decode_text(output, self.infer_vocabs)
            return output_text


if __name__ == "__main__":
    m = Model(
        'G:/data/pro/couplet/raw\\train\\in.txt',
        'G:/data/pro/couplet/raw\\train\\out.txt',
        'G:/data/pro/couplet/raw\\test\\in.txt',
        'G:/data/pro/couplet/raw\\test\\out.txt',
        'G:/data/pro/couplet/raw\\vocabs',
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='./models/output_couplet',
        restore_model=False)

    m.train(5)
