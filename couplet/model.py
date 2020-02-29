from os import path

import tensorflow as tf

from couplet.Config import Config
from couplet.DataLoader import SeqReader
from couplet.Seq2SeqModel import Seq2SeqModel


class Model(Config):
    Seq2Seq = Seq2SeqModel()

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
            self.train_reader = SeqReader(train_input_file, train_target_file, vocab_file, batch_size)
            self.train_reader.start()
            self.train_data = self.train_reader.read()
            self.eval_reader = SeqReader(test_input_file, test_target_file, vocab_file, batch_size)
            self.eval_reader.start()
            self.eval_data = self.eval_reader.read()

        self.model_file = path.join(Config.output_dir, 'model.ckpl')
        self.log_writter = tf.summary.FileWriter(Config.output_dir)

        if init_train:
            self._init_train()
            self._init_eval()

    def gpu_session_config(self):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        return tf_config

    def _init_train(self):
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self.train_in_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.train_in_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_target_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.train_target_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])

            output = self.Seq2Seq.seq2seq_model(self.train_in_seq, self.train_in_seq_len, self.train_target_seq,
                                                self.train_target_seq_len, len(self.train_reader.vocab_file),
                                                self.num_units, self.layers, self.dropout)

            self.train_output = tf.argmax(tf.nn.softmax(output), 2)

            self.loss = self.Seq2Seq.seq_loss(output, self.train_target_seq, self.train_target_seq_len)

            params = tf.trainable_variables

            gradients = tf.gradients(self.loss, params)

            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).apply_gradients(
                zip(clipped_gradients))

            if self.param_histogram:
                for v in tf.trainable_variables():
                    tf.summary.histogram('train_' + v.name, v)

            tf.summary.scalar('loss', self.loss)

            self.train_summary = tf.summary.merge_all()
            self.train_init = tf.global_variables_initializer()
            self.train_saver = tf.train.Saver()
        self.train_session = tf.Session(graph=self.train_graph, config=self.gpu_session_config())

    def _init_eval(self):
        self.eval_graph = tf.Graph()
        with self.eval_graph.as_default():
            self.eval_in_seq = tf.placeholder(tf.int32, shape=[self.batch_size, None])
            self.eval_in_seq_len = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.eval_output = self.Seq2Seq.seq2seq_model(self.eval_in_seq, self.eval_in_seq_len, None, None,
                                                          len(self.eval_reader.vocabs), self.num_units, self.layers,
                                                          self.dropout)
            if self.param_histogram:
                for v in tf.trainable_variables():
                    tf.summary.histogram('eval_' + v.name, v)
            self.eval_summary = tf.summary.merge_all()
            self.eval_saver = tf.train.Saver()
        self.eval_session = tf.Session(graph=self.eval_graph, config=self.gpu_session_config())

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
                target_seq_len = data['target_seq_leq']
                output, loss, train, summary = self.train_session.run(
                    [self.train_output, self.loss, self.train_op, self.train_summary],
                    feed_dict={
                        self.train_in_seq: in_seq,
                        self.train_in_seq_len: in_seq_len,
                        self.train_target_seq: target_seq,
                        self.train_target_seq_len: target_seq_len}
                )

                total_loss += loss

                self.log_writter.add_summary(summary, step)





