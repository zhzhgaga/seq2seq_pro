import queue as Queue
import random



def padding_seq(seq):
    results = []
    max_len = 0
    for i in seq:
        if max_len < len(seq):
            max_len = len(seq)

    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for j in range(l)])
    return results


def encoder_text(words, vocab_indices):
    return [vocab_indices[word] for word in words if word in vocab_indices]


def decoder_text(label, vocabs, end_tokens='</s>'):
    results = []
    for idx in label:
        word = vocabs[idx]
        if word == end_tokens:
            return " ".join(results)
        results.append(word)
    return ' '.join(results)


def load_vocab(vocab_file):
    f = open(vocab_file, 'rb')
    vocabs = [line.decode('utf8')[:-1] for line in f]
    f.close()
    return vocabs


class SeqReader(object):

    def __init__(self, input_file, target_file, vocab_file, batch_size, queue_size=2048, worker_size=2,
                 end_token='</s>', start_token='<s>', padding=True, max_len=50):
        self.input_file = input_file
        self.target_file = target_file
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.worker_size = worker_size
        self.start_token = start_token
        self.end_token = end_token
        self.padding = padding
        self.max_len = max_len

        self.vocabs = load_vocab(vocab_file)
        self.vocab_indice = dict((c, i) for i, c in enumerate(self.vocabs))
        # self.data_queue = Queue(queue_size)

        with open(self.input_file, 'rb') as f:
            for i, l in enumerate(f):
                pass
            f.close()
            self.single_line = i + 1
        self.data_size = int(self.single_line / batch_size)
        self.data_pos = 0
        self._init_reader()

    def _init_reader(self):
        self.data = []
        input_f = open(self.input_file, 'rb')
        target_f = open(self.target_file, 'rb')
        for input_line in input_f:
            input_line = input_line.decode('utf-8')[:-1]
            target_line = target_f.readline().decode('utf-8')[:-1]
            input_words = [x for x in input_line.split(" ") if x != ' ']
            if len(input_words) >= self.max_len:
                input_words = input_words[:self.max_len - 1]
            input_words.append(self.end_token)

            target_words = [x for x in target_line.split(' ') if x != ' ']
            if len(target_words) >= self.max_len:
                target_words = target_words[: self.max_len - 1]
            target_words = ['<s>', ] + target_words

            in_seq = encoder_text(input_words, self.vocab_indice)
            target_seq = encoder_text(target_words, self.vocab_indice)

            self.data.append({
                'in_seq': in_seq,
                'in_seq_len': len(in_seq),
                'target_seq': target_seq,
                'target_seq_len': len(target_seq) - 1
            })
        input_f.close()
        target_f.close()
        self.data_pos = len(self.data)

    def start(self):
        return

    def read_single_data(self):
        if self.data_pos >= len(self.data):
            random.shuffle(self.data)
            self.data_pos = 0
        result = self.data[self.data_pos]
        self.data_pos += 1
        return result

    def read(self):
        while True:
            batch = {
                'in_seq': [],
                'in_seq_len': [],
                'target_seq': [],
                'target_seq_len': []
            }

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
