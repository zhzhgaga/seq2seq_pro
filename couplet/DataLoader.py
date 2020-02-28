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


