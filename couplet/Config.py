from os import path


class CoupletConfig():
    output_dir = '/data/'
    en_file_path = ''
    raw_path = '/data/pro/couplet/raw'
    train_in_path = path.join(raw_path, 'train', 'in.txt')
    train_out_path = path.join(raw_path, 'train', 'out.txt')
    test_in_path = path.join(raw_path, 'test', 'in.txt')
    test_out_path = path.join(raw_path, 'test', 'out.txt')
    vocabs_path = path.join(raw_path, 'vocabs')
