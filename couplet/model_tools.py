from couplet.Config import CoupletConfig
from couplet.model import SeqModel

if __name__ == "__main__":
    config = CoupletConfig()
    m = SeqModel(
        config.train_in_path,
        config.train_out_path,
        config.test_in_path,
        config.test_out_path,
        config.vocabs_path,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.001,
        output_dir='./models/output_couplet',
        restore_model=False)

    m.train(5)
