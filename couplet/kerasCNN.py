from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Concatenate, Dense
from keras.optimizers import Adam


class NgramCNN(object):

    def __init__(self, num_units, max_features, embedding_dims, nums_class, max_len, filters_size, kernel_size,
                 batch_size):
        self.num_units = num_units
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.nums_class = nums_class
        self.max_len = max_len
        self.filters_size = filters_size
        self.kernel_size = kernel_size
        self.batch_size = batch_size

    def get_model(self):
        input = Input(self.max_len)
        embedding = Embedding(input_dim=self.max_features, output_dim=self.embedding_dims, input_length=self.max_len)(
            input)
        convs = []
        for kernel in self.kernel_size:
            c = Conv1D(filters=self.filters_size, kernel=kernel, activation='relu')(embedding)
            c = GlobalMaxPooling1D(c)
            convs.append(c)
        x = Concatenate(convs)
        output = Dense(units=self.nums_class, activation='softmax')(x)
        model = Model(input=input, output=output)
        optimizer = Adam()
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train, y_train, x_test, y_test):
        callbacks = [
            ModelCheckpoint('./cnn_model.h5'),
            EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
        ]

        model = self.get_model()
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=5, callbacks=callbacks,
                  validation_data=(x_test, y_test))
