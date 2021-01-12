import math

import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import PReLU
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from matplotlib import pyplot
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import GRU, Embedding
from tensorflow.keras.optimizers import SGD

nr_cuv_diferite = 5000
dim_max = 500

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=nr_cuv_diferite)
X_train = sequence.pad_sequences(X_train, maxlen=dim_max)
X_test = sequence.pad_sequences(X_test, maxlen=dim_max)

# HyperParameters
epochNr = 120
batchSize = 128
max_learning_rate =  0.005
base_learning_rate = 0.0001


# define model
def define_model():
    kernel = tensorflow.keras.initializers.glorot_normal()
    model = Sequential()
    # First GRU layer with Dropout regularisation
    model.add(Embedding(nr_cuv_diferite, 30, input_length=dim_max))
    model.add(GRU(units=64, return_sequences=True,
                  activation='tanh', kernel_initializer=kernel))
    model.add(Dropout(0.5))
    # Second GRU layer
    model.add(GRU(units=64, return_sequences=True,
                  activation='tanh', kernel_initializer=kernel))
    model.add(Dropout(0.4))

    # third GRU layer
    model.add(GRU(units=64, return_sequences=True,
                  activation='tanh', kernel_initializer=kernel))
    model.add(Dropout(0.3))

    # third GRU layer
    model.add(GRU(units=32, return_sequences=True,
                  activation='tanh', kernel_initializer=kernel))
    model.add(Dropout(0.3))
    
    model.add(GRU(units=32, return_sequences=True,
                  activation='tanh', kernel_initializer=kernel))
    model.add(Dropout(0.3))
    # 5th bc, why not??
    model.add(GRU(units=16,
                  activation='tanh', kernel_initializer=kernel))
    model.add(Dropout(0.1))
    # The output layer
    model.add(Dense(units=1))
    # Compiling the RNN
    opt = tensorflow.keras.optimizers.Adam(learning_rate=max_learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])

    return model


model = define_model()
modelNr = 0
# model = load_model("model_accuracy_{}.h5".format(modelNr))
print(model.summary())


class IntUpdatable:
    def __init__(self, x):
        self.value = x


max_accuracy = IntUpdatable(-1e5)
epoch_real_nr = IntUpdatable(0)
step_real_nr = IntUpdatable(0)


class CustomCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, max_accuracy=max_accuracy, epoch_real_nr_local=epoch_real_nr):
        epoch_real_nr_local.value = epoch_real_nr_local.value + 1
        if logs["val_accuracy"] > max_accuracy.value:
            max_accuracy.value = logs["val_accuracy"]
            print("UPDATE max accuracy, {}".format(max_accuracy.value))
            self.model.save("model_accuracy_{}.h5".format(epoch + modelNr))

#     def on_batch_end(self, batch, logs=None, epoch_real_nr_local=epoch_real_nr, step_real_nr_local=step_real_nr):
#         step_real_nr_local.value = step_real_nr_local.value + 1
#         x = (step_real_nr_local.value % ((5 * X_train.shape[0] // batchSize) + 1)) / (5 * X_train.shape[0] // batchSize)
#         # from 0 to 1, every 5 epochs
#         if (step_real_nr_local.value % (10 * X_train.shape[0] // batchSize)) > (5 * X_train.shape[0] // batchSize):
#             x = 1 - x
#             # from 1 to 0 5 - 10 epochs every 10 epochs

#         new_lr = base_learning_rate + (max_learning_rate - base_learning_rate) * min(max(0, x), 1)
#         tensorflow.keras.backend.set_value(self.model.optimizer.lr, new_lr)


history = model.fit(X_train, Y_train,
                    epochs=epochNr, batch_size=batchSize,
                    callbacks=[CustomCallback()],
                    validation_data=(X_test, Y_test),
                    verbose=2)

print("value updated {}".format(max_accuracy.value))
# evaluate model
_, acc = model.evaluate(X_test, Y_test, verbose=0)
print('> %.3f' % (acc * 100.0))


def summarize_diagnostics():
    # plot accuracy
    pyplot.subplot(313)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

    pyplot.show()


summarize_diagnostics()
