import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense
from matplotlib import pyplot
from tensorflow.keras.layers import GRU, Embedding, Input, Add, Dropout
from tensorflow.python.keras.utils.vis_utils import plot_model

nr_cuv_diferite = 5000
dim_max = 500

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=nr_cuv_diferite)
X_train = sequence.pad_sequences(X_train, maxlen=dim_max)
X_test = sequence.pad_sequences(X_test, maxlen=dim_max)

# HyperParameters
epochNr = 100
batchSize = 128
max_learning_rate = 0.1

# define model
def define_model():
    kernel = tensorflow.keras.initializers.LecunNormal()
    activation = tensorflow.keras.activations.tanh
    # First GRU layer with Dropout regularisation
    input = Input(shape=(dim_max))
    modelIn = Embedding(nr_cuv_diferite, 35, input_length=dim_max)(input)

    Gru1 = GRU(units=40, return_sequences=True,
               activation=activation, kernel_initializer=kernel)(modelIn)

    # Second GRU layer
    Gru2 = GRU(units=40, return_sequences=True,
               activation=activation, kernel_initializer=kernel)(modelIn)

    Gru3 = GRU(units=40, return_sequences=True,
               activation=activation, kernel_initializer=kernel)(modelIn)

    # add layer
    model = Add()([Gru1, Gru2, Gru3])

    # third GRU layer
    Gru4 = GRU(units=35, return_sequences=True,
               activation=activation, kernel_initializer=kernel)(model)

    # third GRU layer
    Gru5 = GRU(units=35, return_sequences=True,
               activation=activation, kernel_initializer=kernel)(model)
    # add layer
    model = Add()([Gru4, Gru5])
    model = Dropout(0.2)(model)

    Gru6 = GRU(units=35, return_sequences=True,
               activation=activation, kernel_initializer=kernel)(model)

    Gru7 = GRU(units=35,
               activation=activation, kernel_initializer=kernel)(model)

    model = Add()([Gru6, Gru7, model, Gru4, Gru5])
    model = Dropout(0.1)(model)
    # The output layer
    output = Dense(units=1, activation=tensorflow.keras.activations.sigmoid)(model)
    # Compiling the RNN
    opt = tensorflow.keras.optimizers.Adam(learning_rate=max_learning_rate / 10, amsgrad=True)

    model = tensorflow.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])

    return model


model = define_model()
modelNr = 0
print(model.summary())
# exit(0)
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


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
    pyplot.subplot(313)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

    pyplot.show()


summarize_diagnostics()
