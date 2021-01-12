from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import load_model
from tensorflow.python.keras import Input

# if __name__ == '__main__':
#     # Load the dataset
#     (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
#     model = load_model("model.h5")
#     scores = model.evaluate(X_test, to_categorical(Y_test))
#     model.summary()
#     print('Loss: %.3f' % scores[0])
#     print('Accuracy: %.3f' % scores[1])

# DELETE EVERYTHING IS BELOW THIS!!

import math

import tensorflow
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import BatchNormalization, Lambda, LayerNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import PReLU
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling1D
from matplotlib import pyplot
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import GRU, Embedding, Add
from tensorflow.keras.optimizers import SGD

nr_cuv_diferite = 5000
dim_max = 500
max_learning_rate = 15


def define_model():
    conv_nr = 8
    kernel_conv1D = tensorflow.keras.initializers.he_uniform()
    kernel = tensorflow.keras.initializers.LecunNormal()
    activation = tensorflow.keras.activations.tanh
    # First GRU layer with Dropout regularisation
    input = Input(shape=nr_cuv_diferite)
    model = Embedding(nr_cuv_diferite, 25, input_length=dim_max)(input)

    Gru1 = GRU(units=64, return_sequences=True,
                activation=activation, kernel_initializer=kernel)(model)
    model = (LayerNormalization(axis=1, center=True, scale=True))(Gru1)
    model = Dropout(0.4)(model)
    # Second GRU layer
    Gru2 = GRU(units=64, return_sequences=True,
                activation=activation, kernel_initializer=kernel)(model)
    #model = LayerNormalization(axis=1, center=True, scale=True)(Gru2)
    model = Add()([model, Gru2])
    model = Dropout(0.3)(model)

    # third GRU layer
    Gru3 = GRU(units=64, return_sequences=True,
                  activation=activation, kernel_initializer=kernel)(model)
    model = LayerNormalization(axis=1, center=True, scale=True)(Gru3)
    model = Dropout(0.2)(model)

    model = Add()([model, Gru2, Gru1])
    Gru4 = GRU(units=64, return_sequences=True,
                  activation=activation, kernel_initializer=kernel)(model)
    model = LayerNormalization(axis=1, center=True, scale=True)(Gru4)
    model = Dropout(0.1)(model)

    model = Add()([model, Gru2, Gru3])
    # The output layer
    output = Dense(units=1, activation=tensorflow.keras.activations.sigmoid)(model)
    # Compiling the RNN
    opt = tensorflow.keras.optimizers.Adam(learning_rate=max_learning_rate, amsgrad=True)

    model = tensorflow.keras.Model(inputs=input, outputs=output, name="mnist_model")
    model.compile(optimizer=opt, loss='mean_squared_error',
                  metrics=['accuracy'])

    return model

print((define_model()).summary())

