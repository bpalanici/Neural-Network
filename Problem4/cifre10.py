import math
import keras
from keras.datasets import mnist
from keras.layers import BatchNormalization
from keras.layers import Dropout, Lambda
from keras.initializers import Constant
from keras.layers import PReLU
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from matplotlib import pyplot
from keras import Sequential
from keras.layers import Conv2D
from tensorflow.python.keras.callbacks import LearningRateScheduler

(x_train, y_train), (x_test, y_test) = mnist.load_data()

trainY = to_categorical(y_train)
testY = to_categorical(y_test)

trainX = (x_train.astype('float32') / 255.0)
testX = (x_test.astype('float32') / 255.0)
image_size = x_train.shape[1]

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# TODO : REMOVE THIS!!!

# HyperParameters
epochNr = 400
batchSize = 128
learning_rate = 0.01


# define model
def define_model():
    kernel_conv2D = keras.initializers.he_uniform()

    conv_nr = 32

    model = Sequential()
    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same',
                     input_shape=(image_size, image_size, 1), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(conv_nr * 2, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 2, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 2, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(conv_nr * 4, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 4, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 2, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128,
                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(2048)),
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.4))

    model.add(Dense(64,
                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(2048)),
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax', bias_initializer='zeros'))
    # compile model
    # opt = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)#, epsilon=) TO CHANGE THIS IN CASE OF UPGRADE!!!
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    # opt = keras.optimizers.RMSprop(lr=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = define_model()

print(model.summary())


# exit(2)


# fit model


class IntUpdatable:
    def __init__(self, x):
        self.value = x


max_accuracy = IntUpdatable(-1e5)
min_loss = IntUpdatable(1e5)


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, max_accuracy=max_accuracy, min_loss=min_loss):
        if epoch > 20:
            if logs["val_loss"] < min_loss.value:
                min_loss.value = logs["val_loss"]
                print("UPDATE min loss, {}".format(min_loss.value))
                self.model.save("model_loss_{}.h5".format(epoch))
            if logs["val_accuracy"] > max_accuracy.value:
                max_accuracy.value = logs["val_accuracy"]
                print("UPDATE min accuracy, {}".format(max_accuracy.value))
                self.model.save("model_accuracy_{}.h5".format(epoch))


def lr_schedule(epoch):
    lrate = learning_rate
    if epoch > epochNr / 4:
        lrate = learning_rate / 2
    if epoch > epochNr / 2:
        lrate = learning_rate / 4
    if epoch > ((epochNr / 4) * 3):
        lrate = learning_rate / 8
    return lrate


history = model.fit(trainX, trainY, epochs=epochNr, batch_size=batchSize,
                    callbacks=[LearningRateScheduler(lr_schedule)], validation_data=(testX, testY),
                    verbose=2)
print("values updated {}, {}".format(min_loss.value, max_accuracy.value))
# evaluate model
_, acc = model.evaluate(x_test, testY, verbose=0)
print('> %.3f' % (acc * 100.0))


def summarize_diagnostics():
    # plot loss
    pyplot.subplot(311)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(313)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')

    pyplot.show()


summarize_diagnostics()
