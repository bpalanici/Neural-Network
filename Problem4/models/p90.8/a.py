import math
import keras
from keras.datasets import cifar10
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.initializers import Constant
from keras.layers import PReLU
from keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from matplotlib import pyplot
from keras import Sequential
from keras.layers import Conv2D
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
#TRAIN : first ~200(250 in my case) epochs with image generator
#next 200 epochs without image generator
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainY = to_categorical(trainY)
testY = to_categorical(testY)

# HyperParameters
epochNr = 200
batchSize = 128
learning_rate = 0.01


# define model
def define_model():
    kernel_conv2D = keras.initializers.he_uniform()

    conv_nr = 32

    model = Sequential()
    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same',
                     input_shape=(32, 32, 3), bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr / 2, (1, 1), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_nr * 2, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 2, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr, (1, 1), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 2, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_nr * 4, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 4, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 2, (1, 1), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr * 4, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64,
                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(2048)),
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.6))

    model.add(Dense(10, activation='softmax', bias_initializer='zeros'))
    # compile model
    # opt = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)#, epsilon=) TO CHANGE THIS IN CASE OF UPGRADE!!!
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    # opt = keras.optimizers.RMSprop(lr=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model = define_model()
model = load_model("model_accuracy_322.h5")
print(model.summary())
#exit(2)


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
                self.model.save("model_loss_{}.h5".format(epoch + 322))
            if logs["val_accuracy"] > max_accuracy.value:
                max_accuracy.value = logs["val_accuracy"]
                print("UPDATE max accuracy, {}".format(max_accuracy.value))
                self.model.save("model_accuracy_{}.h5".format(epoch + 322))


def lr_schedule(epoch):
    lrate = learning_rate
    if epoch > epochNr / 4:
        lrate = learning_rate / 2
    if epoch > epochNr / 2:
        lrate = learning_rate / 4
    if epoch > ((epochNr / 4) * 3):
        lrate = learning_rate / 6
    return lrate


# data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(trainX)

history = model.fit(trainX, trainY,
                    epochs=epochNr, batch_size=batchSize,
                    callbacks=[CustomCallback(), LearningRateScheduler(lr_schedule)],
                    validation_data=(testX, testY),
                    verbose=2)

print("values updated {}, {}".format(min_loss.value, max_accuracy.value))
# evaluate model
_, acc = model.evaluate(testX, testY, verbose=0)
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