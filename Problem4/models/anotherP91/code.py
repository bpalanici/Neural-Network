import math
import keras
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Lambda
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
#train : ~120 - 150 generated, ~ 100 - 150 normal
# ~100 generated, then back to normal ~100
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainY = to_categorical(trainY)
testY = to_categorical(testY)

# HyperParameters
epochNr = 150
batchSize = 128
learning_rate = 0.15


# define model
def define_model():
    kernel_conv2D = keras.initializers.he_uniform()

    conv_nr = 32

    model = Sequential()
    model.add(keras.Input(shape=(32, 32, 3)))
    model.add(Lambda(lambda x: x / 255.0))
    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr, (3, 3), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))

    model.add(Conv2D(conv_nr, (1, 1), kernel_initializer=kernel_conv2D, padding='same', bias_initializer='zeros'))
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

    model.add(Dense(32,
                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(1024)),
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=Constant(value=0.25)))
    model.add(Dropout(0.5))

    model.add(Dense(10, kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(64)),
                    activation='softmax', bias_initializer='zeros'))
    # compile model
    # opt = tf.keras.optimizers.Adam(lr=learning_rate, amsgrad=True)#, epsilon=) TO CHANGE THIS IN CASE OF UPGRADE!!!
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.5, nesterov=True)
    # opt = keras.optimizers.RMSprop(lr=learning_rate)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model = define_model()
modelNr = 312
model = load_model("model_accuracy_{}.h5".format(modelNr))
print(model.summary())


class IntUpdatable:
    def __init__(self, x):
        self.value = x


max_accuracy = IntUpdatable(-1e5)


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, max_accuracy=max_accuracy):
        if epoch > 20:
            if logs["val_accuracy"] > max_accuracy.value:
                max_accuracy.value = logs["val_accuracy"]
                print("UPDATE max accuracy, {}".format(max_accuracy.value))
                self.model.save("model_accuracy_{}.h5".format(epoch + modelNr))


def lr_schedule(epoch):
    if epoch <= epochNr / 3:
        lrate = (learning_rate / 15) * (1 + 14 * (epoch + 1) / (epochNr / 3))
        # linear from lr / 15 to lr
    elif epoch <= 2 * epochNr / 3:
        lrate = learning_rate * (1 - (14 / 15) * (epoch - epochNr / 3) / (epochNr / 3))
        # linear from lr to lr / 15
    else:
        # linear from lr / 15 to lr / 150
        lrate = learning_rate / 15 * (1 - (9 / 10) * (epoch - 2 * epochNr / 3) / (epochNr / 3))
    return lrate


# data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(trainX)

# history = model.fit(datagen.flow(trainX, trainY, batch_size=batchSize),
#                     steps_per_epoch=trainX.shape[0] // batchSize,
#                     epochs=epochNr, batch_size=batchSize,
#                     callbacks=[CustomCallback(), LearningRateScheduler(lr_schedule)],
#                     validation_data=(testX, testY),
#                     verbose=2)

history = model.fit(trainX, trainY,
                    epochs=epochNr, batch_size=batchSize,
                    callbacks=[CustomCallback(), LearningRateScheduler(lr_schedule)],
                    validation_data=(testX, testY),
                    verbose=2)

print("value updated {}".format(max_accuracy.value))
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
