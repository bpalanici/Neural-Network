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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# train : ~120 - 150 generated, ~ 100 - 150 normal
# ~100 generated, then back to normal ~100
# ~ 70 gamerated 40, 0.4, 0.4, then normal ~ 70
(trainX, trainY), (testX, testY) = cifar10.load_data()

trainY = to_categorical(trainY)
testY = to_categorical(testY)

# HyperParameters
epochNr = 150
batchSize = 128
max_learning_rate = 0.1
base_learning_rate = 0.001


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
    opt = keras.optimizers.SGD(momentum=0.5, nesterov=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# model = define_model()
modelNr = 523
model = load_model("model_accuracy_{}.h5".format(modelNr))
print(model.summary())


class IntUpdatable:
    def __init__(self, x):
        self.value = x


max_accuracy = IntUpdatable(-1e5)
epoch_real_nr = IntUpdatable(0)
step_real_nr = IntUpdatable(0)


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, max_accuracy=max_accuracy, epoch_real_nr_local=epoch_real_nr):
        epoch_real_nr_local.value = epoch_real_nr_local.value + 1
        if epoch > 20:
            if logs["val_accuracy"] > max_accuracy.value:
                max_accuracy.value = logs["val_accuracy"]
                print("UPDATE max accuracy, {}".format(max_accuracy.value))
                self.model.save("model_accuracy_{}.h5".format(epoch + modelNr))

    def on_batch_end(self, batch, logs=None, epoch_real_nr_local=epoch_real_nr, step_real_nr_local=step_real_nr):
        step_real_nr_local.value = step_real_nr_local.value + 1
        x = (step_real_nr_local.value % ((5 * trainX.shape[0] // batchSize) + 1)) / (5 * trainX.shape[0] // batchSize)
        # from 0 to 1, every 5 epochs
        if (step_real_nr_local.value % (10 * trainX.shape[0] // batchSize)) > (5 * trainX.shape[0] // batchSize):
            x = 1 - x
            # from 1 to 0 5 - 10 epochs every 10 epochs

        new_lr = base_learning_rate + (max_learning_rate - base_learning_rate) * min(max(0, x), 1)
        keras.backend.set_value(self.model.optimizer.lr, new_lr)


# data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(trainX)

# history = model.fit(datagen.flow(trainX, trainY, batch_size=batchSize),
#                     steps_per_epoch=trainX.shape[0] // batchSize,
#                     epochs=epochNr, batch_size=batchSize,
#                     callbacks=[CustomCallback()],
#                     validation_data=(testX, testY),
#                     verbose=2)

history = model.fit(trainX, trainY,
                    epochs=epochNr, batch_size=batchSize,
                    callbacks=[CustomCallback()],
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
