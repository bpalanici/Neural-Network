import gzip
import math
import pickle
import numpy as np
import time
from random import sample


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp)


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()
# Reading the data set

w_perceptrons_hidden = (1.0 / math.sqrt(784)) * np.random.randn(100, 784)
momentum_w_hidden = np.zeros(shape=(100, 784))
delta_w_hidden = np.zeros(shape=(100, 784))

w_perceptrons_last = (1.0 / math.sqrt(100)) * np.random.randn(10, 100)
momentum_w_last = np.zeros(shape=(10, 100))
delta_w_last = np.zeros(shape=(10, 100))

bias_hidden = (1.0 / math.sqrt(784)) * np.random.randn(100, 1)
momentum_bias_hidden = np.zeros(shape=(100, 1))
delta_b_hidden = np.zeros(shape=(100, 1))

bias_last = (1.0 / math.sqrt(100)) * np.random.randn(10, 1)
momentum_bias_last = np.zeros(shape=(10, 1))
delta_b_last = np.zeros(shape=(10, 1))

real_number_array = np.zeros(shape=(10, 1))

eta = 0.01
nr_epochs = 1000  # FINAL : 1000
momentum_miu = 0.9
lambda_reg = 5
batch_size = 1000
mini_batch_train = 16
l2_reg = eta * lambda_reg / batch_size

indexList = list([i for i in range(len(train_set[0]))])


def check():
    nr_total = len(train_set[0])
    nr_good = 0
    for image, real_number in zip(train_set[0], train_set[1]):
        z_all_perceptrons_hidden = w_perceptrons_hidden.dot(image.reshape(-1, 1)) + bias_hidden
        y_all_perceptrons_hidden = sigmoid(z_all_perceptrons_hidden)
        z_all_perceptrons_final = w_perceptrons_last.dot(y_all_perceptrons_hidden) + bias_last
        if np.argmax(z_all_perceptrons_final) == real_number:
            nr_good += 1

    train_percent = nr_good / nr_total * 100
    print("Train % : ", train_percent)

    nr_good = 0
    nr_total = len(valid_set[0])
    for image, real_number in zip(valid_set[0], valid_set[1]):
        z_all_perceptrons_hidden = w_perceptrons_hidden.dot(image.reshape(-1, 1)) + bias_hidden
        y_all_perceptrons_hidden = sigmoid(z_all_perceptrons_hidden)
        z_all_perceptrons_final = w_perceptrons_last.dot(y_all_perceptrons_hidden) + bias_last
        if np.argmax(z_all_perceptrons_final) == real_number:
            nr_good += 1

    valid_percent = nr_good / nr_total * 100
    print("Valid % : ", valid_percent)

    nr_good = 0
    nr_total = len(test_set[0])
    for image, real_number in zip(test_set[0], test_set[1]):
        z_all_perceptrons_hidden = w_perceptrons_hidden.dot(image.reshape(-1, 1)) + bias_hidden
        y_all_perceptrons_hidden = sigmoid(z_all_perceptrons_hidden)
        z_all_perceptrons_final = w_perceptrons_last.dot(y_all_perceptrons_hidden) + bias_last
        if np.argmax(z_all_perceptrons_final) == real_number:
            nr_good += 1
    test_percent = nr_good / nr_total * 100

    print("Test % : ", test_percent)


start = time.process_time()

start_time = time.time()

for epoch in range(1, nr_epochs + 1):
    delta_w_hidden.fill(0)
    delta_w_last.fill(0)
    delta_b_hidden.fill(0)
    delta_b_last.fill(0)
    step = 0
    for image_index in sample(indexList, batch_size):
        image = train_set[0][image_index]
        real_number = train_set[1][image_index]
        step += 1
        real_number_array.fill(0)
        real_number_array[real_number][0] = 1

        z_all_perceptrons_hidden = w_perceptrons_hidden.dot(image.reshape(-1, 1)) + bias_hidden
        y_all_perceptrons_hidden = sigmoid(z_all_perceptrons_hidden)
        z_all_perceptrons_last = w_perceptrons_last.dot(y_all_perceptrons_hidden) + bias_last
        # calculating w * x + b
        y_all_perceptrons_final = softmax(z_all_perceptrons_last)
        # y = activation(z) ,in our case softMax
        error_last_layer = (real_number_array - y_all_perceptrons_final)
        # error using SoftMax * cross Entropy
        error_hidden_layer = w_perceptrons_last.T.dot(error_last_layer) * \
                             y_all_perceptrons_hidden * (1 - y_all_perceptrons_hidden)
        # Cross Entropy * sigmoid error

        delta_w_last += error_last_layer.dot(y_all_perceptrons_hidden.T) * eta
        delta_b_last += error_last_layer * eta
        delta_w_hidden += error_hidden_layer.dot(image.reshape(-1, 1).T) * eta
        delta_b_hidden += error_hidden_layer * eta

        if (step % mini_batch_train) == 0:
            momentum_w_last = momentum_miu * momentum_w_last + delta_w_last
            w_perceptrons_last = (1.0 - l2_reg) * w_perceptrons_last + momentum_w_last

            momentum_bias_last = momentum_miu * momentum_bias_last + delta_b_last
            bias_last = (1.0 - l2_reg) * bias_last + momentum_bias_last

            momentum_w_hidden = momentum_miu * momentum_w_hidden + delta_w_hidden
            w_perceptrons_hidden = (1.0 - l2_reg) * w_perceptrons_hidden + momentum_w_hidden

            momentum_bias_hidden = momentum_miu * momentum_bias_hidden + delta_b_hidden
            bias_hidden = (1.0 - l2_reg) * bias_hidden + momentum_bias_hidden

            delta_w_hidden.fill(0)
            delta_w_last.fill(0)
            delta_b_hidden.fill(0)
            delta_b_last.fill(0)

    # print("Epoch : ", epoch)
    # print("Time in s :", time.time() - start_time)
    if epoch % int(nr_epochs / 10) == 0:
        print("Epoch : ", epoch)
        print("Time in s (without model check):", time.time() - start_time)
        check()
        start_time = time.time()
    if epoch % int((nr_epochs + 3) / 3) == 0:
        eta /= 10
        batch_size *= 2
        lambda_reg *= 2
        l2_reg = eta * lambda_reg / batch_size
        print("UPDATE!!", eta, batch_size, lambda_reg)

print("\nTime in seconds to run full training : {}\n".format(str(time.process_time() - start)))

start = time.process_time()
check()
print("Time in seconds to run training + validation + test: " + str(time.process_time() - start))
