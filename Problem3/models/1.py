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


start = time.process_time()

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

eta_max = 0.0005
eta_min = 0.0005
eta = 0.5
# eta_min = 0.000003
nr_epocs = 500
momentum_miu = 0.9
lambda_reg = 0.8
batch_size = 1000

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

    print("Train % : ", nr_good / nr_total * 100)

    nr_total = len(valid_set[0])
    nr_good = 0
    for image, real_number in zip(valid_set[0], valid_set[1]):
        z_all_perceptrons_hidden = w_perceptrons_hidden.dot(image.reshape(-1, 1)) + bias_hidden
        y_all_perceptrons_hidden = sigmoid(z_all_perceptrons_hidden)
        z_all_perceptrons_final = w_perceptrons_last.dot(y_all_perceptrons_hidden) + bias_last
        if np.argmax(z_all_perceptrons_final) == real_number:
            nr_good += 1

    print("Valid % : ", nr_good / nr_total * 100)

    nr_total = len(test_set[0])
    nr_good = 0
    for image, real_number in zip(test_set[0], test_set[1]):
        z_all_perceptrons_hidden = w_perceptrons_hidden.dot(image.reshape(-1, 1)) + bias_hidden
        y_all_perceptrons_hidden = sigmoid(z_all_perceptrons_hidden)
        z_all_perceptrons_final = w_perceptrons_last.dot(y_all_perceptrons_hidden) + bias_last
        if np.argmax(z_all_perceptrons_final) == real_number:
            nr_good += 1

    print("Test % : ", nr_good / nr_total * 100)


for epoch in range(1, nr_epocs + 1):
    delta_w_hidden.fill(0)
    delta_w_last.fill(0)
    delta_b_hidden.fill(0)
    delta_b_last.fill(0)
    step = 0
    start_time = time.time()
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

        eta = (eta_min - eta_max) / batch_size * step + eta_max
        # calculating the η(eta) to decrease from eta_max to eta_min lineally

        delta_w_last += error_last_layer.dot(y_all_perceptrons_hidden.T) * eta
        delta_b_last += error_last_layer * eta
        delta_w_hidden += error_hidden_layer.dot(image.reshape(-1, 1).T) * eta
        delta_b_hidden += error_hidden_layer * eta

    eta = eta_max
    momentum_w_last = momentum_miu * momentum_w_last + delta_w_last
    w_perceptrons_last = (1.0 - eta * lambda_reg / 100) * w_perceptrons_last + momentum_w_last

    momentum_bias_last = momentum_miu * momentum_bias_last + delta_b_last
    bias_last = (1.0 - eta * lambda_reg / 100) * bias_last + momentum_bias_last

    momentum_w_hidden = momentum_miu * momentum_w_hidden + delta_w_hidden
    w_perceptrons_hidden = (1.0 - eta * lambda_reg / 784) * w_perceptrons_hidden + momentum_w_hidden

    momentum_bias_hidden = momentum_miu * momentum_bias_hidden + delta_b_hidden
    bias_hidden = (1.0 - eta * lambda_reg / 784) * bias_hidden + momentum_bias_hidden

    # print("Epoch : ", epoch)
    # print("Time in s :", time.time() - start_time)
    if epoch % int(nr_epocs / 10) == 0:
        print("Epoch : ", epoch)
        check()
    if epoch % int(nr_epocs / 3) == 0:
        print("UPDATE!!")
        eta /= 10
        batch_size *= 2

print("Time in seconds : " + str(time.process_time() - start))

start = time.process_time()
check()
print("Time in seconds to run training + validation + test: " + str(time.process_time() - start))
