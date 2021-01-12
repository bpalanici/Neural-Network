import gzip
import pickle
import numpy as np
import time


def basic_function(x):
    return x > 0


start = time.process_time()

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()
# Reading the data set

# w_perceptrons = np.zeros(shape=(10, 784))
w_perceptrons = np.random.uniform(-1, 1, 10 * 784).reshape((10, 784))
bias = np.zeros(shape=(10, 1))
delta_w = np.zeros(shape=(10, 784))
delta_b = np.zeros(shape=(10, 1))
real_number_array = np.zeros(shape=(10, 1))
η = 0.005
nr_iterations = 10


# variables

def check():
    nr_total = len(train_set[0])
    nr_good = 0
    for image, real_number in zip(train_set[0], train_set[1]):
        z_all_perceptrons = w_perceptrons.dot(image.reshape(-1, 1)) + bias
        if np.argmax(z_all_perceptrons) == real_number:
            nr_good += 1

    print("Training : ", nr_good / nr_total * 100)

    nr_total = len(valid_set[0])
    nr_good = 0
    for image, real_number in zip(valid_set[0], valid_set[1]):
        z_all_perceptrons = w_perceptrons.dot(image.reshape(-1, 1)) + bias
        if np.argmax(z_all_perceptrons) == real_number:
            nr_good += 1

    print("Validation : ", nr_good / nr_total * 100)

    nr_total = len(test_set[0])
    nr_good = 0
    for image, real_number in zip(test_set[0], test_set[1]):
        z_all_perceptrons = w_perceptrons.dot(image.reshape(-1, 1)) + bias
        if np.argmax(z_all_perceptrons) == real_number:
            nr_good += 1

    print("Test : ", nr_good / nr_total * 100)


while nr_iterations >= 1:
    delta_w.fill(0)
    delta_b.fill(0)
    for image, real_number, step in zip(train_set[0], train_set[1], range(0, len(train_set[0]))):
        real_number_array.fill(0)
        real_number_array[real_number][0] = 1
        z_all_perceptrons = w_perceptrons.dot(image.reshape(-1, 1)) + bias
        # calculating w * x + b
        y_all_perceptrons = basic_function(z_all_perceptrons)
        # y = activation(z) ,in our case x > 0 ? 1 : 0
        delta_w += ((real_number_array - y_all_perceptrons).dot(image.reshape(-1, 1).T) * η)
        delta_b += ((real_number_array - y_all_perceptrons) * η)
        if step % 15 == 0:  # once every 15 images we update the perceptrons
            w_perceptrons += delta_w
            bias += delta_b
            delta_w.fill(0)
            delta_b.fill(0)
    nr_iterations -= 1
    print(nr_iterations)
    # check()

print("Time in seconds to run training : " + str(time.process_time() - start))

check()
print("Time in seconds to run training + check : " + str(time.process_time() - start))
