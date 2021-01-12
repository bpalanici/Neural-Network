import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

positive_x = np.random.randint(low=55, high=101, size=50)
positive_y = np.random.randint(low=0, high=101, size=50)
negative_x = np.random.randint(low=0, high=46, size=50)
negative_y = np.random.randint(low=0, high=101, size=50)

positive = np.column_stack((positive_x, positive_y, np.full(50, 1))).astype(np.float)
negative = np.column_stack((negative_x, negative_y, np.full(50, 1))).astype(np.float)

precision = 40  # (low_ab, high_ab) => (0, 2 ^ precision)
low_ab = -100
high_ab = 100
interval_length = high_ab - low_ab + 1
end_range = 2 ** precision
scale_factor = interval_length / end_range
to_draw_x = [0, 100]
nr_epoch = 1000
err = 1e-9
print_updates = False
update = np.array([0, 0, 0]).reshape((3, 1))


def prinT(delim):
    plt.clf()
    plt.scatter(positive[:, 0], positive[:, 1])
    plt.scatter(negative[:, 0], negative[:, 1])
    if abs(delim[1]) <= err:
        plt.plot((-delim[2] / delim[0], -delim[2] / delim[0]), (0, 100),
                 marker='o')
    else:
        plt.plot(to_draw_x, (to_draw_x * delim[0] + delim[2]) / (-delim[1]),
                 marker='o')
    plt.show()


def get_result():
    global update
    delimiter = np.random.randint(low=low_ab, high=high_ab, size=(3, 1)).astype(np.float)
    delimiter[2] = np.random.randint(0, 101, 1)
    # updating the i bit in the representation
    # (nr - low_ab) / interval_length * end_range => 0, (2 ** precision)
    # (nr - low_ab) / interval_length * end_range +- 2 ** i => 0, 2 ** i
    # ((nr - low_ab) / interval_length * end_range +- 2 ** i) * interval_length / end_range + low_ab
    # => (low_ab, high_ab)
    # nr +- (2 ** i) * interval_length / end_range
    # fin: nr +- (2 ** i) * scale_factor?

    # a > 0, ax + by + c > 0 => dreapta (cele cu 1)
    # a < 0, ax + by + c > 0 => stanga (cele cu 1)
    # => cele din dreapta (cu 1 definite) :
    # a * (ax + by + c) > 0

    is_updated = True
    while is_updated:
        is_updated = False
        positive_correct = np.count_nonzero((positive.dot(delimiter) * delimiter[0]) > 0)
        negative_correct = np.count_nonzero((negative.dot(delimiter) * delimiter[0]) <= 0)
        if positive_correct + negative_correct == 100:
            break
        for ia in range(-1, precision):
            if is_updated:
                break
            if ia == -1:
                ia = -100
            update[0] = (2 ** ia) * scale_factor
            for ib in range(-1, precision):
                if is_updated:
                    break
                if ib == -1:
                    ib = -100
                update[1] = (2 ** ib) * scale_factor
                for sign in range(0, 4):
                    update[0] *= (-1) ** (sign % 2)
                    update[1] *= (-1) ** ((sign // 2) % 2)
                    delimiter += update
                    next_positive_correct = np.count_nonzero((positive.dot(delimiter) * delimiter[0]) > 0)
                    next_negative_correct = np.count_nonzero((negative.dot(delimiter) * delimiter[0]) <= 0)
                    if next_positive_correct + next_negative_correct > positive_correct + negative_correct:
                        is_updated = True
                        break
                    else:
                        delimiter -= update
                    update[0] *= (-1) ** (sign % 2)
                    update[1] *= (-1) ** ((sign // 2) % 2)

    positive_correct = np.count_nonzero((positive.dot(delimiter) * delimiter[0]) > 0)
    negative_correct = np.count_nonzero((negative.dot(delimiter) * delimiter[0]) <= 0)
    return delimiter, positive_correct + negative_correct


start = time.time()
final_delimiter, final_result = get_result()

for i in range(nr_epoch - 1):
    print("Epoch:", i + 2)
    delimiter, result = get_result()
    if result > final_result:
        final_delimiter = delimiter
        final_result = result
        print("update:", result)
        if print_updates:
            prinT(final_delimiter)
    if result == positive.shape[0] + negative.shape[0]:
        print((time.time() - start) / (i + 2))
        break
    #

print("Final delimiter :", final_delimiter)
print("Time :", time.time() - start)
prinT(final_delimiter)

