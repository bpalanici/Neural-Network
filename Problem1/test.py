import numpy as np
import matplotlib.pyplot as plt

delimiter = np.random.randint(low=-101, high=101, size=(3, 1)).astype(np.float)
delimiter[2] = 50

for i in range(1000):
    plt.clf()
    delimiter[0] -= i
    plt.plot([0, 100], [-delimiter[2] / delimiter[1], -100 * delimiter[0] - delimiter[2] / delimiter[1]],
             marker='o')
    plt.show()
    delimiter[0] += i
