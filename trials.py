import numpy as np
import matplotlib.pyplot as plt

global_step = range(1000)
clr = []
for i in global_step:
    step_size = 100
    learning_rate = 0.0001
    max_lr = 0.0005
    counter = 1 + i / (2 * step_size)
    cycle = np.floor(counter)
    x = abs((i / step_size) - 2 * cycle + 1)

    if counter == (cycle + 0.5):
        c_lr = -(learning_rate + (max_lr - learning_rate) * max(0, 1 - x))
    else:
        c_lr = learning_rate + (max_lr - learning_rate) * max(0, 1 - x)
    clr.append(c_lr)
print(np.mean(clr))
plt.plot(global_step, clr)
plt.show()
