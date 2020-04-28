import matplotlib.pyplot as plt
import numpy as np

file = np.load('doubletrouble_alpha_1.npz')
x_test = file['data_test']
k = file['k']
rf = file['rf']

for i in range(x_test.shape[0]):
    plt.plot(rf, 1-x_test[i], label='K = {}'.format(k[i]), linewidth=2)
plt.title('Ensembling: Double trouble on MNIST', fontsize=16)
plt.grid()
plt.ylabel('Test Error', fontsize='15')
plt.xlabel('K', fontsize='15')
plt.xlim(rf[-1], rf[1])
plt.xticks(rf)
plt.legend()
plt.show()