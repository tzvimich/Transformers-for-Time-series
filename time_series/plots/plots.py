import numpy as np
import matplotlib.pyplot as plt

# 30 2 (1 head) zero initialization with PE and with reconnecting with norm - train 36.4 test 44.6
# 15 4 (1 head) zero initialization with PE and with reconnecting with norm - train 35.11 test 44.33
# 12 5 (1 head) zero initialization with PE and with reconnecting with norm - train 34.51 test 43.98
# 10 6 (1 head) zero initialization with PE and with reconnecting with norm - train 37.61 test 46.01
# 6 12 (1 head) zero initialization with PE and with reconnecting with norm - train 38.66 test 46.52

x = [30, 15, 12, 10, 6]
y = [44.6, 44.33, 43.98, 46.01, 46.52]

# plt.subplot(1, 2, 1)
# plt.plot(x, y)
# plt.title('Loss as function of token size')
# plt.ylabel('Test Loss')
# plt.xlabel('Input token size')

x = [32, 64, 128, 256]
y = [38.2, 33.7, 49.8, 61.3]

# plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title('Loss as function of embedding size')
plt.ylabel('Test Loss')
plt.xlabel('Embedding size')

plt.savefig('Influence of embedding size on Loss.png')

plt.show()