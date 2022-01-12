import matplotlib.pyplot as plt
import numpy as np

data = np.load('losses-1.npy')

x, y = zip(*data)

plt.title("Training Losses Over 1.5^6 Frames")
plt.xlabel("Frames")
plt.ylabel("Loss")
plt.scatter(x,y)

plt.show()

