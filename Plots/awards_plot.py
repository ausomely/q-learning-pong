import matplotlib.pyplot as plt
import numpy as np

data = np.load('awards-1.npy')
frames, awards = zip(*data)



plt.title("Training Awards Over 1.5^6 Frames")
plt.xlabel("Frames")
plt.ylabel("Awards")



plt.scatter(frames, awards)

plt.show()

