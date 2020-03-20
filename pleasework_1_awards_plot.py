import matplotlib.pyplot as plt
import numpy as np

data = np.load('pleasework_award_1.npy')
frames, awards = zip(*data)



plt.title("Training Awards")
plt.xlabel("Frames")
plt.ylabel("Awards")


plt.plot(frames,awards,linewidth = 2.0)
# plt.scatter(frames, awards)
plt.show()

