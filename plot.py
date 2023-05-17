import matplotlib.pyplot as plt
import numpy as np

def get_data(fname):
    data = np.loadtxt(fname, delimiter=",")
    data[:, 1:] = data[:, 0, None] / (data[:, 1:] / 1000)
    data[:, 0] *= 1000000
    return data

fig, axes = plt.subplots(1, 2, figsize=[15, 7])
fnames = ["result_32.csv", "result_64.csv"]
titles = ["Sort 32-bit key-value pairs", "Sort 64-bit key-value pairs"]
for i in range(2):
    data = get_data(fnames[i])
    axes[i].grid(which="major")
    axes[i].set_xscale("log")
    axes[i].plot(data[:, 0], data[:, 1], data[:, 0], data[:, 2], data[:, 0], data[:, 3])
    axes[i].legend(["CUB merge sort", "CUB radix sort", "Bitonic sort"])
    axes[i].set_xlabel("Number of elements")
    axes[i].set_ylabel("Sort rate (million elements / second)")
    axes[i].set_title(titles[i])
    
fig.suptitle("Performance comparison of different sorting algorithms")
plt.tight_layout()
plt.savefig("benchmark.png")