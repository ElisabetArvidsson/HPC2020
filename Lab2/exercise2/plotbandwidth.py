import matplotlib.pyplot as plt
import numpy as np

f = open("bandwidth-beskow.txt", "r")

count = 0
sum_band = 0

threads = [1,2,4,8,12,16,20,24,28,32]
bandwidth = []
error_bars = []
bandwidth_threads = []

for line in f:

    count+=1
    print(line.strip())
    bandwidth_threads.append(float(line.strip()))

    if count ==5:
        mean = np.mean(np.array(bandwidth_threads))
        bandwidth.append(mean)

        bndwidth_std = np.std(np.array(bandwidth_threads))
        error_bars.append(bndwidth_std)

        bandwidth_threads = []
        count = 0

plt.plot(threads, bandwidth)
plt.errorbar(threads, bandwidth, yerr=error_bars, capthick=1, capsize=3, ecolor="k")
plt.xlabel("Number of threads")
plt.ylabel("Bandwidth, MB/s")
plt.show()
