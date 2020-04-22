import matplotlib.pyplot as plt
import numpy as np

f = open("bandwidth-schedule.txt", "r")

count = 0

schedule = ["Dynamic", "Static", "Guided"]
bandwidth_mean = []
error_bars = []
bandwidth = []

all_bandwidths = []

for line in f:

    count +=1
    bandwidth.append(float(line.strip()))

    if count ==5:
        all_bandwidths.append(bandwidth)
    #    mean = np.mean(np.array(bandwidth))
    #    bandwidth_mean.append(mean)

    #    bndwidth_std = np.std(np.array(bandwidth))
    #    error_bars.append(bndwidth_std)

        bandwidth = []
        count = 0
plt.boxplot(all_bandwidths, labels=schedule)
#plt.scatter(schedule, bandwidth_mean)
#plt.errorbar(schedule, bandwidth_mean, yerr=error_bars, linestyle = "None")
plt.xlabel("Type of schedule")
plt.ylabel("Bandwidth, MB/s")
#plt.ylim(30000,55000)
plt.show()
