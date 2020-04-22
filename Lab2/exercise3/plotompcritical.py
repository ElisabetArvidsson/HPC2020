import matplotlib.pyplot as plt
import numpy as np

def plotGraph(f, name):
    count = 0
    sum_band = 0

    threads = [1,2,4,8,12,16,20,24,28,32]

    all_mean = []
    error_bars = []
    time = []
    first_val = 0;

    for line in f:

        count+=1
        time.append(float(line.strip()))

        if count ==5:
            mean = np.mean(np.array(time))
            all_mean.append(mean)

            time_std = np.std(np.array(time))
            error_bars.append(time_std)

            time = []
            count = 0

    all_mean = np.divide(all_mean[0], np.array(all_mean))
#    plt.title(name)
   # plt.plot(threads, all_mean, label = name)
    plt.errorbar(threads, all_mean, capsize=3, capthick=1,yerr=error_bars, label=name)

critical = open("ompcritical.txt", "r")
wo_critical = open("parallel-without-critical.txt", "r")
para_array = open("parallel-array.txt", "r")
padding = open("parallel-padding.txt", "r")

plotGraph(critical, "Parallel with omp critical")
plotGraph(wo_critical, "Incorrect parallel")
plotGraph(para_array, "Parallel with array")
plotGraph(padding, "Parallel with padding")
plt.xlabel("Number of threads")
plt.ylabel("Speed up")
plt.legend()
plt.show()
