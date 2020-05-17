import matplotlib.pyplot as plt
import numpy as np


def plotRuntime(filename):
    f = open(filename, "r")

    runtime = []
    threads = [8,16,32,64,128]
    runtime_threads = []
    count = 0

    error_bars = []

    for line in f:

        count+=1
        runtime_threads.append(float(line.strip()))

        if count ==5:
            mean = np.mean(np.array(runtime_threads))
            runtime.append(mean)

            time_std = np.std(np.array(runtime_threads))
            error_bars.append(time_std)

            runtime_threads = []
            count = 0

    speedup = np.divide(runtime[0], runtime)
    dspeedup = np.multiply(np.divide(-1, np.power(runtime,2)), error_bars)
    plt.subplot(2,1,1)
    plt.plot(threads, runtime)
    plt.errorbar(threads, runtime, yerr=dspeedup, capthick=1, capsize=3, ecolor="k")
    plt.ylabel("Time, s")
    plt.subplot(2,1,2)
    plt.errorbar(threads, speedup, yerr=dspeedup, capthick=1, capsize=3, ecolor="k")
    plt.ylabel("Speedup")
    plt.xlabel("Threads")



if __name__ == "__main__":
    plotRuntime("runtime_binary.txt")
    plotRuntime("runtime_linear.txt")
    plt.show()
