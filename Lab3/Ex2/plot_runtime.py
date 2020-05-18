import matplotlib.pyplot as plt
import numpy as np


def plotRuntime(filename, color, axs, name):
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
   # plt.subplot(2,1,1)
   # plt.plot(threads, runtime, c=color)
    axs[0].errorbar(threads, runtime, yerr=dspeedup, capthick=1, capsize=3, ecolor="k", c=color, label = name)
   # plt.ylabel("Time, s")
   # plt.subplot(2,1,2)
    axs[1].errorbar(threads, speedup, yerr=dspeedup, capthick=1, capsize=3, ecolor="k", c = color, label = name)
    #plt.ylabel("Speedup")
    #plt.xlabel("Threads")



if __name__ == "__main__":
    fig, axs = plt.subplots(2, 1)
    axs[0].set_ylabel("Time, s")
    axs[1].set_ylabel("Speedup")
    axs[1].set_xlabel("Threads")
    plotRuntime("runtime_binary.txt", "b", axs, "Binary")
    plotRuntime("runtime_linear.txt", "r", axs, "Linear")
    plotRuntime("runtime_nonblocking.txt", "g", axs, "Nonblocking")
    axs[0].legend(loc="upper right")
    axs[1].legend(loc="lower right")
    plt.show()
