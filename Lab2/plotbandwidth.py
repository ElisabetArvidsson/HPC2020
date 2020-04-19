import matplotlib.pyplot as plt


f = open("bandwidth.txt", "r")

count = 0
sum_band = 0

threads = [1,2,4,8,12,16,20,24,28,32]
bandwidth = []


for line in f:

    count+=1
    sum_band+=float(line.strip())
    print(count)
    if count ==5:
        mean = sum_band/5
        bandwidth.append(mean)
        sum_band = 0
        count = 0

plt.plot(threads, bandwidth)
plt.xlabel("Number of threads")
plt.ylabel("Bandwidth, MB/s")
plt.show()
