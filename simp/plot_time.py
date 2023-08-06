# %%
import numpy as np
import matplotlib.pyplot as plt

time = np.loadtxt('time.txt')

print(time)

n = [i for i in range(1, 13)]

plt.figure()
plt.plot(n, time, 'o-')
plt.xlabel('Number of processors')
plt.ylabel('Time (s)')
plt.title('Time vs Number processors (30 samples)')
plt.savefig('time.png')

speedup = time / time[0]

plt.figure()
plt.plot(n, speedup, 'o-')
plt.xlabel('Number of processors')
plt.ylabel('Time (s)')
plt.title('Time vs Number processors (30 samples)')
plt.savefig('time.png')