import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 3, 1])
y = np.array([1, 2, 2, 1, 1])
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.plot(x, y, color='g', linewidth=1, marker='o', markersize=6)

plt.title("Zadatak 1")
plt.xlabel("x os")
plt.ylabel("y os")

plt.grid(True)

plt.show()
