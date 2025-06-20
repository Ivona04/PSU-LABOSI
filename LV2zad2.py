import numpy as np
import matplotlib.pyplot as plt

podaci = np.loadtxt(open("mtcars.csv","rb"), usecols=(1,2,3,4,5,6),
                    delimiter=",", skiprows=1)

print("Minimalna potrošnja (mpg):", min(podaci[:, 0]))
print("Maksimalna potrošnja (mpg):", max(podaci[:, 0]))
print("Prosječna potrošnja (mpg):", sum(podaci[:, 0]) / len(podaci[:, 0]))

filter_sest_cilindara = podaci[:, 1] == 6

plt.scatter(podaci[:, 0], podaci[:, 3], c='lime', edgecolor='black', s=podaci[:, 5]*16, marker="h")

for i, oznaka in enumerate(podaci[:, 5]):
    plt.text(podaci[i, 0], podaci[i, 3] + 5, str(podaci[i, 5]))

print("Minimalna potrošnja za 6 cilindara:", min(podaci[filter_sest_cilindara, 0]))
print("Maksimalna potrošnja za 6 cilindara:", max(podaci[filter_sest_cilindara, 0]))
print("Prosječna potrošnja za 6 cilindara:", sum(podaci[filter_sest_cilindara, 0]) / len(podaci[filter_sest_cilindara, 0]))

plt.show()
